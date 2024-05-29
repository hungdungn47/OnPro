from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
from utils.rotation_transform import Rotation
from utils import my_transform as TL
from losses.loss import Supervised_NT_xent_n, Supervised_NT_xent_uni
from copy import deepcopy
from modules.OPE import OPELoss
from modules.APF import AdaptivePrototypicalFeedback
from utils.util import Memory
from utils.util import MySampler
from torch.utils.data import DataLoader
from experiment.dataset import ModifyTensorDataset
from utils.util import *

device = "cuda" if torch.cuda.is_available() else "cpu"

pdist = torch.nn.PairwiseDistance(p=2).cuda()


class TrainLearner(object):
    def __init__(self, model, buffer, optimizer, n_classes_num, class_per_task, input_size, args, fea_dim=128):
        print("Use CLOOD")
        self.model = model
        self.optimizer = optimizer
        self.oop_base = n_classes_num
        self.oop = args.oop
        self.n_classes_num = n_classes_num
        self.fea_dim = fea_dim
        self.classes_mean = torch.zeros((n_classes_num, fea_dim), requires_grad=False).cuda()
        self.class_per_task = class_per_task
        self.class_holder = []
        self.mixup_base_rate = args.mixup_base_rate
        self.ins_t = args.ins_t
        self.proto_t = args.proto_t

        self.buffer = buffer
        self.buffer_batch_size = args.buffer_batch_size
        self.buffer_per_class = 7

        self.clood_buffer_dataset = Memory(args.clood_buffer_size)
        self.clood_buffer = None
        self.clood_optimizer = None

        self.OPELoss = OPELoss(self.class_per_task, temperature=self.proto_t)

        self.dataset = args.dataset
        if args.dataset == "cifar10":
            self.sim_lambda = 0.5
            self.total_samples = 10000
        elif "cifar100" in args.dataset:
            self.sim_lambda = 1.0
            self.total_samples = 5000
        elif args.dataset == "tiny_imagenet":
            self.sim_lambda = 1.0
            self.total_samples = 10000
        elif args.dataset == "colorectal":
            self.sim_lambda = 0.5
            self.total_samples = 1000
        elif args.dataset == "ham10000":
            self.sim_lambda = 0.5
            self.total_samples = 1000
        self.print_num = self.total_samples // 10

        hflip = TL.HorizontalFlipLayer().cuda()
        with torch.no_grad():
            resize_scale = (0.3, 1.0)
            color_gray = TL.RandomColorGrayLayer(p=0.25).cuda()
            resize_crop = TL.RandomResizedCropLayer(scale=resize_scale,
                                                    size=[input_size[1], input_size[2], input_size[0]]).cuda()
            self.transform = torch.nn.Sequential(
                hflip,
                color_gray,
                resize_crop)

        self.APF = AdaptivePrototypicalFeedback(self.buffer, args.mixup_base_rate, args.mixup_p, args.mixup_lower,
                                                args.mixup_upper,
                                                args.mixup_alpha, self.class_per_task)

        self.scaler = GradScaler()

        self.args = args
        print("Num classes: ", self.n_classes_num)

        self.args.mean, self.args.cov, self.args.cov_inv = {}, {}, {}
        self.output_list, self.feature_list, self.label_list, self.pred_list = [], [], [], []

    def preprocess_task(self, train_loader):
        # Add new embeddings for HAT
        self.model.append_wp_head()

        # Prepare memory loader if memory data exist
        if len(self.clood_buffer_dataset.data) > 0:
            self.sampler = MySampler(
                len(self.clood_buffer_dataset),
                len(train_loader.dataset),
                self.args.batch_size,
                1,
            )
            # We don't use minibatch. Use upsampling.
            self.clood_buffer = DataLoader(self.clood_buffer_dataset,
                                           batch_size=self.args.batch_size,
                                           sampler=self.sampler,
                                           num_workers=self.args.n_workers,
                                           pin_memory=True)
            self.clood_buffer_iter = iter(self.clood_buffer)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, betas=(0.9, 0.99),
                                          weight_decay=1e-4)

    def clood_observe(self, inputs, labels, task_id):

        n_samples = len(inputs)
        normalized_labels = labels % self.class_per_task

        if self.clood_buffer:
            inputs_bf, labels_bf = next(self.clood_buffer_iter)
            inputs_bf = inputs_bf.to(device)

            labels_bf = torch.zeros_like(labels_bf).to(device) + self.class_per_task
            normalized_labels_bf = labels_bf
            inputs = torch.cat([inputs, inputs_bf])
            labels = torch.cat([labels, labels_bf])
            normalized_labels = torch.cat([normalized_labels, normalized_labels_bf])

        with torch.cuda.amp.autocast(enabled=True):
            features = self.model.forward_features(self.transform(inputs))
            outputs = self.model.forward_classifier(features, task_id)
            loss = F.cross_entropy(outputs, normalized_labels)

        return loss

    def train_task0(self, task_id, train_loader, task_loader):

        self.preprocess_task(train_loader)

        for epoch in range(self.args.num_epochs):
            loss_save = []
            num_d = 0
            for batch_idx, (x, y) in enumerate(train_loader):
                num_d += x.shape[0]

                Y = deepcopy(y)
                for j in range(len(Y)):
                    if Y[j] not in self.class_holder:
                        self.class_holder.append(Y[j].detach())

                with autocast():
                    x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                    x = x.requires_grad_()

                    rot_x = Rotation(x)
                    rot_x_aug = self.transform(rot_x)
                    images_pair = torch.cat([rot_x, rot_x_aug], dim=0)

                    rot_sim_labels = torch.cat([y + self.oop_base * i for i in range(self.oop)], dim=0)

                    features, projections = self.model(images_pair, use_proj=True)
                    projections = F.normalize(projections)

                    # instance-wise contrastive loss in OCM
                    features = F.normalize(features)
                    dim_diff = features.shape[1] - projections.shape[1]  # 512 - 128
                    dim_begin = torch.randperm(dim_diff)[0]
                    dim_len = projections.shape[1]

                    sim_matrix = torch.matmul(projections, features[:, dim_begin:dim_begin + dim_len].t())
                    sim_matrix += torch.mm(projections, projections.t())

                    ins_loss = Supervised_NT_xent_n(sim_matrix, labels=rot_sim_labels, temperature=self.ins_t)

                    if batch_idx != 0:
                        buffer_x, buffer_y = self.sample_from_buffer_for_prototypes()
                        buffer_x.requires_grad = True
                        buffer_x, buffer_y = buffer_x.cuda(), buffer_y.cuda()
                        buffer_x_pair = torch.cat([buffer_x, self.transform(buffer_x)], dim=0)

                        proto_seen_loss, _, _, _ = self.cal_buffer_proto_loss(buffer_x, buffer_y, buffer_x_pair,
                                                                              task_id)
                    else:
                        proto_seen_loss = 0

                    z = projections[:rot_x.shape[0]]
                    zt = projections[rot_x.shape[0]:]
                    proto_new_loss, cur_new_proto_z, cur_new_proto_zt = self.OPELoss(z[:x.shape[0]], zt[:x.shape[0]], y,
                                                                                     task_id, True)

                    OPE_loss = proto_new_loss + proto_seen_loss

                    y_pred = self.model.forward_features(self.transform(x))
                    y_pred = self.model.forward_classifier(y_pred, task_id)
                    ce = F.cross_entropy(y_pred, y)

                    # clood_loss = self.clood_observe(x, y, task_id)

                    loss = ce + ins_loss + OPE_loss

                    loss_save.append(loss.item())

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.buffer.add_reservoir(x=x.detach(), y=y.detach(), logits=None, t=task_id)

                if num_d % self.print_num == 0 or batch_idx == 1:
                    print(
                        '==>>> it: {}, loss: ce {:.2f} + ins {:.4f} + OPE {:.4f} = {:.6f}, {}%'
                        .format(batch_idx, ce, ins_loss, OPE_loss, loss, 100 * (num_d / self.total_samples)))

            print("Epoch: {}, loss: {:.6f}".format(epoch, np.array(loss_save).mean()))
            # self.test(task_id, task_loader)

    def train_other_tasks(self, task_id, train_loader, task_loader):

        self.preprocess_task(train_loader)

        for epoch in range(self.args.num_epochs):
            num_d = 0
            loss_save = []
            for batch_idx, (x, y) in enumerate(train_loader):
                num_d += x.shape[0]

                Y = deepcopy(y)
                for j in range(len(Y)):
                    if Y[j] not in self.class_holder:
                        self.class_holder.append(Y[j].detach())

                with autocast():
                    x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                    x = x.requires_grad_()
                    buffer_batch_size = min(self.buffer_batch_size, self.buffer_per_class * len(self.class_holder))

                    ori_mem_x, ori_mem_y, bt = self.buffer.sample(buffer_batch_size, exclude_task=None)
                    if batch_idx != 0:
                        mem_x, mem_y, mem_y_mix = self.APF(ori_mem_x, ori_mem_y, buffer_batch_size, self.classes_mean,
                                                           task_id)
                        rot_sim_labels = torch.cat([y + self.oop_base * i for i in range(self.oop)], dim=0)
                        rot_sim_labels_r = torch.cat([mem_y + self.oop_base * i for i in range(self.oop)], dim=0)
                        rot_mem_y_mix = torch.zeros(rot_sim_labels_r.shape[0], 3).cuda()
                        rot_mem_y_mix[:, 0] = torch.cat([mem_y_mix[:, 0] + self.oop_base * i for i in range(self.oop)],
                                                        dim=0)
                        rot_mem_y_mix[:, 1] = torch.cat([mem_y_mix[:, 1] + self.oop_base * i for i in range(self.oop)],
                                                        dim=0)
                        rot_mem_y_mix[:, 2] = mem_y_mix[:, 2].repeat(self.oop)
                    else:
                        mem_x = ori_mem_x
                        mem_y = ori_mem_y

                        rot_sim_labels = torch.cat([y + self.oop_base * i for i in range(self.oop)], dim=0)
                        rot_sim_labels_r = torch.cat([mem_y + self.oop_base * i for i in range(self.oop)], dim=0)

                    mem_x = mem_x.requires_grad_()

                    rot_x = Rotation(x)
                    rot_x_r = Rotation(mem_x)
                    rot_x_aug = self.transform(rot_x)
                    rot_x_r_aug = self.transform(rot_x_r)
                    images_pair = torch.cat([rot_x, rot_x_aug], dim=0)
                    images_pair_r = torch.cat([rot_x_r, rot_x_r_aug], dim=0)

                    all_images = torch.cat((images_pair, images_pair_r), dim=0)

                    features, projections = self.model(all_images, use_proj=True)

                    projections_x = projections[:images_pair.shape[0]]
                    projections_x_r = projections[images_pair.shape[0]:]

                    projections_x = F.normalize(projections_x)
                    projections_x_r = F.normalize(projections_x_r)

                    # instance-wise contrastive loss in OCM
                    features_x = F.normalize(features[:images_pair.shape[0]])
                    features_x_r = F.normalize(features[images_pair.shape[0]:])

                    dim_diff = features_x.shape[1] - projections_x.shape[1]
                    dim_begin = torch.randperm(dim_diff)[0]
                    dim_begin_r = torch.randperm(dim_diff)[0]
                    dim_len = projections_x.shape[1]

                    sim_matrix = self.sim_lambda * torch.matmul(projections_x,
                                                                features_x[:, dim_begin:dim_begin + dim_len].t())
                    sim_matrix_r = self.sim_lambda * torch.matmul(projections_x_r,
                                                                  features_x_r[:,
                                                                  dim_begin_r:dim_begin_r + dim_len].t())

                    sim_matrix += self.sim_lambda * torch.mm(projections_x, projections_x.t())
                    sim_matrix_r += self.sim_lambda * torch.mm(projections_x_r, projections_x_r.t())

                    loss_sim_r = Supervised_NT_xent_uni(sim_matrix_r, labels=rot_sim_labels_r, temperature=self.ins_t)
                    loss_sim = Supervised_NT_xent_n(sim_matrix, labels=rot_sim_labels, temperature=self.ins_t)

                    ins_loss = loss_sim_r + loss_sim

                    # y_pred = self.model(self.transform(mem_x))

                    y_pred = self.model.forward_features(self.transform(mem_x))
                    y_pred = self.model.forward_classifier(y_pred, task_id)

                    buffer_x = ori_mem_x
                    buffer_y = ori_mem_y
                    buffer_x_pair = torch.cat([buffer_x, self.transform(buffer_x)], dim=0)
                    proto_seen_loss, cur_buffer_z1_proto, cur_buffer_z2_proto, cur_buffer_z = self.cal_buffer_proto_loss(
                        buffer_x, buffer_y, buffer_x_pair, task_id)

                    z = projections_x[:rot_x.shape[0]]
                    zt = projections_x[rot_x.shape[0]:]
                    proto_new_loss, cur_new_proto_z, cur_new_proto_zt = self.OPELoss(z[:x.shape[0]], zt[:x.shape[0]], y,
                                                                                     task_id, True)

                    OPE_loss = proto_new_loss + proto_seen_loss

                    if batch_idx != 0:
                        ce = self.loss_mixup(y_pred, mem_y_mix)
                    else:
                        ce = F.cross_entropy(y_pred, mem_y)

                    # clood_loss = self.clood_observe(x, y, task_id)

                    loss = ce + ins_loss + OPE_loss

                    loss_save.append(loss.item())

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.buffer.add_reservoir(x=x.detach(), y=y.detach(), logits=None, t=task_id)

                if num_d % self.print_num == 0 or batch_idx == 1:
                    print(
                        '==>>> it: {}, loss: ce {:.2f} + ins {:.4f} + OPE {:.4f} = {:.6f}, {}%'
                        .format(batch_idx, ce, ins_loss, OPE_loss, loss, 100 * (num_d / self.total_samples)))

            print("Epoch: {}, loss: {:.6f}".format(epoch, np.array(loss_save).mean()))
            # self.test(task_id, task_loader)

    def clood_observe_wp(self, inputs, labels, task_id):

        # normalized_labels = labels % self.class_per_task
        normalized_labels = labels

        with torch.no_grad():
            features = self.model.forward_features(self.transform(inputs))
        outputs = self.model.forward_classifier(features, task_id)
        loss = F.cross_entropy(outputs, normalized_labels)

        self.clood_optimizer.zero_grad()
        loss.backward()
        self.clood_optimizer.step()

        return loss

    def clood_finetune_wp(self, task_id, train_loader):
        print("Fine-tunning WP head")
        self.clood_optimizer = torch.optim.Adam(self.model.wp_head[task_id].parameters(), lr=self.args.lr,
                                                betas=(0.9, 0.99), weight_decay=1e-4)
        for epoch in range(self.args.finetune_wp_epochs):
            loss_save = []
            num_d = 0
            for batch_idx, (x, y) in enumerate(train_loader):
                num_d += x.shape[0]
                x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                x = x.requires_grad_()
                loss = self.clood_observe_wp(x, y, task_id)
                loss_save.append(loss.item())

                if num_d % self.print_num == 0 or batch_idx == 1:
                    print('==>>> it: {}, loss = {:.6f}, {}%'
                          .format(batch_idx, loss, 100 * (num_d / self.total_samples)))

            print("Epoch: {}, loss: {:.6f}".format(epoch, np.array(loss_save).mean()))

    def preprocess_finetune_ood(self, task_id, train_loader):
        if len(self.model.tp_head) < task_id + 1:
            self.model.append_tp_head()

        self.clood_optimizer = torch.optim.Adam(
            self.model.tp_head[task_id].parameters(),
            lr=self.args.lr, betas=(0.9, 0.99), weight_decay=1e-4
        )

        self.sampler = MySampler(
            len(self.clood_buffer_dataset),
            len(train_loader.dataset),
            self.args.batch_size,
            self.args.finetune_ood_epochs
        )

        self.clood_buffer = DataLoader(
            self.clood_buffer_dataset,
            batch_size=self.args.batch_size,
            sampler=self.sampler,
            num_workers=self.args.n_workers,
            pin_memory=True
        )
        self.clood_buffer_iter = iter(self.clood_buffer)

    def clood_observe_ood(self, inputs, labels, task_id):
        # normalized_labels = labels % self.class_per_task
        normalized_labels = labels

        inputs_bf, labels_bf = next(self.clood_buffer_iter)
        inputs_bf = inputs_bf.to(device)
        labels_bf = labels_bf.to(device)

        # print(task_id, labels, labels_bf)

        # labels_bf_ood = torch.zeros_like(labels_bf) + self.class_per_task
        labels_bf_ood = torch.zeros_like(labels_bf) + self.n_classes_num
        inputs = torch.cat([inputs, inputs_bf])
        normalized_labels = torch.cat([normalized_labels, labels_bf])

        with torch.no_grad():
            features = self.model.forward_features(self.transform(inputs))
        outputs = self.model.forward_tp_head(features, task_id)

        loss = F.cross_entropy(outputs, normalized_labels)

        self.clood_optimizer.zero_grad()
        loss.backward()
        self.clood_optimizer.step()

        return loss

    def clood_finetune_ood(self, task_id, last_task_id=None, train_loader=None):
        print(f"Fine-tuning an OOD head task={task_id}")
        self.preprocess_finetune_ood(task_id, train_loader)

        num_d = 0
        pos_print = self.args.finetune_ood_epochs // 10
        for epoch in range(self.args.finetune_ood_epochs):
            loss_save = []
            for batch_idx, (x, y) in enumerate(train_loader):
                num_d += x.shape[0]
                x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                x = x.requires_grad_()

                loss = self.clood_observe_ood(x, y, task_id=task_id)
                loss_save.append(loss.item())

                # if ((epoch + 1) % pos_print == 0 or epoch == 0) and batch_idx == 1:
                #     print('==>>> epoch: {}, loss = {:.6f}'
                #           .format(epoch, loss))
            if ((epoch + 1) % pos_print == 0 or epoch == 0):
                print("Epoch: {}, loss: {:.6f}".format(epoch, np.array(loss_save).mean()))

    def clood_finetune_all_ood(self, task_id, train_loader):
        sample_per_cls = Counter(self.clood_buffer_dataset.targets)
        print("Number of samples per class: ", sample_per_cls)
        sample_per_cls = sample_per_cls[0]

        current_loader_copy = deepcopy(train_loader)
        buffer_copy = deepcopy(self.clood_buffer_dataset)

        data = np.concatenate([train_loader.dataset.data, self.clood_buffer_dataset.data])
        targets = np.concatenate([train_loader.dataset.targets, self.clood_buffer_dataset.targets])

        for p_task_id in range(task_id + 1):
            ind = set(np.arange(self.class_per_task[p_task_id], self.class_per_task[p_task_id + 1]))

            ind_list, ood_list = [], []
            for y in range(self.class_per_task[task_id + 1]):
                idx = np.where(targets == y)[0]
                np.random.shuffle(idx)
                idx = idx[:sample_per_cls]

                if y in ind:
                    ind_list.append(idx)
                else:
                    ood_list.append(idx)

            # train_loader.dataset.targets = targets[np.concatenate(ind_list)]
            # train_loader.dataset.data = data[np.concatenate(ind_list)]
            dataset_new_train = ModifyTensorDataset(torch.from_numpy(data[np.concatenate(ind_list)]),
                                                    torch.from_numpy(targets[np.concatenate(ind_list)]))
            train_loader = torch.utils.data.DataLoader(
                dataset_new_train,
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=self.args.n_workers,
            )
            self.clood_buffer_dataset.data = data[np.concatenate(ood_list)]
            self.clood_buffer_dataset.targets = targets[np.concatenate(ood_list)]

            self.clood_finetune_ood(p_task_id, task_id, train_loader)
            if p_task_id != task_id:
                self.clood_finetune_wp(p_task_id, train_loader)

        train_loader = current_loader_copy
        self.clood_buffer_dataset = buffer_copy

    def end_task(self, task_id, train_loader):
        self.model.eval()
        ys = list(sorted(set(train_loader.dataset.targets)))
        print("End task: ", task_id, ", Labels: ", ys)
        self.clood_buffer_dataset.update(train_loader.dataset)
        self.output_list, self.feature_list, self.label_list, self.pred_list = [], [], [], []

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.cuda(), target.cuda()
                self.label_list.append(target.data.cpu())
                features = self.model.forward_features(data)
                self.feature_list.append(features.data.cpu().numpy())

        self.feature_list = np.concatenate(self.feature_list)
        self.label_list = np.concatenate(self.label_list)
        cov_list = []
        ys = list(sorted(set(self.label_list)))
        print("Task: ", task_id, ", Labels: ", ys)
        for y in ys:
            idx = np.where(self.label_list == y)[0]
            f = self.feature_list[idx]

            mean = np.mean(f, 0)
            self.args.mean[y] = mean
            # np.save(
            #     os.path.join(self.args.logger.dir(), f'{self.args.mean_label_name}_{y}'),
            #     mean
            # )

            cov = np.cov(f.T)
            cov_list.append(cov)
        cov = np.array(cov_list).mean(0)
        self.args.cov[task_id] = cov
        self.args.cov_inv[task_id] = np.linalg.inv(0.8 * cov + 0.2 * np.eye(len(cov)))

    def train(self, task_id, train_loader, task_loader):
        self.model.train()
        for epoch in range(1):
            if task_id == 0:
                self.train_task0(task_id, train_loader, task_loader)
                self.clood_finetune_wp(task_id, train_loader)
            else:
                self.train_other_tasks(task_id, train_loader, task_loader)
                self.clood_finetune_wp(task_id, train_loader)
                self.clood_finetune_all_ood(task_id, train_loader)

            self.end_task(task_id, train_loader)

    def test(self, i, task_loader):
        print("Test with Temperature: ", self.args.T)
        if self.args.use_md:
            print("Test with MD")
        self.model.eval()
        with torch.no_grad():
            acc_list = np.zeros(len(task_loader))
            for j in range(i + 1):
                if self.args.til:
                    acc = self.test_model(task_loader[j]['test'], j)
                else:
                    acc = self.test_model_with_clood(task_loader[j]['test'], j, i)
                acc_list[j] = acc.item()

            print(f"tasks acc:{acc_list}")
            print(f"tasks avg acc:{acc_list[:i + 1].mean()}")

        return acc_list

    def test_on_train_set(self, i, task_loader):
        print('=' * 50)
        print("Test on Train set")
        print("Test with Temperature: ", self.args.T)
        if self.args.use_md:
            print("Test with MD")
        print("")

        self.model.eval()
        with torch.no_grad():
            acc_list = np.zeros(len(task_loader))
            for j in range(i + 1):
                if self.args.til:
                    acc = self.test_model(task_loader[j]['train'], j)
                else:
                    acc = self.test_model_with_clood(task_loader[j]['train'], j, i)
                acc_list[j] = acc.item()

            print(f"tasks acc:{acc_list}")
            print(f"tasks avg acc:{acc_list[:i + 1].mean()}")

        return acc_list

    def compute_md_by_task(self, net_id, features):
        """
            Compute Mahalanobis distance of features to the Gaussian distribution of task == net_id
            return: scores_md of np.array of ndim == (B, 1) if cov_inv is available
                    None if cov_inv is not available (e.g. task=0 or cov_inv is not yet computed)
        """
        md_list, dist_list = [], []
        if len(self.args.cov_inv) >= net_id + 1:
            for y in range(self.class_per_task[net_id], self.class_per_task[net_id + 1]):
                mean, cov_inv = self.mean_cov(y, net_id)
                distance = md(features, mean, cov_inv, inverse=True)

                scores_md = 1 / distance
                md_list.append(scores_md)
                dist_list.append(-distance)

            scores_md = np.concatenate(md_list, axis=1)
            dist_list = np.concatenate(dist_list, axis=1)
            scores_md = scores_md.max(1, keepdims=True)
            dist_list = dist_list.max(1)
            return scores_md, dist_list
        return None, None

    def mean_cov(self, y, net_id, inverse=True):
        if inverse:
            cov = self.args.cov_inv[net_id]
        else:
            cov = self.args.cov[net_id]
        return self.args.mean[y], cov

    def evaluate(self, inputs, labels, total_learned_task_id):
        """
        Evaluate the model for both TIL and CIL. Prepare and save outputs for various purposes

        Args:
            total_learned_task_id: int, the last task_id the model has learned so far
        """
        out_list, output_ood = [], []
        use_two_heads = total_learned_task_id > 0
        with torch.no_grad():
            entropy_list, md_score_list, logit_output = [], [], []
            for t in range(total_learned_task_id + 1):
                features = self.model.forward_features(inputs)

                out = self.model.forward_classifier(features, t)[:, :self.class_per_task]
                # out = self.model.forward_classifier(features, t)[:, self.class_per_task[t]:self.class_per_task[t + 1]]

                out = F.softmax(out / self.args.T, dim=1)

                if self.args.compute_md:
                    scores, _ = self.compute_md_by_task(t, features)
                    if scores is not None: md_score_list.append(scores)

                if use_two_heads:
                    out_ood = self.model.forward_tp_head(features, t)
                    out_ood = F.softmax(out_ood / self.args.T, dim=1)
                    # out_ood = out_ood[:, :self.class_per_task]
                    out_ood = out_ood[:, self.class_per_task[t]:self.class_per_task[t + 1]]
                    out = out * torch.max(out_ood, dim=-1, keepdim=True)[0]

                logit_output.append(out.data)

                out_list.append(out)
                output_ood.append(out)

        out_list = torch.cat(out_list, dim=1)
        output_ood = torch.cat(output_ood, dim=1)
        logit_output = torch.cat(logit_output, dim=1)

        if len(md_score_list) > 0:
            md_score_list = np.concatenate(md_score_list, axis=1)

            md_score_list = torch.from_numpy(md_score_list)
            if self.args.use_md and total_learned_task_id + 1 == md_score_list.size(1):
                md_score_list = md_score_list.to(device).unsqueeze(-1)
                md_score_list = md_score_list / md_score_list.sum(dim=1, keepdims=True)

                md_score_matrix = []
                # print(out_list.size())
                for t in range(total_learned_task_id + 1):
                    md_score_matrix.append(md_score_list[:, t].repeat(1, self.class_per_task[t + 1] - self.class_per_task[t]))
                md_score_matrix = torch.cat(md_score_matrix, dim=1)
                # print(md_score_matrix.size())
                out_list *= md_score_matrix
                # out_list = out_list.view(out_list.size(0), total_learned_task_id + 1, -1) * md_score_list
                # out_list = out_list.view(out_list.size(0), -1)

        _, pred = out_list.max(1)
        correct = pred.eq(labels).sum()

        # print(out_list)
        # print(labels)
        # print(output_ood)

        return correct

    def test_model_with_clood(self, loader, i, total_learned_task_id):
        correct = torch.full([], 0).cuda()
        num = torch.full([], 0).cuda()
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.cuda(), target.cuda()
            num += data.size()[0]
            correct += self.evaluate(data, target, total_learned_task_id)

        test_accuracy = (100. * correct / num)
        print('Test task {}: Accuracy: {}/{} ({:.2f}%)'.format(i, correct, num, test_accuracy))
        return test_accuracy

    def test_model(self, loader, i):
        correct = torch.full([], 0).cuda()
        num = torch.full([], 0).cuda()
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.cuda(), target.cuda()
            pred = self.model(data)

            # pred = self.model.forward_features(data)
            # pred = self.model.forward_classifier(pred, i)
            # target = target % self.class_per_task

            Pred = pred.data.max(1, keepdim=True)[1]
            num += data.size()[0]
            correct += Pred.eq(target.data.view_as(Pred)).sum()

        test_accuracy = (100. * correct / num)
        print('Test task {}: Accuracy: {}/{} ({:.2f}%)'.format(i, correct, num, test_accuracy))
        return test_accuracy

    def cal_buffer_proto_loss(self, buffer_x, buffer_y, buffer_x_pair, task_id):
        buffer_fea, buffer_z = self.model(buffer_x_pair, use_proj=True)
        buffer_z_norm = F.normalize(buffer_z)
        buffer_z1 = buffer_z_norm[:buffer_x.shape[0]]
        buffer_z2 = buffer_z_norm[buffer_x.shape[0]:]

        buffer_proto_loss, buffer_z1_proto, buffer_z2_proto = self.OPELoss(buffer_z1, buffer_z2, buffer_y, task_id)
        self.classes_mean = (buffer_z1_proto + buffer_z2_proto) / 2

        return buffer_proto_loss, buffer_z1_proto, buffer_z2_proto, buffer_z_norm

    def sample_from_buffer_for_prototypes(self):
        b_num = self.buffer.x.shape[0]
        if b_num <= self.buffer_batch_size:
            buffer_x = self.buffer.x
            buffer_y = self.buffer.y
            _, buffer_y = torch.max(buffer_y, dim=1)
        else:
            buffer_x, buffer_y, _ = self.buffer.sample(self.buffer_batch_size, exclude_task=None)

        return buffer_x, buffer_y

    def loss_mixup(self, logits, y):
        criterion = F.cross_entropy
        loss_a = criterion(logits, y[:, 0].long(), reduction='none')
        loss_b = criterion(logits, y[:, 1].long(), reduction='none')
        return ((1 - y[:, 2]) * loss_a + y[:, 2] * loss_b).mean()
