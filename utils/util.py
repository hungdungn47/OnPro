import torch.distributed as dist
import numpy as np
from scipy.stats import sem
import scipy.stats as stats
from torch.utils.data import DataLoader, Dataset, Sampler, Subset
import math
from PIL import Image
import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n):
        self.sum += val * n
        self.count += n

    def avg(self):
        if self.count == 0:
            return 0
        return float(self.sum) / self.count


def all_reduce_tensor(tensor, op=dist.ReduceOp.SUM, world_size=1, norm=True):
    tensor = tensor.clone()
    dist.all_reduce(tensor, op)
    if norm:
        tensor.div_(world_size)

    return tensor


def compute_performance(end_task_acc_arr):
    """
    Given test accuracy results from multiple runs saved in end_task_acc_arr,
    compute the average accuracy, forgetting, and task accuracies as well as their confidence intervals.

    :param end_task_acc_arr:       (list) List of lists
    :param task_ids:                (list or tuple) Task ids to keep track of
    :return:                        (avg_end_acc, forgetting, avg_acc_task)
    """
    n_run, n_tasks = end_task_acc_arr.shape[:2]
    t_coef = stats.t.ppf((1+0.95) / 2, n_run-1)     # t coefficient used to compute 95% CIs: mean +- t *

    # compute average test accuracy and CI
    end_acc = end_task_acc_arr[:, -1, :]                         # shape: (num_run, num_task)
    avg_acc_per_run = np.mean(end_acc, axis=1)      # mean of end task accuracies per run
    avg_end_acc = (np.mean(avg_acc_per_run), t_coef * sem(avg_acc_per_run))

    # compute forgetting
    best_acc = np.max(end_task_acc_arr, axis=1)
    final_forgets = best_acc - end_acc
    avg_fgt = np.mean(final_forgets, axis=1)
    avg_end_fgt = (np.mean(avg_fgt), t_coef * sem(avg_fgt))

    # compute ACC
    acc_per_run = np.mean((np.sum(np.tril(end_task_acc_arr), axis=2) /
                           (np.arange(n_tasks) + 1)), axis=1)
    avg_acc = (np.mean(acc_per_run), t_coef * sem(acc_per_run))


    # compute BWT+
    bwt_per_run = (np.sum(np.tril(end_task_acc_arr, -1), axis=(1,2)) -
                  np.sum(np.diagonal(end_task_acc_arr, axis1=1, axis2=2) *
                         (np.arange(n_tasks, 0, -1) - 1), axis=1)) / (n_tasks * (n_tasks - 1) / 2)
    bwtp_per_run = np.maximum(bwt_per_run, 0)
    avg_bwtp = (np.mean(bwtp_per_run), t_coef * sem(bwtp_per_run))

    # compute FWT
    fwt_per_run = np.sum(np.triu(end_task_acc_arr, 1), axis=(1,2)) / (n_tasks * (n_tasks - 1) / 2)
    avg_fwt = (np.mean(fwt_per_run), t_coef * sem(fwt_per_run))
    return avg_end_acc, avg_end_fgt, avg_acc, avg_bwtp, avg_fwt

class MySampler(Sampler):
    """
        Sampler for dataset whose length is different from that of the target dataset.
        This can be useful when we need oversampling/undersampling because
        the target dataset has more/less samples than the dataset of interest.
        Generate indices whose length is same as that of the target length * maximum number of epochs.
    """
    def __init__(self, current_length, target_length, batch_size, max_epoch):
        self.current = current_length
        self.length = math.ceil(target_length / batch_size) * batch_size * max_epoch

    def __iter__(self):
        self.indices = np.array([], dtype=int)
        while len(self.indices) < self.length:
            idx = np.random.permutation(self.current)
            self.indices = np.concatenate([self.indices, idx])
        self.indices = self.indices[:self.length]
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

class Memory(Dataset):
    """
        Replay buffer. Keep balanced samples. Data must be compatible with Image.
        Currently, MNIST and CIFAR are compatible.
    """
    def __init__(self, buffer_size):
        # self.args = args
        self.buffer_size = buffer_size

        self.data_list = {}
        self.targets_list = {}

        self.data, self.targets = [], []

        self.n_cls = len(self.data_list)
        self.n_samples = self.buffer_size

    def update(self, dataset):
        # self.args.logger.print("Updating Memory")
        # self.transform = dataset.transform

        ys = list(sorted(set(dataset.targets)))
        for y in ys:
            idx = np.where(dataset.targets == y)[0]
            self.data_list[y] = dataset.data[idx]
            self.targets_list[y] = dataset.targets[idx]

            self.n_cls = len(self.data_list)

        self.n_samples = self.buffer_size // self.n_cls
        for y, data in self.data_list.items():
            idx = np.random.permutation(len(data))
            idx = idx[:self.n_samples]
            self.data_list[y] = self.data_list[y][idx]
            self.targets_list[y] = self.targets_list[y][idx]

        self.data, self.targets = [], []
        for (k, data), (_, targets) in zip(self.data_list.items(), self.targets_list.items()):
            self.data.append(data)
            self.targets.append(targets)
        self.data = np.concatenate(self.data)
        self.targets = np.concatenate(self.targets)

    def is_empty(self):
        return len(self.data) == 0

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = Image.fromarray(img)

        # if self.transform is not None:
        #     img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)

def md(data, mean, mat, inverse=False):
    if isinstance(data, torch.Tensor):
        data = data.data.cpu().numpy()
    if data.ndim == 1:
        data.reshape(1, -1)
    delta = (data - mean)

    if not inverse:
        mat = np.linalg.inv(mat)

    distance = np.dot(np.dot(delta, mat), delta.T)
    return np.sqrt(np.diagonal(distance)).reshape(-1, 1)
