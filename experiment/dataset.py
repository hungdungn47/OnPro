import os
import numpy as np
import torch
from torchvision import datasets, transforms
from experiment.tinyimagenet import MyTinyImagenet
from torch.utils.data import TensorDataset
from torch import Tensor
from PIL import Image
import pandas as pd


class ModifyTensorDataset(TensorDataset):
    def __init__(self, *tensors: Tensor) -> None:
        super().__init__(tensors[0], tensors[1])
        self.data = tensors[0].numpy()
        self.targets = tensors[1].numpy()


def get_data(dataset_name, batch_size, n_workers, args):
    if "cifar" in dataset_name:
        return get_cifar_data(dataset_name, batch_size, n_workers)
    elif dataset_name == "tiny_imagenet":
        return get_tinyimagenet(batch_size, n_workers)
    elif dataset_name == "colorectal":
        return get_iColorectal_data(batch_size, n_workers, args)
    elif dataset_name == "ham10000":
        return get_ham10000_data(batch_size, n_workers)
    else:
        raise Exception('unknown dataset!')


def get_cifar_data(dataset_name, batch_size, n_workers):
    data = {}
    size = [3, 32, 32]
    class_per_task = [0, 2, 4, 6, 8, 10]
    if dataset_name == "cifar10":
        task_num = 5
        class_num = 10
        data_dir = './data/binary_cifar_/'
        class_per_task = [0, 2, 4, 6, 8, 10]
    elif dataset_name == "cifar100":
        task_num = 10
        class_num = 100
        data_dir = './data/binary_cifar100_10/'
        class_per_task = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
        dataset_path = './data/'
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
        dataset = {}
        if dataset_name == "cifar10":
            dataset['train'] = datasets.CIFAR10(dataset_path, train=True, download=True, transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean, std)]))
            dataset['test'] = datasets.CIFAR10(dataset_path, train=False, download=True, transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean, std)]))
        elif dataset_name == "cifar100" or dataset_name == "cifar100_50":
            dataset['train'] = datasets.CIFAR100(dataset_path, train=True, download=True, transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean, std)]))
            dataset['test'] = datasets.CIFAR100(dataset_path, train=False, download=True, transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean, std)]))
        for task_id in range(task_num):
            data[task_id] = {}
            for data_type in ['train', 'test']:
                loader = torch.utils.data.DataLoader(dataset[data_type], batch_size=1, shuffle=False)
                data[task_id][data_type] = {'x': [], 'y': []}
                for image, target in loader:
                    label = target.numpy()[0]
                    if label in range(class_per_task[task_id], class_per_task[task_id + 1]):
                        data[task_id][data_type]['x'].append(image)
                        data[task_id][data_type]['y'].append(label)

        # save
        for task_id in data.keys():
            for data_type in ['train', 'test']:
                data[task_id][data_type]['x'] = torch.stack(data[task_id][data_type]['x']).view(-1, size[0], size[1],
                                                                                                size[2])
                data[task_id][data_type]['y'] = torch.LongTensor(
                    np.array(data[task_id][data_type]['y'], dtype=int)).view(-1)
                torch.save(data[task_id][data_type]['x'],
                           os.path.join(os.path.expanduser(data_dir), 'data' + str(task_id) + data_type + 'x.bin'))
                torch.save(data[task_id][data_type]['y'],
                           os.path.join(os.path.expanduser(data_dir), 'data' + str(task_id) + data_type + 'y.bin'))

    # Load binary files
    data = {}
    ids = list(np.arange(task_num))
    print('Task order =', ids)
    for i in range(task_num):
        data[i] = dict.fromkeys(['train', 'test'])
        for s in ['train', 'test']:
            data[i][s] = {'x': [], 'y': []}
            data[i][s]['x'] = torch.load(
                os.path.join(os.path.expanduser(data_dir), 'data' + str(ids[i]) + s + 'x.bin'))
            data[i][s]['y'] = torch.load(
                os.path.join(os.path.expanduser(data_dir), 'data' + str(ids[i]) + s + 'y.bin'))

    Loader = {}
    for t in range(task_num):
        Loader[t] = dict.fromkeys(['train', 'test'])

        dataset_new_train = ModifyTensorDataset(data[t]['train']['x'], data[t]['train']['y'])
        dataset_new_test = ModifyTensorDataset(data[t]['test']['x'], data[t]['test']['y'])
        train_loader = torch.utils.data.DataLoader(
            dataset_new_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=n_workers,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset_new_test,
            batch_size=64,
            shuffle=True,
            num_workers=n_workers,
        )
        Loader[t]['train'] = train_loader
        Loader[t]['test'] = test_loader

    print("Data and loader is prepared")
    return data, class_num, class_per_task, Loader, size


def get_tinyimagenet(batch_size, n_workers):
    data = {}
    size = [3, 64, 64]
    task_num = 100
    class_num = 200
    class_per_task = class_num // task_num

    base_path = './data/TINYIMG'
    data_dir = './data/binary_tiny200_100'

    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
        dat = {}
        transform = transforms.Normalize((0.4802, 0.4480, 0.3975),
                                         (0.2770, 0.2691, 0.2821))

        test_transform = transforms.Compose(
            [transforms.ToTensor(), transform])

        train = MyTinyImagenet(base_path, train=True, download=True, transform=test_transform)
        test = MyTinyImagenet(base_path, train=False, download=True, transform=test_transform)

        dat['train'] = train
        dat['test'] = test
        for t in range(task_num):
            data[t] = {}
            for s in ['train', 'test']:
                loader = torch.utils.data.DataLoader(dat[s], batch_size=1, shuffle=False)
                data[t][s] = {'x': [], 'y': []}
                for image, target in loader:
                    label = target.numpy()[0]
                    if label in range(class_per_task * t, class_per_task * (t + 1)):
                        data[t][s]['x'].append(image)
                        data[t][s]['y'].append(label)

        # and save
        for t in data.keys():
            for s in ['train', 'test']:
                data[t][s]['x'] = torch.stack(data[t][s]['x']).view(-1, size[0], size[1], size[2])
                data[t][s]['y'] = torch.LongTensor(np.array(data[t][s]['y'], dtype=int)).view(-1)
                torch.save(data[t][s]['x'],
                           os.path.join(os.path.expanduser(data_dir), 'data' + str(t) + s + 'x.bin'))
                torch.save(data[t][s]['y'],
                           os.path.join(os.path.expanduser(data_dir), 'data' + str(t) + s + 'y.bin'))
    # Load binary files
    data = {}
    ids = list(np.arange(task_num))
    print('Task order =', ids)
    for i in range(task_num):
        data[i] = dict.fromkeys(['train', 'test'])
        for s in ['train', 'test']:
            data[i][s] = {'x': [], 'y': []}
            data[i][s]['x'] = torch.load(
                os.path.join(os.path.expanduser(data_dir), 'data' + str(ids[i]) + s + 'x.bin'))
            data[i][s]['y'] = torch.load(
                os.path.join(os.path.expanduser(data_dir), 'data' + str(ids[i]) + s + 'y.bin'))

    Loader = {}
    for t in range(task_num):
        Loader[t] = dict.fromkeys(['train', 'test'])

        dataset_new_train = ModifyTensorDataset(data[t]['train']['x'], data[t]['train']['y'])
        dataset_new_test = ModifyTensorDataset(data[t]['test']['x'], data[t]['test']['y'])
        train_loader = torch.utils.data.DataLoader(
            dataset_new_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=n_workers,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset_new_test,
            batch_size=64,
            shuffle=True,
            num_workers=n_workers,
        )
        Loader[t]['train'] = train_loader
        Loader[t]['test'] = test_loader

    print("Data and loader is prepared")
    return data, class_num, class_per_task, Loader, size


def get_iColorectal_data(batch_size, n_workers, args):
    data = {}
    size = [3, 150, 150]
    class_num = 8
    data_dir = './data/colorectal'
    task_num = 3
    class_per_task = [0, 4, 6, 8]
    if args.class_per_task == 1:
        task_num = 5
        class_per_task = [0, 4, 5, 6, 7, 8]
    class_order = [2, 3, 0, 4, 5, 6, 7, 1]
    # class_order = [1, 5, 6, 0, 3, 2, 4, 7]
    # class_order = [1, 4, 5, 7, 2, 3, 6, 0]
    print("Class order: ", class_order)
    map_class = {}
    for i, id in enumerate(class_order):
        map_class[id] = i
    print("Map class: ", map_class)
    # os.makedirs(data_dir)
    dataset = {}

    def get_dataset(data_path, train=True):
        if train is True:
            metadata_path = os.path.join(data_path, 'train.csv')
        else:
            metadata_path = os.path.join(data_path, 'test.csv')

        df = pd.read_csv(metadata_path)
        df['image'] = df['image'].apply(lambda x: os.path.join(data_path, 'images/' + x))
        data = {}
        data['x'] = np.array(df['image'])
        data['y'] = np.array(df['label'])

        return data

    dataset['train'] = get_dataset(data_dir, train=True)
    dataset['test'] = get_dataset(data_dir, train=False)
    #     if dataset_name == "cifar10":
    #         dataset['train'] = datasets.CIFAR10(dataset_path, train=True, download=True, transform=transforms.Compose(
    #             [transforms.ToTensor(), transforms.Normalize(mean, std)]))
    #         dataset['test'] = datasets.CIFAR10(dataset_path, train=False, download=True, transform=transforms.Compose(
    #             [transforms.ToTensor(), transforms.Normalize(mean, std)]))
    #     elif dataset_name == "cifar100" or dataset_name == "cifar100_50":
    #         dataset['train'] = datasets.CIFAR100(dataset_path, train=True, download=True, transform=transforms.Compose(
    #             [transforms.ToTensor(), transforms.Normalize(mean, std)]))
    #         dataset['test'] = datasets.CIFAR100(dataset_path, train=False, download=True, transform=transforms.Compose(
    #             [transforms.ToTensor(), transforms.Normalize(mean, std)]))
    trsf = transforms.Compose([
        transforms.Resize((150, 150), Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    data = {}
    # for task_id in range(task_num):
    #     data[task_id] = {}
    #     for data_type in ['train', 'test']:
    #         # loader = torch.utils.data.DataLoader(dataset[data_type], batch_size=1, shuffle=False)
    #         data[task_id][data_type] = {'x': [], 'y': []}
    #         for image, target in zip(dataset[data_type]['x'], dataset[data_type]['y']):
    #             if target in range(class_per_task[task_id], class_per_task[task_id + 1]):
    #                 img = Image.open(image).convert("RGB")
    #                 img = trsf(img)
    #                 data[task_id][data_type]['x'].append(img)
    #                 data[task_id][data_type]['y'].append(target)
    #         data[task_id][data_type]['x'] = torch.stack(data[task_id][data_type]['x'], dim=0)

    for task_id in range(task_num):
        data[task_id] = {}
        for data_type in ['train', 'test']:
            # loader = torch.utils.data.DataLoader(dataset[data_type], batch_size=1, shuffle=False)
            data[task_id][data_type] = {'x': [], 'y': []}
            for image, target in zip(dataset[data_type]['x'], dataset[data_type]['y']):
                if target in class_order[class_per_task[task_id]: class_per_task[task_id + 1]]:
                    img = Image.open(image).convert("RGB")
                    img = trsf(img)
                    data[task_id][data_type]['x'].append(img)
                    data[task_id][data_type]['y'].append(map_class[target])
            data[task_id][data_type]['x'] = torch.stack(data[task_id][data_type]['x'], dim=0)

    #
    #     # save
    #     for task_id in data.keys():
    #         for data_type in ['train', 'test']:
    #             data[task_id][data_type]['x'] = torch.stack(data[task_id][data_type]['x']).view(-1, size[0], size[1], size[2])
    #             data[task_id][data_type]['y'] = torch.LongTensor(np.array(data[task_id][data_type]['y'], dtype=int)).view(-1)
    #             torch.save(data[task_id][data_type]['x'],
    #                        os.path.join(os.path.expanduser(data_dir), 'data' + str(task_id) + data_type + 'x.bin'))
    #             torch.save(data[task_id][data_type]['y'],
    #                        os.path.join(os.path.expanduser(data_dir), 'data' + str(task_id) + data_type + 'y.bin'))
    #
    # # Load binary files
    # data = {}
    # ids = list(np.arange(task_num))
    # print('Task order =', ids)
    # for i in range(task_num):
    #     data[i] = dict.fromkeys(['train', 'test'])
    #     for s in ['train', 'test']:
    #         data[i][s] = {'x': [], 'y': []}
    #         data[i][s]['x'] = torch.load(
    #             os.path.join(os.path.expanduser(data_dir), 'data' + str(ids[i]) + s + 'x.bin'))
    #         data[i][s]['y'] = torch.load(
    #             os.path.join(os.path.expanduser(data_dir), 'data' + str(ids[i]) + s + 'y.bin'))

    Loader = {}
    for t in range(task_num):
        Loader[t] = dict.fromkeys(['train', 'test'])
        print(len(data[t]['train']['x']))
        print(type(data[t]['train']['y']))

        dataset_new_train = ModifyTensorDataset(data[t]['train']['x'],
                                                torch.from_numpy(np.asarray(data[t]['train']['y'])))
        dataset_new_test = ModifyTensorDataset(data[t]['test']['x'],
                                               torch.from_numpy(np.asarray(data[t]['test']['y'])))
        train_loader = torch.utils.data.DataLoader(
            dataset_new_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=n_workers,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset_new_test,
            batch_size=64,
            shuffle=True,
            num_workers=n_workers,
        )
        Loader[t]['train'] = train_loader
        Loader[t]['test'] = test_loader

    print("Data and loader is prepared")
    return data, class_num, class_per_task, Loader, size


def get_ham10000_data(batch_size, n_workers):
    data = {}
    size = [3, 150, 150]
    task_num = 3
    class_num = 7
    data_dir = './data/ham10000'
    class_per_task = [0, 3, 5, 7]
    class_order = [0, 2, 3, 4, 5, 6, 1]
    # class_order = [5, 4, 0, 1, 3, 6, 2]
    # class_order = [1, 5, 0, 6, 2, 4, 3]
    print("Class order: ", class_order)
    map_class = {}
    for i, id in enumerate(class_order):
        map_class[id] = i
    print("Map class: ", map_class)
    # os.makedirs(data_dir)
    dataset = {}

    def get_dataset(data_path, train=True):
        if train is True:
            metadata_path = os.path.join(data_path, 'train.csv')
        else:
            metadata_path = os.path.join(data_path, 'val.csv')

        df = pd.read_csv(metadata_path)
        df['image'] = df['image'].apply(lambda x: os.path.join(data_path, 'images', x+'.jpg'))
        data = {}
        data['x'] = np.array(df['image'])
        data['y'] = np.argmax(np.array(df.drop(columns=['image'])), axis=1)

        return data

    dataset['train'] = get_dataset(data_dir, train=True)
    dataset['test'] = get_dataset(data_dir, train=False)
    #     if dataset_name == "cifar10":
    #         dataset['train'] = datasets.CIFAR10(dataset_path, train=True, download=True, transform=transforms.Compose(
    #             [transforms.ToTensor(), transforms.Normalize(mean, std)]))
    #         dataset['test'] = datasets.CIFAR10(dataset_path, train=False, download=True, transform=transforms.Compose(
    #             [transforms.ToTensor(), transforms.Normalize(mean, std)]))
    #     elif dataset_name == "cifar100" or dataset_name == "cifar100_50":
    #         dataset['train'] = datasets.CIFAR100(dataset_path, train=True, download=True, transform=transforms.Compose(
    #             [transforms.ToTensor(), transforms.Normalize(mean, std)]))
    #         dataset['test'] = datasets.CIFAR100(dataset_path, train=False, download=True, transform=transforms.Compose(
    #             [transforms.ToTensor(), transforms.Normalize(mean, std)]))
    trsf = transforms.Compose([
        transforms.Resize((150, 150), Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    data = {}
    # for task_id in range(task_num):
    #     data[task_id] = {}
    #     for data_type in ['train', 'test']:
    #         # loader = torch.utils.data.DataLoader(dataset[data_type], batch_size=1, shuffle=False)
    #         data[task_id][data_type] = {'x': [], 'y': []}
    #         for image, target in zip(dataset[data_type]['x'], dataset[data_type]['y']):
    #             if target in range(class_per_task[task_id], class_per_task[task_id + 1]):
    #                 img = Image.open(image).convert("RGB")
    #                 img = trsf(img)
    #                 data[task_id][data_type]['x'].append(img)
    #                 data[task_id][data_type]['y'].append(target)
    #         data[task_id][data_type]['x'] = torch.stack(data[task_id][data_type]['x'], dim=0)

    for task_id in range(task_num):
        data[task_id] = {}
        for data_type in ['train', 'test']:
            # loader = torch.utils.data.DataLoader(dataset[data_type], batch_size=1, shuffle=False)
            data[task_id][data_type] = {'x': [], 'y': []}
            for image, target in zip(dataset[data_type]['x'], dataset[data_type]['y']):
                if target in class_order[class_per_task[task_id]: class_per_task[task_id + 1]]:
                    img = Image.open(image).convert("RGB")
                    img = trsf(img)
                    data[task_id][data_type]['x'].append(img)
                    data[task_id][data_type]['y'].append(map_class[target])
            data[task_id][data_type]['x'] = torch.stack(data[task_id][data_type]['x'], dim=0)

    #
    #     # save
    #     for task_id in data.keys():
    #         for data_type in ['train', 'test']:
    #             data[task_id][data_type]['x'] = torch.stack(data[task_id][data_type]['x']).view(-1, size[0], size[1], size[2])
    #             data[task_id][data_type]['y'] = torch.LongTensor(np.array(data[task_id][data_type]['y'], dtype=int)).view(-1)
    #             torch.save(data[task_id][data_type]['x'],
    #                        os.path.join(os.path.expanduser(data_dir), 'data' + str(task_id) + data_type + 'x.bin'))
    #             torch.save(data[task_id][data_type]['y'],
    #                        os.path.join(os.path.expanduser(data_dir), 'data' + str(task_id) + data_type + 'y.bin'))
    #
    # # Load binary files
    # data = {}
    # ids = list(np.arange(task_num))
    # print('Task order =', ids)
    # for i in range(task_num):
    #     data[i] = dict.fromkeys(['train', 'test'])
    #     for s in ['train', 'test']:
    #         data[i][s] = {'x': [], 'y': []}
    #         data[i][s]['x'] = torch.load(
    #             os.path.join(os.path.expanduser(data_dir), 'data' + str(ids[i]) + s + 'x.bin'))
    #         data[i][s]['y'] = torch.load(
    #             os.path.join(os.path.expanduser(data_dir), 'data' + str(ids[i]) + s + 'y.bin'))

    Loader = {}
    for t in range(task_num):
        Loader[t] = dict.fromkeys(['train', 'test'])
        print(len(data[t]['train']['x']))
        print(type(data[t]['train']['y']))

        dataset_new_train = ModifyTensorDataset(data[t]['train']['x'],
                                                torch.from_numpy(np.asarray(data[t]['train']['y'])))
        dataset_new_test = ModifyTensorDataset(data[t]['test']['x'],
                                               torch.from_numpy(np.asarray(data[t]['test']['y'])))
        train_loader = torch.utils.data.DataLoader(
            dataset_new_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=n_workers,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset_new_test,
            batch_size=64,
            shuffle=True,
            num_workers=n_workers,
        )
        Loader[t]['train'] = train_loader
        Loader[t]['test'] = test_loader

    print("Data and loader is prepared")
    return data, class_num, class_per_task, Loader, size