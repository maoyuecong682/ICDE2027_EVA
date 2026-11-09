import time
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os 
import pickle

########################################################################################################################
# Load Data
########################################################################################################################

def load_data(args):
    """
    Load data for training and testing.

    Returns:
        train_loader: DataLoader for training data.
        test_loader: DataLoader for testing data.
    """
    train_loader, test_loader = load_dataset(args)
    return train_loader, test_loader

def load_dataset(args):
    """
    Load dataset based on the specified dataset in args.

    Returns:
        train_loader: DataLoader for training data.
        test_loader: DataLoader for testing data.
    """
    if args.dataset == 'cifar10' and args.dataset not in _eva_names():
        train_loader, test_loader = load_cifar10(args)
    elif args.dataset == 'cifar100' and args.dataset not in _eva_names():
        # cifar100 交给适配层：原生测试集 10000 张，EVA 的划分是 val5000/test5000，
        # 走原生分支会让 baseline 与主实验不在同一测试集上。
        train_loader, test_loader = load_cifar100(args)
    elif args.dataset == 'adult' and args.dataset not in _eva_names():
        # Adult 表格数据：DUAL 的动力学采集与模态无关，只需换成表格 Dataset
        from adult_data import load_adult
        train_loader, test_loader = load_adult(args)
    # elif args.dataset == 'tiny-imagenet':
    #     train_loader, test_loader = load_tiny(args)
    elif args.dataset in _eva_names():
        train_loader, test_loader = load_eva(args)
    else:
        raise NotImplementedError("Dataset not supported: {}".format(args.dataset))
    return train_loader, test_loader

def load_cifar10(args, mask=None, score=None, target_probs=None):
    """
    Load CIFAR-10 dataset.

    Returns:
        train_loader: DataLoader for training data.
        test_loader: DataLoader for testing data.
    """
    print('Loading CIFAR-10... ', end='')
    time_start = time.time()
    
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    train_data = dset.CIFAR10(args.data_path, train=True, transform=train_transform, download=True)
    target_index = [[train_data.targets[i], i] for i in range(len(train_data.targets))]
    train_data.targets = target_index
    
    train_loader = torch.utils.data.DataLoader(train_data, args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    test_data = dset.CIFAR10(args.data_path, train=False, transform=test_transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)
    
    print(f"done in {time.time() - time_start:.2f} seconds.")
    return train_loader, test_loader

def load_cifar100(args):
    """
    Load CIFAR-100 dataset.

    Returns:
        train_loader: DataLoader for training data.
        test_loader: DataLoader for testing data.
    """
    print('Loading CIFAR-100... ', end='')
    time_start = time.time()
    
    mean = [x / 255 for x in [129.3, 124.1, 112.4]]
    std = [x / 255 for x in [68.2, 65.4, 70.4]]
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    train_data = dset.CIFAR100(args.data_path, train=True, transform=train_transform, download=True)
    target_index = [[train_data.targets[i], i] for i in range(len(train_data.targets))]
    train_data.targets = target_index
    
    train_loader = torch.utils.data.DataLoader(train_data, args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    test_data = dset.CIFAR100(args.data_path, train=False, transform=test_transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)
    
    print(f"done in {time.time() - time_start:.2f} seconds.")
    return train_loader, test_loader


def _eva_names():
    """走适配层的数据集名。

    改走 eva_names：它在 import 阶段就把仓库根与 /root/eva_output 补进 sys.path。
    原实现 `except Exception: return set()` 会在 exp_cifar/ 启动时因 ImportError
    静默返回空集，表现为 "Dataset not supported"。现在导入失败直接抛错。
    """
    from eva_names import ADAPTER_DATASETS
    return set(ADAPTER_DATASETS)


def load_eva(args):
    """EVA 预处理数据集。

    DUAL 的训练动力学代码要求 train 侧每个样本是 (x, [label, index])，由
    eva_adapter.Indexed 提供（惰性构造，避免 465k 行的 covtype 撑爆内存）。
    CPU 上 pin_memory 无意义且会多占一份 pinned 缓冲，这里显式关掉。
    """
    import eva_adapter as EVA

    m = EVA.load(args.dataset)
    # train 侧必须带下标：train.py 的动力学循环要 target[1] 才有法记录 {epoch}_Index.npy。
    # test 侧必须保持二元组：评估循环直接 `target.to(device)`，套 Indexed 会拿到 list。
    # 两侧协议不同是有意的，不要「统一」掉。
    train_data = EVA.Indexed(m["train"], m)
    test_data = EVA.Wrapped(m["test"], m)

    bs = int(getattr(args, "batch_size", 128) or 128)
    nw = int(getattr(args, "workers", 0) or 0)
    train_loader = DataLoader(train_data, batch_size=bs, shuffle=True,
                              num_workers=nw, pin_memory=False)
    test_loader = DataLoader(test_data, batch_size=bs, shuffle=False,
                             num_workers=nw, pin_memory=False)
    print("[eva] %s train=%d test=%d classes=%d tabular=%s"
          % (args.dataset, len(train_data), len(test_data),
             m["num_classes"], m["is_tabular"]))
    return train_loader, test_loader
