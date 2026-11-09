# -*- coding: utf-8 -*-
"""adult_data.py —— dual-pruning 的 Adult 表格数据接入

DUAL 的 train.py 约定：`(input, target)` 中 `target` 是 `[label, index]` 两元组
（因为原实现把 CIFAR 的 `train_data.targets` 改写成 `[[label, index], ...]`）。
这里用 `AdultPairDataset` 复现同样的约定，无需改动 DUAL 的训练/动力学采集逻辑。

采样策略（random / stratified / beta / 默认）直接复用 data_subset.py 里的函数，
它们只依赖数组长度与掩码，与模态无关。
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from adult_adapter import AdultDataset, load_adult_arrays
from data_subset import beta_sampling, stratified_sampling


class AdultPairDataset(AdultDataset):
    """返回 (x, [label, index])，对齐 DUAL train.py 的 target[0] / target[1] 约定。"""

    def __getitem__(self, i):
        x, y = AdultDataset.__getitem__(self, i)
        return x, [y, int(i)]


class AdultEvalDataset(AdultDataset):
    """评估用，返回 (x, label)。"""


def get_adult_targets():
    """训练集标签数组，供 importance_evaluation.rearrange 使用。"""
    d = load_adult_arrays()
    return d["train_y"]


def _make_loaders(train_set, args):
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=getattr(args, "workers", 0),
        pin_memory=False,
    )
    d = load_adult_arrays()
    test_set = AdultEvalDataset(d["test_x"], d["test_y"])
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=getattr(args, "workers", 0),
        pin_memory=False,
    )
    return train_loader, test_loader


def load_adult(args):
    """Step1：收集训练动力学用，全量训练集。"""
    print("Loading Adult (tabular)... ", end="")
    d = load_adult_arrays()
    train_set = AdultPairDataset(d["train_x"], d["train_y"])
    train_loader, test_loader = _make_loaders(train_set, args)
    print("done. train=%d test=%d" % (len(train_set), len(test_loader.dataset)))
    return train_loader, test_loader


def load_adult_sub(args, target_probs=None, score=None, data_mask=None):
    """Step3：在剪枝后的子集上训练。

    注意：train_subset.py 的 train() 直接对 target 调 .to(device)，
    期望的是纯标签张量（不像 train.py 需要 [label, index] 二元组），
    所以这里用 AdultEvalDataset。
    """
    print("Loading Adult subset (tabular)... ", end="")
    d = load_adult_arrays()
    train_set = AdultEvalDataset(d["train_x"], d["train_y"])
    total = len(train_set)

    if args.sample == "random":
        subset_mask = np.random.choice(total, int(args.subset_rate * total), replace=False)
    elif args.sample == "stratified":
        mis_num = int(args.mis_ratio * total)
        easy_index = data_mask[mis_num:]
        selected = stratified_sampling(
            torch.tensor(score[easy_index]), int(args.subset_rate * total))
        subset_mask = easy_index[selected]
    elif args.sample == "beta":
        subset_mask = beta_sampling(
            1 - args.subset_rate, args.c_d, target_probs, data_mask, score)
    else:
        subset_mask = data_mask[-int(args.subset_rate * len(data_mask)):]

    subset_mask = np.asarray(subset_mask, dtype=np.int64)
    data_set = Subset(train_set, subset_mask)
    train_loader, test_loader = _make_loaders(data_set, args)
    print("done. subset=%d / %d" % (len(data_set), total))
    return train_loader, test_loader
