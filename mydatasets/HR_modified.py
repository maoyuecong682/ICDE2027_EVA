"""HR_modified.py -- HR 加载器的修复版。

原 mydatasets/HR.py 有两处会直接毁掉 test 指标的问题，本文件按**原意图**
（8:2 切分 train/valid、StandardScaler 标准化、13 维特征）重写，不改任务定义。

问题 1：test.csv 的列序与 train.csv 不一致
    train: ... Education Sex Race Relationship MaritalStatus Country Age HoursPerWeek
              fnlgwt CapitalLoss EdNum **Occupation CapitalGain** Income
    test : ... Education Sex Race Relationship MaritalStatus Country Age HoursPerWeek
              fnlgwt CapitalLoss EdNum **CapitalGain Occupation** Income
  `Occupation` 与 `CapitalGain` 两列在 test 里互换了位置。原实现按下标取特征，
  于是 test 的第 12、13 维喂反了 → 模型在 test 上系统性预测成另一类。
  实测：val acc 0.8269，test acc 0.2760，test 上 1297 条里 1282 条被判成类别 1。

问题 2：标准化做了两遍，而且只错了 train/valid
  ```
  train_dataset, valid_dataset = random_split(train_dataset, ...)
  train_dataset.dataset.features = scaler.fit_transform(...)   # 底层同一个对象
  valid_dataset.dataset.features = scaler.transform(...)       # 又做了一遍
  test_dataset.features          = scaler.transform(...)       # 只做一遍
  ```
  `random_split` 返回的 Subset 共享 `.dataset`，所以验证集那次 transform 作用在
  **已经被标准化过的数据**上（HR 的特征本来就在 [0,1]，二次变换约等于 5z-1.5），
  而 test 只做一次 → train/valid 与 test 分布错位，指标全废。

修复后的口径：
  * 先按 train 的表头把 test 的列重排好；
  * StandardScaler 在**整份 train.csv** 上 fit（与原实现一致），
    然后 train / valid / test 各 transform **恰好一次**；
  * 切分仍为 8:2，用固定种子 torch.Generator(manual_seed=48) 复现；
  * 返回 9 元组，顺序与原实现一致；im_size 用 list（build_model 要求）。
"""
import os

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import float32, long, tensor
from torch.utils.data import Dataset, random_split

SPLIT_SEED = 48
VAL_RATIO = 0.2


class HRDataset(Dataset):
    """与 HR.py 中的同名类语义一致（features / labels 两个属性）。"""

    def __init__(self, features, labels):
        self.features = tensor(features, dtype=float32)
        self.labels = tensor(labels, dtype=long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def _align_test_columns(train_df, test_df):
    """把 test 的特征列按 train 的表头重新排序（修复 Occupation/CapitalGain 互换）。"""
    cols = list(train_df.columns)
    if list(test_df.columns) == cols:
        return test_df
    missing = [c for c in cols if c not in test_df.columns]
    if missing:
        raise RuntimeError("HR: test.csv 缺少列 %s，无法对齐" % missing)
    fixed = test_df[cols]
    print("[HR][fix] test.csv 列序与 train 不一致，已按 train 重排：%s -> %s"
          % (list(test_df.columns), cols), flush=True)
    return fixed


def HR(data_path, balance=True):
    train_df = pd.read_csv(os.path.join(data_path, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_path, "test.csv"))
    test_df = _align_test_columns(train_df, test_df)

    X = train_df.iloc[:, :-1].to_numpy(dtype="float32")
    y = train_df.iloc[:, -1].to_numpy(dtype="int64")
    Xte = test_df.iloc[:, :-1].to_numpy(dtype="float32")
    yte = test_df.iloc[:, -1].to_numpy(dtype="int64")

    # 原实现是在整份 train.csv 上 fit 的，这里保持不变（只是为了修掉重复 transform）。
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X).astype("float32")
    Xts = scaler.transform(Xte).astype("float32")

    n = len(Xs)
    n_val = int(VAL_RATIO * n)
    n_tr = n - n_val
    full = HRDataset(Xs, y)
    train_dataset, valid_dataset = random_split(
        full, [n_tr, n_val], generator=torch.Generator().manual_seed(SPLIT_SEED))
    test_dataset = HRDataset(Xts, yte)

    num_classes = int(len(np.unique(y)))
    class_names = [str(i) for i in range(num_classes)]

    pos_tr = float((y[train_dataset.indices] == 1).mean())
    print("[HR][fix] train/val/test = %d/%d/%d，特征 %d 维，类别 %d，"
          "train 正类比 %.3f"
          % (len(train_dataset), len(valid_dataset), len(test_dataset),
             Xs.shape[1], num_classes, pos_tr), flush=True)

    return (1, [Xs.shape[1]], num_classes, class_names,
            scaler.mean_, scaler.scale_,
            train_dataset, valid_dataset, test_dataset)
