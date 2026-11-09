# -*- coding: utf-8 -*-
"""adult_adapter.py —— 把预处理好的 Adult 表格数据接入 COPS / GM-Matching / dual-pruning

放在任意 baseline 目录下即可独立使用，不依赖 EVA 主工程。

设计要点
--------
1. 数据来自 `adult_processed.npz`（由 mydatasets/prepare_adult.py 生成），
   默认使用 **z-score 归一化后** 的特征（X_*_norm），与 EVA 的 AdultsDataset 口径一致。
2. 骨干网络提供两个，均复刻 EVA `nets/` 下的实现，保证与 EVA 主实验可比：
     - TabNet  ：13 -> 128 -> 128 -> attention -> 128 -> 2   （EVA all.sh 中 adult 的配置）
     - MLP     ：13 -> 128 -> 128 -> 2
3. Dataset 的 `__getitem__` 返回 `(x_tensor, int_label)`。
   返回 python int 而不是 tensor 很重要：GM-Matching 的 `run_coreset_select`
   直接遍历 dataset 并用 label 当 dict key，若返回 0-dim tensor 会退化成 N 个类别。
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

DEFAULT_NPZ = "/root/autodl-tmp/EVA/data/adult/adult_processed.npz"


# --------------------------------------------------------------------------- #
# 环境
# --------------------------------------------------------------------------- #
def set_threads(n=8):
    """CPU 环境下限制线程数，避免 128 核上线程争抢。"""
    try:
        torch.set_num_threads(int(n))
        torch.set_num_interop_threads(1)
    except Exception:
        torch.set_num_threads(int(n))


# --------------------------------------------------------------------------- #
# 骨干网络（与 EVA nets/tabnet.py、nets/mlp.py 完全同构）
# --------------------------------------------------------------------------- #
class TabNetAdult(nn.Module):
    """复刻 EVA nets/tabnet.py 的 TabNet。"""

    def __init__(self, input_features=13, num_classes=2, hidden=128):
        super(TabNetAdult, self).__init__()
        self.input_layer = nn.Linear(input_features, hidden)
        self.shared_layer_1 = nn.Linear(hidden, hidden)
        self.shared_layer_2 = nn.Linear(hidden, hidden)

        self.attention_1 = nn.Linear(hidden, 1)
        self.attention_2 = nn.Linear(hidden, 1)

        self.decision_1 = nn.Linear(hidden, hidden)
        self.decision_2 = nn.Linear(hidden, num_classes)
        self.fc = self.decision_2

    def _shared(self, x):
        out = torch.flatten(x, 1)
        out = F.relu(self.input_layer(out))
        out = F.relu(self.shared_layer_1(out))
        out = F.relu(self.shared_layer_2(out))
        a1 = torch.sigmoid(self.attention_1(out))
        a2 = torch.sigmoid(self.attention_2(out))
        return out * a1 * a2

    def forward(self, x):
        out = self._shared(x)
        out = F.relu(self.decision_1(out))
        return self.fc(out)

    def feature_extractor(self, x):
        """返回 decision 层之前的嵌入（GM-Matching 的 embeddings 用这个）。"""
        return self._shared(x)

    def get_last_layer(self):
        return self.fc


class MLPAdult(nn.Module):
    """复刻 EVA nets/mlp.py。"""

    def __init__(self, input_features=13, num_classes=2, hidden=128):
        super(MLPAdult, self).__init__()
        self.fc_1 = nn.Linear(input_features, hidden)
        self.fc_2 = nn.Linear(hidden, hidden)
        self.fc_3 = nn.Linear(hidden, num_classes)
        self.fc = self.fc_3

    def forward(self, x):
        out = torch.flatten(x, 1)
        out = F.relu(self.fc_1(out))
        out = F.relu(self.fc_2(out))
        return self.fc_3(out)

    def feature_extractor(self, x):
        out = torch.flatten(x, 1)
        out = F.relu(self.fc_1(out))
        return F.relu(self.fc_2(out))

    def get_last_layer(self):
        return self.fc_3


def build_model(model_name="tabnet", input_features=13, num_classes=2):
    name = str(model_name).lower()
    if name in ("tabnet", "tabnetadult"):
        return TabNetAdult(input_features, num_classes)
    if name in ("mlp", "mlpadult"):
        return MLPAdult(input_features, num_classes)
    raise ValueError("未知模型：%s（可选 tabnet / mlp）" % model_name)


# --------------------------------------------------------------------------- #
# 数据
# --------------------------------------------------------------------------- #
def load_adult_arrays(npz_path=DEFAULT_NPZ, normalized=True):
    """读取预处理好的 npz，返回 dict。

    normalized=True  -> 用 X_*_norm（z-score，训练集 mean/std）
    normalized=False -> 用 X_*（原始数值）
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            "找不到 %s，请先运行 mydatasets/prepare_adult.py" % npz_path
        )
    z = np.load(npz_path, allow_pickle=True)
    suffix = "_norm" if normalized else ""

    out = {
        "train_x": z["X_train" + suffix].astype(np.float32),
        "train_y": z["y_train"].astype(np.int64),
        "val_x": z["X_val" + suffix].astype(np.float32),
        "val_y": z["y_val"].astype(np.int64),
        "test_x": z["X_test" + suffix].astype(np.float32),
        "test_y": z["y_test"].astype(np.int64),
        "mean": z["mean"],
        "std": z["std"],
        "feature_names": [str(v) for v in z["feature_names"]],
        "class_names": [str(v) for v in z["class_names"]],
    }
    out["num_features"] = int(out["train_x"].shape[1])
    out["num_classes"] = int(len(out["class_names"]))
    return out


class AdultDataset(Dataset):
    """返回 (x_float32_tensor, int_label)。

    attributes 额外暴露 `data` / `targets`，供 dual-pruning 之类
    依赖 `dataset.targets` 的代码直接使用。
    """

    def __init__(self, x, y):
        self.data = np.asarray(x, dtype=np.float32)
        self.labels = np.asarray(y, dtype=np.int64)
        assert len(self.data) == len(self.labels)
        # dual-pruning 需要 [[label, index], ...] 结构
        self.targets = [[int(self.labels[i]), i] for i in range(len(self.labels))]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        if isinstance(index, (list, np.ndarray)):
            index = int(index[0])
        return torch.from_numpy(self.data[index]), int(self.labels[index])


def split_train_probe(x, y, probe_ratio=0.2, seed=48):
    """按类别把训练集拆成 probe(train_X_1) 与 sampling(train_X_2)。

    对齐 COPS 的 cifar_split.py：CIFAR10 每类 5000 张里 1000 张做 probe（20%）。
    """
    rng = np.random.RandomState(seed)
    probe_idx, sample_idx = [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_probe = max(1, int(round(len(idx) * probe_ratio)))
        probe_idx.extend(idx[:n_probe].tolist())
        sample_idx.extend(idx[n_probe:].tolist())
    probe_idx = np.array(sorted(probe_idx), dtype=np.int64)
    sample_idx = np.array(sorted(sample_idx), dtype=np.int64)
    return (x[probe_idx], y[probe_idx]), (x[sample_idx], y[sample_idx])


# --------------------------------------------------------------------------- #
# 自检
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    set_threads(8)
    d = load_adult_arrays()
    print("features :", d["num_features"])
    print("classes  :", d["num_classes"], d["class_names"])
    print("train/val/test:", d["train_x"].shape, d["val_x"].shape, d["test_x"].shape)
    print("train mean/std (should be ~0/1):", round(float(d["train_x"].mean()), 4),
          round(float(d["train_x"].std()), 4))
    for name in ("tabnet", "mlp"):
        m = build_model(name, d["num_features"], d["num_classes"])
        n_par = sum(p.numel() for p in m.parameters())
        out = m(torch.from_numpy(d["test_x"][:4]))
        emb = m.feature_extractor(torch.from_numpy(d["test_x"][:4]))
        print("%-7s params=%-6d logits=%s emb=%s" % (name, n_par, tuple(out.shape), tuple(emb.shape)))
    ds = AdultDataset(d["train_x"], d["train_y"])
    x0, y0 = ds[0]
    print("dataset[0]:", tuple(x0.shape), x0.dtype, "label", y0, type(y0).__name__)
    print("targets[0]:", ds.targets[0])
