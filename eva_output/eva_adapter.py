# -*- coding: utf-8 -*-
"""eva_adapter.py —— 把 EVA 预处理好的 9 个数据集统一喂给 COPS / GM-Matching / dual-pruning。

放在基座仓库外面（`/root/eva_output/eva_adapter.py`），三个仓库各自软链接一份，
避免三份拷贝漂移。用法：

    import eva_adapter as EVA
    EVA.DATASETS                       # 支持的数据集名 -> 规格
    EVA.load("mnist")                  # 一次性拿到 train/val/test + 元信息
    EVA.part("mnist", "train")         # 只要其中一个 split
    EVA.cops_views("mnist")            # COPS 要的 (probe, sampling, test, meta)
    EVA.build_model("mnist")           # 与三个 baseline 共用的骨干网络
    EVA.select(ds, idx)                # 按索引取子集（核心集），带 .targets

为什么不让三个仓库各自解析原始数据
--------------------------------
三个仓库对同一份数据有不同期待：COPS 要 probe/sampling/test 三段 npy、
GM 要 (dataset, n_classes)、DUAL 要 train 侧 `.targets = [[label, idx], ...]`。
如果各写一套预处理，同一数据集在三个 baseline 里就会落到不同的划分/不同的归一化上，
横向比较立刻失效。这里只保留**一份**解析逻辑，且它直接调用 EVA 自己的
`mydatasets/*_modified.py`，因此 train/val/test 与 EVA 主实验逐位相同。

三条硬约定
----------
1. `__getitem__` 一律返回 `(float32 CHW tensor, int label)`。
   返回 python int 而不是 0-dim tensor：GM-Matching 的选核集代码直接拿 label 当 dict key，
   若给 tensor 会退化成 N 个"类别"。
2. 不做任何 uint8 往返。图像直接用 EVA 数据集吐出的、已经 Normalize 过的张量，
   所以不存在 `(x*std+mean)*255` 再 round 的像素级误差。
3. 内存：EVA 的图像数据集是惰性的（存路径/JPEG 字节，取时才解码），
   本模块只做惰性包装，不 materialize。容器 cgroup 上限 2GB，这点很关键。

关于 COPS 的 probe / sampling 切分
---------------------------------
`split_train_probe` 复刻 `COPS-main/adult_adapter.py` 的口径：按类各取 20% 做 probe
（COPS 用它训练不确定性模型），其余 80% 做 sampling 池（COPS 从这个池子里选核心集）。
`probe_ratio=0.2, seed=48` 与 adult 那次完全一致，所以跨数据集可比。
"""
import importlib.util
import os
import random as _pyrandom
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

EVA_ROOT = os.environ.get("EVA_ROOT", "/root/chaichengliang/VLDB2027/EVA/EVA")
DATA_ROOT = os.environ.get("EVA_DATA_ROOT", "/root/autodl-tmp/EVA/data")
MYDATASETS = os.path.join(EVA_ROOT, "mydatasets")

# name -> (脚本文件, 加载函数, data_path 相对 DATA_ROOT；"." 表示 DATA_ROOT 本身)
# 本次要跑 baseline 的九个数据集。它们必须全部经由本适配层取数，原因有二：
#   (1) 划分一致：cifar100 在 COPS / DUAL 里原本有原生分支，但原生用的是官方
#       10000 张测试集，而 EVA 的划分是 val 5000 / test 5000。若走原生分支，
#       baseline 与 EVA 主实验就不在同一个测试集上，数值不可比。
#   (2) 预处理一致：归一化、通道数、表格标准化都在 mydatasets/*_modified.py 里，
#       绕开适配层就会和主实验差一个预处理。
ADAPTER_DATASETS = (
    "cifar100", "covtype", "fashionmnist", "HR", "mnist", "qmnist", "SVHN",
    "tinyimagenet", "mnist2mnistm", "cifar10", "adult", "bike", "House",
    "news", "imdb", "imdb0", "imdb_large", "imagenet",
)

REGISTRY = {
    "cifar100":     ("cifar100_modified.py",     "CIFAR100",     "cifar100"),
    "covtype":      ("covtype_modified.py",      "covtype",      "covtype"),
    "fashionmnist": ("fashionmnist_modified.py", "FashionMNIST", "fashionmnist"),
    "HR":           ("HR_modified.py",           "HR",           "HR"),
    "mnist":        ("mnist_modified.py",        "MNIST",        "mnist"),
    "qmnist":       ("qmnist_modified.py",       "QMNIST",       "qmnist"),
    "SVHN":         ("svhn_modified.py",         "SVHN",         "SVHN"),
    "tinyimagenet": ("tinyimagenet_modified.py", "TinyImageNet", "tinyimagenet"),
    "mnist2mnistm": ("mnist2mnistm_modified.py", "Mnist2MnistM", "mnist2mnistm"),
    # cifar10 / adult 三个仓库本来就能跑，这里一并登记，方便统一驱动脚本
    "cifar10":      ("cifar10.py",               "CIFAR10",      "cifar10"),
    "adult":        ("adult_modified.py",        "adult",        "adult"),
    # ---- 本轮扩到 18 个数据集：新增的 7 个 ----
    "bike":       ("bike_modified.py",              "Bike",         "bike"),
    "House":      ("House_modified.py",             "House",        "House"),
    "news":       ("news.py",                       "news",         "news"),
    "imdb":       ("imdb_modified.py",              "imdb",         "imdb"),
    "imdb0":      ("imdb0_modified.py",             "imdb",         "imdb"),
    "imdb_large": ("imdb_large_modified.py",        "IMDBLarge",    "imdb/imdb_large"),
    "imagenet":   ("imagenet100.py",                "ImageNet",     "imagenet100"),
}

DATASETS = tuple(REGISTRY)

# 数据集内部 train/val 切分用的固定种子。有些加载器用 random_split 却没传
# generator（见 _call_loader 的注释），切分依赖 torch 全局 RNG，不固定就会
# 每次运行都不一样。取 48 与 COPS probe 切分的 seed 保持一致，便于记忆。
SPLIT_SEED = 48

_CACHE = {}


def set_threads(n=1):
    """0.5 核环境下多余线程只会互相抢；GPU 机器上留 1 也无害。"""
    try:
        torch.set_num_threads(int(n))
        torch.set_num_interop_threads(1)
    except Exception:
        torch.set_num_threads(int(n))


# --------------------------------------------------------------------------- #
# 加载
# --------------------------------------------------------------------------- #
def _load_module(name):
    fname, fname_func, rel = REGISTRY[name]
    path = os.path.join(MYDATASETS, fname)
    if not os.path.exists(path):
        raise FileNotFoundError("找不到 %s（数据集 %s）" % (path, name))
    if MYDATASETS not in sys.path:
        sys.path.insert(0, MYDATASETS)   # valtest_split 就在同目录
    mod_name = "_eva_ds_" + os.path.splitext(fname)[0]
    if mod_name in sys.modules:
        return sys.modules[mod_name], fname_func, rel
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod, fname_func, rel


def _smoke_limit():
    """读 EVA_SMOKE_LIMIT；未设置或非法一律返回 0（=不截断，正式实验的默认行为）。"""
    raw = os.environ.get("EVA_SMOKE_LIMIT", "").strip()
    if not raw:
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def _truncate(meta, total):
    """按类等比例把 train/val/test 各截到 total 张附近，仅供 CPU 冒烟。

    保持每类样本数相同，是为了让 num_classes、per-class probe 切分、指标计算
    在截断后仍然成立（否则某些类会一个样本都不剩，直接除零）。
    """
    out = dict(meta)
    for split in ("train", "val", "test"):
        base = meta.get(split)
        if base is None:
            continue
        if meta.get("is_regression"):
            keep = np.sort(np.random.RandomState(SPLIT_SEED).permutation(len(base))[:total])
            out[split] = _wrap(base, meta, indices=keep)
            continue
        labs = _labels_of(base)
        if labs is None:
            continue
        labs = np.asarray(labs).reshape(-1)
        cls = np.unique(labs)
        per = max(1, int(total) // max(1, len(cls)))
        keep = [np.where(labs == c)[0][:per] for c in cls]
        out[split] = _wrap(base, meta, indices=np.sort(np.concatenate(keep)))
    return out


def _call_loader(func, data_path, balance):
    """固定种子后再调用数据集加载函数，调用完把三套全局 RNG 状态还原。

    为什么必须这么做
    ----------------
    各数据集的 train/val 切分散落在三种互不相干的随机源上，而且**都没设种子**：

      * torch.utils.data.random_split  -> torch 全局 RNG
        HR.py:90、mnist2mnistm.py:172
      * np.random.shuffle              -> numpy 全局 RNG
        covtype.py:46 的 classwise_split（balance=True 分支走的就是它）
      * train_test_split(..., random_state=42)  这一处反而是固定的

    实测（同一份数据、连跑三次、每次独立进程）：
        HR        train labels = a50b41040e24 / 3cd17fc920c1 / ce32e3cc7790
        covtype   train labels = 33586b284c98 / c996211a21c5 / ...
    —— 每跑一次切分都不同。

    后果不是崩溃，而是**静默地不可复现**：主实验与三个 baseline 是各自独立的
    进程，它们看到的 HR / covtype / mnist2mnistm 的 val 划分互不相同，于是
    同一份核心集在不同方法下的分数没有可比性，重跑也对不上。

    为什么在适配层改、而不是改数据集源码
    ------------------------------------
    `*_modified.py` 的职责是"只修原作者写错的地方"（写死路径、写死类别数），
    不该顺手改人家的切分语义；而且 HR_modified 是靠正则删一行实现的，再往里塞
    generator 会让那个正则更脆。放在唯一的取数入口上，数据集无关、一处生效。

    为什么用完要还原
    ----------------
    调用方（baseline 自己的 train.py）在后面会设自己的种子并要求随机性可复现。
    如果这里留下被改过的全局状态，等于替调用方改了种子，反而制造新的不可复现。
    """
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if dev.type == "cuda" else None
    np_state = np.random.get_state()
    py_state = _pyrandom.getstate()

    torch.manual_seed(SPLIT_SEED)
    np.random.seed(SPLIT_SEED)
    _pyrandom.seed(SPLIT_SEED)
    try:
        return func(data_path, balance)
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        np.random.set_state(np_state)
        _pyrandom.setstate(py_state)


def load(name, balance=True):
    """返回 dict：channel / im_size / num_classes / class_names / mean / std / train / val / test。

    结果按 name 缓存：COPS 的 stage1 与两组重训是三个独立进程，各自加载一次；
    但同一进程内多次调用（例如 GM 先取 train 再取 test）不该重复解析。
    """
    _lim = _smoke_limit()
    _key = (name, bool(balance), _lim, os.environ.get("EVA_REG_BINS", "0"))
    if _key in _CACHE:
        return _CACHE[_key]
    if name not in REGISTRY:
        raise KeyError("未知数据集 %r；可选：%s" % (name, ", ".join(sorted(REGISTRY))))

    mod, func_name, rel = _load_module(name)
    func = getattr(mod, func_name, None)
    if func is None:
        raise AttributeError("%s 里没有 %s()" % (mod.__file__, func_name))
    data_path = DATA_ROOT if rel == "." else os.path.join(DATA_ROOT, rel)

    info = _call_loader(func, data_path, balance)
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_val, dst_test = info
    meta = {
        "name": name,
        "channel": int(channel),
        "im_size": list(im_size) if hasattr(im_size, "__len__") else int(im_size),
        "num_classes": int(num_classes),
        "class_names": list(class_names),
        "mean": [float(v) for v in np.asarray(mean).reshape(-1)],
        "std": [float(v) for v in np.asarray(std).reshape(-1)],
        "train": dst_train,
        "val": dst_val,
        "test": dst_test,
    }
    meta["is_tabular"] = _is_tabular(meta)
    # 回归判定：一维向量 + 单输出。bike / House 属于这一类。
    meta["is_regression"] = bool(meta["num_classes"] == 1 and meta["is_tabular"])
    if meta["is_regression"]:
        meta = _apply_reg_bins(meta)
    if _lim:
        # 注意放在 _is_tabular 之后：表格判定要看真实样本的维度，截断不影响它，
        # 但先判形态再截断语义更清楚。
        meta = _truncate(meta, _lim)
    _CACHE[_key] = meta
    return meta


def _is_tabular(meta):
    """表格数据判定：把单个样本的最后一维当作特征数。

    判据是"样本是不是一维向量且不是灰度图"——covtype 是 (54,)、HR 是 (13,)，
    而 mnist 是 (1,28,28)。用一次真实取样判断，比维护一张手写表更不容易出错。
    """
    x, _ = _raw_item(meta["train"], 0)
    return x.dim() == 1


def _raw_item(ds, i):
    out = ds[i]
    x, y = out[0], out[1]
    if not torch.is_tensor(x):
        x = torch.as_tensor(np.asarray(x), dtype=torch.float32)
    else:
        x = x.float()
    return x, int(y)


def part(name, which):
    if which not in ("train", "val", "test"):
        raise ValueError("which 必须是 train/val/test，收到 %r" % which)
    return _wrap(load(name)[which], load(name))


# --------------------------------------------------------------------------- #
# 数据集包装
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# 回归数据集（bike / House）
# --------------------------------------------------------------------------- #
def _targets_float(ds):
    """取连续目标值（回归）。优先向量化读 `.labels`/`.targets`，取不到再逐条读。

    退化路径很慢（House 的 20195 条要逐条 `__getitem__`），但只在底层数据集
    既没有 `.labels` 也没有 `.targets` 时才会走到，属于兜底。
    """
    vals = getattr(ds, "values", None)
    if vals is not None:
        return np.asarray(vals, dtype=np.float64).reshape(-1)
    if isinstance(ds, Wrapped):
        base_vals = _targets_float(ds.base)
        return base_vals if ds.indices is None else base_vals[np.asarray(ds.indices)]
    for attr in ("labels", "targets"):
        if hasattr(ds, attr):
            raw = getattr(ds, attr)
            if torch.is_tensor(raw):
                raw = raw.detach().cpu().numpy()
            arr = np.asarray(raw, dtype=np.float64)
            if arr.size and arr.reshape(-1).size == len(ds):
                return arr.reshape(-1)
    out = np.empty(len(ds), dtype=np.float64)
    for i in range(len(ds)):
        y = ds[i][1]
        out[i] = float(np.asarray(y).reshape(-1)[0])
    return out


def _apply_reg_bins(meta, k=None):
    """按训练集 k 分位把回归目标离散成 k 类（仅用于核心集选择）。

    分位点只从 **train** 算，再套到 val/test 上，避免分布泄漏；
    重复分位点会被 np.unique 合并，所以实际箱数可能少于 k（bike 的目标是整数
    计数，低分段大量并列值就会合并）。
    """
    k = int(k if k is not None else (os.environ.get("EVA_REG_BINS", "0") or 0))
    if k <= 1:
        return meta
    vals = _targets_float(meta["train"])
    qs = np.linspace(0.0, 1.0, k + 1)[1:-1]
    edges = np.unique(np.quantile(vals, qs))
    out = dict(meta)
    out["reg_bin_edges"] = edges
    out["num_classes"] = int(len(edges) + 1)
    print("[eva_adapter] %s 是回归任务：按训练集分位切成 %d 个箱（仅用于核心集选择）"
          % (meta.get("name"), out["num_classes"]))
    return out


class Wrapped(Dataset):
    """惰性包装 EVA 数据集：统一成 (float32 tensor, int label)，并暴露 .targets。

    `.targets` 是 DUAL 与 GM 都需要的：DUAL 要求 `[[label, idx], ...]` 结构，
    GM 的部分代码按 label 分组。这里从 EVA 数据集的 `.targets`/`.labels` 取，
    取不到就退化成逐样本读取（慢，但正确）。
    """

    def __init__(self, base, meta=None, indices=None):
        self.base = base
        self.meta = meta or {}
        self.indices = None if indices is None else np.asarray(indices, dtype=np.int64)
        # 回归：标签是连续目标值，绝不能走下面 int 截断那条路
        # （bike 的目标是 z-score，int(1.13)=1 / int(-1.03)=-1，会产生非法类别 -1）。
        # 判定放在 Wrapped 里而不是子类里，是因为 Weighted / Indexed 会抽 ds.base
        # 重建 Wrapped，挂在子类上的能力会被它们抹掉。
        self.is_regression = bool(self.meta.get("is_regression"))
        if self.is_regression:
            vals = _targets_float(base)
            if self.indices is not None:
                vals = vals[self.indices]
            self.values = np.asarray(vals, dtype=np.float64).reshape(-1)
            edges = self.meta.get("reg_bin_edges")
            if edges is None:
                self.bins = None
                # 不分箱时 labels 只在 DUAL/GM 的"按类分组"里用，回归不走那条路，
                # 给全 0 占位，避免下游拿到截断值后当真。
                self.labels = np.zeros(len(self.values), dtype=np.int64)
            else:
                self.bins = np.digitize(self.values, edges).astype(np.int64)
                self.labels = self.bins
            self._targets = None
            return
        labs = _labels_of(base)
        if labs is None:
            labs = np.array([_raw_item(base, i)[1] for i in range(len(base))], dtype=np.int64)
        if self.indices is not None:
            labs = labs[self.indices]
        self.labels = np.asarray(labs, dtype=np.int64).reshape(-1)
        self._targets = None

    # DUAL 要求 `.targets` 是 [[label, index], ...]。但这个结构是 Python 对象的
    # 列表——covtype 的 465k 行要占上百 MB，而 COPS / GM 根本不用它。
    # 改成首次访问才构造，并在 __init__ 里保持为 None。
    @property
    def targets(self):
        if self._targets is None:
            self._targets = [[int(self.labels[i]), i] for i in range(len(self.labels))]
        return self._targets

    @targets.setter
    def targets(self, value):
        self._targets = value

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        if isinstance(i, (list, np.ndarray, torch.Tensor)):
            i = int(np.asarray(i).reshape(-1)[0])
        real = int(self.indices[i]) if self.indices is not None else int(i)
        if self.is_regression:
            x, _ = _raw_item(self.base, real)
            if self.bins is not None:
                return x, int(self.bins[i])          # 选择阶段：返回箱号
            # 评估阶段：返回 (1,) 浮点张量，collate 成 (B,1) 才能和 model(x) 的
            # (B,1) 输出对上 MSELoss
            return x, torch.tensor([float(self.values[i])], dtype=torch.float32)
        x, y = _raw_item(self.base, real)
        return x, int(y)


def _wrap(base, meta=None, indices=None):
    """统一的包装工厂（等价于 Wrapped(...)，只为让 5 个构造点读起来一致）。"""
    return Wrapped(base, meta, indices)


def _labels_of(ds):
    """取标签数组；兼容 .targets / .labels / Subset。"""
    if isinstance(ds, Wrapped):
        return ds.labels          # 短路，避免触发 Wrapped.targets 的惰性构造
    for attr in ("targets", "labels"):
        if hasattr(ds, attr):
            raw = getattr(ds, attr)
            if torch.is_tensor(raw):
                raw = raw.detach().cpu().numpy()
            arr = np.asarray(raw)
            if arr.ndim == 2 and arr.shape[1] == 10:
                # QMNIST 的原始标签是 N x 10 的标签记录矩阵，第 0 列才是类别
                # （mydatasets/qmnist.py:14 做过同样的切片）
                arr = arr[:, 0]
            arr = arr.reshape(-1)
            if len(arr) == len(ds):
                return arr
    if hasattr(ds, "indices") and hasattr(ds, "dataset"):
        sub = _labels_of(ds.dataset)
        if sub is not None:
            return sub[np.asarray(ds.indices)]
    return None


def select(ds, indices):
    """按索引取子集（核心集）。返回 Wrapped，带正确的 .targets。"""
    meta = getattr(ds, "meta", {})
    base = ds.base if isinstance(ds, Wrapped) else ds
    if isinstance(ds, Wrapped) and ds.indices is not None:
        indices = np.asarray(ds.indices)[np.asarray(indices)]
    return _wrap(base, meta, indices)


class Weighted(Wrapped):
    """多返回一个逐样本损失权重，供 COPS 的 reweight_clip 使用。

    只换 `__getitem__` 的返回元数（3 元组），其余与 Wrapped 一致。
    DataLoader 默认 collate 会把 float 权重堆成 (B,) 张量，正好对上
    `retraining_withy.py` 里 `inputs, targets, tr_weight = ...` 的解包。
    """

    def __init__(self, ds, weights):
        super(Weighted, self).__init__(
            ds.base if isinstance(ds, Wrapped) else ds,
            getattr(ds, "meta", {}),
            getattr(ds, "indices", None),
        )
        self.w = np.asarray(weights, dtype=np.float32).reshape(-1)
        assert len(self.w) == len(self), "权重数 %d != 样本数 %d" % (len(self.w), len(self))


    def __getitem__(self, i):
        if isinstance(i, (list, np.ndarray, torch.Tensor)):
            i = int(np.asarray(i).reshape(-1)[0])
        x, y = Wrapped.__getitem__(self, i)
        return x, y, float(self.w[i])


class Indexed(Wrapped):
    """多返回一个样本下标，供 DUAL 记录训练动力学用。

    DUAL 的动力学训练循环是：
        for input, target in loader:
            y = target[0]        # 标签批次
            index = target[1]    # 下标批次，存进 {epoch}_Index.npy
    即每个样本必须自带下标。Wrapped 只给 (x, y)，直接喂进去 `target[0]` 会退化为
    一个 0 维标量，cross_entropy 报
        ValueError: Expected input batch_size (16) to match target batch_size (0)

    下标取**局部**下标（0..N-1，按 Wrapped 的顺序），不是底层数据集里的绝对位置。
    因为 importance_evaluation 生成的 dual_mask_T30 长度是 len(Wrapped)，
    下游 `mask[-k:]` 取 top-k 时用的也是这个顺序 —— 两者必须同一套编号。

    只给 train.py（train_subset.py 用的是 `target.to(device)` 的二元组协议，
    套上本类会直接把 list 喂进 .to()，反而更错）。
    """

    def __getitem__(self, i):
        if isinstance(i, (list, np.ndarray, torch.Tensor)):
            i = int(np.asarray(i).reshape(-1)[0])
        x, y = Wrapped.__getitem__(self, i)
        return x, [int(y), int(i)]


# --------------------------------------------------------------------------- #
# COPS: probe / sampling / test
# --------------------------------------------------------------------------- #
def split_train_probe(y, probe_ratio=0.2, seed=48):
    """按类把训练集拆成 (probe_idx, sampling_idx)。

    与 `COPS-main/adult_adapter.py::split_train_probe` 完全同构：
    每类打乱后取 round(20%) 做 probe。COPS 用 probe 训练不确定性模型，
    用 sampling 池选核心集 —— 两者不能重叠，否则不确定性分数是被测样本自己算的。
    """
    y = np.asarray(y).reshape(-1)
    rng = np.random.RandomState(seed)
    probe_idx, sample_idx = [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_probe = max(1, int(round(len(idx) * probe_ratio)))
        probe_idx.extend(idx[:n_probe].tolist())
        sample_idx.extend(idx[n_probe:].tolist())
    return np.array(sorted(probe_idx), dtype=np.int64), np.array(sorted(sample_idx), dtype=np.int64)


def cops_views(name, probe_ratio=0.2, seed=48):
    """返回 (probe, sampling, test, meta)，供 COPS 的 stage1 与两组重训使用。"""
    meta = load(name)
    train = _wrap(meta["train"], meta)
    probe_idx, sample_idx = split_train_probe(train.labels, probe_ratio, seed)
    probe = select(train, probe_idx)
    sampling = select(train, sample_idx)
    test = _wrap(meta["test"], meta)
    return probe, sampling, test, meta


# --------------------------------------------------------------------------- #
# 骨干网络
# --------------------------------------------------------------------------- #
def _resnet18(in_ch, num_classes, small_input):
    """torchvision ResNet18 的小输入改造版。三个 baseline 共用，保证横比干净。

    small_input=True（32x32 / 28x28）时把 7x7 stride2 主干换成 3x3 stride1 且去掉
    maxpool —— 这是 CIFAR 系列的标准做法，否则 28x28 的 MNIST 到 layer1 只剩 7x7。
    """
    from torchvision.models import resnet18
    net = resnet18(num_classes=num_classes)
    if small_input:
        net.conv1 = nn.Conv2d(in_ch, 64, kernel_size=3, stride=1, padding=1, bias=False)
        net.maxpool = nn.Identity()
    else:
        net.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return net


class TabMLP(nn.Module):
    """表格骨干：与 nets/mlp.py、adult_adapter.MLPAdult 同构（in ->128 ->128 ->C）。"""

    def __init__(self, input_features, num_classes, hidden=128):
        super(TabMLP, self).__init__()
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
        return F.relu(self.fc_2(F.relu(self.fc_1(out))))

    def get_last_layer(self):
        return self.fc_3


def build_model(name, arch=None, num_classes=None):
    """按数据集挑骨干。arch 可显式覆盖为 resnet18 / mlp。"""
    meta = name if isinstance(name, dict) else load(name)
    arch = (arch or ("mlp" if meta["is_tabular"] else "resnet18")).lower()
    nc = int(num_classes or meta["num_classes"])

    if arch in ("mlp", "tabnet", "tabmlp"):
        feat = int(np.prod(meta["im_size"])) if not meta["is_tabular"] else int(meta["im_size"][0] if isinstance(meta["im_size"], list) else meta["im_size"])
        return TabMLP(feat, nc)
    if arch.startswith("resnet18"):
        ch = int(meta["channel"])
        size = meta["im_size"][0] if isinstance(meta["im_size"], list) else meta["im_size"]
        return _resnet18(ch, nc, small_input=int(size) <= 64)
    raise ValueError("未知骨干 %r（可选 resnet18 / mlp）" % arch)


def input_dim(meta):
    if meta["is_tabular"]:
        return int(meta["im_size"][0] if isinstance(meta["im_size"], list) else meta["im_size"])
    return int(meta["channel"]) * int(np.prod(meta["im_size"]))


def describe(name):
    m = load(name)
    return ("%-13s channel=%d im_size=%s classes=%-3d tabular=%-5s "
            "train=%-7d val=%-6d test=%-6d"
            % (name, m["channel"], m["im_size"], m["num_classes"], m["is_tabular"],
               len(m["train"]), len(m["val"]), len(m["test"])))


# --------------------------------------------------------------------------- #
# 自检
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    set_threads(1)
    names = sys.argv[1:] or list(DATASETS)
    for nm in names:
        if nm not in REGISTRY:
            print("%-13s 未登记，跳过" % nm)
            continue
        try:
            m = load(nm)
            print(describe(nm), flush=True)
            x, y = m["train"][0]
            x = torch.as_tensor(x).float() if not torch.is_tensor(x) else x.float()
            print("    sample: shape=%s dtype=%s label=%s"
                  % (tuple(x.shape), x.dtype, y), flush=True)
            net = build_model(m)
            n_par = sum(p.numel() for p in net.parameters())
            with torch.no_grad():
                out = net(x.unsqueeze(0).float())
            print("    model : %-9s params=%-9d out=%s"
                  % (type(net).__name__, n_par, tuple(out.shape)), flush=True)
            if not m["is_tabular"] and m["channel"] == 3:
                p, s, t, _ = cops_views(nm)
                ov = len(set(np.asarray(p.indices).tolist()) & set(np.asarray(s.indices).tolist()))
                print("    cops  : probe=%d sampling=%d test=%d  索引交集=%d"
                      % (len(p), len(s), len(t), ov), flush=True)
        except Exception as exc:
            import traceback
            print("%-13s FAILED: %s: %s" % (nm, type(exc).__name__, exc), flush=True)
            traceback.print_exc()
