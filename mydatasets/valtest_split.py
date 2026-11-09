"""在“官方 test 集”上按类分层切出 val / test 的公共工具。

口径与 `mydatasets/cifar10.py:38-52` 完全一致：

    dst_train = 官方 train 全量（不缩水）
    官方 test 按类 50/50 切成 dst_val / dst_test

为什么需要它
------------
`main0.py:122` 一次性解包 9 个值
`(channel, im_size, num_classes, class_names, mean, std, dst_train, dst_val, dst_test)`，
而 `mnist` 之外的 `qmnist` / `fashionmnist` / `svhn` 只返回 8 个（没有 val），
会直接 `ValueError: not enough values to unpack`。

val 在 EVA 里不是装饰：`methods/*.py` 的 `train_model()` 每个 epoch 在 val 上评估，
`acc > best_acc` 才 `torch.save()`，`patience` 次不升就早停；训练结束后加载那个
checkpoint 再在 test 上报数。所以 val 必须与 train、test 都不相交——**不能拿 test 当 val**。

与 cifar10.py 的两处有意差别
----------------------------
1) 固定随机种子（默认 0），不依赖全局 RNG。
   `cifar10.py` 用的是 `torch.utils.data.random_split`（走全局 RNG），而 `main0.py`
   从未设置 `torch.manual_seed`，所以 CIFAR 的 val/test 划分**每次运行都不一样**、
   不可复现。本模块改成由 `numpy.random.RandomState(seed)` 决定，跨进程稳定。
2) 划分结果第一次生成后写进 `<data_path>/split_valtest_<name>.json`，
   之后优先按该文件切分。只存**索引**，不复制任何图像数据。
"""
import json
import os

import numpy as np
from torch.utils.data import Subset


def _as_1d_array(labels):
    if hasattr(labels, "detach"):          # torch.Tensor
        labels = labels.detach().cpu().numpy()
    elif type(labels).__name__ == "Series":  # pandas.Series
        labels = labels.values
    return np.asarray(labels).reshape(-1)


def labels_of(dataset):
    """取数据集的标签数组；兼容 .targets / .labels 两种命名。"""
    for attr in ("targets", "labels"):
        if hasattr(dataset, attr):
            return _as_1d_array(getattr(dataset, attr))
    raise AttributeError(
        "%s 既没有 .targets 也没有 .labels，无法分层切分" % type(dataset).__name__
    )


def stratified_half(labels, num_classes, seed=0):
    """每个类各自随机打乱后对半切，返回 (val_idx, test_idx)，均为升序 list。"""
    arr = _as_1d_array(labels)
    rng = np.random.RandomState(seed)
    val_idx, test_idx = [], []
    for c in range(num_classes):
        idx = np.where(arr == c)[0]
        if len(idx) == 0:
            raise ValueError("类别 %d 在 test 集中一个样本都没有，无法按类对半切" % c)
        rng.shuffle(idx)
        half = len(idx) // 2
        val_idx.extend(idx[:half].tolist())
        test_idx.extend(idx[half:].tolist())
    val_idx.sort()
    test_idx.sort()
    return val_idx, test_idx


def _attach_idx(dataset, full_dataset, indices):
    """给 Subset 挂上 .targets / .labels，方便下游按类统计。"""
    labels = labels_of(full_dataset)
    sub = labels[np.asarray(indices)]
    dataset.targets = sub
    dataset.labels = sub
    return dataset


def split_test(full_test, num_classes, seed=0, data_path=None, name=None, verbose=True):
    """把官方 test 分层切成 (dst_val, dst_test) 两个 Subset。"""
    n = len(full_test)
    meta_path = None
    if data_path is not None and name is not None:
        meta_path = os.path.join(data_path, "split_valtest_%s.json" % name)

    val_idx = test_idx = None
    if meta_path and os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            if (meta.get("n") == n and meta.get("num_classes") == num_classes
                    and meta.get("seed") == seed):
                v, t = meta["val_idx"], meta["test_idx"]
                if v and t and max(max(v), max(t)) < n and not (set(v) & set(t)):
                    val_idx, test_idx = v, t
                    if verbose:
                        print("[valtest_split] %s: 复用 %s (val=%d, test=%d)"
                              % (name, meta_path, len(v), len(t)))
        except Exception as exc:
            if verbose:
                print("[valtest_split] %s: 忽略损坏的 %s (%r)" % (name, meta_path, exc))

    if val_idx is None:
        val_idx, test_idx = stratified_half(labels_of(full_test), num_classes, seed)
        if meta_path:
            try:
                with open(meta_path, "w", encoding="utf-8") as fh:
                    json.dump({
                        "dataset": name, "policy": "official test split 50/50 per class",
                        "seed": seed, "num_classes": num_classes, "n": n,
                        "n_val": len(val_idx), "n_test": len(test_idx),
                        "val_idx": val_idx, "test_idx": test_idx,
                    }, fh)
                if verbose:
                    print("[valtest_split] %s: 写入 %s" % (name, meta_path))
            except Exception as exc:
                if verbose:
                    print("[valtest_split] %s: 无法写 %s (%r)" % (name, meta_path, exc))

    dst_val = _attach_idx(Subset(full_test, val_idx), full_test, val_idx)
    dst_test = _attach_idx(Subset(full_test, test_idx), full_test, test_idx)
    if verbose:
        print("[valtest_split] %s: val=%d test=%d (每类约各半)"
              % (name, len(dst_val), len(dst_test)))
    return dst_val, dst_test


def rebuild_with_valtest(info, data_path, name, seed=0, verbose=True):
    """把原 loader 的返回值改写成“train 全量 + test 切半”的 9 元组。

    原 loader 可能返回 8 个（train, test）或 9 个（train, val, test）；
    两种情况都只保留 train / test，中间的 val 丢弃，改用官方 test 对半切。
    若原 dst_train 是个 Subset（作者从 train 里切走过一部分当 val），
    这里会还原成它的底层全量数据集——否则那部分样本既不在 train 也不在 val，白丢。
    """
    head = tuple(info[:6])
    rest = tuple(info[6:])
    if len(rest) < 2:
        raise ValueError("%s: 原 loader 至少要返回 train 和 test" % name)

    num_classes = head[2]
    dst_train, full_test = rest[0], rest[-1]
    full_train = getattr(dst_train, "dataset", dst_train)
    if full_train is not dst_train and verbose:
        print("[valtest_split] %s: train 由 %d 还原为全量 %d"
              % (name, len(dst_train), len(full_train)))

    dst_val, dst_test = split_test(
        full_test, num_classes, seed=seed, data_path=data_path, name=name, verbose=verbose
    )
    return head + (full_train, dst_val, dst_test)
