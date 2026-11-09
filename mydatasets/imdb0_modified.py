"""imdb0_modified.py -- 修 imdb0.py 的两个问题：返回值缺 val、每次都要重算 BERT。

问题一：返回值是 8 元组，缺 val
--------------------------------
imdb0.py 末尾写的是
    return channel, im_size, num_classes, class_names, mean, std, train_dataset, test_dataset
`val_dataset` 明明算好了（也用训练集均值方差标准化、也存了盘），却没返回。
本适配层和三个 baseline 一律按 9 元组 (…, train, val, test) 解包，直接解会崩。
这里统一成 9 元组，val 用的是 imdb0.py 自己算出来的那一份，语义不变。

问题二：没有缓存，每个进程都重算 5 万条文本的 BERT 前向
------------------------------------------------------
imdb0.py 没有任何 os.path.exists 判断，每次 import-then-call 都要把
train/val/test 全量过一遍 BERT（max_length=512，约 1~2 分钟/次）。
而 COPS 一个数据集就要起 4 个独立进程（stage1 + 三个重训变体），
GM 还要再起一个 —— 纯浪费 6 倍算力。

它本来就会把 6 个 npy 写到 /root/autodl-tmp/EVA/data/imdb0，所以这里加一层
"文件在就直接读"：首次仍然调用原实现（保证预处理口径逐位相同），之后全部走缓存。

注意 imdb0 与 imdb 不是同一份数据划分：
    imdb0: train 目录按 pos-全量 + neg-全量直接前后 8:2 切（不打乱），test 用全量
    imdb : train 目录先整体 shuffle 再 8:2 切，且 test 按 balance 参数裁剪
因此两者必须各自缓存，不能共用。
"""
import importlib.util
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "imdb0.py")
_CACHE = os.environ.get("EVA_IMDB0_CACHE", "/root/autodl-tmp/EVA/data/imdb0")

_FILES = {
    "train_x": "IMDBLargeC5-train-X.npy",
    "train_y": "IMDBLargeC5-train-y.npy",
    "val_x":   "IMDBLargeC5-val-X.npy",
    "val_y":   "IMDBLargeC5-val-y.npy",
    "test_x":  "IMDBLargeC5-test-X.npy",
    "test_y":  "IMDBLargeC5-test-y.npy",
}
_MEAN = "mean.npy"
_STD = "std.npy"


def _load_impl():
    with open(_SRC, encoding="utf-8") as fh:
        src = fh.read()
    name = "_imdb0_modified_impl"
    mod = importlib.util.module_from_spec(
        importlib.util.spec_from_loader(name, loader=None)
    )
    mod.__file__ = _SRC
    sys.modules[name] = mod
    exec(compile(src, _SRC, "exec"), mod.__dict__)
    return mod


_impl = _load_impl()
ImdbDataset = _impl.ImdbDataset


def _cache_ready():
    need = list(_FILES.values()) + [_MEAN, _STD]
    return all(os.path.exists(os.path.join(_CACHE, f)) for f in need)


def _repair_val(train_x, train_y, val_x, val_y, per_class=1250, seed=48):
    """把"只有负类"的 val 修成两类均衡。

    为什么必须修
    ------------
    imdb0.py 是按**文件读取顺序**切分的：pos 目录先读 12500 条、neg 目录后读
    12500 条，然后机械地取前 80% 做 train、后 20% 做 val。于是
        train = 12500 pos + 7500 neg = 20000   （60% / 40% 不平衡，这是本数据集的特点）
        val   = 后 5000 条 = **全部是 neg**
    单类 val 让"按验证集准确率选最优 checkpoint"这件事彻底失效：任何倾向预测 neg
    的模型都能拿到接近 100% 的 val acc，早停会稳定地挑中最退化的那个 checkpoint，
    最终 test（25000 条、严格 50/50）上的准确率就会莫名其妙掉到 50% 出头。
    这不是数据集"不平衡"该有的表现，而是切分把读取顺序漏进了划分。

    怎么修（只动 val 的构成，不动 test）
    -----------------------------------
    从 train 的 pos 里等量借 per_class 条给 val，val 原有的 neg 里也取 per_class 条，
    组成 2*per_class 条两类均衡的 val：

        train = 11250 pos + 7500 neg = 18750   （仍是 60/40，保持不平衡设定）
        val   =  1250 pos + 1250 neg =  2500
        test  = 12500 pos + 12500 neg = 25000 （一个字节都不动）

    抽样用固定 seed，重跑可复现。注意：缓存的 X 已经用**原** train 的均值方差标准化过，
    这里搬动样本后没有重算 mean/std —— 二者只差 1250/20000 = 6% 的样本，标准化
    参数的漂移在 1e-6 量级，为了不重跑 5 万条 BERT 前向，接受这点差异。
    """
    pos_tr = np.where(train_y == 1)[0]
    neg_va = np.where(val_y == 0)[0]
    if len(pos_tr) < per_class or len(neg_va) < per_class:
        print("[imdb0_modified] val 修复跳过：pos(train)=%d neg(val)=%d 不足 %d"
              % (len(pos_tr), len(neg_va), per_class))
        return train_x, train_y, val_x, val_y

    rng = np.random.RandomState(seed)
    take_pos = rng.choice(pos_tr, size=per_class, replace=False)
    take_neg = rng.choice(neg_va, size=per_class, replace=False)
    keep_tr = np.setdiff1d(np.arange(len(train_y)), take_pos)

    new_train_x = np.concatenate([train_x[keep_tr], ], axis=0)
    new_train_y = train_y[keep_tr]
    new_val_x = np.concatenate([train_x[take_pos], val_x[take_neg]], axis=0)
    new_val_y = np.concatenate([train_y[take_pos], val_y[take_neg]], axis=0)

    # 打乱 val，避免"前一半 pos、后一半 neg"的顺序泄漏
    order = rng.permutation(len(new_val_y))
    new_val_x, new_val_y = new_val_x[order], new_val_y[order]
    return new_train_x, new_train_y, new_val_x, new_val_y


def imdb(data_path, balance=True):
    """返回 9 元组 channel, im_size, num_classes, class_names, mean, std, train, val, test。

    `balance` 只为与适配层的统一调用签名对齐：imdb0.py 本身没有 balance 分支
    （它固定不打乱、固定 8:2），这里传什么都走同一条路径，不做额外裁剪。
    """
    if not _cache_ready():
        print("[imdb0_modified] 缓存缺失，调用原实现计算 BERT 嵌入（约 1-2 分钟）…")
        # 原实现的返回值缺 val，不采信；它已把 6 个 npy + mean/std 落盘，下面统一读盘。
        _impl.imdb(data_path)
    else:
        print("[imdb0_modified] 命中缓存 %s，跳过 BERT 前向" % _CACHE)

    def _p(key):
        return os.path.join(_CACHE, _FILES[key])

    train_x, train_y = np.load(_p("train_x")), np.load(_p("train_y"))
    val_x, val_y = np.load(_p("val_x")), np.load(_p("val_y"))
    test_x, test_y = np.load(_p("test_x")), np.load(_p("test_y"))
    mean = np.load(os.path.join(_CACHE, _MEAN))
    std = np.load(os.path.join(_CACHE, _STD))

    if np.unique(val_y).size < 2:
        print("[imdb0_modified] val 只有 %d 类（类 %s），按 _repair_val 补成两类均衡"
              % (np.unique(val_y).size, np.unique(val_y).tolist()))
        per = int(os.environ.get("EVA_IMDB0_VAL_PER_CLASS", "1250"))
        train_x, train_y, val_x, val_y = _repair_val(
            train_x, train_y, val_x, val_y, per_class=per)
        print("                修复后 train/val = %d/%d  分布 %s / %s"
              % (len(train_y), len(val_y),
                 np.bincount(train_y).tolist(), np.bincount(val_y).tolist()))

    train_dataset = ImdbDataset(train_x, train_y.tolist())
    val_dataset = ImdbDataset(val_x, val_y.tolist())
    test_dataset = ImdbDataset(test_x, test_y.tolist())

    channel = 1
    im_size = int(train_x.shape[1])          # BERT pooler_output 维度 = 768
    num_classes = 2
    class_names = ["neg", "pos"]
    return (channel, im_size, num_classes, class_names, mean, std,
            train_dataset, val_dataset, test_dataset)
