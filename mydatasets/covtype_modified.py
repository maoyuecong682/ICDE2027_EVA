"""covtype_modified.py -- 让 covtype.py 能用本机这份原始 covtype.data。

covtype.py 的 `load_covertype_dataset()` 只认两个 joblib 文件：
    <data_path>/covertype/samples_py3
    <data_path>/covertype/targets_py3
而本机只有原始的 UCI covertype 明文：<data_path>/covtype.data
（581012 行 x 55 列，前 54 列是特征，最后一列是 1..7 的类别标签，无表头）。

本脚本先由原始文件生成那两个 joblib 文件（幂等，已存在则跳过），再委托原 covtype()，
因此切分、类别平衡、标准化、返回顺序都与原实现完全一致。

生成规则：X = 前 54 列（float32），y = 最后一列（int64，保持 1..7，由原函数减 1）。

【修正 dst_test = dst_val 的数据泄漏】
原 `covtype.py` 在构造完 val 之后直接写：

    dst_val = ConvertypeDataset(data, labels, val_indices)
    # Test data can be the same as the validation data in this setup
    dst_test = dst_val

也就是**验证集和测试集是同一批 116205 个样本**（还是同一个对象）。而 EVA 的
`train_model()` 是拿 val 选 checkpoint / 早停、最后再在 test 上报数的，所以这等于
用挑模型的那批样本当考卷 —— 任何 covtype 数字都会被系统性高估。

【连带发现：归一化被执行了两次】
因为 `dst_test is dst_val`，紧随其后的三行

    dst_train.data = (dst_train.data - mean) / (std + 1e-8)
    dst_val.data   = (dst_val.data   - mean) / (std + 1e-8)
    dst_test.data  = (dst_test.data  - mean) / (std + 1e-8)   # <- 又对同一块池子做了一遍

使 val/test 池被归一化了两次，而 train 只归一化一次 —— train 与 test 的尺度根本不一致。
二次归一化后的值 y = ((x-m)/s - m)/s，还原成只归一化一次只需 z = y*s + m。
本脚本在包装层做这一步还原，再用 `valtest_split.py` 按类 50/50 切成真正的 val / test。

【注意：covtype 的 val/test 成员每次运行会变】
`classwise_split()` 末尾有一句 `np.random.shuffle(val_indexes)`（未设种子），
所以那份 val 池的**顺序**每次运行都不同；而这里是按位置在其上切分，成员随之变化。
按用户 2026-10-25 的决定（"随机种子不做处理，就当现在写死的是随机的"），不固定种子，
因此**不为 covtype 写切分缓存文件**（写了也会因底层顺序变动而失真）。
"""
import importlib.util
import os
import sys

import joblib
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from valtest_split import rebuild_with_valtest  # noqa: E402

_SRC = os.path.join(_HERE, "covtype.py")
_SEED = 0


def _load_impl():
    name = "_covtype_impl"
    mod = importlib.util.module_from_spec(
        importlib.util.spec_from_loader(name, loader=None)
    )
    mod.__file__ = _SRC
    sys.modules[name] = mod
    with open(_SRC, encoding="utf-8") as fh:
        src = fh.read()
    exec(compile(src, _SRC, "exec"), mod.__dict__)
    return mod


def ensure_joblib(data_path, verbose=True):
    """把 covtype.data 转成 covtype.py 期望的 samples_py3 / targets_py3。返回是否真的转换了。"""
    root = os.path.join(data_path, "covertype")
    samples = os.path.join(root, "samples_py3")
    targets = os.path.join(root, "targets_py3")
    if os.path.exists(samples) and os.path.exists(targets):
        return False

    raw = os.path.join(data_path, "covtype.data")
    if not os.path.exists(raw):
        raise FileNotFoundError(
            "covtype: 既没有 %s + %s，也没有原始文件 %s" % (samples, targets, raw)
        )

    # 581012 x 55 float32 ≈ 128MB，用 pandas 比 np.loadtxt 快一个量级
    import pandas as pd

    arr = pd.read_csv(raw, header=None, dtype=np.float32).to_numpy()
    if arr.shape[1] != 55:
        raise ValueError("covtype.data 期望 55 列，实际 %d 列" % arr.shape[1])

    os.makedirs(root, exist_ok=True)
    joblib.dump(arr[:, :54], samples)
    joblib.dump(arr[:, 54].astype(np.int64), targets)
    if verbose:
        print(
            "[covtype_modified] built samples_py3/targets_py3 from covtype.data: "
            "X=%s y=%s" % (arr[:, :54].shape, arr[:, 54].shape)
        )
    return True


def covtype(data_path, balance=True):
    ensure_joblib(data_path)
    info = list(_load_impl().covtype(data_path, balance))

    # 原实现 dst_test is dst_val，归一化被对同一块池子执行了两次；还原成一次。
    mean, std = np.asarray(info[4]), np.asarray(info[5])
    pool = info[8]
    pool.data = pool.data * (std + 1e-8) + mean
    info[8] = pool

    # data_path=None：不写切分缓存（val 池顺序本身每次运行都变，缓存会失真）
    return rebuild_with_valtest(tuple(info), None, "covtype", seed=_SEED)
