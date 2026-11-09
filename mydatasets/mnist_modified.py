"""mnist_modified.py -- 口径统一为“官方 test 按类对半切成 val/test”。

原 `mnist.py` 是从 **train** 里切 20% 当 val（dst_train 只剩 48000），
与 `cifar10.py` 的口径不一致（cifar10 保持 train 全量、把官方 test 对半切）。

本文件**不改原文件一行**，只在外面包一层：
  1. 调 `mnist.py` 的 `MNIST(data_path)` 拿原返回值；
  2. 丢弃它从 train 切出来的 val；
  3. 把被切走的 12000 张还原回 train（`Subset.dataset` 就是那 60000 张全量）；
  4. 从官方 test(10000) 按类对半切出 5000 val + 5000 test。

附带解决一个调用陷阱
--------------------
原签名 `MNIST(data_path, permuted=False, permutation_seed=None)`，而 `main0.py`
用 `dataset_func(args.data_path, args.balance_target)` **位置传参**，第 2 个参数
会绑到 `permuted` 上——按默认 `--balance_target 1` 运行就会静默启用像素置换，
数据集被悄悄改掉。本包装的签名是 `(data_path, balance=True)`，第 2 个位置参数无害。
"""
import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from valtest_split import rebuild_with_valtest  # noqa: E402

_SRC = os.path.join(_HERE, "mnist.py")
_SEED = 0


def _load_impl():
    name = "_mnist_modified_impl"
    spec = importlib.util.spec_from_file_location(name, _SRC)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_impl = _load_impl()


def MNIST(data_path, balance=True):
    info = _impl.MNIST(data_path)
    return rebuild_with_valtest(info, data_path, "mnist", seed=_SEED)


def permutedMNIST(data_path, permutation_seed=None):
    info = _impl.permutedMNIST(data_path, permutation_seed)
    return rebuild_with_valtest(info, data_path, "mnist_permuted", seed=_SEED)
