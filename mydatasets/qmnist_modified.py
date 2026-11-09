"""qmnist_modified.py -- 给 QMNIST 补出 val 划分。

原 `qmnist.py` 的 `QMNIST(data_path)` 只返回 8 个值（train, test），而
`main0.py:122` 一次性解包 9 个，会直接
`ValueError: not enough values to unpack (expected 9, got 8)`。

补法完全按 `cifar10.py` 的口径：train 保持全量，官方 test 按类对半切成 val/test。
QMNIST 的官方 test 有 60000 张（NIST 手写字符的扩展测试集），因此切成
30000 val + 30000 test，每类各半。

签名改成 `(data_path, balance=True)`：原签名只有 `data_path`，而 main0 会
位置传入第 2 个参数（`--balance_target`），不加这个形参会直接 TypeError。
"""
import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from valtest_split import rebuild_with_valtest  # noqa: E402

_SRC = os.path.join(_HERE, "qmnist.py")
_SEED = 0


def _load_impl():
    name = "_qmnist_modified_impl"
    spec = importlib.util.spec_from_file_location(name, _SRC)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_impl = _load_impl()


def QMNIST(data_path, balance=True):
    info = _impl.QMNIST(data_path)
    return rebuild_with_valtest(info, data_path, "qmnist", seed=_SEED)
