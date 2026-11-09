"""fashionmnist_modified.py -- 给 FashionMNIST 补出 val 划分。

原 `fashionmnist.py` 的 `FashionMNIST(data_path)` 只返回 8 个值（train, test），
`main0.py:122` 解包 9 个会直接
`ValueError: not enough values to unpack (expected 9, got 8)`。

补法按 `cifar10.py` 的口径：train 保持全量(60000)，官方 test(10000) 按类对半切，
得到 5000 val + 5000 test，每类各 500。

签名改成 `(data_path, balance=True)`：原签名只有 `data_path`，main0 会位置
传入第 2 个参数（`--balance_target`），不加会 TypeError。
"""
import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from valtest_split import rebuild_with_valtest  # noqa: E402

_SRC = os.path.join(_HERE, "fashionmnist.py")
_SEED = 0


def _load_impl():
    name = "_fashionmnist_modified_impl"
    spec = importlib.util.spec_from_file_location(name, _SRC)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_impl = _load_impl()


def FashionMNIST(data_path, balance=True):
    info = _impl.FashionMNIST(data_path)
    return rebuild_with_valtest(info, data_path, "fashionmnist", seed=_SEED)
