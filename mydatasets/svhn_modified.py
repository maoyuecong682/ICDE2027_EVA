"""svhn_modified.py -- 给 SVHN 补出 val 划分。

原 `svhn.py` 的 `SVHN(data_path)` 只返回 8 个值（train, test），
`main0.py:122` 解包 9 个会直接
`ValueError: not enough values to unpack (expected 9, got 8)`。

补法按 `cifar10.py` 的口径：train 保持全量(73257)，官方 test(26032) 按类对半切，
得到 13016 val + 13016 test。

注意 SVHN 的类别分布极不均衡（数字 1 远多于其他），所以只能**按类**分别对半切，
不能整体对半切——整体切会让 val/test 的类别比例随随机性抖动。
（官方各数字的占比本来就偏，按类切分保留了这一点，与 cifar10 的处理一致。）

签名改成 `(data_path, balance=True)`：原签名只有 `data_path`，main0 会位置
传入第 2 个参数（`--balance_target`），不加会 TypeError。

依赖：`data/SVHN/train_32x32.mat` 与 `test_32x32.mat` 必须是斯坦福原版，
否则 torchvision 的 `_check_integrity()` 会拒绝并重新联网下载（本机不可达）。
    train_32x32.mat  md5 e26dedcc434d2e4c54c9b2d4a06d8373
    test_32x32.mat   md5 eb5a983be6a315427106f1b164d9cef3
"""
import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from valtest_split import rebuild_with_valtest  # noqa: E402

_SRC = os.path.join(_HERE, "svhn.py")
_SEED = 0


def _load_impl():
    name = "_svhn_modified_impl"
    spec = importlib.util.spec_from_file_location(name, _SRC)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_impl = _load_impl()


def SVHN(data_path, balance=True):
    info = _impl.SVHN(data_path)
    return rebuild_with_valtest(info, data_path, "svhn", seed=_SEED)
