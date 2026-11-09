"""cifar100_modified.py -- 修正 cifar100.py 把 num_classes 写死为 10 的错误。

cifar100.py 第 10 行写着：
    num_classes = 10
但 CIFAR-100 有 100 类。这个错误不会报错，只会静默产生两个后果：
  1. 第 42 行 `for c in range(num_classes)` 只遍历类别 0..9，于是 val/test 只从
     10 个类里各取 50 张 —— 实测 500/500（正确应为 5000/5000），测试集缺 90% 的类；
  2. 返回的 num_classes=10 会让 EVA 建 10 维输出头去拟合 100 类的标签。

本脚本只把这一个常数改成 100，其余（切分方式、targets/labels 赋值、返回顺序）完全沿用原实现。
"""
import importlib.util
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "cifar100.py")
_BAD = re.compile(r'^([ \t]*num_classes[ \t]*=[ \t]*)10([ \t]*)$', re.M)


def _load_impl():
    with open(_SRC, encoding="utf-8") as fh:
        src = fh.read()
    src, n = _BAD.subn(r"\g<1>100\g<2>", src)
    if n != 1:
        raise RuntimeError(
            "cifar100_modified: 期望恰好 1 处 'num_classes = 10'，实际匹配 %d 处" % n
        )
    name = "_cifar100_modified_impl"
    mod = importlib.util.module_from_spec(
        importlib.util.spec_from_loader(name, loader=None)
    )
    mod.__file__ = _SRC
    try:
        from .cifar_memmap import MemoryMappedCIFAR100
    except ImportError:
        from cifar_memmap import MemoryMappedCIFAR100
    src = src.replace('datasets.CIFAR100(', '_memorymapped_cifar100(')
    mod._memorymapped_cifar100 = MemoryMappedCIFAR100
    sys.modules[name] = mod
    exec(compile(src, _SRC, "exec"), mod.__dict__)
    return mod


_impl = _load_impl()

CIFAR100 = _impl.CIFAR100
