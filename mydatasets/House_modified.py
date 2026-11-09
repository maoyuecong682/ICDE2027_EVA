"""House_modified.py -- 修正 House.py 内部把 data_path 覆盖成原作者绝对路径的问题。

House.py 第 35 行函数体里写着：
    data_path = "/data/dengqiyan/EVA/data/House"
这是原作者机器上的路径，在本机不存在，因此传进来的 data_path 被静默丢弃。

本脚本只删掉这一行，其余逻辑完全沿用原实现，因此与 House.py 语义等价。
"""
import importlib.util
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "House.py")
_OVERRIDE = re.compile(
    r'^[ \t]*data_path\s*=\s*"/data/dengqiyan/EVA/data/House"\s*$\n?', re.M
)


def _load_impl():
    with open(_SRC, encoding="utf-8") as fh:
        src = fh.read()
    src, n = _OVERRIDE.subn("", src)
    if n != 1:
        raise RuntimeError(
            "House_modified: 期望恰好 1 处硬编码 data_path，实际匹配 %d 处" % n
        )
    name = "_house_modified_impl"
    mod = importlib.util.module_from_spec(
        importlib.util.spec_from_loader(name, loader=None)
    )
    mod.__file__ = _SRC
    sys.modules[name] = mod
    exec(compile(src, _SRC, "exec"), mod.__dict__)
    return mod


_impl = _load_impl()

House = _impl.House
