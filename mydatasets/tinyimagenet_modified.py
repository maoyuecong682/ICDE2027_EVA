"""Load TinyImageNet pickles from the caller-provided dataset directory.

Preserve the original loader's tensors and PIL images. Encoded image bytes
are also supported and decoded to RGB when accessed.
"""
import importlib.util
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "tinyimagenet.py")

_OVERRIDE = re.compile(
    r'^[ \t]*data_path\s*=\s*"/root/autodl-tmp/EVA/data/TinyImageNet"\s*$\n?', re.M
)

# 插入点：__getitem__ 里把 Tensor 转 PIL 之前
_ANCHOR = "        # 如果 img 是 torch.Tensor 类型，转换为 PIL.Image 类型\n"
_PATCH = (
    "        # [modified] pkl 内为原始 JPEG 字节，解码并统一为 RGB\n"
    "        if isinstance(img, (bytes, bytearray)):\n"
    "            img = Image.open(io.BytesIO(img)).convert('RGB')\n"
) + _ANCHOR


def _load_impl():
    with open(_SRC, encoding="utf-8") as fh:
        src = fh.read()

    src, n1 = _OVERRIDE.subn("", src)
    if n1 != 1:
        raise RuntimeError(
            "tinyimagenet_modified: 期望恰好 1 处硬编码 data_path，实际匹配 %d 处" % n1
        )

    if _ANCHOR not in src:
        raise RuntimeError(
            "tinyimagenet_modified: 找不到 __getitem__ 的 Tensor->PIL 锚点，"
            "tinyimagenet.py 可能已被改动"
        )
    if "isinstance(img, (bytes, bytearray))" not in src:
        src = src.replace(_ANCHOR, _PATCH, 1)

    if "\nimport io" not in src:
        src = "import io\n" + src

    name = "_tinyimagenet_modified_impl"
    mod = importlib.util.module_from_spec(
        importlib.util.spec_from_loader(name, loader=None)
    )
    mod.__file__ = _SRC
    sys.modules[name] = mod
    exec(compile(src, _SRC, "exec"), mod.__dict__)
    return mod


_impl = _load_impl()

TinyImageNet = _impl.TinyImageNet
load_pkl_data = _impl.load_pkl_data
