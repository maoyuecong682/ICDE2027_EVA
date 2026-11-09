"""tinyimagenet_modified.py -- 修正 tinyimagenet.py 的两处问题。

【修正 1】目录名大小写硬编码
tinyimagenet.py 第 57 行函数体里写着：
    data_path = "/root/autodl-tmp/EVA/data/TinyImageNet"
而本机数据实际落在 /root/autodl-tmp/EVA/data/tinyimagenet（全小写）。
大小写敏感的文件系统上两者不是同一个目录，于是 train.pkl 找不到。
→ 删掉这一行，改用调用方传入的 data_path。

【修正 2】支持 pkl 中存放原始 JPEG 字节
本容器被 cgroup 限死在 2GB 内存（memory.max = 2147483648）。若 train.pkl
按“10 万个已解码的 64x64x3 PIL 图像”存储，光列表就约 1.3GB，构建与加载
时都会越过 2GB 而被 SIGKILL。因此
`evaluation/build_tinyimagenet_pkl.py` 改为存放**原始 JPEG 字节**
（train 约 205MB），代价是 `__getitem__` 需要自行解码。
→ 在取图处插入一个 bytes 分支：Image.open(BytesIO(b)).convert("RGB")。
  convert("RGB") 同时修掉官方数据里约 0.45% 灰度图（mode='L'）会让
  ToTensor() 产出单通道、与 channel=3 冲突的问题。

除上述两点外，pkl 读取、downsize、返回顺序与类别映射完全沿用原实现。
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
