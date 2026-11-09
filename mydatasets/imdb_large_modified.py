"""imdb_large_modified.py -- 修正 imdb_large.py 的路径硬编码与拼写错误。

原文件 `IMDBLarge()` 里有三处会直接炸掉的问题：

1) 路径完全没有使用入参 data_path，而是硬编码成大写目录
       /root/autodl-tmp/EVA/data/IMDB/...
   本机实际目录是全小写 `imdb`，Linux 大小写敏感 → FileNotFoundError。

2) 其中两行路径字符串带**前导空格**：
       train_labels_path =" /root/..."
       val_labels_path   =" /root/..."
   即使目录存在，`" /root/..."` 也不是合法路径。

3) 变量名拼错：
       test_features_pat= "/root/.../IMDBLargeC5-test-X.npy"
   随后 `IMDBLargeDataset(test_features_path, ...)` 引用的
   `test_features_path` 从未被定义 → NameError。

本脚本把这 6 行统一改写成基于 data_path 的 os.path.join，其余（标签 -1->5、
StandardScaler 标准化、返回顺序、num_classes 推断）完全沿用原实现。

【注意：内存】
原实现在 `IMDBLargeDataset.__init__` 里 `np.load` 整个矩阵再转 float32：

    train X (10561348, 41) float64 = 3.46 GB
    -> tensor(float32)            = 1.73 GB
    -> StandardScaler.fit_transform 再要 1.73 GB

峰值约 7 GB。本容器无卡模式下 cgroup 限制为 2 GB（memory.max=2147483648），
因此**即便数据补齐，原实现也无法在此环境运行**；需要换到有卡的实例
（内存更大），或把加载改成 np.load(mmap_mode='r') 逐行取。
本脚本保持原语义未做该改造，以免与论文数字不可比。
"""
import importlib.util
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "imdb_large.py")

_BLOCK = re.compile(
    r'^[ \t]*train_features_path[ \t]*=[ \t]*"[^"]*"\n'
    r'[ \t]*train_labels_path[ \t]*=[ \t]*"[^"]*"\n'
    r'[ \t]*val_features_path[ \t]*=[ \t]*"[^"]*"\n'
    r'[ \t]*val_labels_path[ \t]*=[ \t]*"[^"]*"\n'
    r'[ \t]*test_features_pat[h]?[ \t]*=[ \t]*"[^"]*"\n'
    r'[ \t]*test_labels_path[ \t]*=[ \t]*"[^"]*"\n',
    re.M,
)

_NEW_BLOCK = """    train_features_path = os.path.join(data_path, "IMDBLargeC5-train-X.npy")
    train_labels_path = os.path.join(data_path, "IMDBLargeC5-train-y.npy")
    val_features_path = os.path.join(data_path, "IMDBLargeC5-val-X.npy")
    val_labels_path = os.path.join(data_path, "IMDBLargeC5-val-y.npy")
    test_features_path = os.path.join(data_path, "IMDBLargeC5-test-X.npy")
    test_labels_path = os.path.join(data_path, "IMDBLargeC5-test-y.npy")
"""


def _load_impl():
    with open(_SRC, encoding="utf-8") as fh:
        src = fh.read()

    src, n = _BLOCK.subn(lambda m: _NEW_BLOCK, src)
    if n != 1:
        raise RuntimeError(
            "imdb_large_modified: 期望恰好 1 处六行路径块，实际匹配 %d 处；"
            "请检查 imdb_large.py 是否已被改动" % n
        )

    if not re.search(r"^import os\b", src, re.M):
        src = "import os\n" + src

    name = "_imdb_large_modified_impl"
    mod = importlib.util.module_from_spec(
        importlib.util.spec_from_loader(name, loader=None)
    )
    mod.__file__ = _SRC
    sys.modules[name] = mod
    exec(compile(src, _SRC, "exec"), mod.__dict__)
    return mod


_impl = _load_impl()

IMDBLarge = _impl.IMDBLarge
IMDBLargeDataset = _impl.IMDBLargeDataset
