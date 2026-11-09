"""number_modified.py -- 修正 number.py 的标签属性错配。

`Number()` 是 **MNIST -> SVHN** 的跨域数字分类任务：
    dst_train/dst_val 来自 torchvision MNIST（只有 `.targets`）
    dst_test          来自 torchvision SVHN （只有 `.labels`）

而结尾的
    set_labels_to_targets(dst_val, full_train_dataset)   # Subset of MNIST
内部在 Subset 分支里写的是 `full_dataset.labels[i]`：

    AttributeError: 'MNIST' object has no attribute 'labels'

（`set_targets_to_labels` 用的全是 `.targets`，所以 dst_train 没事；
  而 `set_labels_to_targets` 的 else 分支又只认 `.targets`，对 SVHN 也不稳。）

修法：在 `set_labels_to_targets` 里按“哪个属性存在就用哪个”解析一次标签源，
再统一赋给 dataset.labels / dataset.targets（兼容 pandas Series 的 `.values`）。
其余逻辑（transform、80/20 切分、返回顺序）一行未动。

注：本数据集此前失败的直接原因是 torchvision 要从 ufldl.stanford.edu 下载
SVHN 的 test_32x32.mat（本机网络不可达）。现在
/root/autodl-tmp/EVA/data/SVHN/test_32x32.mat 已是斯坦福原版
（md5 = eb5a983be6a315427106f1b164d9cef3，与 torchvision 内置值一致），
download=True 会直接通过校验，不再联网。
"""
import importlib.util
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "number.py")

_OLD = (
    "    def set_labels_to_targets(dataset, full_dataset):\n"
    "        if isinstance(dataset, Subset):\n"
    "            indices = dataset.indices\n"
    "            dataset.labels = [full_dataset.labels[i] for i in indices]\n"
    "            dataset.targets = tensor([full_dataset.labels[i] for i in indices], dtype=long)\n"
    "        else:\n"
    "            dataset.labels = full_dataset.targets\n"
    "            dataset.targets = tensor(full_dataset.targets, dtype=long)\n"
)
_NEW = (
    "    def set_labels_to_targets(dataset, full_dataset):\n"
    "        # [modified] MNIST 只有 .targets(Tensor)，SVHN 只有 .labels(ndarray)\n"
    "        _src = full_dataset.labels if hasattr(full_dataset, 'labels') "
    "else full_dataset.targets\n"
    "        # Tensor / ndarray / pandas.Series 都有 tolist()；注意不能判断 .values，\n"
    "        # torch.Tensor.values 是个内置方法，会把 _src 变成 method\n"
    "        _src = _src.tolist() if hasattr(_src, 'tolist') else list(_src)\n"
    "        if isinstance(dataset, Subset):\n"
    "            indices = dataset.indices\n"
    "            dataset.labels = [_src[i] for i in indices]\n"
    "            dataset.targets = tensor([_src[i] for i in indices], dtype=long)\n"
    "        else:\n"
    "            dataset.labels = _src\n"
    "            dataset.targets = tensor(_src, dtype=long)\n"
)


def _load_impl():
    with open(_SRC, encoding="utf-8") as fh:
        src = fh.read()
    src, n = re.subn(re.escape(_OLD), lambda m: _NEW, src)
    if n != 1:
        raise RuntimeError(
            "number_modified: 期望恰好 1 处 set_labels_to_targets，实际匹配 %d 处；"
            "请检查 number.py 是否已被改动" % n
        )
    name = "_number_modified_impl"
    mod = importlib.util.module_from_spec(
        importlib.util.spec_from_loader(name, loader=None)
    )
    mod.__file__ = _SRC
    sys.modules[name] = mod
    exec(compile(src, _SRC, "exec"), mod.__dict__)
    return mod


_impl = _load_impl()


def Number(data_path, balance=True):
    """MNIST -> SVHN 跨域数字分类，**保留原实现的 val 来源**。

    原实现：MNIST 60000 按 80/20 切成 48000 train + 12000 val（同域），
            test = SVHN 官方 test 全量 26032（跨域）。

    这里刻意不改成“从 SVHN 测试池对半切 val”：那是域偏移任务，若 val 取自目标域，
    等于用目标域的带标签样本挑 checkpoint，与“不使用目标域标签”的主张冲突。
    据用户 2026-10-25 的决定，保持 val 与 train 同域。

    注意：原实现的 80/20 用的是全局 RNG 且未设种子，val 的具体成员每次运行会变。
    """
    return _impl.Number(data_path)

