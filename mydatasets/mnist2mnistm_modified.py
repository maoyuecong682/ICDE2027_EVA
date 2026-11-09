"""mnist2mnistm_modified.py -- MNIST->MNIST-M 加载器的修正版。

【修正 1】标签属性错配（这是本数据集当初跑不起来的直接原因）
原 `MNISTM` 类只定义 `self.labels`（pandas Series），没有 `self.targets`；而
`set_labels_to_targets()` 的非 Subset 分支写成了：

    dataset.labels = full_dataset.targets          # AttributeError
    dataset.targets = tensor(full_dataset.targets, dtype=long)

`dst_test` 正是完整的 `MNISTM` 实例（非 Subset），所以必然抛
`AttributeError: 'MNISTM' object has no attribute 'targets'`。
→ 改成按“哪个属性存在就用哪个”解析标签源，再统一赋给 labels / targets。

【修正 2】签名
原签名 `Mnist2MnistM(data_path, permuted=False, permutation_seed=None)`，而 `main0.py`
用 `dataset_func(args.data_path, args.balance_target)` **位置传参**，第 2 个参数会
绑到 `permuted` 上。本包装签名是 `(data_path, balance=True)`，第 2 个位置参数无害。

【刻意不做的事：val 保持从 train 切】
这个任务是域偏移（train=MNIST，test=MNIST-M）。若把 val 也取自 MNIST-M 的测试池，
就等于用目标域的带标签样本挑 checkpoint，与"不使用目标域标签"的主张冲突。
所以这里**故意保留原实现的 80/20**：

    dst_train = MNIST 的 80% = 48000
    dst_val   = MNIST 的 20% = 12000   （与 train 同域）
    dst_test  = MNIST-M 官方 test 全量 10000（跨域）

这与 cifar10 的"test 切半"口径不同，是有意为之；据用户 2026-10-25 的决定。
注意：原实现的 80/20 用的是全局 RNG 且未设种子，因此 val 的具体成员每次运行会变。
"""
import importlib.util
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "mnist2mnistm.py")

_OLD = (
    "            dataset.labels = full_dataset.targets\n"
    "            dataset.targets = tensor(full_dataset.targets, dtype=long)\n"
)
_NEW = (
    "            _src = full_dataset.labels if hasattr(full_dataset, 'labels') "
    "else full_dataset.targets\n"
    "            # Tensor / ndarray / pandas.Series 都有 tolist()；不能判断 .values，\n"
    "            # torch.Tensor.values 是内置方法，会把 _src 变成 method\n"
    "            _src = _src.tolist() if hasattr(_src, 'tolist') else list(_src)\n"
    "            dataset.labels = _src\n"
    "            dataset.targets = tensor(_src, dtype=long)\n"
)


def _load_impl():
    with open(_SRC, "r", encoding="utf-8") as fh:
        code = fh.read()

    code, n = re.subn(re.escape(_OLD), lambda m: _NEW, code)
    if n != 1:
        raise RuntimeError(
            "mnist2mnistm_modified: 期望恰好 1 处 target/label 掉换，实际匹配 %d 处；"
            "请检查 mnist2mnistm.py 是否已被改动" % n
        )

    name = "_mnist2mnistm_modified_impl"
    mod = importlib.util.module_from_spec(
        importlib.util.spec_from_loader(name, loader=None)
    )
    mod.__file__ = _SRC
    sys.modules[name] = mod
    code=code.replace('import pandas as pd\n','')
    code=code.replace('datasets.MNIST(', '_memorymapped_mnist(')
    code=code.replace('tensor([full_dataset.targets[i] for i in indices], dtype=long)',
                      'full_dataset.targets[tensor(indices, dtype=long)].clone()')
    code=code.replace('[full_dataset.targets[i] for i in indices]', 'dataset.targets.tolist()')
    exec(compile(code, _SRC, "exec"), mod.__dict__)
    try:
        from .mnistm_memmap import MemoryMappedMNISTM,MemoryMappedMNIST
    except ImportError:
        from mnistm_memmap import MemoryMappedMNISTM,MemoryMappedMNIST
    mod.MNISTM=MemoryMappedMNISTM
    mod._memorymapped_mnist=MemoryMappedMNIST
    return mod


_impl = _load_impl()


def Mnist2MnistM(data_path, balance=True):
    """返回原实现的 9 元组：train/val 同域(MNIST 80/20)，test 跨域(MNIST-M)。"""
    return _impl.Mnist2MnistM(data_path)
