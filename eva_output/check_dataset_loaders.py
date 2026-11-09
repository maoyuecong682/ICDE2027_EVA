"""按 main0/main1 的调用方式真正加载数据集，确认返回的是可用的 9 元组。

为什么要单独验证：`verify_main_datasets.py` 只做静态解析和函数解析，能发现
"名字打错""用了原始模块"，但不能发现**运行时**问题，例如
  * 位置传参把 balance_target 绑到别的参数上（MNIST 的 permuted 陷阱）
  * loader 返回的元素个数不是 9（main0.py 里是一行 9 个变量解包）
  * 类别数与文件名不符（原始 cifar100.py 把 num_classes 写死成 10）

一次只加载一个数据集，由调用方用 shell 循环分批跑 —— 每个数据集一个独立进程，
避免 2GB 配额下多个大图数据集叠加导致 OOM。

用法：
    python check_dataset_loaders.py <数据集> [仓库根目录]
"""
import importlib
import os
import sys

DATASET = sys.argv[1]
ROOT = sys.argv[2] if len(sys.argv) > 2 else "/root/chaichengliang/VLDB2027/EVA/EVA"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# main0/main1 里的登记（与 patch_main_datasets.py 保持一致）
MODULE_OF = {
    "cifar100": ("mydatasets.cifar100_modified", "CIFAR100"),
    "covtype": ("mydatasets.covtype_modified", "covtype"),
    "fashionmnist": ("mydatasets.fashionmnist_modified", "FashionMNIST"),
    "HR": ("mydatasets.HR_modified", "HR"),
    "mnist": ("mydatasets.mnist_modified", "MNIST"),
    "qmnist": ("mydatasets.qmnist_modified", "QMNIST"),
    "SVHN": ("mydatasets.svhn_modified", "SVHN"),
    "tinyimagenet": ("mydatasets.tinyimagenet_modified", "TinyImageNet"),
    "mnist2mnistm": ("mydatasets.mnist2mnistm_modified", "Mnist2MnistM"),
}

# 各数据集在 data 根目录下的相对子目录（与 eva_adapter.REGISTRY 一致）
SUB = {
    "cifar100": "cifar100", "covtype": "covtype", "fashionmnist": "fashionmnist",
    "HR": "HR", "mnist": "mnist", "qmnist": "qmnist", "SVHN": "SVHN",
    "tinyimagenet": "tinyimagenet", "mnist2mnistm": "mnist2mnistm",
}
DATA_ROOT = os.environ.get("EVA_DATA_ROOT", "/root/autodl-tmp/EVA/data")

if DATASET not in MODULE_OF:
    sys.exit("未知数据集 %r，可选：%s" % (DATASET, ", ".join(MODULE_OF)))

modname, fnname = MODULE_OF[DATASET]
data_path = os.path.join(DATA_ROOT, SUB[DATASET])

mod = importlib.import_module(modname)
fn = getattr(mod, fnname)

# 关键：完全照 main0 的位置传参方式调用，balance_target=1
info = fn(data_path, 1)

assert isinstance(info, (tuple, list)), "返回值不是元组，而是 %r" % type(info)
assert len(info) == 9, "main0 按 9 个变量解包，实际返回 %d 个元素" % len(info)

channel, im_size, num_classes, class_names, mean, std, dst_train, dst_val, dst_test = info

print("[%-13s] channel=%s im_size=%s classes=%-3s train=%-6d val=%-6d test=%-6d"
      % (DATASET, channel, im_size, num_classes,
         len(dst_train), len(dst_val), len(dst_test)))

# 类别数必须与数据集定义相符：原始 cifar100.py 写死 10，是最容易静默出错的地方
EXPECT_CLASSES = {"cifar100": 100, "covtype": 7, "fashionmnist": 10, "HR": 2,
                  "mnist": 10, "qmnist": 10, "SVHN": 10, "tinyimagenet": 200,
                  "mnist2mnistm": 10}
exp = EXPECT_CLASSES[DATASET]
assert int(num_classes) == exp, \
    "num_classes=%s 与预期 %d 不符（原始脚本可能写死了类别数）" % (num_classes, exp)

assert len(dst_val) > 0, "val 为空：*_modified 应当从官方 test 切出 val"
assert len(dst_test) > 0, "test 为空"

# 取一个样本确认张量形状与通道数一致
x, y = dst_train[0]
shape = tuple(getattr(x, "shape", ()))

# 表格数据（HR、covtype）与图像走两套语义：
#   * 图像：样本形如 (C, H, W)，首维才等于 channel
#   * 表格：样本形如 (F,)，首维是**特征数**而不是通道数
# 判定只看样本本身的维数，**不要**看 im_size 的类型 —— 各 *_modified 之间并不统一：
# HR 返回 im_size=(13,)，covtype 返回 im_size=54（整数），拿它去 len() 会直接抛
# TypeError。样本 ndim 才是唯一可靠的信号。
is_tabular = len(shape) == 1
if is_tabular:
    assert shape[0] > 0, "表格样本形状异常：%s" % (shape,)
    if isinstance(im_size, (tuple, list)):
        assert shape == tuple(im_size), \
            "表格样本形状 %s 与 im_size=%s 不符" % (shape, tuple(im_size))
    else:
        assert shape == (int(im_size),), \
            "表格样本形状 %s 与 im_size=%s 不符" % (shape, im_size)
else:
    assert len(shape) == 3, "图像样本应为三维 (C,H,W)，实际 %s" % (shape,)
    assert shape[0] == int(channel), \
        "样本首维 %d 与 channel=%s 不符，shape=%s" % (shape[0], channel, shape)
    if isinstance(im_size, (tuple, list)):
        assert shape[1:] == tuple(im_size), \
            "样本空间尺寸 %s 与 im_size=%s 不符" % (shape[1:], tuple(im_size))

assert 0 <= int(y) < int(num_classes), "标签 %r 越界" % (y,)

print("           样本 shape=%s label=%s  表格=%s  OK"
      % (shape, int(y), is_tabular))
print("RESULT OK %s" % DATASET)
