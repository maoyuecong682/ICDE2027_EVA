"""imagenet100.py -- ImageNet-100（100 类 ILSVRC 子集）的 EVA 加载器。

为什么不用原来的 imagenet.py
---------------------------
1) 原实现走 `torchvision.datasets.ImageNet(data_path, split="train"/"val")`，
   要求的是**完整 ILSVRC2012** 的目录布局（train/n0*/、val/、meta.mat），
   而本机只有 ImageNet-100 的扁平布局 `imagenet100/<wnid>/*.JPEG`（100 类，
   每类 1300 张，本身不带 train/val 划分）。
2) 原实现只返回 8 元组（train, test），没有 val。而三个 baseline 与 EVA 主实验
   一律按 9 元组解包，且 COPS 的 probe 切分、DUAL 的中间评估都要用 val。
3) 原实现 `im_size = (224, 224)` 是 tuple。适配层 `build_model` 里写的是
   `size = im_size[0] if isinstance(im_size, list) else im_size`，tuple 会掉进
   else 分支变成 `int((224,224))` → TypeError。这里直接给 list。

划分口径
--------
ImageNet-100 官方无 train/val/test 划分，coreset 文献（如 DM/AS/ECS 在
ImageNet 子集上的做法）通常自己按类分层切。这里固定 **每类 8:1:1**：

    1300 张/类  ->  train 1040 / val 130 / test 130

按类分层保证 num_classes=100 在三段里都成立（否则某些类会整段缺失，
per-class 采样数算不出来，accuracy 也会退化）。种子固定为 48，
与适配层 SPLIT_SEED、COPS probe 切分的种子一致，重跑可复现。

预处理与缓存
------------
科学口径是 `Resize(256) + CenterCrop(224) + ToTensor + ImageNet mean/std`，
与原 imagenet.py 完全一致（原实现对 train 用的也是 CenterCrop、不做随机增广）。

**关键工程点**：这三步全是确定性的，所以可以预先算好 224 中心裁剪的结果，
缓存成 uint8 `(N,3,224,224)`（约 15.2 GB），训练时只做 `/255` + 标准化。
原因是实测在线解码时 COPS stage1 只有 **24 分钟/epoch**（75 分钟才到 epoch 3），
照那样 imagenet 要跑 40 小时；GPU 利用率却常年只有一半 —— 瓶颈是 10 万次
JPEG 解码。缓存后一轮回到 20 秒量级，且与在线解码**逐像素等价**。

缓存位置：`<data_path>/_cache224/{train,val,test}_{x,y}.npy`（可用环境变量
`EVA_IMAGENET_CACHE` 覆盖；设 `EVA_IMAGENET_NO_CACHE=1` 可强制走在线解码）。
缓存不存在时自动退化为在线解码，行为与之前一致。
"""
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

SPLIT_SEED = 48
_SPLIT = (0.8, 0.1, 0.1)          # train / val / test，按类分层

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

SPLITS = ("train", "val", "test")


def _cache_dir(data_path):
    """有完整缓存就返回目录，否则 None。"""
    if os.environ.get("EVA_IMAGENET_NO_CACHE"):
        return None
    d = os.environ.get("EVA_IMAGENET_CACHE") or os.path.join(data_path, "_cache224")
    for sp in SPLITS:
        if not (os.path.exists(os.path.join(d, "%s_x.npy" % sp))
                and os.path.exists(os.path.join(d, "%s_y.npy" % sp))):
            return None
    return d


class ImageNet100Dataset(Dataset):
    """ImageNet-100 样本访问。

    两种模式：
      * 在线解码（默认退化路径）：存 (路径, 标签)，取样本时 `Image.open` + transform；
      * 缓存模式（cachedir 给定时）：从 uint8 memmap 读 (3,224,224)，只做 /255 + 标准化。
    两种模式的输出逐像素等价（Resize(256)+CenterCrop(224) 是确定性的）。
    """

    def __init__(self, paths, labels, transform, cachedir=None, split=None):
        self.paths = list(paths)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.transform = transform
        self._x = None
        self._mean = None
        self._std = None
        if cachedir is not None:
            self._x = np.load(os.path.join(cachedir, "%s_x.npy" % split), mmap_mode="r")
            if len(self._x) != len(self.paths):
                raise RuntimeError(
                    "imagenet100: 缓存 %s 长度 %d != 划分长度 %d，缓存与划分不一致，"
                    "请删掉缓存重建" % (split, len(self._x), len(self.paths)))
            self._mean = torch.tensor(MEAN, dtype=torch.float32).view(3, 1, 1)
            self._std = torch.tensor(STD, dtype=torch.float32).view(3, 1, 1)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        if self._x is not None:
            x = torch.from_numpy(np.asarray(self._x[i], dtype=np.uint8))
            x = x.float().div_(255.0)
            x = (x - self._mean) / self._std
            return x, int(self.labels[i])
        img = Image.open(self.paths[i])
        if img.mode != "RGB":
            img = img.convert("RGB")
        with img:
            x = self.transform(img)
        return x, int(self.labels[i])

    # 适配层 / DUAL / GM 都会读 .targets；给 numpy 数组，避免构造 10 万个 list
    @property
    def targets(self):
        return self.labels


def _resolve_root(data_path):
    """兼容两种落盘：<data_path>/imagenet100/<wnid> 或 <data_path>/<wnid>。"""
    cand = os.path.join(data_path, "imagenet100")
    if os.path.isdir(cand):
        return cand
    return data_path


def _scan(root):
    """逐类列出 JPEG，返回 {wnid: [abs paths]}；类按 wnid 排序保证索引稳定。"""
    classes = sorted(
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d)) and not d.startswith(".")
    )
    if not classes:
        raise RuntimeError("imagenet100: %s 下没有类别子目录" % root)
    table = {}
    for c in classes:
        cdir = os.path.join(root, c)
        files = sorted(
            f for f in os.listdir(cdir)
            if f.lower().endswith((".jpeg", ".jpg", ".png"))
        )
        if files:
            table[c] = [os.path.join(cdir, f) for f in files]
    classes = [c for c in classes if c in table]
    if len(classes) != len(table):
        raise RuntimeError("imagenet100: 类别目录存在但为空，数据不完整")
    return classes, table


def ImageNet(data_path, balance=True):
    """返回 9 元组（与 eva_adapter 的契约一致）。

    balance 在本数据集上无意义（每类张数天然相等），保留只为统一签名。
    """
    root = _resolve_root(data_path)
    classes, table = _scan(root)
    n_cls = len(classes)

    tr_p, tr_y, va_p, va_y, te_p, te_y = [], [], [], [], [], []
    rng = np.random.RandomState(SPLIT_SEED)

    for ci, c in enumerate(classes):
        paths = table[c]
        n = len(paths)
        idx = rng.permutation(n)
        n_tr = int(round(n * _SPLIT[0]))
        n_va = int(round(n * _SPLIT[1]))
        # 至少给 val/test 各留 1 张，否则该类在评估段整段消失
        n_tr = min(n_tr, n - 2)
        blocks = (idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:])
        for (dst_p, dst_y), blk in zip(
            ((tr_p, tr_y), (va_p, va_y), (te_p, te_y)), blocks
        ):
            for j in blk:
                dst_p.append(paths[j])
                dst_y.append(ci)

    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    cd = _cache_dir(data_path)
    dst_train = ImageNet100Dataset(tr_p, tr_y, tf, cd, "train")
    dst_val = ImageNet100Dataset(va_p, va_y, tf, cd, "val")
    dst_test = ImageNet100Dataset(te_p, te_y, tf, cd, "test")

    channel = 3
    im_size = [224, 224]        # 必须是 list：见模块 docstring 第 3 条
    num_classes = n_cls
    class_names = list(classes)
    print("[imagenet100] %d 类, train/val/test = %d/%d/%d, cache=%s" % (
        n_cls, len(dst_train), len(dst_val), len(dst_test), cd or "off"))
    return (channel, im_size, num_classes, class_names, list(MEAN), list(STD),
            dst_train, dst_val, dst_test)
