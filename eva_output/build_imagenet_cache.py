#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""为 ImageNet-100 建 uint8 预裁剪缓存，消掉每轮 10 万次 JPEG 解码。

为什么必须做
------------
COPS stage1 在 imagenet 上实测 **24 分钟/epoch**（75 分钟才到 epoch 3），
按 4 模型 × 25 epoch 算要 40 小时。瓶颈不是 GPU（利用率只有 50% 上下），
而是每轮都要重新解码 101351 张 224px 的 JPEG。

现有预处理是 `Resize(256) + CenterCrop(224) + ToTensor + Normalize`，
**完全确定性、无随机增广**（原 imagenet.py 对 train 也是 CenterCrop）。
所以可以把 `Resize+CenterCrop` 的结果一次性缓存成 uint8 (N,3,224,224)，
训练时只做 `/255` 与标准化 —— 结果与在线解码逐像素等价。

体积：101351 × 3 × 224 × 224 × 1B ≈ 15.2 GB（数据盘还有 80 GB）。
"""
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from PIL import Image
from torchvision import transforms

OUT = "/root/autodl-tmp/EVA/data/imagenet100/_cache224"
_np_cache = {}
_np_y = {}


def _tf():
    """只保留确定性的两步；ToTensor/Normalize 在 Dataset 里用 float 做。"""
    return transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])


def _work(arg):
    i, p = arg
    img = Image.open(p)
    if img.mode != "RGB":
        img = img.convert("RGB")
    with img:
        x = _tf()(img)
    arr = np.asarray(x, dtype=np.uint8)            # HWC
    return i, np.ascontiguousarray(arr.transpose(2, 0, 1))   # CHW


def build(name, paths, labels):
    n = len(paths)
    xs = np.lib.format.open_memmap(os.path.join(OUT, "%s_x.npy" % name),
                                   mode="w+", dtype=np.uint8, shape=(n, 3, 224, 224))
    t0 = time.time()
    done = 0
    with Pool(16) as pool:
        for i, arr in pool.imap_unordered(_work, list(enumerate(paths)), chunksize=64):
            xs[i] = arr
            done += 1
            if done % 5000 == 0:
                el = time.time() - t0
                print("  %s %d/%d  %.0f img/s  eta %.0f s"
                      % (name, done, n, done / el, (n - done) / max(1e-9, done / el)),
                      flush=True)
    xs.flush()
    del xs
    np.save(os.path.join(OUT, "%s_y.npy" % name), np.asarray(labels, dtype=np.int64))
    print("[cache] %s 完成 n=%d  %.1f s" % (name, n, time.time() - t0), flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    sys.path.insert(0, "/root/chaichengliang/VLDB2027/EVA/EVA/mydatasets")
    os.environ["EVA_IMAGENET_NO_CACHE"] = "1"
    import imagenet100 as M

    _, _, _, _, _, _, tr, va, te = M.ImageNet("/root/autodl-tmp/EVA/data/imagenet100")
    for name, ds in (("train", tr), ("val", va), ("test", te)):
        build(name, ds.paths, ds.labels)


if __name__ == "__main__":
    main()
