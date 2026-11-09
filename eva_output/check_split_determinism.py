# -*- coding: utf-8 -*-
"""check_split_determinism.py —— 同一个数据集在两个独立进程里是否切出同一份 val

存在的意义：`mydatasets/HR.py`、`mydatasets/mnist2mnistm.py` 里用的是
    random_split(dataset, [train_size, val_size])
**没有传 generator**，因此切分依赖 torch 的全局 RNG。torch 不 manual_seed 时
全局 RNG 由进程启动时的熵初始化 —— 也就是说每次运行、每个进程，切出来的
train/val 都可能不同。

后果不是崩溃，而是**静默地不可复现**：主实验和三个 baseline 是各自独立的进程，
它们看到的 val 不同，于是同一份核心集在不同方法下的分数没有可比性，重跑也对不上。

用法：python check_split_determinism.py <数据集>
连跑两次，比较 train/val 的标签哈希与前若干行特征。
"""
import hashlib
import sys

import numpy as np

sys.path.insert(0, "/root/eva_output")
import eva_adapter as EVA

ds = sys.argv[1]
m = EVA.load(ds)


def fp(part):
    # meta[part] 是原始数据集（可能是 Subset），要过一层 Wrapped 才有 .labels
    w = EVA.Wrapped(m[part], m)
    lab = np.asarray(w.labels).reshape(-1)
    h = hashlib.md5(lab.tobytes()).hexdigest()[:12]
    # feats 只是参照：cifar100 / SVHN / tinyimagenet 的 **train** transform 带随机
    # 增广（RandomCrop/Flip），同一份切分取两次也会不同 —— 那不是切分不一致。
    # 判定切分是否一致请只看 labels=<hash>（顺序敏感，能同时反映成员与顺序）。
    x0 = np.asarray(w[0][0], dtype=np.float64).ravel()
    x1 = np.asarray(w[1][0], dtype=np.float64).ravel()
    hx = hashlib.md5((x0.tobytes() + x1.tobytes())).hexdigest()[:12]
    return "%s len=%-6d labels=%s feats=%s" % (part, len(lab), h, hx)


print("[%s] %s" % (ds, fp("train")))
print("[%s] %s" % (ds, fp("val")))
