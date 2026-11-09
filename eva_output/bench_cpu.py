"""在 0.5 核 cgroup 配额下实测 ResNet18 的训练吞吐，用于估算全量实验耗时。

为什么要先测：容器 cpu.max = 50000/100000（0.5 核），但 nproc 谎报 128。
任何"按 GPU 经验估时间"的做法都会差两个数量级。这里直接跑 forward+backward。
"""
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch
import torch.nn as nn

torch.set_num_threads(1)

import torchvision.models as tvm

BATCH = 128
REPS = 5

for name, insize in [("resnet18@32", 32), ("resnet18@224", 224)]:
    model = tvm.resnet18(num_classes=100).train()
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    crit = nn.CrossEntropyLoss()
    x = torch.randn(BATCH, 3, insize, insize)
    y = torch.randint(0, 100, (BATCH,))

    # warmup
    for _ in range(1):
        opt.zero_grad()
        crit(model(x), y).backward()
        opt.step()

    t0 = time.time()
    for _ in range(REPS):
        opt.zero_grad()
        crit(model(x), y).backward()
        opt.step()
    dt = (time.time() - t0) / REPS
    print("%-14s batch=%d  %.3f s/batch  -> %.2f img/s"
          % (name, BATCH, dt, BATCH / dt), flush=True)

    if insize == 32:
        print("   推算：cifar100 50000 张 = 391 batch/epoch -> %.1f 分钟/epoch"
              % (391 * dt / 60), flush=True)
        print("   推算：tinyimagenet 100000 张 = 782 batch/epoch -> %.1f 分钟/epoch"
              % (782 * dt / 60), flush=True)
