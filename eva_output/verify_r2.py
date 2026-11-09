# -*- coding: utf-8 -*-
"""verify_r2.py —— 确认第二轮的两处修复在运行时真的生效

只看 grep 不够：Indexed/Weighted 必须能被 DataLoader 正确 collate，
否则错误会推迟到训练循环内部才暴露（这正是 HR 那次踩的坑）。
"""
import sys

sys.path.insert(0, "/root/eva_output")

import numpy as np
import torch
from torch.utils.data import DataLoader

import eva_adapter as E

ok = True


def chk(cond, msg):
    global ok
    print(("  [PASS] " if cond else "  [FAIL] ") + msg)
    if not cond:
        ok = False


print("== 1. 类是否就位")
chk(hasattr(E, "Weighted"), "eva_adapter.Weighted 存在")
chk(hasattr(E, "Indexed"), "eva_adapter.Indexed 存在")

print("== 2. Indexed 经 DataLoader 后的批次形状（DUAL 动力学协议）")
m = E.load("HR")
ds = E.Indexed(m["train"], m)
x0, t0 = ds[0]
chk(isinstance(t0, list) and len(t0) == 2, "单样本 target 是 [label, index]，实际 %r" % (t0,))
chk(t0[1] == 0, "首个样本局部下标为 0，实际 %r" % (t0[1],))

dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
xb, tb = next(iter(dl))
chk(torch.is_tensor(tb[0]) and tb[0].shape == (16,),
    "target[0] 是长度 16 的标签张量，实际 %s" % (getattr(tb[0], "shape", None),))
chk(torch.is_tensor(tb[1]) and tb[1].shape == (16,),
    "target[1] 是长度 16 的下标张量，实际 %s" % (getattr(tb[1], "shape", None),))
chk(tb[1].tolist()[:3] == [0, 1, 2], "下标是 0..15 连续，实际 %r" % (tb[1].tolist()[:3],))
chk(torch.equal(tb[0].long(), torch.as_tensor(ds.labels[:16], dtype=torch.long)),
    "target[0] 与 Wrapped.labels 前 16 个一致")

print("== 3. Weighted 经 DataLoader 后的三元组（COPS reweight_clip 协议）")
w = np.random.RandomState(0).rand(len(ds)).astype(np.float32)
wds = E.Weighted(ds, w)
xb2, yb2, wb2 = next(iter(DataLoader(wds, batch_size=16, shuffle=False, num_workers=0)))
chk(torch.is_tensor(wb2) and wb2.shape == (16,),
    "权重是长度 16 的张量，实际 %s" % (getattr(wb2, "shape", None),))
chk(abs(float(wb2[3]) - float(w[3])) < 1e-6, "第 4 个权重与输入一致")

print("== 4. select 之后再 Weighted，权重与样本仍然对齐")
idx = np.array([5, 7, 9, 11], dtype=np.int64)
sub = E.select(ds, idx)
sw = E.Weighted(sub, w[idx])
chk(len(sw) == 4, "子集长度 4，实际 %d" % len(sw), )
chk(abs(float(sw[2][2]) - float(w[9])) < 1e-6, "子集第 3 个样本的权重取自原第 9 个")

print("\nRESULT %s" % ("OK" if ok else "BAD"))
sys.exit(0 if ok else 1)
