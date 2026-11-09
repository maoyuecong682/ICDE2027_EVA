#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断 HR / adult 的评估指标自洽性：打印混淆矩阵 + sklearn 各口径指标。

怀疑点：acc=0.2783 与 precision_weighted=0.732 对同一个预测向量不可能同时成立
（HR 测试集 952 负 / 345 正，多数类基线就是 0.734）。所以要看清到底是
「模型真的在反着猜」还是「指标口径不一致」。
"""
import os
import sys

os.environ.pop("EVA_REG_BINS", None)
sys.path.insert(0, "/root/eva_output")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

import eva_adapter as EVA


class Plain(torch.utils.data.Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        x, y = self.ds[i]
        return x, int(y)


def run(ds_name, n_ep=60, lr=0.01):
    meta = EVA.load(ds_name)
    print("=" * 60)
    print(ds_name, "num_classes =", meta["num_classes"], "is_tabular =", meta["is_tabular"])

    idx = np.load("/root/eva_output/COPS/%s/weights_cops_wy/selected_indices.npy" % ds_name).reshape(-1)
    coreset = EVA.select(EVA._wrap(meta["train"], meta), idx)
    val = EVA._wrap(meta["val"], meta)
    test = EVA._wrap(meta["test"], meta)
    print("coreset =", len(coreset))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(48)
    model = EVA.build_model(meta, arch=None).to(device)
    loader = DataLoader(Plain(coreset), batch_size=128, shuffle=True, num_workers=0)
    crit = nn.CrossEntropyLoss().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                          weight_decay=5e-4, nesterov=True)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_ep)
    for ep in range(n_ep):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = crit(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sch.step()

    def preds_of(d):
        model.eval()
        P, Y = [], []
        with torch.no_grad():
            for x, y in DataLoader(Plain(d), batch_size=256, shuffle=False):
                P.append(model(x.to(device)).argmax(1).cpu().numpy())
                Y.append(y.numpy())
        return np.concatenate(P), np.concatenate(Y)

    for tag, d in (("val", val), ("test", test)):
        p, y = preds_of(d)
        print("[%s] n=%d  预测分布=%s  真实分布=%s"
              % (tag, len(y), dict(zip(*np.unique(p, return_counts=True))),
                 dict(zip(*np.unique(y, return_counts=True)))))
        print("     acc=%.4f  prec_w=%.4f  rec_w=%.4f  f1_w=%.4f"
              % (accuracy_score(y, p),
                 precision_score(y, p, average="weighted", zero_division=0),
                 recall_score(y, p, average="weighted", zero_division=0),
                 f1_score(y, p, average="weighted", zero_division=0)))
        print("     混淆矩阵(行=真,列=测):\n", confusion_matrix(y, p))
        if tag == "test":
            print(classification_report(y, p, zero_division=0))


if __name__ == "__main__":
    for name in (sys.argv[1:] or ["HR", "adult"]):
        run(name)
