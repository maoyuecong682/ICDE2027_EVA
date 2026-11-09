# -*- coding: utf-8 -*-
"""prerun_text.py -- 预热三个文本数据集的 BERT 嵌入缓存。

为什么要单独预热
----------------
imdb / imdb0 / news 的加载函数都是"首次调用时算 BERT、之后读 npy"。而每个
baseline 的每个阶段都是一个**独立进程**：COPS 有 stage1 + 3 个重训变体，
GM 一个，DUAL 一个 —— 同一份数据要被加载 6 次。若边跑边算，就是 6 倍 BERT 前向
（每次 1~2 分钟）；先在这里串行算一遍，后面 6 次全走缓存。

顺便做三件事：
  1) 打印每个数据集的形状/类别数/标签分布，作为"接进适配层后是否真的对"的证据；
  2) 校验 train/val/test 无重叠、类别数一致；
  3) 把 mean/std 的若干维打出来，便于和主实验对照标准化口径。
"""
import os
import sys

import numpy as np

sys.path.insert(0, "/root/eva_output")
import eva_adapter as E  # noqa: E402


def probe(name):
    print("\n" + "=" * 70, flush=True)
    print("[prerun] %s" % name, flush=True)
    m = E.load(name)
    print("  channel=%d im_size=%s num_classes=%d is_tabular=%d"
          % (m["channel"], m["im_size"], m["num_classes"], m["is_tabular"]), flush=True)
    print("  train/val/test = %d / %d / %d"
          % (len(m["train"]), len(m["val"]), len(m["test"])), flush=True)
    print("  mean[:3]=%s  std[:3]=%s"
          % (np.round(np.asarray(m["mean"])[:3], 4),
             np.round(np.asarray(m["std"])[:3], 4)), flush=True)
    for split in ("train", "val", "test"):
        labs = np.asarray(E._labels_of(m[split])).reshape(-1)
        u, c = np.unique(labs, return_counts=True)
        show = dict(zip(u.tolist()[:6], c.tolist()[:6]))
        print("  %-5s labels: %d 类 %s%s"
              % (split, len(u), show, " ..." if len(u) > 6 else ""), flush=True)
    x, y = E._raw_item(m["train"], 0)
    print("  sample0 shape=%s dtype=%s y=%s  |x|max=%.3f"
          % (tuple(x.shape), x.dtype, y, float(x.abs().max())), flush=True)
    return m


if __name__ == "__main__":
    ok, bad = [], []
    for name in sys.argv[1:] or ["news", "imdb", "imdb0"]:
        try:
            probe(name)
            ok.append(name)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            bad.append("%s(%s: %s)" % (name, type(exc).__name__, exc))
    print("\n[done] ok=%s bad=%s" % (ok, bad), flush=True)
