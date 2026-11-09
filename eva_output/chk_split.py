"""核对 mnist / qmnist 官方 test 的每类数量，以及 split_valtest 缓存是否自洽。

用途：`mnist` 汇总为 4999/5001、`qmnist` 为 29998/30002，而这两个的官方 test
按说每类等量（1000 / 6000）。差异只可能来自两处：
  a) 官方 test 本身某些类不是偶数；或
  b) 复用的缓存 JSON 与当前标签数组对不上。
本脚本把两者都打出来对比。
"""
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from torchvision import datasets

MDIR = "/root/chaichengliang/VLDB2027/EVA/EVA/mydatasets"
if MDIR not in sys.path:
    sys.path.insert(0, MDIR)

D = "/root/autodl-tmp/EVA/data"

CASES = [
    ("mnist",  D + "/mnist",  datasets.MNIST,  {}),
    ("qmnist", D + "/qmnist", datasets.QMNIST, {"what": "test"}),
]

for name, path, cls, kw in CASES:
    d = cls(path, train=False, download=False, **kw)
    t = getattr(d, "targets", None)
    if t is None:
        t = getattr(d, "labels")
    t = np.asarray(t)
    # QMNIST 的原始 .targets 是 N x 10 的标签记录矩阵，第 0 列才是数字类别
    # （见 mydatasets/qmnist.py:14 的 `targets = targets[:, 0]`）。
    # 直接调 datasets.QMNIST 会绕过那句话，这里补上同样的切片。
    if t.ndim > 1:
        t = t[:, 0]
    t = t.reshape(-1)
    print("%s: n=%d per_class=%s" % (name, len(d), np.bincount(t, minlength=10).tolist()),
          flush=True)

    mp = os.path.join(path, "split_valtest_%s.json" % name)
    if not os.path.exists(mp):
        print("  no cache", flush=True)
        continue
    with open(mp) as fh:
        m = json.load(fh)
    v = np.asarray(m["val_idx"])
    te = np.asarray(m["test_idx"])
    print("  cache: n=%d val=%d test=%d overlap=%d" % (m["n"], len(v), len(te),
                                                       len(np.intersect1d(v, te))), flush=True)
    print("  val_per_class =%s" % np.bincount(t[v], minlength=10).tolist(), flush=True)
    print("  test_per_class=%s" % np.bincount(t[te], minlength=10).tolist(), flush=True)

    # 用同一 seed 重算一遍，看和缓存是否逐位一致
    from valtest_split import stratified_half
    v2, t2 = stratified_half(t, 10, seed=0)
    print("  recompute(seed=0): val=%d test=%d  identical_to_cache=%s"
          % (len(v2), len(t2), bool(len(v) == len(v2) and np.array_equal(v, np.asarray(v2)))),
          flush=True)
