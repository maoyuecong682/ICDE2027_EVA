"""在 2GB cgroup 限制下验证 covtype_modified 的三个不变量。

上一版为什么会被 SIGKILL
------------------------
`/sys/fs/cgroup/memory.max = 2147483648`，而 `memory.current` 平时就在 1.2GB 左右
（大半是读数据留下的 page cache，但匿名内存的预算只剩 ~800MB）。`covtype.py` 内部

    dst_train.data = np.array(data)[train_indices]      # np.array(data) 先整块拷 125MB
    mean = np.mean(data[train_indices], axis=0)        # 又拷 100MB
    std  = np.std (data[train_indices], axis=0)        # 再拷 100MB
    dst_train.data = (dst_train.data - mean) / (std + 1e-8)   # 又生成 100MB 临时量

峰值远高于剩余预算。所以本脚本自己不做任何整块拷贝，只在 loader 返回之后
读取**小对象**（indices / labels），并全程 flush，死在哪一步能看见。

三个不变量
----------
1. 不再泄漏：`dst_val is dst_test` 必须为 False，且两者索引不相交。
2. 划分完整：|val| + |test| == 原 val 池大小（116205），且每类两边的数目相差 <= 1。
3. 归一化只做一次：池子经 `* (std+1e-8) + mean` 还原后，特征均值应贴近 0、
   标准差贴近 1。若还原失败（仍是二次归一化），均值会落在 -mean/std 量级
   —— covtype 的 elevation 一列 mean/std ≈ 10.6，一眼可辨。
"""
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

MDIR = "/root/chaichengliang/VLDB2027/EVA/EVA/mydatasets"
DATA = "/root/autodl-tmp/EVA/data/covtype"

if MDIR not in sys.path:
    sys.path.insert(0, MDIR)

import numpy as np  # noqa: E402


def rss_mb():
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) // 1024
    return -1


def cg():
    try:
        with open("/sys/fs/cgroup/memory.current") as fh:
            return int(fh.read().strip()) // (1024 * 1024)
    except Exception:
        return -1


def p(*args):
    print("[rss=%dMB cg=%dMB]" % (rss_mb(), cg()), *args, flush=True)


p("start; importing torch ...")
import torch  # noqa: E402

torch.set_num_threads(1)
p("torch", torch.__version__, "ok")

from covtype_modified import covtype  # noqa: E402

p("calling covtype(%r, balance=True) ..." % DATA)
info = covtype(DATA, True)
p("returned; len(info) =", len(info))

ch, im, nc, cn, mean, std, tr, va, te = info
p("head:", "channel=%r im_size=%r num_classes=%r" % (ch, im, nc))

report = {
    "n_tuple": len(info),
    "channel": int(ch),
    "im_size": int(im),
    "num_classes": int(nc),
    "len_train": len(tr),
    "len_val": len(va),
    "len_test": len(te),
    "type_train": type(tr).__name__,
    "type_val": type(va).__name__,
    "type_test": type(te).__name__,
    "val_is_test_obj": bool(va is te),
    "val_dataset_is_test_dataset": bool(getattr(va, "dataset", None) is getattr(te, "dataset", None)),
}

# ---- 只碰 indices / labels，不碰 .data ----------------------------------
vi = np.asarray(va.indices)
ti = np.asarray(te.indices)
report["val_idx_unique"] = int(len(np.unique(vi)))
report["test_idx_unique"] = int(len(np.unique(ti)))
report["overlap"] = int(np.intersect1d(vi, ti).size)
n_pool = len(va.dataset)
report["pool_len"] = int(n_pool)
report["cover_ok"] = bool(len(vi) + len(ti) == n_pool)

pool_labels = np.asarray(va.dataset.labels).reshape(-1)
report["pool_label_min"] = int(pool_labels.min())
report["pool_label_max"] = int(pool_labels.max())

val_counts = np.bincount(pool_labels[vi], minlength=int(nc))
test_counts = np.bincount(pool_labels[ti], minlength=int(nc))
report["val_per_class"] = val_counts.tolist()
report["test_per_class"] = test_counts.tolist()
report["per_class_max_diff"] = int(np.abs(val_counts - test_counts).max())

# 每个样本真的能取出来（走 __getitem__ -> torch.tensor）
x0, y0 = va[0]
x1, y1 = te[0]
report["sample_val"] = [list(x0.shape), str(x0.dtype), int(y0)]
report["sample_test"] = [list(x1.shape), str(x1.dtype), int(y1)]
report["sample_finite"] = bool(np.isfinite(x0.numpy()).all() and np.isfinite(x1.numpy()).all())

# ---- 归一化只做一次？池子整体统计量 ------------------------------------
pool = va.dataset.data
m = np.asarray(pool.mean(axis=0), dtype=np.float64)
s = np.asarray(pool.std(axis=0), dtype=np.float64)
report["pool_mean_abs_max"] = float(np.abs(m).max())
report["pool_mean_abs_mean"] = float(np.abs(m).mean())
report["pool_std_min"] = float(s.min())
report["pool_std_max"] = float(s.max())
report["pool_std_mean"] = float(s.mean())
# 若还原失败（仍在二次归一化状态），均值量级应当是 |mean/std|
ms = np.asarray(mean, dtype=np.float64) / (np.asarray(std, dtype=np.float64) + 1e-8)
report["mean_over_std_abs_max"] = float(np.abs(ms).max())
# 期望：pool_mean_abs_max 远小于 mean_over_std_abs_max -> 说明还原生效
report["norm_looks_single"] = bool(report["pool_mean_abs_max"] < 0.2)

print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)

out = "/root/eva_output/verify_covtype_report.json"
with open(out, "w", encoding="utf-8") as fh:
    json.dump(report, fh, indent=2, ensure_ascii=False)
p("written", out)

ok = (report["n_tuple"] == 9
      and report["val_is_test_obj"] is False
      and report["overlap"] == 0
      and report["cover_ok"] is True
      and report["per_class_max_diff"] <= 1
      and report["norm_looks_single"] is True)
p("ALL_INVARIANTS_OK =", ok)
print("[exit-marker] done", flush=True)
