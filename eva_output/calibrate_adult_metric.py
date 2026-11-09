# -*- coding: utf-8 -*-
"""calibrate_adult_metric.py —— 为 COPS 标定 constant_1

COPS 里的 `--constant_1` 是 metric 的裁剪上限，属于**依赖数据量纲**的超参：
官方 CIFAR 脚本用 0.005，adult 的 metric 尺度不同，直接照搬会让所有样本
被 clip 成同一个常数 → 采样退化成均匀随机，COPS 失效。

这里复现 `retraining_withy.py` 中 `cal_un_wy` 的算法，统计 metric 分布，
并给出一个尺度无关的默认阈值：**中位数**（即裁掉不确定性较高的一半）。
同时输出各分位数，便于按需换成其他分位。

用法：
    python calibrate_adult_metric.py --npy_folder <adult_npy> --out metric_stats.json
"""

import argparse
import json
import os

import numpy as np


def cal_un_wy(logit, label):
    """与 retraining_withy.py 完全一致的实现（含除零保护）。"""
    mu = logit - np.mean(logit, axis=0, keepdims=True)
    covariance = np.matmul(mu.transpose()[1:, :], mu[:, 1:]) / max(len(logit) - 1, 1)
    z = logit - np.max(logit, 1, keepdims=True)
    pred = np.mean(np.exp(z) / np.sum(np.exp(z), 1, keepdims=True), 0)
    s = np.reshape((pred - label)[1:], (len(label) - 1, 1))
    met = np.matmul(s.transpose(), np.matmul(covariance, s))
    return np.sqrt(met.ravel()[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy_folder", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quantile", type=float, default=0.5,
                    help="用哪个分位数作为 constant_1（默认 0.5 即中位数）")
    args = ap.parse_args()

    logit = np.load(os.path.join(args.npy_folder, "train_UN_2_raw.npy"))  # (M, N, C)
    y = np.load(os.path.join(args.npy_folder, "train_Y_2.npy"))
    n_models, n, n_cls = logit.shape
    print("logits shape:", logit.shape, " -> model_num =", n_models)

    oh = np.zeros((n, n_cls), dtype=np.float64)
    oh[np.arange(n), y] = 1.0

    per_sample = logit.transpose((1, 0, 2))  # (N, M, C)
    metric = np.array([cal_un_wy(per_sample[k], oh[k]) for k in range(n)])

    qs = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    quantiles = {("p%g" % (q * 100)): float(np.quantile(metric, q)) for q in qs}
    c1 = float(np.quantile(metric, args.quantile))

    # 采用该阈值时会被裁剪（封顶）的样本比例，用于判断阈值是否合理
    capped_ratio = float((metric > c1).mean())

    stats = {
        "n_models": int(n_models),
        "n_samples": int(n),
        "num_classes": int(n_cls),
        "metric": {
            "min": float(metric.min()),
            "max": float(metric.max()),
            "mean": float(metric.mean()),
            "std": float(metric.std()),
            "quantiles": quantiles,
        },
        "quantile_used": args.quantile,
        "constant_1": c1,
        "capped_ratio": capped_ratio,
        "nan_count": int(np.isnan(metric).sum()),
        "note": "constant_1 取 metric 的中位数，尺度无关；如需其他口径改 --quantile",
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
