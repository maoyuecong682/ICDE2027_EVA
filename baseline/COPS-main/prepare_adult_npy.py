# -*- coding: utf-8 -*-
"""prepare_adult_npy.py —— 为 COPS 生成 adult_npy

对齐 COPS 官方 split_data/cifar_split.py 的协议：
    train_X_1 / train_Y_1 : probe 集（每类 20% 训练样本）
    train_X_2 / train_Y_2 : sampling 集（其余 80%，COPS 从这里选核心集）
    test_X  / test_Y      : 真实测试集

用法：
    python prepare_adult_npy.py --out_dir /root/autodl-tmp/EVA/output/COPS/adult_npy
"""

import argparse
import json
import os

import numpy as np

from adult_adapter import load_adult_arrays, split_train_probe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="/root/autodl-tmp/EVA/data/adult/adult_processed.npz")
    ap.add_argument("--out_dir", default="/root/autodl-tmp/EVA/output/COPS/adult_npy")
    ap.add_argument("--probe_ratio", type=float, default=0.2,
                    help="probe 集占训练集比例，CIFAR10 官方为 1000/5000=0.2")
    ap.add_argument("--seed", type=int, default=48, help="对齐 cifar_split.py 的 seed=48")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    d = load_adult_arrays(args.npz, normalized=True)

    (px, py), (sx, sy) = split_train_probe(
        d["train_x"], d["train_y"], probe_ratio=args.probe_ratio, seed=args.seed)

    np.save(os.path.join(args.out_dir, "train_X_1.npy"), px)
    np.save(os.path.join(args.out_dir, "train_Y_1.npy"), py)
    np.save(os.path.join(args.out_dir, "train_X_2.npy"), sx)
    np.save(os.path.join(args.out_dir, "train_Y_2.npy"), sy)
    np.save(os.path.join(args.out_dir, "test_X.npy"), d["test_x"])
    np.save(os.path.join(args.out_dir, "test_Y.npy"), d["test_y"])

    meta = {
        "npz": args.npz,
        "normalized": True,
        "probe_ratio": args.probe_ratio,
        "seed": args.seed,
        "num_features": d["num_features"],
        "num_classes": d["num_classes"],
        "class_names": d["class_names"],
        "probe": {"n": int(len(py)), "label_counts": np.bincount(py, minlength=2).tolist()},
        "sampling": {"n": int(len(sy)), "label_counts": np.bincount(sy, minlength=2).tolist()},
        "test": {"n": int(len(d["test_y"])), "label_counts": np.bincount(d["test_y"], minlength=2).tolist()},
    }
    with open(os.path.join(args.out_dir, "adult_npy_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[prepare_adult_npy] ->", os.path.abspath(args.out_dir))
    print("  probe   :", px.shape, py.shape, "counts", meta["probe"]["label_counts"])
    print("  sampling:", sx.shape, sy.shape, "counts", meta["sampling"]["label_counts"])
    print("  test    :", d["test_x"].shape, d["test_y"].shape, "counts", meta["test"]["label_counts"])


if __name__ == "__main__":
    main()
