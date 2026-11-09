# -*- coding: utf-8 -*-
"""prepare_adult.py —— 用 adult_modified 预处理 Adult 数据并落盘

输出目录（默认与原始数据同路径）：<data_dir>/adult/
    adult_train_processed.csv   清洗后的训练原始表（全数值，与 raw train.csv 同 32561 行）
    adult_test_processed.csv    清洗后的测试原始表（全数值，与 raw test.csv 同 16281 行）
    adult_train_split.csv       adult() 实际使用的训练子集
    adult_val_split.csv         adult() 实际使用的验证子集
    adult_test_split.csv        adult() 实际使用的测试子集
    adult_processed.npz         X/y 数组（raw + 归一化）、mean/std、特征名
    adult_preprocess_meta.json  形状 / 类别分布 / 校验信息

注意：不会覆盖原始 train.csv / test.csv。
"""

import json
import os
import sys

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "adult_modified", os.path.join(BASE_DIR, "mydatasets", "adult_modified.py")
)
am = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(am)


def main(data_dir, out_dir=None):
    out_dir = out_dir or os.path.join(data_dir, "adult")
    os.makedirs(out_dir, exist_ok=True)

    # ---------- 1. 清洗原始表 ----------
    df_train_raw, df_test_raw = am.load_adult_dataset(data_dir)
    df_train = am.dataCleaning(df_train_raw)
    df_test = am.dataCleaning(df_test_raw)

    # 原始表（未做标签过滤），保证与 raw csv 行数一一对应
    train_proc_path = os.path.join(out_dir, "adult_train_processed.csv")
    test_proc_path = os.path.join(out_dir, "adult_test_processed.csv")
    df_train.to_csv(train_proc_path, index=False)
    df_test.to_csv(test_proc_path, index=False)

    # 幂等性检查：对已处理的数据再跑一次 dataCleaning，结果必须不变
    again = am.dataCleaning(pd.read_csv(train_proc_path))
    idempotent = bool(
        again.shape == df_train.shape
        and np.allclose(again.to_numpy(), df_train.to_numpy())
    )

    # ---------- 2. 走完整流程拿到划分 ----------
    (channel, im_size, num_classes, class_names,
     mean, std, train_dataset, val_dataset, test_dataset) = am.adult(data_dir, balance=True)

    feat_names = [c for c in df_train.columns if c not in am.LABEL_COLUMNS]

    # 重新从划分结果还原 DataFrame（标签过滤后）
    dtr = am._filter_valid_labels(df_train, "money", verbose=False)
    dte = am._filter_valid_labels(df_test, "money1", verbose=False)
    l0 = dtr[dtr["money"] == 0]
    l1 = dtr[dtr["money"] == 1]
    t0, v0 = am._safe_split(l0, 0.2)
    t1, v1 = am._safe_split(l1, 0.2)
    sp_train = pd.concat([t0, t1]).reset_index(drop=True)
    sp_val = pd.concat([v0, v1]).reset_index(drop=True)
    sp_test = dte.reset_index(drop=True)

    sp_train.to_csv(os.path.join(out_dir, "adult_train_split.csv"), index=False)
    sp_val.to_csv(os.path.join(out_dir, "adult_val_split.csv"), index=False)
    sp_test.to_csv(os.path.join(out_dir, "adult_test_split.csv"), index=False)

    # ---------- 3. 导出 npz ----------
    def to_xy(df, label_col):
        x = df[feat_names].to_numpy(dtype=np.float64)
        y = df[label_col].to_numpy(dtype=np.int64)
        return x.astype(np.float32), y

    X_train, y_train = to_xy(sp_train, "money")
    X_val, y_val = to_xy(sp_val, "money")
    X_test, y_test = to_xy(sp_test, "money1")

    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    norm = lambda x: (x - mean) / (std + 1e-8)

    npz_path = os.path.join(out_dir, "adult_processed.npz")
    np.savez_compressed(
        npz_path,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        X_train_norm=norm(X_train), X_val_norm=norm(X_val), X_test_norm=norm(X_test),
        mean=mean, std=std,
        feature_names=np.array(feat_names, dtype=object),
        class_names=np.array(class_names, dtype=object),
    )

    # ---------- 4. meta ----------
    meta = {
        "data_dir": os.path.abspath(data_dir),
        "out_dir": os.path.abspath(out_dir),
        "raw_rows": {"train": int(len(df_train_raw)), "test": int(len(df_test_raw))},
        "processed_rows": {"train": int(len(df_train)), "test": int(len(df_test))},
        "split_rows": {"train": int(len(sp_train)), "val": int(len(sp_val)), "test": int(len(sp_test))},
        "num_features": int(im_size),
        "num_classes": int(num_classes),
        "channel": int(channel),
        "feature_names": feat_names,
        "class_names": list(class_names),
        "label_distribution": {
            "train_split": {str(k): int(v) for k, v in sp_train["money"].value_counts().sort_index().items()},
            "val_split": {str(k): int(v) for k, v in sp_val["money"].value_counts().sort_index().items()},
            "test_split": {str(k): int(v) for k, v in sp_test["money1"].value_counts().sort_index().items()},
        },
        "dtype": {"X": "float32", "y": "int64"},
        "mean": [float(v) for v in mean],
        "std": [float(v) for v in std],
        "nan_in_processed": bool(df_train.isna().any().any() or df_test.isna().any().any()),
        "sentinel_minus1_total": {c: int((df_train[c] == -1).sum()) for c in df_train.columns},
        "idempotent_dataCleaning": idempotent,
        "files": [
            "adult_train_processed.csv", "adult_test_processed.csv",
            "adult_train_split.csv", "adult_val_split.csv", "adult_test_split.csv",
            "adult_processed.npz",
        ],
    }
    meta_path = os.path.join(out_dir, "adult_preprocess_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # ---------- 5. 报告 ----------
    print("=" * 62)
    print("输出目录 :", os.path.abspath(out_dir))
    print("特征数   : %d   类别数: %d   类别名: %s" % (im_size, num_classes, class_names))
    print("清洗后   : train %d 行 / test %d 行（raw %d / %d）"
          % (len(df_train), len(df_test), len(df_train_raw), len(df_test_raw)))
    print("划分     : train %d / val %d / test %d"
          % (len(sp_train), len(sp_val), len(sp_test)))
    print("标签分布 : train %s  val %s  test %s"
          % (dict(sp_train["money"].value_counts().sort_index()),
             dict(sp_val["money"].value_counts().sort_index()),
             dict(sp_test["money1"].value_counts().sort_index())))
    print("异常值   : NaN=%s   -1 兜底合计=%d"
          % (meta["nan_in_processed"], sum(meta["sentinel_minus1_total"].values())))
    print("幂等性   : dataCleaning 二次处理不变 =", idempotent)
    print("=" * 62)
    for name in meta["files"] + ["adult_preprocess_meta.json"]:
        p = os.path.join(out_dir, name)
        print("  %-32s %8.1f KB" % (name, os.path.getsize(p) / 1024))
    return meta


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/EVA/data"
    main(d)
