#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把三个 baseline 各自的 *_result.csv 汇总成 evaluation3090.csv。

版式严格照 evaluation/evaluation3090.md：

    数据集(18个,分别占一行) ||  COPS                          ||  GM Matching                 ||  DUAL
    news(本质是分类)        ||  RANDOM acc f1 recall precision ||  RANDOM acc f1 recall precision ||  RANDOM acc f1 recall precision
                            ||  FULL   acc f1 recall precision ||  FULL   ...                    ||  FULL   ...
    bike(本质是回归)        ||  MSE MAE                        ||  MSE MAE                      ||  MSE MAE

也就是说：
  * 分类数据集占**两行**（RANDOM 一行、FULL 一行），每个 baseline 四列 acc/f1/recall/precision；
  * 回归数据集占**一行**，每个 baseline 两列 MSE/MAE（后两列留空）。
列顺序按 md 里写的 acc f1 recall precision。

用法：
    python make_eval3090.py [--root /root/autodl-tmp/EVA/evaluation]
"""
import argparse
import csv
import os

ORDER = ["news", "adult", "bike", "cifar10", "cifar100", "covtype", "fashionmnist",
         "House", "HR", "imdb", "mnist", "qmnist", "SVHN", "tinyimagenet", "imdb0",
         "imdb_large", "mnist2mnistm", "imagenet"]
REG = {"bike", "House"}
BASELINES = [("COPS", "COPS"), ("GM_matching", "GM Matching"), ("DUAL", "DUAL")]

# 每行表头：dataset, 类型标记, 然后每个 baseline 4 列
HEADER = ["dataset", "routine"]
for _, disp in BASELINES:
    for m in ("acc", "f1", "recall", "precision"):
        HEADER.append("%s_%s" % (disp, m))


def load(root, bl):
    p = os.path.join(root, bl, "%s_result.csv" % bl)
    out = {}
    if not os.path.exists(p):
        return out
    with open(p, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out[r["dataset"]] = r
    return out


def fmt(v, nd=4):
    if v in (None, "", "None"):
        return ""
    try:
        return "%.*f" % (nd, float(v))
    except (TypeError, ValueError):
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/root/autodl-tmp/EVA/evaluation")
    a = ap.parse_args()

    data = {bl: load(a.root, bl) for bl, _ in BASELINES}
    rows = []
    for ds in ORDER:
        if ds in REG:
            row = {"dataset": "%s(本质是回归)" % ds, "routine": ""}
            for bl, disp in BASELINES:
                r = data[bl].get(ds, {})
                row["%s_acc" % disp] = fmt(r.get("kmeans_mse"))
                row["%s_f1" % disp] = fmt(r.get("kmeans_mae"))
                row["%s_recall" % disp] = ""
                row["%s_precision" % disp] = ""
            rows.append(row)
            continue
        for routine, tag in (("random", "RANDOM"), ("full", "FULL")):
            row = {"dataset": "%s(本质是分类)" % ds if routine == "random" else "",
                   "routine": tag}
            for bl, disp in BASELINES:
                r = data[bl].get(ds, {})
                row["%s_acc" % disp] = fmt(r.get("%s_accuracy" % routine))
                row["%s_f1" % disp] = fmt(r.get("%s_f1" % routine))
                row["%s_recall" % disp] = fmt(r.get("%s_recall" % routine))
                row["%s_precision" % disp] = fmt(r.get("%s_precision" % routine))
            rows.append(row)

    out = os.path.join(a.root, "evaluation3090.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=HEADER)
        w.writeheader()
        w.writerows(rows)

    # 同时落一份人读的 markdown
    md = os.path.join(a.root, "evaluation3090_table.md")
    with open(md, "w", encoding="utf-8") as fh:
        fh.write("| 数据集 | | " + " | ".join("%s acc / f1 / recall / precision" % d
                                              for _, d in BASELINES) + " |\n")
        fh.write("|---|---|" + "---|" * len(BASELINES) + "\n")
        for r in rows:
            cells = []
            is_reg = "回归" in r["dataset"]
            for _, disp in BASELINES:
                v = [r["%s_%s" % (disp, m)] for m in ("acc", "f1", "recall", "precision")]
                if is_reg:
                    # 回归行：md 模板把这 4 列写成 "MSE MAE"，列名却是 acc/f1，
                    # 这里显式标出来，免得读者把 MSE 当成 accuracy。
                    cells.append("MSE %s / MAE %s" % (v[0] or "—", v[1] or "—"))
                else:
                    cells.append(" / ".join(x if x else "—" for x in v))
            fh.write("| %s | %s | %s |\n" % (r["dataset"] or "→", r["routine"],
                                             " | ".join(cells)))

    n_done = sum(1 for ds in ORDER
                 if all(ds in data[bl] and any(v for k, v in data[bl][ds].items()
                                               if k != "dataset")
                        for bl, _ in BASELINES))
    print("[ok] 写出 %s（%d 行）" % (out, len(rows)))
    print("[ok] 写出 %s" % md)
    print("[进度] %d/%d 个数据集三个 baseline 都有数" % (n_done, len(ORDER)))
    miss = [ds for ds in ORDER if ds not in {d for bl, _ in BASELINES for d in data[bl]}]
    if miss:
        print("[缺] " + " ".join(miss))


if __name__ == "__main__":
    main()
