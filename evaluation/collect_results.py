#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collect the adult results of all four sources into one comparable table.

Sources and the criteria each one actually uses (they do **not** agree, which is
why the criterion is recorded in its own column):

  COPS    output/COPS/logs/{03_uniform,04_cops_wy,05_cops_woy}.log
          per-epoch "Real Test Loss Acc" -> best and last.  Best = selected on test.
  GM      output/GM_matching/logs/run.log  "test_acc: x" -> best and last.
          (gm_matching.csv is truncated per repetition, so run.log is the record.)
  DUAL    output/DUAL/logs/01_dynamics.log      full-data reference
          output/DUAL/logs/03_pruned_train.log  10% pruned subset
          "Best : Accuracy=" is validation-selected; "**Test** Prec@1" is the
          accuracy of the final epoch.
  EVA     output/evaluation/summary.csv  random / full / kmeans

Run on the server:
    /root/miniconda3/bin/python evaluation/collect_results.py
"""
import argparse
import csv
import os
import re


def _read(path):
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()


def best_last(values):
    if not values:
        return None, None
    return max(values), values[-1]


def parse_cops(path):
    text = _read(path)
    # COPS already logs percentages here ("Real Test Loss Acc :  84.4174")
    vals = [float(v) for v in re.findall(r"Real Test Loss Acc\s*:\s*([\d.]+)", text)]
    best, last = best_last(vals)
    return {
        "best": round(best, 2) if best is not None else None,
        "last": round(last, 2) if last is not None else None,
        "criterion": "best-on-test / last epoch",
        "epochs": len(vals),
        "secs": "",
    }


def parse_gm(path):
    text = _read(path)
    vals = [float(v) for v in re.findall(r"test_acc:\s*([\d.]+)", text)]
    best, last = best_last(vals)
    return {
        "best": round(best * 100, 2) if best is not None else None,
        "last": round(last * 100, 2) if last is not None else None,
        "criterion": "best-on-test / last epoch",
        "epochs": len(vals),
        "secs": "",
    }


def parse_dual(path, label):
    text = _read(path)
    best = [float(v) for v in re.findall(r"Best\s*:\s*Accuracy=([\d.]+)", text)]
    # "**Test** Prec@1 85.105" -- take the summary lines, not the per-batch ones
    test = [float(v) for v in re.findall(r"\*\*Test\*\*\s*Prec@1\s+([\d.]+)", text)]
    b, _ = best_last(best)      # b = max
    _, t = best_last(test)      # t = the last epoch's test accuracy, not the max
    return {
        "best": round(b, 2) if b is not None else None,
        "last": round(t, 2) if t is not None else None,
        "criterion": "best-on-val / last-epoch test (%s)" % label,
        "epochs": len(best),
        "secs": "",
    }


def parse_eva(path):
    rows = []
    text = _read(path)
    if not text.strip():
        return rows
    reader = csv.DictReader(text.strip().splitlines())
    for r in reader:
        try:
            acc = float(r["accuracy"])
        except (KeyError, ValueError, TypeError):
            continue
        rows.append({
            "method": r["method"],
            "best": round(acc * 100, 2),
            "last": round(acc * 100, 2),
            "criterion": "single run, see method source for checkpoint rule",
            # summary.csv has no epoch column, so leave it blank rather than
            # smuggling elapsed_sec in here (that used to make the CSV lie)
            "epochs": "",
            "secs": r.get("elapsed_sec", ""),
        })
    return rows


VARIANTS = {
    "10pct": {
        "fraction": 0.1,
        "cops_dir": "COPS/logs",
        "gm_log": "GM_matching/logs/run.log",
        "dual_sub_log": "DUAL/logs/03_pruned_train.log",
        "eva_summary": "evaluation/summary.csv",
        "out": "evaluation/baselines_summary.csv",
    },
    "1pct": {
        "fraction": 0.01,
        "cops_dir": "COPS/logs_1pct",
        "gm_log": "GM_matching/logs_1pct/run.log",
        "dual_sub_log": "DUAL/logs/03_pruned_train_1pct.log",
        "eva_summary": "evaluation/fraction0.01/summary.csv",
        "out": "evaluation/baselines_summary_1pct.csv",
    },
    # 1% coreset, but every method spends the same ~1100 gradient updates
    "aligned1pct": {
        "fraction": 0.01,
        "cops_dir": "COPS/logs_aligned",
        "cops_files": ("uniform.log", "cops_wy.log", "cops_woy.log"),
        "gm_log": "GM_matching/logs_aligned/run.log",
        "dual_sub_log": "DUAL/logs/03_pruned_train_aligned.log",
        "eva_summary": "evaluation/aligned1pct/summary.csv",
        "out": "evaluation/baselines_summary_aligned1pct.csv",
    },
}

DEFAULT_COPS_FILES = ("03_uniform.log", "04_cops_wy.log", "05_cops_woy.log")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/root/autodl-tmp/EVA/output")
    ap.add_argument("--variant", default="10pct", choices=sorted(VARIANTS))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = args.root
    cfg = VARIANTS[args.variant]
    frac = cfg["fraction"]
    out = args.out or os.path.join(root, cfg["out"])

    rows = []

    def add(method, coreset_fraction, kind, stats):
        if stats.get("best") is None:
            return
        rows.append({
            "method": method,
            "coreset_fraction": coreset_fraction,
            "best_acc_pct": stats["best"],
            "last_acc_pct": stats["last"],
            "selection_criterion": stats["criterion"],
            "kind": kind,
            "epochs_logged": stats["epochs"],
            "train_elapsed_sec": stats.get("secs", ""),
        })

    cops_dir = os.path.join(root, cfg["cops_dir"])
    f_uniform, f_wy, f_woy = cfg.get("cops_files", DEFAULT_COPS_FILES)
    add("COPS-uniform", frac, "coreset",
        parse_cops(os.path.join(cops_dir, f_uniform)))
    add("COPS-with-y", frac, "coreset",
        parse_cops(os.path.join(cops_dir, f_wy)))
    add("COPS-without-y", frac, "coreset",
        parse_cops(os.path.join(cops_dir, f_woy)))
    add("GM-Matching", frac, "coreset",
        parse_gm(os.path.join(root, cfg["gm_log"])))
    # the full-data reference is independent of the coreset fraction
    add("DUAL (full data)", 1.0, "reference",
        parse_dual(os.path.join(root, "DUAL", "logs", "01_dynamics.log"), "full data"))
    add("DUAL (%s pruned)" % args.variant.replace("pct", "% "), frac, "coreset",
        parse_dual(os.path.join(root, cfg["dual_sub_log"]), "%s subset" % args.variant))

    for r in parse_eva(os.path.join(root, cfg["eva_summary"])):
        rows.append({
            "method": "EVA-" + r["method"],
            "coreset_fraction": 1.0 if r["method"] == "full" else frac,
            "best_acc_pct": r["best"],
            "last_acc_pct": r["last"],
            "selection_criterion": r["criterion"],
            "kind": "reference" if r["method"] == "full" else "coreset",
            "epochs_logged": r["epochs"],
            "train_elapsed_sec": r["secs"],
        })

    rows.sort(key=lambda r: (-(r["best_acc_pct"] or 0)))

    fields = ["method", "coreset_fraction", "kind", "best_acc_pct",
              "last_acc_pct", "selection_criterion", "epochs_logged",
              "train_elapsed_sec"]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"adult [{args.variant}], test set = 16281 rows, majority-class accuracy = 76.38%")
    print(f"{'method':<22}{'frac':>6}{'best':>8}{'last':>8}  criterion")
    print("-" * 92)
    for r in rows:
        print(f"{r['method']:<22}{r['coreset_fraction']:>6}{r['best_acc_pct']:>8}"
              f"{r['last_acc_pct'] if r['last_acc_pct'] is not None else '':>8}"
              f"  {r['selection_criterion']}")
    print("-" * 92)
    print(f"written -> {out}")


if __name__ == "__main__":
    main()
