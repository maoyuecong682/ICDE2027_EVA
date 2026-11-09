#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Aggregate the 1% repeated runs into mean +- std.

EVA   output/evaluation/seeds/f1pct_s{0,1,2}/summary.csv
COPS  output/COPS/logs_1pct_s{1,2,3}/{03_uniform,04_cops_wy,05_cops_woy}.log
"""
import csv
import glob
import os
import re
import statistics
import sys

ROOT = "/root/autodl-tmp/EVA/output"


def read(path):
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()


def eva_seeds():
    out = {}
    for path in sorted(glob.glob(os.path.join(ROOT, "evaluation", "seeds", "f1pct_s*", "summary.csv"))):
        seed = re.search(r"f1pct_s(\d+)", path).group(1)
        text = read(path)
        if not text.strip():
            continue
        for r in csv.DictReader(text.strip().splitlines()):
            try:
                acc = float(r["accuracy"]) * 100
            except (KeyError, ValueError, TypeError):
                continue
            out.setdefault("EVA-" + r["method"], []).append((seed, round(acc, 2)))
    return out


def cops_seeds():
    out = {}
    label = {"03_uniform": "COPS-uniform", "04_cops_wy": "COPS-with-y",
             "05_cops_woy": "COPS-without-y"}
    for d in sorted(glob.glob(os.path.join(ROOT, "COPS", "logs_1pct_s*"))):
        seed = re.search(r"logs_1pct_s(\d+)", d).group(1)
        for key, name in label.items():
            text = read(os.path.join(d, key + ".log"))
            vals = [float(v) for v in re.findall(r"Real Test Loss Acc\s*:\s*([\d.]+)", text)]
            if vals:
                out.setdefault(name, []).append((seed, round(vals[-1], 2)))
    return out


def main():
    groups = {}
    groups.update(cops_seeds())
    groups.update(eva_seeds())

    rows = []
    for name, pairs in groups.items():
        vals = [v for _, v in pairs]
        if not vals:
            continue
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        rows.append({
            "method": name,
            "n": len(vals),
            "mean": round(mean, 2),
            "std": round(std, 2),
            "min": round(min(vals), 2),
            "max": round(max(vals), 2),
            "runs": " ".join(f"{s}:{v}" for s, v in pairs),
        })
    rows.sort(key=lambda r: -r["mean"])

    out_csv = os.path.join(ROOT, "evaluation", "seeds_1pct_stats.csv")
    with open(out_csv, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["method", "n", "mean", "std", "min", "max", "runs"])
        w.writeheader()
        w.writerows(rows)

    print("adult, 1% coreset, repeated runs (test = 16281, majority = 76.38%)")
    print("COPS rows use the LAST-epoch test accuracy -- best-on-test would be optimistic")
    print("and is the single most likely reason COPS looked strong at 10%.")
    print(f"{'method':<20}{'n':>3}{'mean':>8}{'std':>7}{'min':>8}{'max':>8}  runs")
    print("-" * 96)
    for r in rows:
        print(f"{r['method']:<20}{r['n']:>3}{r['mean']:>8.2f}{r['std']:>7.2f}"
              f"{r['min']:>8.2f}{r['max']:>8.2f}  {r['runs']}")
    print("-" * 96)
    print(f"written -> {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
