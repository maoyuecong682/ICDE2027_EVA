#!/bin/bash
# run_preprocess_all.sh -- 逐个执行 mydatasets 预处理（排除 news / imdb0 / imdb 的 BERT 路径）。
# 每个数据集独立进程 + 独立超时；单点失败不影响其余。
set -u
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1

PY=/root/miniconda3/bin/python
REPO=/root/chaichengliang/VLDB2027/EVA/EVA
DRV=$REPO/evaluation/run_preprocess.py
OUT=/root/eva_output/preprocess
LOG=${LOG:-$OUT/run.log}

mkdir -p "$OUT"
: > "$LOG"

# 先跑已就绪的，把需要修复/可能失败的放后面
# 可用 ORDER="HR House" 只重跑指定子集
ORDER=${ORDER:-"cifar10 cifar100 mnist qmnist fashionmnist HR House bike tinyimagenet number svhn covtype IMDBLarge mnist2mnistm"}

for d in $ORDER; do
    echo "===== $d =====" | tee -a "$LOG"
    timeout 3600 $PY -u "$DRV" --only "$d" 2>&1 | tee -a "$LOG"
    echo | tee -a "$LOG"
done

echo "PREPROCESS DONE" | tee -a "$LOG"
