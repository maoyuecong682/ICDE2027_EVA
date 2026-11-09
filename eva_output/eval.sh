#!/bin/bash
# 通用评估启动器 v2：每个（baseline, 数据集）一个独立 python 进程。
#
# 为什么按数据集拆进程：random/full 例程一旦在某个数据集上触发 CUDA
# device-side assert，同进程内所有后续 CUDA 调用都会失败，连下一个数据集的
# EVA.load() 都会崩掉（bike 事故：bike 三列全空 + House 完全没跑到）。
# 拆进程后单个数据集的失败只影响它自己，--resume 保证其余行不受影响。
#
#   eval.sh "news bike House"          # 三个 baseline
#   eval.sh "news" "COPS"              # 只跑一个 baseline
#   eval.sh "news" "" "full"           # 只用 full 例程（第三个参数可覆盖例程）
set -u
DSS="${1:?用法: eval.sh \"ds1 ds2 ...\" [\"B1 B2\"] [\"routine ...\"]}"
BASELINES="${2:-COPS GM_matching DUAL}"
ROUTINES="${3:-random full kmeans}"
TAG=$(echo "$DSS" | tr ' ' '_')
mkdir -p /root/eva_output/logs

# 串行锁：三个 baseline 的 *_result.csv 是「读整表 → 改一行 → 整表覆写」，
# 两路评估同时跑会互相覆盖（--resume 也救不了，因为读到的是旧内容）。
# 用 flock 把整个脚本串起来，后到的会等前一个跑完。
exec 9>/root/eva_output/.eval.lock
flock 9

LOG="/root/eva_output/logs/eval_${TAG}.log"
: > "$LOG"
cd /root/autodl-tmp/EVA/evaluation || exit 1
for B in $BASELINES; do
  for ds in $DSS; do
    echo "[$(date +%F_%T)] >>>>>> $B / $ds" >> "$LOG"
    EVA_ROOT=/root/chaichengliang/VLDB2027/EVA/EVA PYTHONPATH=/root/eva_output \
      /root/miniconda3/bin/python -u eval_baselines.py \
        --baseline "$B" --datasets "$ds" --routines $ROUTINES \
        --epochs 60 --fraction 0.1 --resume >> "$LOG" 2>&1
    echo "[$(date +%F_%T)] <<<<<< $B / $ds rc=$?" >> "$LOG"
  done
done
echo "[$(date +%F_%T)] [ALL DONE] $DSS" >> "$LOG"
