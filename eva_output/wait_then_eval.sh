#!/bin/bash
# 等待指定 pid 退出后，再串行跑一组评估（eval.sh 自带 flock）
# 用法: wait_then_eval.sh <pid> "ds1 ds2 ..."
PID=$1; shift
while kill -0 $PID 2>/dev/null; do sleep 15; done
echo "[$(date +%F_%T)] 前置进程 $PID 已退出，开始评估 $*" >> /root/eva_output/logs/eval_queue.log
bash /root/eva_output/eval.sh "$*"
echo "[$(date +%F_%T)] 完成 $*" >> /root/eva_output/logs/eval_queue.log
