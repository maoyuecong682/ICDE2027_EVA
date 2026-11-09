#!/bin/bash
# 训练完成后自动收尾：补评 12 个缺失数据集 + 聚合出 54/54 的 evaluation3090.csv。
# 由 launch.sh 脱离 SSH 启动；训练进程（pid 在下方）结束后自动继续。
set -u
EVA_TOOLS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOG=/root/autodl-tmp/eva_output/logs/finish_pipeline.log
TRAIN_PID=299423
DONE=/root/autodl-tmp/eva_output/FINAL_DONE.txt
MISSING="covtype SVHN tinyimagenet mnist2mnistm imagenet"
: > "$LOG"
log(){ echo "[$(date +%F_%T)] $*" | tee -a "$LOG"; }

log "finish_pipeline 启动，等待训练进程 pid=$TRAIN_PID 结束 ..."

# 轮询等待训练结束（最多 ~20 小时），避免与仍在写的权重/索引抢跑。
for i in $(seq 1 1200); do
  if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
    log "训练进程已退出（轮询第 $i 次，约 $((i)) 分钟）"
    break
  fi
  sleep 60
done
if kill -0 "$TRAIN_PID" 2>/dev/null; then
  log "WARNING: 超时训练进程仍在跑，仍对已完成项补评"
fi

# 补评 12 个缺失（eval.sh 自带 flock 串行锁；--resume 保留已有 42 行，只更新这 12 个）
log "开始补评缺失数据集：$MISSING"
bash /root/autodl-tmp/eva_output/eval.sh "$MISSING" >> "$LOG" 2>&1

# 聚合最终表
log "聚合 evaluation3090.csv ..."
cd /root/autodl-tmp/EVA/evaluation
/root/miniconda3/bin/python "$EVA_TOOLS_DIR/make_eval3090.py" 2>&1 | tee -a "$LOG"

PROG=$(grep -oE '进度\] [0-9]+/18' "$LOG" | tail -1)
log "===== 收尾完成：$PROG ====="
echo "$PROG" > "$DONE"
echo "完成时间 $(date +%F_%T)" >> "$DONE"
