#!/usr/bin/env bash
# 脱离 SSH 通道启动长任务。
#
# 为什么需要它：之前把 python 直接接在 ssh 命令后面（哪怕用 run_in_background），
# 一旦 SSH 通道关闭，远端 bash 收到 SIGHUP 会把 python 一起带走 —— 而
# `python ... | tail` 的退出码是 tail 的 0，于是表现为「退出码 0、日志正常、产物为空」，
# 极难判断。这里用 setsid + nohup 彻底断开进程组，并把真实退出码写进日志尾部。
#
# 用法：
#   bash launch.sh <日志路径> <工作目录> "<命令>"
set -u
LOG="${1:?需要日志路径}"
WD="${2:?需要工作目录}"
CMD="${3:?需要命令}"

mkdir -p "$(dirname "$LOG")"
cd "$WD" || { echo "工作目录不存在: $WD"; exit 1; }

setsid nohup bash -c "$CMD; echo \"[PY_EXIT=\$?] finished at \$(date +%F_%T)\"" \
    > "$LOG" 2>&1 < /dev/null &

echo "LAUNCHED pid=$! log=$LOG"
