#!/usr/bin/env bash
# One command after the GPU arrives. Detached, locked, resumable; never falls back to CPU.
set -euo pipefail
MATRIX_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MATRIX_PYTHON="${MATRIX_PYTHON:-/root/miniconda3/bin/python}"
MATRIX_OUTPUT="${MATRIX_OUTPUT:-$MATRIX_DIR/production_runs}"
export CUDA_VISIBLE_DEVICES="${MATRIX_CUDA_DEVICE:-${CUDA_VISIBLE_DEVICES:-0}}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 WANDB_MODE=offline
export PYTHONPATH=/root/autodl-tmp/eva_output
mkdir -p "$MATRIX_OUTPUT/launcher_logs"

# Take the lock before checking/launching; the detached child inherits FD 9.
if [[ "${MATRIX_LOCK_HELD:-0}" != 1 ]]; then
    exec 9>"$MATRIX_OUTPUT/.run.lock"
    if ! flock -n 9; then
        printf 'Another matrix run already holds %s/.run.lock\n' "$MATRIX_OUTPUT" >&2
        exit 75
    fi
    export MATRIX_LOCK_HELD=1
fi

if [[ "${1:-}" != "--foreground" ]]; then
    "$MATRIX_PYTHON" "$MATRIX_DIR/runtime_check.py" --require-cuda --out "$MATRIX_OUTPUT/readiness.json"
    MATRIX_TAG="$(date -u +%Y%m%dT%H%M%SZ)"
    MATRIX_LOG="$MATRIX_OUTPUT/launcher_logs/$MATRIX_TAG.log"
    nohup setsid bash "$MATRIX_DIR/run_all.sh" --foreground >"$MATRIX_LOG" 2>&1 </dev/null &
    MATRIX_PID=$!
    printf '%s\n' "$MATRIX_PID" >"$MATRIX_OUTPUT/launcher.pid"
    printf '%s\n' "$MATRIX_LOG" >"$MATRIX_OUTPUT/latest_log.txt"
    printf 'Started PID %s\nLog: %s\n' "$MATRIX_PID" "$MATRIX_LOG"
    exit 0
fi

"$MATRIX_PYTHON" "$MATRIX_DIR/runtime_check.py" --require-cuda --out "$MATRIX_OUTPUT/readiness.json"
exec "$MATRIX_PYTHON" -u "$MATRIX_DIR/pipeline.py" --mode run --out "$MATRIX_OUTPUT" --timeout 86400
