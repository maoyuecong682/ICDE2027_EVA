#!/usr/bin/env bash
# Run EVA's own reference methods (random / full / kmeans) on adult and collect
# the results into <OUT_ROOT>/summary.{json,csv}.
#
# Mirrors the protocol used by all.sh for adult (TabNet, 100 epochs, fraction 0.1)
# so the numbers are directly comparable with the COPS / GM-Matching / DUAL runs
# in ../COPS, ../GM_matching and ../DUAL.
#
# Usage:
#   bash evaluation/run_evaluation.sh                 # default: lr 0.1 (all.sh)
#   SELECTION_LR=0.01 bash evaluation/run_evaluation.sh
set -uo pipefail

EVA_ROOT="${EVA_ROOT:-/root/chaichengliang/VLDB2027/EVA/EVA}"
PY="${PY:-/root/miniconda3/bin/python}"
DATA_PATH="${DATA_PATH:-/root/autodl-tmp/EVA/data/adult}"
OUT_ROOT="${OUT_ROOT:-/root/autodl-tmp/EVA/output/evaluation}"

EPOCHS="${EPOCHS:-100}"
PATIENCE="${PATIENCE:-100}"
FRACTION="${FRACTION:-0.1}"
SELECTION_LR="${SELECTION_LR:-0.1}"
BALANCE_TARGET="${BALANCE_TARGET:-1}"
METHODS="${METHODS:-full,random,kmeans}"
SEED="${SEED:-}"
STEP_BUDGET="${STEP_BUDGET:-0}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

mkdir -p "$OUT_ROOT/logs" "$OUT_ROOT/best_models"
LOG="$OUT_ROOT/run_evaluation.log"
: > "$LOG"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "EVA reference methods on adult"
log "  root      = $EVA_ROOT"
log "  out       = $OUT_ROOT"
log "  methods   = $METHODS"
log "  epochs    = $EPOCHS  patience = $PATIENCE"
log "  fraction  = $FRACTION  lr = $SELECTION_LR  balance_target = $BALANCE_TARGET"

cd "$EVA_ROOT" || exit 1

# clear previous method logs so every run starts from a clean slate
rm -f "$OUT_ROOT/logs/adult/random/$FRACTION/1.log" 2>/dev/null
rm -f "$OUT_ROOT/logs/adult/TabNet/$FRACTION/"*.log 2>/dev/null

"$PY" -u evaluation/eval_runner.py \
    --data_path "$DATA_PATH" \
    --out_root "$OUT_ROOT" \
    --dataset adult \
    --model TabNet \
    --methods "$METHODS" \
    --fraction "$FRACTION" \
    --balance_target "$BALANCE_TARGET" \
    --epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --lr "$SELECTION_LR" \
    --workers 0 \
    ${SEED:+--seed "$SEED"} \
    --step_budget "$STEP_BUDGET" \
    >> "$LOG" 2>&1
RC=$?

log "eval_runner exit=$RC"
if [ -f "$OUT_ROOT/summary.csv" ]; then
    log "---- summary.csv ----"
    cat "$OUT_ROOT/summary.csv" | tee -a "$LOG"
fi
log "EVALUATION DONE (rc=$RC)"
exit $RC
