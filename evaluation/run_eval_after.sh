#!/usr/bin/env bash
# Chain: wait for the three-baseline run to finish, then run the EVA reference
# methods (random / full / kmeans) on adult.
#
# Launched detached so it survives the SSH session.  Progress goes to
# /root/autodl-tmp/EVA/output/eval_chain.log
set -uo pipefail

MAIN_LOG="/root/autodl-tmp/EVA/output/run_all.log"
CHAIN_LOG="/root/autodl-tmp/EVA/output/eval_chain.log"
EVA_ROOT="/root/chaichengliang/VLDB2027/EVA/EVA"

say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$CHAIN_LOG"; }

say "chain started; waiting for 'ALL DONE' in $MAIN_LOG"

# up to 8 hours, polled every 30 s
for _ in $(seq 1 960); do
    if grep -q 'ALL DONE' "$MAIN_LOG" 2>/dev/null; then
        say "main run reported ALL DONE"
        break
    fi
    sleep 30
done

if ! grep -q 'ALL DONE' "$MAIN_LOG" 2>/dev/null; then
    say "TIMEOUT waiting for the main run -- evaluation NOT started"
    exit 1
fi

# make sure nothing from the baseline run is still holding the CPU
sleep 20

say "starting evaluation of random / full / kmeans"
cd "$EVA_ROOT" || { say "cannot cd $EVA_ROOT"; exit 1; }
bash evaluation/run_evaluation.sh >> "$CHAIN_LOG" 2>&1
rc=$?
say "evaluation finished rc=$rc"
say "==================== CHAIN DONE ===================="
exit $rc
