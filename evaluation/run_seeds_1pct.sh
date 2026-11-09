#!/bin/bash
# run_seeds_1pct.sh —— 1% 比例下的重复实验，用于估计单次运行的波动
#
# 目的：10% 时各方法挤在 81-85，1% 时 COPS 从 84.42 掉到 76.46、EVA-random 恰好
# 等于多数类 76.38（塌缩）。单个种子无法判断这是方法问题还是噪声，这里补重复。
#
#   EVA  random/kmeans  1%  seeds 0,1,2   注意：两者选核集时各自写死了种子
#                                          (random.py 42 / KMeans random_state=0)，
#                                          所以这三次只反映**训练**波动。
#   COPS uniform/wy/woy 1%  seeds 1,2,3   这里种子会影响采样，波动包含选点。
#
set -u
export OMP_NUM_THREADS=1
export WANDB_MODE=offline

PY=/root/miniconda3/bin/python
BASE=/root/autodl-tmp/EVA/baseline
OUT=/root/autodl-tmp/EVA/output
COPS=$OUT/COPS
EVA_ROOT=/root/chaichengliang/VLDB2027/EVA/EVA

SAMPLE_PER_CLASS=130      # 1% / 2 类
EPOCHS_COPS_RETRAIN=50

log() { echo "[$(date '+%F %T')] $*"; }

C1=$($PY -c "import json;print(json.load(open('$COPS/logs/adult_metric_stats.json'))['constant_1'])" 2>/dev/null)
[ -z "${C1:-}" ] && C1=1e9

log "==== 1% 重复实验开始 (constant_1=$C1) ===="

########################################################################
# EVA random / kmeans：训练波动
########################################################################
cd "$EVA_ROOT" || exit 1
for s in 0 1 2; do
    log "[EVA] seed=$s"
    SEED=$s FRACTION=0.01 OUT_ROOT="$OUT/evaluation/seeds/f1pct_s$s" \
        METHODS=random,kmeans \
        bash evaluation/run_evaluation.sh > /dev/null 2>&1
    log "  exit=$?"
done

########################################################################
# COPS：选点 + 训练波动
########################################################################
cd "$BASE/COPS-main" || exit 1
for s in 1 2 3; do
    d="$COPS/logs_1pct_s$s"
    mkdir -p "$d"

    log "[COPS] seed=$s uniform"
    $PY -u retraining_withy.py --seed $s --lr 1e-3 --dataset adult --model tabnet \
        --test_mode random --sample_number $SAMPLE_PER_CLASS --loss_type base \
        --epochs $EPOCHS_COPS_RETRAIN \
        --un_folder "$COPS/adult_npy" --weight_folder "$COPS/wu_s$s" \
        > "$d/03_uniform.log" 2>&1

    log "[COPS] seed=$s with-y"
    $PY -u retraining_withy.py --seed $s --lr 1e-3 --dataset adult --model tabnet \
        --test_mode oracle_sampling_cut --constant_1 "$C1" --constant_2 10 \
        --sample_number $SAMPLE_PER_CLASS --loss_type reweight_clip \
        --epochs $EPOCHS_COPS_RETRAIN \
        --un_folder "$COPS/adult_npy" --weight_folder "$COPS/wy_s$s" \
        > "$d/04_cops_wy.log" 2>&1

    log "[COPS] seed=$s without-y"
    $PY -u retraining_woy.py --seed $s --lr 1e-3 --dataset adult --model tabnet \
        --test_mode oracle_sampling_cut --constant_1 "$C1" --constant_2 10 \
        --sample_number $SAMPLE_PER_CLASS --loss_type reweight_clip \
        --epochs $EPOCHS_COPS_RETRAIN \
        --un_folder "$COPS/adult_npy" --weight_folder "$COPS/wo_s$s" \
        > "$d/05_cops_woy.log" 2>&1
    log "  seed=$s done"
done

log "==================== SEEDS DONE ===================="
