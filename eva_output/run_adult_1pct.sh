#!/bin/bash
# run_adult_1pct.sh —— 与 run_adult_all.sh 同协议，但核心集比例改为 1%
#
# 只重跑「依赖核心集比例」的步骤，复用 10% 实验里已经算好的产物：
#   - COPS  复用 01_uncertainty(10 模型 logits) 和 constant_1 标定，只重跑三组重训
#   - GM    重新选核集（比例变了）
#   - DUAL  **复用** 训练动力学和 dual_mask_T30（mask 与比例无关），只重跑 7c 子集训练
#   - EVA   random/full/kmeans 用 fraction=0.01 重跑
#
# 产物写到 *_1pct 路径，不会覆盖 10% 的结果。
set -u
export OMP_NUM_THREADS=1
export WANDB_MODE=offline

PY=/root/miniconda3/bin/python
BASE=/root/autodl-tmp/EVA/baseline
OUT=/root/autodl-tmp/EVA/output
COPS=$OUT/COPS
DUAL=$OUT/DUAL
GM=$OUT/GM_matching
EVA_ROOT=/root/chaichengliang/VLDB2027/EVA/EVA

FRACTION=0.01
N_TRAIN=26049
SAMPLE_PER_CLASS=$(( N_TRAIN * 1 / 100 / 2 ))     # 1% / 2 类 = 130
EPOCHS_COPS_RETRAIN=50
EPOCHS_GM=200
EPOCHS_DUAL=200

mkdir -p "$COPS/logs_1pct" "$GM/logs_1pct" "$OUT/evaluation/fraction0.01"

log() { echo "[$(date '+%F %T')] $*"; }

C1=$($PY -c "import json;print(json.load(open('$COPS/logs/adult_metric_stats.json'))['constant_1'])" 2>/dev/null)
[ -z "${C1:-}" ] && { C1=1e9; log "读不到 constant_1，退回 1e9"; }
log "==== 1% 实验开始：sample/class=$SAMPLE_PER_CLASS  constant_1=$C1 ===="

########################################################################
cd "$BASE/COPS-main" || exit 1

log "[1/4] COPS uniform (1%)"
$PY -u retraining_withy.py --seed 1 --lr 1e-3 --dataset adult --model tabnet \
    --test_mode random --sample_number $SAMPLE_PER_CLASS --loss_type base \
    --epochs $EPOCHS_COPS_RETRAIN \
    --un_folder "$COPS/adult_npy" --weight_folder "$COPS/weights_uniform" \
    > "$COPS/logs_1pct/03_uniform.log" 2>&1
log "  uniform exit=$?"

log "[2/4] COPS with y (1%)"
$PY -u retraining_withy.py --seed 1 --lr 1e-3 --dataset adult --model tabnet \
    --test_mode oracle_sampling_cut --constant_1 "$C1" --constant_2 10 \
    --sample_number $SAMPLE_PER_CLASS --loss_type reweight_clip \
    --epochs $EPOCHS_COPS_RETRAIN \
    --un_folder "$COPS/adult_npy" --weight_folder "$COPS/weights_cops_wy" \
    > "$COPS/logs_1pct/04_cops_wy.log" 2>&1
log "  cops_wy exit=$?"

log "[3/4] COPS without y (1%)"
$PY -u retraining_woy.py --seed 1 --lr 1e-3 --dataset adult --model tabnet \
    --test_mode oracle_sampling_cut --constant_1 "$C1" --constant_2 10 \
    --sample_number $SAMPLE_PER_CLASS --loss_type reweight_clip \
    --epochs $EPOCHS_COPS_RETRAIN \
    --un_folder "$COPS/adult_npy" --weight_folder "$COPS/weights_cops_woy" \
    > "$COPS/logs_1pct/05_cops_woy.log" 2>&1
log "  cops_woy exit=$?"

########################################################################
log "[4/4a] GM-Matching (1%, ${EPOCHS_GM} epoch x 3 reps)"
cd "$BASE/GM-Matching-main" || exit 1
$PY -u run_classify_coreset.py \
    --dataset adult --coreset_size_dataset $FRACTION --coreset_method gm_matching \
    --num_repetitions 3 --epochs $EPOCHS_GM --batch_size 128 --num_workers 0 \
    --embeds_dir "$GM/embeds" \
    --log_filename "$GM/logs_1pct/gm_matching.csv" \
    > "$GM/logs_1pct/run.log" 2>&1
log "  GM exit=$?"

########################################################################
log "[4/4b] DUAL 剪枝子集 (1%, 复用已有 mask)"
cd "$BASE/dual-pruning-main/exp_cifar" || exit 1
CKPT=$DUAL/checkpoint
MASK=$CKPT/adult/42/generated_mask
$PY -u train_subset.py --data_path dummy --dataset adult --arch tabnet \
    --epochs $EPOCHS_DUAL --learning_rate 0.1 --batch-size 128 --manualSeed 42 \
    --subset_rate $FRACTION \
    --target-probs-path "$MASK/target_probs.npy" \
    --score-path "$MASK/dual_mask_T30.npy" \
    --mask-path "$MASK/dual_mask_T30.npy" \
    --c_d 4 --sample beta --save_path "$DUAL/pruned_1pct" --workers 0 \
    > "$DUAL/logs/03_pruned_train_1pct.log" 2>&1
log "  DUAL subset exit=$?"

########################################################################
log "[4/4c] EVA random/full/kmeans (fraction=0.01)"
cd "$EVA_ROOT" || exit 1
FRACTION=$FRACTION OUT_ROOT="$OUT/evaluation/fraction0.01" \
    bash evaluation/run_evaluation.sh > "$OUT/evaluation/fraction0.01/run.log" 2>&1
log "  EVA exit=$?"

log "==================== 1PCT DONE ===================="
