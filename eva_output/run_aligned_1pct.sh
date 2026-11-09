#!/bin/bash
# run_aligned_1pct.sh —— 1% 核心集，但把**总梯度步数**对齐到 1100 步
#
# 为什么需要这个：1% 时各方法每轮步数差异巨大，按 epoch 比较是伪对齐。
#   核心集 260 条，batch 32（EVA random 用）-> 9 步/epoch
#                     batch 128（kmeans/GM/DUAL）-> 3 步/epoch
#                     batch 256（COPS 原生）     -> 2 步/epoch
# 于是 epoch 数相同不代表优化量相同。1100 步取自 10% 参考跑的真实预算
# （random: 2604/256=11 步 × 100 epoch ≈ 1100）。
#
# 各方法 epoch = ceil(1100 / 步数每轮)：
#   COPS    2 -> 550
#   GM      3 -> 367
#   DUAL    3 -> 367
#   EVA     step_budget=1100 自动推导（random 123 / kmeans 367）
#
# EVA 的 lr 用 0.01（对齐后 batch 只有 32，0.1 会发散；已单独验证 0.01 可用）。
set -u
export OMP_NUM_THREADS=1
export WANDB_MODE=offline

PY=/root/miniconda3/bin/python
BASE=/root/autodl-tmp/EVA/baseline
OUT=/root/eva_output
COPS=$OUT/COPS
DUAL=$OUT/DUAL
GM=$OUT/GM_matching
EVA_ROOT=/root/chaichengliang/VLDB2027/EVA/EVA

STEPS=1100
COPS_EP=550
GM_EP=367
DUAL_EP=367
SAMPLE_PER_CLASS=130

log() { echo "[$(date '+%F %T')] $*"; }

C1=$($PY -c "import json;print(json.load(open('$COPS/logs/adult_metric_stats.json'))['constant_1'])" 2>/dev/null)
[ -z "${C1:-}" ] && C1=1e9

mkdir -p "$COPS/logs_aligned" "$GM/logs_aligned" "$OUT/evaluation/aligned1pct"
log "==== 1% 步数对齐实验：目标 $STEPS 步  (constant_1=$C1) ===="

########################################################################
cd "$BASE/COPS-main" || exit 1
for spec in "uniform:random:base:retraining_withy:wu_al" \
            "cops_wy:oracle_sampling_cut:reweight_clip:retraining_withy:wy_al" \
            "cops_woy:oracle_sampling_cut:reweight_clip:retraining_woy:wo_al"; do
    IFS=':' read -r name mode loss script wdir <<< "$spec"
    log "[COPS] $name  epochs=$COPS_EP"
    $PY -u $script.py --seed 1 --lr 1e-3 --dataset adult --model tabnet \
        --test_mode $mode --constant_1 "$C1" --constant_2 10 \
        --sample_number $SAMPLE_PER_CLASS --loss_type $loss \
        --epochs $COPS_EP \
        --un_folder "$COPS/adult_npy" --weight_folder "$COPS/$wdir" \
        > "$COPS/logs_aligned/$name.log" 2>&1
    log "  $name exit=$?"
done

########################################################################
log "[GM] epochs=$GM_EP"
cd "$BASE/GM-Matching-main" || exit 1
$PY -u run_classify_coreset.py \
    --dataset adult --coreset_size_dataset 0.01 --coreset_method gm_matching \
    --num_repetitions 1 --epochs $GM_EP --batch_size 128 --num_workers 0 \
    --embeds_dir "$GM/embeds" \
    --log_filename "$GM/logs_aligned/gm_matching.csv" \
    > "$GM/logs_aligned/run.log" 2>&1
log "  GM exit=$?"

########################################################################
log "[DUAL] subset epochs=$DUAL_EP"
cd "$BASE/dual-pruning-main/exp_cifar" || exit 1
MASK=$DUAL/checkpoint/adult/42/generated_mask
$PY -u train_subset.py --data_path dummy --dataset adult --arch tabnet \
    --epochs $DUAL_EP --learning_rate 0.1 --batch-size 128 --manualSeed 42 \
    --subset_rate 0.01 \
    --target-probs-path "$MASK/target_probs.npy" \
    --score-path "$MASK/dual_mask_T30.npy" \
    --mask-path "$MASK/dual_mask_T30.npy" \
    --c_d 4 --sample beta --save_path "$DUAL/pruned_aligned" --workers 0 \
    > "$DUAL/logs/03_pruned_train_aligned.log" 2>&1
log "  DUAL exit=$?"

########################################################################
log "[EVA] step_budget=$STEPS  lr=0.01"
cd "$EVA_ROOT" || exit 1
STEP_BUDGET=$STEPS SELECTION_LR=0.01 FRACTION=0.01 SEED=0 \
    OUT_ROOT="$OUT/evaluation/aligned1pct" METHODS=random,kmeans \
    bash evaluation/run_evaluation.sh > "$OUT/evaluation/aligned1pct/run.log" 2>&1
log "  EVA exit=$?"

log "==================== ALIGNED DONE ===================="
