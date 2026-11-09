#!/bin/bash
# run_adult_all.sh —— 三个 baseline 在 adult 上的全量实验驱动脚本
#
# 串行执行（容器只有 0.5 核 CPU 配额，并行只会互相抢占）：
#   COPS  : 不确定性生成(10 模型 x 30 epoch) -> metric 标定 -> 三组重训
#   GM    : geometric-median matching 选核集 -> 200 epoch 训练 x 3 次重复
#   DUAL  : 动力学 200 epoch -> 重要性评估 -> 剪枝子集 200 epoch
#
# 所有产物落在 /root/autodl-tmp/EVA/output/{COPS,DUAL,GM_matching}

set -u
export OMP_NUM_THREADS=1
export WANDB_MODE=offline

PY=/root/miniconda3/bin/python
BASE=/root/autodl-tmp/EVA/baseline
OUT=/root/autodl-tmp/EVA/output
COPS=$OUT/COPS
DUAL=$OUT/DUAL
GM=$OUT/GM_matching

FRACTION=0.1
N_TRAIN=26049
SAMPLE_PER_CLASS=$(( N_TRAIN * 1 / 10 / 2 ))     # 0.1 比例 / 2 类 = 1302
EPOCHS_UNC=30
EPOCHS_COPS_RETRAIN=50
EPOCHS_GM=200
EPOCHS_DUAL=200

mkdir -p "$COPS/logs" "$COPS/weights" "$COPS/adult_npy" \
         "$DUAL/logs" "$GM/logs" "$GM/embeds"

log() { echo "[$(date '+%F %T')] $*"; }

########################################################################
log "==== [1/7] COPS 不确定性生成 (10 模型 x ${EPOCHS_UNC} epoch) ===="
cd "$BASE/COPS-main" || exit 1
$PY -u uncertainty_generation.py \
    --dataset adult --lr 1e-3 --model tabnet --model_num 10 --num_classes 2 \
    --epochs $EPOCHS_UNC \
    --weight_folder "$COPS/weights" \
    --npy_folder "$COPS/adult_npy" \
    > "$COPS/logs/01_uncertainty.log" 2>&1
log "COPS stage1 exit=$?"

########################################################################
log "==== [2/7] 标定 COPS constant_1 ===="
$PY -u "$OUT/calibrate_adult_metric.py" \
    --npy_folder "$COPS/adult_npy" \
    --out "$COPS/logs/adult_metric_stats.json" \
    > "$COPS/logs/02_calibrate.log" 2>&1
C1=$($PY -c "import json;print(json.load(open('$COPS/logs/adult_metric_stats.json'))['constant_1'])" 2>/dev/null)
if [ -z "${C1:-}" ]; then C1=1e9; log "标定失败，constant_1 退回 1e9（等价不裁剪）"; fi
log "constant_1 = $C1  (sample_number/class = $SAMPLE_PER_CLASS)"

########################################################################
log "==== [3/7] COPS 重训: uniform random 基线 ===="
$PY -u retraining_withy.py --seed 1 --lr 1e-3 --dataset adult --model tabnet \
    --test_mode random --sample_number $SAMPLE_PER_CLASS --loss_type base \
    --epochs $EPOCHS_COPS_RETRAIN \
    --un_folder "$COPS/adult_npy" --weight_folder "$COPS/weights_uniform" \
    > "$COPS/logs/03_uniform.log" 2>&1
log "uniform exit=$?"

log "==== [4/7] COPS 重训: COPS with y ===="
$PY -u retraining_withy.py --seed 1 --lr 1e-3 --dataset adult --model tabnet \
    --test_mode oracle_sampling_cut --constant_1 "$C1" --constant_2 10 \
    --sample_number $SAMPLE_PER_CLASS --loss_type reweight_clip \
    --epochs $EPOCHS_COPS_RETRAIN \
    --un_folder "$COPS/adult_npy" --weight_folder "$COPS/weights_cops_wy" \
    > "$COPS/logs/04_cops_wy.log" 2>&1
log "cops_wy exit=$?"

log "==== [5/7] COPS 重训: COPS without y ===="
$PY -u retraining_woy.py --seed 1 --lr 1e-3 --dataset adult --model tabnet \
    --test_mode oracle_sampling_cut --constant_1 "$C1" --constant_2 10 \
    --sample_number $SAMPLE_PER_CLASS --loss_type reweight_clip \
    --epochs $EPOCHS_COPS_RETRAIN \
    --un_folder "$COPS/adult_npy" --weight_folder "$COPS/weights_cops_woy" \
    > "$COPS/logs/05_cops_woy.log" 2>&1
log "cops_woy exit=$?"

########################################################################
log "==== [6/7] GM-Matching (fraction=$FRACTION, ${EPOCHS_GM} epoch x 3 reps) ===="
cd "$BASE/GM-Matching-main" || exit 1
$PY -u run_classify_coreset.py \
    --dataset adult --coreset_size_dataset $FRACTION --coreset_method gm_matching \
    --num_repetitions 3 --epochs $EPOCHS_GM --batch_size 128 --num_workers 0 \
    --embeds_dir "$GM/embeds" \
    --log_filename "$GM/logs/gm_matching.csv" \
    > "$GM/logs/run.log" 2>&1
log "GM exit=$?"

########################################################################
log "==== [7/7] dual-pruning 三步 ===="
cd "$BASE/dual-pruning-main/exp_cifar" || exit 1
CKPT=$DUAL/checkpoint
MASK=$CKPT/adult/42/generated_mask

log "  [7a] 收集训练动力学 (${EPOCHS_DUAL} epoch)"
$PY -u train.py --data_path dummy --dataset adult --arch tabnet \
    --epochs $EPOCHS_DUAL --learning-rate 0.1 --batch-size 128 --manualSeed 42 \
    --dynamics --save_path "$CKPT" --workers 0 \
    > "$DUAL/logs/01_dynamics.log" 2>&1
log "  [7a] exit=$?"

log "  [7b] 重要性评估"
$PY -u importance_evaluation.py --dataset adult --num_epochs $EPOCHS_DUAL \
    --dynamics_path "$CKPT/adult/42/npy/" --save_path "$MASK" \
    > "$DUAL/logs/02_importance.log" 2>&1
log "  [7b] exit=$?"

log "  [7c] 剪枝子集训练 (subset_rate=$FRACTION)"
$PY -u train_subset.py --data_path dummy --dataset adult --arch tabnet \
    --epochs $EPOCHS_DUAL --learning_rate 0.1 --batch-size 128 --manualSeed 42 \
    --subset_rate $FRACTION \
    --target-probs-path "$MASK/target_probs.npy" \
    --score-path "$MASK/dual_mask_T30.npy" \
    --mask-path "$MASK/dual_mask_T30.npy" \
    --c_d 4 --sample beta --save_path "$DUAL/pruned" --workers 0 \
    > "$DUAL/logs/03_pruned_train.log" 2>&1
log "  [7c] exit=$?"

log "==================== ALL DONE ===================="
