#!/bin/bash
# smoke_adult.sh -- 重启后确认 COPS / GM-Matching / DUAL 在处理后的 adult 上仍可执行。
#
# v2 修正（v1 的 5 个 FAIL 都是本脚本的问题，不是环境问题）：
#   1. COPS 的 npy_folder 必须含 adapter 产出的 train_X_*.npy，v1 给了空目录 -> 先复制已有产物
#   2. GM 走预计算嵌入，v1 给了空 embeds_dir -> 触发 CLIP 分支并 ModuleNotFoundError
#   3. DUAL importance_evaluation 硬编码 T=30，v1 只跑 1 epoch -> 1 维数组 IndexError
#      -> 动力学改成 30 epoch
#   4. train_subset.py 的参数是 --learning_rate（下划线），不是 --learning-rate
set -u
export OMP_NUM_THREADS=1
export WANDB_MODE=offline

PY=/root/miniconda3/bin/python
BASE=/root/autodl-tmp/EVA/baseline
OUT=/root/eva_output/smoke
COPS=$OUT/COPS
GM=$OUT/GM_matching
DUAL=$OUT/DUAL
REF=/root/eva_output

SAMPLE=130
EP_COPS=1
EP_GM=1
EP_DUAL=30

rm -rf "$OUT"
mkdir -p "$COPS/logs" "$COPS/weights" "$COPS/adult_npy" \
         "$GM/logs" "$GM/embeds" \
         "$DUAL/logs"

log() { echo "[$(date '+%F %T')] $*"; }
rc_of() { local r=$1; [ "$r" = "0" ] && echo PASS || echo "FAIL(rc=$r)"; }

log "==== 复用已有中间产物（adapter npy / 预计算嵌入）===="
cp -a "$REF/COPS/adult_npy/." "$COPS/adult_npy/" && log "  adult_npy  copied ($(ls "$COPS/adult_npy" | wc -l) files)"
cp -a "$REF/GM_matching/embeds/adult" "$GM/embeds/" && log "  gm embeds  copied"

log "==== COPS [1/2] 不确定性生成 (1 model x ${EP_COPS} epoch) ===="
cd "$BASE/COPS-main" || exit 1
$PY -u uncertainty_generation.py \
    --dataset adult --lr 1e-3 --model tabnet --model_num 1 --num_classes 2 \
    --epochs $EP_COPS \
    --weight_folder "$COPS/weights" \
    --npy_folder "$COPS/adult_npy" \
    > "$COPS/logs/01_uncertainty.log" 2>&1
R1=$?; log "  stage1      $(rc_of $R1)"

log "==== COPS [2/2] 重训 uniform random 基线 ===="
$PY -u retraining_withy.py --seed 1 --lr 1e-3 --dataset adult --model tabnet \
    --test_mode random --sample_number $SAMPLE --loss_type base \
    --epochs $EP_COPS \
    --un_folder "$COPS/adult_npy" --weight_folder "$COPS/weights_uniform" \
    > "$COPS/logs/02_uniform.log" 2>&1
R2=$?; log "  retrain     $(rc_of $R2)"

log "==== GM-Matching (1 rep x ${EP_GM} epoch) ===="
cd "$BASE/GM-Matching-main" || exit 1
$PY -u run_classify_coreset.py \
    --dataset adult --coreset_size_dataset 0.1 --coreset_method gm_matching \
    --num_repetitions 1 --epochs $EP_GM --batch_size 128 --num_workers 0 \
    --embeds_dir "$GM/embeds" \
    --log_filename "$GM/logs/gm_matching.csv" \
    > "$GM/logs/run.log" 2>&1
R3=$?; log "  gm          $(rc_of $R3)"

log "==== DUAL [1/3] 训练动力学 (${EP_DUAL} epoch) ===="
cd "$BASE/dual-pruning-main/exp_cifar" || exit 1
CKPT=$DUAL/checkpoint
MASK=$CKPT/adult/42/generated_mask
$PY -u train.py --data_path dummy --dataset adult --arch tabnet \
    --epochs $EP_DUAL --learning-rate 0.1 --batch-size 128 --manualSeed 42 \
    --dynamics --save_path "$CKPT" --workers 0 \
    > "$DUAL/logs/01_dynamics.log" 2>&1
R4=$?; log "  dynamics    $(rc_of $R4)"

log "==== DUAL [2/3] 重要性评估 (T=${EP_DUAL}) ===="
$PY -u importance_evaluation.py --dataset adult --num_epochs $EP_DUAL \
    --dynamics_path "$CKPT/adult/42/npy/" --save_path "$MASK" \
    > "$DUAL/logs/02_importance.log" 2>&1
R5=$?; log "  importance  $(rc_of $R5)"

log "==== DUAL [3/3] 剪枝子集训练 ===="
SCORE=$(ls "$MASK"/dual_mask_T*.npy 2>/dev/null | head -1)
if [ -z "$SCORE" ]; then
    SCORE=$REF/DUAL/checkpoint/adult/42/generated_mask/dual_mask_T30.npy
    log "  未生成 mask，回退到全量实验的 T30 mask"
fi
$PY -u train_subset.py --data_path dummy --dataset adult --arch tabnet \
    --epochs $EP_GM --learning_rate 0.1 --batch-size 128 --manualSeed 42 \
    --subset_rate 0.1 \
    --target-probs-path "$(dirname "$SCORE")/target_probs.npy" \
    --score-path "$SCORE" \
    --mask-path "$SCORE" \
    --c_d 4 --sample beta --save_path "$DUAL/pruned" --workers 0 \
    > "$DUAL/logs/03_pruned_train.log" 2>&1
R6=$?; log "  subset      $(rc_of $R6)"

echo
echo "==================== SMOKE SUMMARY ===================="
echo "COPS stage1        : $(rc_of $R1)"
echo "COPS retrain       : $(rc_of $R2)"
echo "GM matching        : $(rc_of $R3)"
echo "DUAL dynamics      : $(rc_of $R4)"
echo "DUAL importance    : $(rc_of $R5)"
echo "DUAL subset train  : $(rc_of $R6)"
echo "mask used          : $SCORE"
echo "SMOKE DONE"
