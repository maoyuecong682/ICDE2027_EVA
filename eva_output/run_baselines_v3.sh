#!/bin/bash
# run_baselines_all.sh —— COPS / GM-Matching / DUAL 在 EVA 预处理数据集上的全量实验驱动
#
# 命名与存放规则见 /root/eva_output/命名规则_output.md：
#     autodl-tmp/EVA/output/<baseline 名字>/<数据集名字>/<结果文件>
# 本脚本严格按此落盘。output 是 /root/eva_output 的软链接。
#
# 用法
# ----
#   # 1) 先在 CPU 上验通路（分钟级），确认三个 baseline 都能跑通
#   SMOKE=1 bash /root/eva_output/run_baselines_all.sh
#
#   # 2) 正式实验（务必在 GPU 机器上，且必须用 launch.sh 脱离 SSH 通道）
#   bash /root/eva_output/launch.sh /root/eva_output/logs/sweep.log \
#        /root/eva_output "bash /root/eva_output/run_baselines_all.sh"
#
#   DATASETS="mnist covtype"  bash run_baselines_all.sh   # 只跑指定数据集
#   BASELINES="COPS"          bash run_baselines_all.sh   # 只跑指定 baseline
#
# 为什么必须脱离 SSH 通道
# -----------------------
# 直接把 python 挂在 ssh 命令后面，通道一断远端 bash 收到 SIGHUP 就把 python 带走。
# 更糟的是 `python ... | tail` 的退出码是 tail 的 0，于是表现为「退出码 0、日志正常、
# 产物为空」。launch.sh 用 setsid + nohup 断开进程组，并把真实退出码写进日志尾部。
#
# 为什么串行
# ----------
# 容器 cpu.max = 50000/100000，即 0.5 核。并行只会互相抢占，总时间不变还更容易 OOM。
# 实测 ResNet18@28x28 单线程约 6.7 张/秒，cifar100 全量一轮约 2 小时。
#
# 为什么 SMOKE 要单独一套预算
# ----------------------------
# SMOKE 把每个 baseline 的训练压到 1 epoch、batch 16，并用 EVA_SMOKE_LIMIT 把每个
# 划分按类截到 2000 张附近，只为证明「数据能进去、产物能出来」。数字不可用于任何结论。

set -u
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export WANDB_MODE=offline

PY=${PY:-/root/miniconda3/bin/python}
BASE=${BASE:-/root/autodl-tmp/EVA/baseline}
OUT=${OUT:-/root/autodl-tmp/EVA/output}
EVA=${EVA:-/root/chaichengliang/VLDB2027/EVA/EVA}

DATASETS=${DATASETS:-"cifar100 covtype fashionmnist HR mnist qmnist SVHN tinyimagenet mnist2mnistm"}
BASELINES=${BASELINES:-"COPS GM_matching DUAL"}
FRACTION=${FRACTION:-0.1}
MODEL=${MODEL:-resnet18}

DATA_ROOT=${EVA_DATA_ROOT:-/root/autodl-tmp/EVA/data}
export EVA_DATA_ROOT=$DATA_ROOT
export EVA_ROOT=$EVA
export PYTHONPATH="$OUT:$BASE/COPS-main:$BASE/GM-Matching-main:$BASE/dual-pruning-main/exp_cifar:${PYTHONPATH:-}"

if [ "${SMOKE:-0}" = "1" ]; then
    EPOCHS_UNC=1;   EPOCHS_COPS=1;  EPOCHS_GM=1;  GM_REPS=1
    # DUAL 的 epoch 数不能压到 1：importance_evaluation.dual() 是 10 帧滑窗
    # （window_size=10）且上层按 T=30 截断 target_probs[:30]。T 不足时窗口为空，
    # torch.stack([]) / 维度断言会直接抛错 —— 那是夹具问题，不是适配层问题。
    EPOCHS_DUAL=${EPOCHS_DUAL_SMOKE:-30}
    BS=16
    # 冒烟也要至少 2 个模型：COPS 的 cal_un_wy 算的是**跨模型**logit 协方差，
    # MODEL_NUM=1 时 mu 去掉自身均值后全为 0 → 协方差恒为 0 → metric 全 0 →
    # constant_1 标定成 0 → reweight_clip 把所有权重裁成 0，采样退化成均匀。
    # 这样冒烟虽然不报错，却验不到 COPS 真正依赖的那条路径。
    MODEL_NUM=${MODEL_NUM_SMOKE:-2}
    # 样本上限：不加这个，0.5 核上连一个数据集的一个 baseline 都要几十分钟
    export EVA_SMOKE_LIMIT=${EVA_SMOKE_LIMIT:-2000}
    echo "[config] SMOKE=1 —— 极简配置（每划分 <=${EVA_SMOKE_LIMIT} 张，${MODEL_NUM} 个模型）"
else
    EPOCHS_UNC=${EPOCHS_UNC:-30}; EPOCHS_COPS=${EPOCHS_COPS:-50}
    EPOCHS_GM=${EPOCHS_GM:-200};  GM_REPS=${GM_REPS:-3}
    EPOCHS_DUAL=${EPOCHS_DUAL:-200}
    BS=${BS:-128};                MODEL_NUM=${MODEL_NUM:-10}
    unset EVA_SMOKE_LIMIT 2>/dev/null || true      # 正式跑绝不允许截断
    echo "[config] 正式配置 FRACTION=$FRACTION MODEL=$MODEL BS=$BS " \
         "UNC=${EPOCHS_UNC}x${MODEL_NUM} COPS=${EPOCHS_COPS} GM=${EPOCHS_GM}x${GM_REPS} DUAL=${EPOCHS_DUAL}"
fi

log() { echo "[$(date '+%F %T')] $*"; }
die() { echo "[$(date '+%F %T')] FATAL: $*" >&2; exit 1; }

# ------------------------------------------------------------------ 工具函数
# 从 eva_adapter 拿 n_train / num_classes / is_tabular，用于推算每类采样数
ds_info() {
    $PY - <<PYEOF 2>/dev/null | tail -1
import eva_adapter as E
m = E.load("$1")
print(len(m["train"]), m["num_classes"], int(m["is_tabular"]),
              int(bool(m.get("is_regression"))))
PYEOF
}

# 每个类要选多少样本：总比例 / 类别数，与 adult 那次 (0.1*26049/2=1302) 同口径
#
# 必须再对「每类实际可用数」取 min。原因：spc 是按**原始** n_train 的比例算的，
# 而 SMOKE 下 eva_adapter 会把 train 截到 EVA_SMOKE_LIMIT 附近（每类只剩十几到
# 几百张）。covtype 尤其明显：0.1*464807/7 = 6640 张/类，而 smoke 下每类只有
# ~285 张 —— 采样数大于可用数会在取子集时直接炸，那是夹具的问题，不是代码的问题，
# 会白白掩盖真正的适配层错误。
# 正式跑时 n_train 远大于 spc，这个 min 是恒等变换，不影响任何数字。
per_class() {
    $PY -c "
n = float('$1'); c = float('$2'); f = float('$FRACTION')
v = max(1, int(round(f * n / c)))      # 按比例想要的每类张数
avail = max(1, int(n // c))            # 当前划分每类实际有多少张
print(min(v, avail))
"
}

# 标定脚本属于**工具链**，不随 OUT 走。OUT 在 SMOKE 下会被指到独立目录，
# 若仍用 \"\$OUT/calibrate_adult_metric.py\" 就会找不到文件、constant_1 静默退回 1e9。
TOOLS=${TOOLS:-/root/eva_output}

# ------------------------------------------------------------------ COPS
run_cops() {
    local ds=$1 n_train=$2 ncls=$3
    local d=$OUT/COPS/$ds
    local spc; spc=$(per_class "$n_train" "$ncls")
    mkdir -p "$d"/{weights,weights_uniform,weights_cops_wy,weights_cops_woy,cops_npy,logs}
    local cd=$BASE/COPS-main

    log "  [COPS/$ds] stage1 不确定性生成 (${MODEL_NUM} 模型 x ${EPOCHS_UNC} epoch, spc=$spc)"
    ( cd "$cd" && $PY -u uncertainty_generation.py \
        --dataset "$ds" --lr 1e-3 --model "$MODEL" --model_num "$MODEL_NUM" \
        --num_classes "$ncls" --epochs "$EPOCHS_UNC" --batch_size "$BS" \
        --weight_folder "$d/weights" --npy_folder "$d/cops_npy" ) \
        > "$d/logs/01_uncertainty.log" 2>&1 \
        || { log "  [COPS/$ds] stage1 失败，跳过后续"; return 1; }

    log "  [COPS/$ds] 标定 constant_1"
    $PY -u "$TOOLS/calibrate_adult_metric.py" --npy_folder "$d/cops_npy" \
        --out "$d/logs/metric_stats.json" > "$d/logs/02_calibrate.log" 2>&1
    local c1
    c1=$($PY -c "import json;print(json.load(open('$d/logs/metric_stats.json'))['constant_1'])" 2>/dev/null)
    [ -z "${c1:-}" ] && { c1=1e9; log "  [COPS/$ds] 标定失败，constant_1 退回 1e9（等价不裁剪）"; }
    log "  [COPS/$ds] constant_1 = $c1"

    ( cd "$cd" && $PY -u retraining_withy.py --seed 1 --lr 1e-3 \
        --dataset "$ds" --model "$MODEL" --batch_size "$BS" \
        --test_mode random --sample_number "$spc" --loss_type base \
        --epochs "$EPOCHS_COPS" --un_folder "$d/cops_npy" \
        --weight_folder "$d/weights_uniform" ) > "$d/logs/03_uniform.log" 2>&1 \
        || { log "  [COPS/$ds] uniform(random) 失败"; return 1; }

    ( cd "$cd" && $PY -u retraining_withy.py --seed 1 --lr 1e-3 \
        --dataset "$ds" --model "$MODEL" --batch_size "$BS" \
        --test_mode oracle_sampling_cut --constant_1 "$c1" --constant_2 10 \
        --sample_number "$spc" --loss_type reweight_clip \
        --epochs "$EPOCHS_COPS" --un_folder "$d/cops_npy" \
        --weight_folder "$d/weights_cops_wy" ) > "$d/logs/04_cops_wy.log" 2>&1 \
        || { log "  [COPS/$ds] cops_wy 失败"; return 1; }

    ( cd "$cd" && $PY -u retraining_woy.py --seed 1 --lr 1e-3 \
        --dataset "$ds" --model "$MODEL" --batch_size "$BS" \
        --test_mode oracle_sampling_cut --constant_1 "$c1" --constant_2 10 \
        --sample_number "$spc" --loss_type reweight_clip \
        --epochs "$EPOCHS_COPS" --un_folder "$d/cops_npy" \
        --weight_folder "$d/weights_cops_woy" ) > "$d/logs/05_cops_woy.log" 2>&1 \
        || { log "  [COPS/$ds] cops_woy 失败"; return 1; }

    return 0
}

# ------------------------------------------------------------------ GM-Matching
run_gm() {
    local ds=$1
    local d=$OUT/GM_matching/$ds
    mkdir -p "$d"/{embeds,logs}
    if [ "$is_tab" = "1" ]; then
        # CLIP 只对图像有定义；表格走"标准化后的原始特征"作 φ(x)。
        # 不预生成的话 GM 会拿 27 维向量去做 interpolate(224,224) -> ValueError。
        $PY -u "$TOOLS/prepare_eva_embed.py" --dataset "$ds" \
            --out_dir "$d/embeds/$ds" >> "$d/logs/00_embed.log" 2>&1 \
            || { log "  [GM/$ds] 表格嵌入预生成失败"; return 1; }
    fi
    log "  [GM/$ds] geometric-median matching (${EPOCHS_GM} epoch x ${GM_REPS} 次)"
    ( cd "$BASE/GM-Matching-main" && $PY -u run_classify_coreset.py \
        --dataset "$ds" --coreset_size_dataset "$FRACTION" --coreset_method gm_matching \
        --num_repetitions "$GM_REPS" --epochs "$EPOCHS_GM" --batch_size "$BS" \
        --num_workers "$NW" --embeds_dir "$d/embeds" \
        --log_filename "$d/logs/gm_matching.csv" ) > "$d/logs/run.log" 2>&1 || return 1
    return 0
}

# ------------------------------------------------------------------ DUAL
run_dual() {
    local ds=$1
    local d=$OUT/DUAL/$ds
    local ckpt=$d/checkpoint
    local mask=$ckpt/$ds/42/generated_mask
    local arch=$MODEL
    mkdir -p "$d"/{checkpoint,pruned,logs}

    log "  [DUAL/$ds] 7a 训练动力学 (${EPOCHS_DUAL} epoch)"
    ( cd "$BASE/dual-pruning-main/exp_cifar" && $PY -u train.py \
        --data_path dummy --dataset "$ds" --arch "$arch" \
        --epochs "$EPOCHS_DUAL" --learning-rate 0.1 --batch-size "$BS" --manualSeed 42 \
        --dynamics --save_path "$ckpt" --workers "$NW" ) > "$d/logs/01_dynamics.log" 2>&1 \
        || { log "  [DUAL/$ds] 7a 失败"; return 1; }

    log "  [DUAL/$ds] 7b 重要性评估"
    ( cd "$BASE/dual-pruning-main/exp_cifar" && $PY -u importance_evaluation.py \
        --dataset "$ds" --num_epochs "$EPOCHS_DUAL" \
        --dynamics_path "$ckpt/$ds/42/npy/" --save_path "$mask" ) \
        > "$d/logs/02_importance.log" 2>&1 \
        || { log "  [DUAL/$ds] 7b 失败"; return 1; }

    log "  [DUAL/$ds] 7c 剪枝子集训练 (subset_rate=$FRACTION)"
    ( cd "$BASE/dual-pruning-main/exp_cifar" && $PY -u train_subset.py \
        --data_path dummy --dataset "$ds" --arch "$arch" \
        --epochs "$EPOCHS_DUAL" --learning_rate 0.1 --batch-size "$BS" --manualSeed 42 \
        --subset_rate "$FRACTION" \
        --target-probs-path "$mask/target_probs.npy" \
        --score-path "$mask/dual_mask_T30.npy" \
        --mask-path "$mask/dual_mask_T30.npy" \
        --c_d 4 --sample beta --save_path "$d/pruned" --workers "$NW" ) \
        > "$d/logs/03_pruned_train.log" 2>&1 \
        || { log "  [DUAL/$ds] 7c 失败"; return 1; }

    return 0
}

########################################################################### main
log "==================== 开始 ===================="
mkdir -p "$OUT"

FAILED_LIST=""

for ds in $DATASETS; do
    read -r n_train ncls is_tab is_reg <<< "$(ds_info "$ds")"
    if [ -z "${n_train:-}" ]; then
        log "[$ds] 适配层拿不到数据集信息，跳过"; continue
    fi
    log "[$ds] n_train=$n_train classes=$ncls tabular=$is_tab"
    # 图像用 4 个 worker 解包（本机实测 2x），表格必须 0（covtype 上 4 worker 慢 2.6x）
    if [ "$is_tab" = "1" ]; then NW=0; else NW=4; fi
    # 回归数据集：选择阶段把连续目标分箱成 10 类，评估阶段不分箱
    if [ "$is_reg" = "1" ]; then export EVA_REG_BINS=${EVA_REG_BINS_SEL:-10}; else unset EVA_REG_BINS; fi
    # spc 必须按选择阶段的类别数算：回归分箱后是 10 类而不是 ds_info 报的 1 类，
    # 否则 spc=0.1*n_train 再乘 10 个箱就会超出采样池大小
    NCLS_SEL=$ncls; if [ "$is_reg" = "1" ]; then NCLS_SEL=${EVA_REG_BINS_SEL:-10}; fi

    for bl in $BASELINES; do
        # 上次已完成就跳过（断点续跑；删掉 $OUT/<bl>/<ds>/.done 可强制重跑）
        if [ -f "$OUT/$bl/$ds/.done" ]; then
            log "  [$bl/$ds] 已完成（.done 存在），跳过"; continue
        fi

        case $bl in
            COPS)        run_cops "$ds" "$n_train" "$NCLS_SEL" ;;
            GM_matching) run_gm "$ds" ;;
            DUAL)        run_dual "$ds" ;;
            *)           log "  [$bl] 未知 baseline，跳过"; continue ;;
        esac
        rc=$?

        # 三重判定：退出码为 0、日志里没有崩溃痕迹、产物目录存在。
        # 只看日志关键字会把「退出码非 0 但没打印 Error」的情况误判为成功。
        if [ $rc -eq 0 ] && [ -d "$OUT/$bl/$ds" ] \
                && ! grep -qiE "Traceback|Killed|核心已转储" "$OUT/$bl/$ds"/logs/*.log 2>/dev/null; then
            date '+%F %T' > "$OUT/$bl/$ds/.done"
            log "  [$bl/$ds] 完成 ✓"
        else
            FAILED_LIST="$FAILED_LIST $bl/$ds(rc=$rc)"
            log "  [$bl/$ds] 未完成（rc=$rc），不写 .done"
        fi
    done
done

log "==================== 结束 ===================="
if [ -n "$FAILED_LIST" ]; then
    log "失败项:${FAILED_LIST}"
    log "（修好后重跑本脚本即可，成功的项会自动跳过）"
else
    log "全部完成"
fi
log "产物：$OUT/<baseline>/<dataset>/"
log "下一步：python $EVA/evaluation/eval_baselines.py --baseline COPS|GM_matching|DUAL"
