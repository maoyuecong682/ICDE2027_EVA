#!/bin/bash
# 为表格数据集预生成 GM 的 φ(x) 缓存（CLIP 对表格无定义，必须先有 embeddings.pt）
set -u
PY=/root/miniconda3/bin/python
OUT=/root/autodl-tmp/EVA/output/GM_matching
TOOLS=/root/eva_output
export EVA_ROOT=/root/chaichengliang/VLDB2027/EVA/EVA
export EVA_DATA_ROOT=/root/autodl-tmp/EVA/data
export PYTHONPATH=":D:\workbuddy\resources\app.asar.unpacked\cli\vendor\shim"
for ds in adult HR covtype bike House news imdb imdb0 imdb_large; do
    d=/$ds
    mkdir -p "$d/embeds/$ds" "$d/logs"
    if [ -f "$d/embeds/$ds/embeddings.pt" ]; then
        echo "[skip] $ds 已有 embeddings.pt"; continue
    fi
    echo "[gen ] $ds"
    $PY -u "$TOOLS/prepare_eva_embed.py" --dataset "$ds" --out_dir "$d/embeds/$ds"         > "$d/logs/00_embed.log" 2>&1
    rc=$?
    echo "[done] $ds rc=$rc $(ls -l "$d/embeds/$ds/embeddings.pt" 2>/dev/null | awk '{print $5" B"}')"
done
