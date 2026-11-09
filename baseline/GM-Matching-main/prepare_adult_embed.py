# -*- coding: utf-8 -*-
"""prepare_adult_embed.py —— 为 GM-Matching 预生成 adult 的嵌入缓存

GM-Matching 默认用 CLIP ViT-B/32 提特征，这对表格数据没有意义。
这里改用 **z-score 标准化后的原始 13 维特征** 作为 φ(x)：
  - 与 GM 原始形式（在特征空间做 geometric median matching）一致
  - 不依赖额外的预训练模型，避免循环论证
  - 完全可复现

输出：<out_dir>/embeddings.pt  —— torch.float32 Tensor, shape (N, 13)
`run_coreset_select` 检测到文件已存在就会跳过 CLIP 计算。
"""

import argparse
import json
import os

import numpy as np
import torch

from adult_adapter import load_adult_arrays


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="/root/autodl-tmp/EVA/data/adult/adult_processed.npz")
    ap.add_argument("--out_dir", default="/root/autodl-tmp/EVA/output/GM_matching/embeds/adult")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    d = load_adult_arrays(args.npz, normalized=True)

    embeddings = torch.from_numpy(d["train_x"].astype(np.float32))
    path = os.path.join(args.out_dir, "embeddings.pt")
    torch.save(embeddings, path)

    meta = {
        "source": args.npz,
        "embedding": "normalized raw features (z-score)",
        "dim": int(embeddings.shape[1]),
        "n": int(embeddings.shape[0]),
        "dtype": str(embeddings.dtype),
        "note": "替代 CLIP ViT-B/32，表格数据上 CLIP 嵌入无意义",
    }
    with open(os.path.join(args.out_dir, "embed_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    back = torch.load(path, map_location="cpu")
    print("[prepare_adult_embed] ->", path)
    print("  shape %s dtype %s  mean %.4f std %.4f"
          % (tuple(back.shape), back.dtype, float(back.mean()), float(back.std())))


if __name__ == "__main__":
    main()
