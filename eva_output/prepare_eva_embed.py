# -*- coding: utf-8 -*-
"""prepare_eva_embed.py —— 为 GM-Matching 预生成 φ(x) 嵌入缓存

为什么要有这一步
----------------
`GM-Matching-main/sampling/sample.py::run_coreset_select` 的逻辑是：
    if not os.path.exists(embedding_file):
        compute_embed(...)      # 调 CLIP ViT-B/32
    embeddings = load_embed(embedding_file)
也就是说，只要 `<embeds_dir>/<dataset>/embeddings.pt` 已经存在，它就直接复用、不再
算 CLIP。预生成带来三个好处：
  1. 每个图像数据集只计算一次 CLIP，GM 的四种比例复用同一份特征；
  2. 换 coreset_size / 换 coreset_method 不必重算嵌入；
  3. 嵌入的行序可以显式对齐适配层的 train 顺序，而不是依赖 GM 自己的 DataLoader。

φ(x) 的口径（用户已定：图像用 GM 原版的 CLIP ViT-B/32）
------------------------------------------------------
* 图像数据集 → CLIP ViT-B/32 图像编码器，输出 (N, 512) 原始特征（**不做 L2 归一化**，
  与 `sampling/embed.py::get_embeddings` 的 `model.encode_image(...)` 完全一致）。
  预处理按 CLIP 论文：缩放到 224、双三次插值、CLIP 的均值方差；单通道复制成三通道。
* 表格数据集（HR、covtype）→ CLIP 对表格没有定义，退回 **标准化后的原始特征**，
  与 adult 那次的 `prepare_adult_embed.py` 同口径。这一条必须在论文/说明里写明，
  否则「三个 baseline 的表示空间不完全一致」会被当成隐藏假设。

顺序对齐是硬要求
----------------
`embeddings.pt` 的第 i 行必须对应 `eva_adapter.load(ds)["train"]` 的第 i 个样本，
否则 GM 选出的索引会指到别的样本上，而且**不会报错**，只是结果整体错位。
脚本最后会重新逐条比对 N、并抽查若干行的 L2 范数，写进 embed_meta.json。

用法
----
    python prepare_eva_embed.py --dataset mnist --out_dir /root/.../embeds/mnist
    python prepare_eva_embed.py --dataset HR    --out_dir /root/.../embeds/HR
    python prepare_eva_embed.py --dataset mnist --out_dir /tmp/e --limit 32   # 冒烟
"""
import argparse
import json
import os
import sys
import time
import hashlib
import shutil

import numpy as np
import torch
import torchvision.transforms as T

sys.path.insert(0, os.environ.get("EVA_OUT", "/root/eva_output"))

import eva_adapter as EVA

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def clip_preprocess():
    """CLIP ViT-B/32 的图像预处理。

    不用 `model.preprocess`（那是 PIL 版）：我们的样本已经是 [0,1] 的 float 张量，
    直接走 tensor 版更省一次 tensor->PIL->tensor 的往返，也不会引入量化误差。
    """
    return T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.Normalize(CLIP_MEAN, CLIP_STD),
    ])


def to_clip_batch(xb):
    """(B,C,H,W) in [0,1] -> (B,3,224,224) in CLIP 归一化空间。"""
    if xb.shape[1] == 1:
        # 灰度图复制成三通道：CLIP 的 patch_embed 固定 3 通道
        xb = xb.repeat(1, 3, 1, 1)
    elif xb.shape[1] != 3:
        raise ValueError("通道数 %d 既不是 1 也不是 3，CLIP 无法处理" % xb.shape[1])
    return clip_preprocess()(xb)


def embed_images(meta, device, batch_size, limit):
    # encode_image uses only this exact official image tower; loading the text
    # tower consumes memory without affecting the image representation.
    from clip_visual_cpu import load_visual
    model=load_visual().to(device)
    if str(device)=='cpu':
        batch_size=min(batch_size,2)
    meta['_clip_backend']='official_visual_tower_original_precision_'+str(device)
    meta['_clip_checkpoint_sha256']=model.checkpoint_sha256
    model.eval()

    n = len(meta["train"])
    if limit:
        n = min(n, limit)

    dim = model.visual.output_dim
    out = np.zeros((n, dim), dtype=np.float32)

    tf = time.time()
    done = 0
    with torch.no_grad():
        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
            xb = torch.stack([meta["train"][i][0] for i in range(s, e)])
            if xb.dtype != torch.float32:
                xb = xb.float()
            # EVA images are already normalized. Undo that normalization before
            # CLIP preprocessing; dividing normalized floats by 255 corrupts them.
            mean = torch.as_tensor(meta['mean'], dtype=xb.dtype).reshape(1,-1,1,1)
            std = torch.as_tensor(meta['std'], dtype=xb.dtype).reshape(1,-1,1,1)
            xb = (xb * std + mean).clamp(0, 1)
            xb = to_clip_batch(xb).to(device)
            out[s:e] = model.encode_image(xb).float().cpu().numpy()
            done = e
            if s == 0 or e == n or (s // batch_size) % 20 == 0:
                print("  [%d/%d] %.1f img/s" % (e, n, e / max(time.time() - tf, 1e-9)))
    return torch.from_numpy(out), "clip-vit-b32"


def embed_tabular(meta, limit):
    n = len(meta["train"])
    if limit:
        n = min(n, limit)
    rows = []
    for i in range(n):
        x, _ = meta["train"][i]
        rows.append(np.asarray(x, dtype=np.float32).reshape(-1))
    arr = np.stack(rows)
    return torch.from_numpy(arr), "raw-features(standardized-by-mydatasets)"


def embed_placeholder(meta, limit, dim):
    """随机占位嵌入。**只用于验证 GM 的通路，绝不能用于任何结果**。

    仅保留给隔离的接口测试。正式矩阵的 supervisor 拒绝这种 method。
    CPU 原生端到端验证也使用真实 CLIP 权重。
    """
    n = len(meta["train"])
    if limit:
        n = min(n, limit)
    rs = np.random.RandomState(48)
    arr = rs.randn(n, dim).astype(np.float32)
    return torch.from_numpy(arr), "PLACEHOLDER-random(不可用于结果)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out_dir", required=True,
                    help="写到 <out_dir>/embeddings.pt，应等于 <embeds_dir>/<dataset>")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="只算前 N 个样本（冒烟用）")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--method", default="auto",
                    choices=["auto", "placeholder"],
                    help="auto=图像走 CLIP / 表格走原始特征；placeholder=随机占位，仅验通路")
    ap.add_argument("--force", action="store_true", help="覆盖已有 embeddings.pt")
    ap.add_argument('--reuse-source',help='Reuse only after every feature value matches current input')
    a = ap.parse_args()

    path = os.path.join(a.out_dir, "embeddings.pt")
    if os.path.exists(path) and not a.force:
        print("[SKIP] 已存在：%s（加 --force 可覆盖）" % path)
        return

    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    meta = EVA.load(a.dataset)
    n = len(meta["train"])
    print("[%s] train=%d classes=%d tabular=%s device=%s"
          % (a.dataset, n, meta["num_classes"], meta["is_tabular"], device))

    if a.method == "placeholder":
        emb, method = embed_placeholder(meta, a.limit, 512)
    elif meta["is_tabular"]:
        emb, method = embed_tabular(meta, a.limit)
    else:
        emb, method = embed_images(meta, device, a.batch_size, a.limit)

    expected_rows=min(n,a.limit) if a.limit else n
    if emb.ndim!=2 or emb.shape[0]!=expected_rows or not torch.isfinite(emb).all():
        raise ValueError('Embedding shape/row count/finiteness validation failed')

    os.makedirs(a.out_dir, exist_ok=True)
    reused_source=None
    if a.reuse_source and meta['is_tabular'] and not os.environ.get('EVA_SMOKE_LIMIT'):
        old=torch.load(a.reuse_source,map_location='cpu',mmap=True,weights_only=True)
        if old.shape==emb.shape and torch.equal(old,emb):
            reused_source=a.reuse_source
            shutil.copyfile(reused_source,path)
    if not reused_source:torch.save(emb, path)

    # 回读校验：mmap 加载（GM 的 load_embed 就是这么读的），确认行数、dtype、
    # 与首段样本的 L2 范数都对得上。顺序错位是最隐蔽的错误，必须实测一次。
    back = torch.load(path, map_location="cpu", mmap=True)
    same_n = int(back.shape[0]) == int(emb.shape[0])
    probes = [0, min(1, emb.shape[0] - 1), emb.shape[0] // 2, emb.shape[0] - 1]
    drift = float(max((back[i].float() - emb[i].float()).abs().max() for i in probes))

    meta_out = {
        "dataset": a.dataset,
        "method": method,
        "n_rows": int(emb.shape[0]),
        "dim": int(emb.shape[1]),
        "dtype": str(emb.dtype),
        "reused_source":reused_source,
        "verified_feature_rows":int(emb.shape[0]) if reused_source else 0,
        "encoder_backend":meta.get('_clip_backend'),
        "checkpoint_sha256":meta.get('_clip_checkpoint_sha256'),
        "adapter_n_train": int(n),
        "row_order": "第 i 行 == eva_adapter.load(dataset)['train'][i]，未打乱",
        "norm_mean": float(emb.float().norm(dim=1).mean()),
        "smoke_limit": int(a.limit),
        "device": device,
        "note": ("图像：CLIP ViT-B/32 原始特征（未 L2 归一化），与 sampling/embed.py 一致；"
                 "表格：CLIP 对表格无定义，退回标准化后的原始特征（同 adult 口径）"),
        "roundtrip_ok": bool(same_n and drift == 0.0),
    }
    with open(os.path.join(a.out_dir, "embed_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta_out, f, ensure_ascii=False, indent=2)

    print("[done] -> %s  shape=%s dtype=%s  row_norm_mean=%.4f  roundtrip=%s"
          % (path, tuple(emb.shape), emb.dtype, meta_out["norm_mean"],
             "OK" if meta_out["roundtrip_ok"] else "BAD"))
    if not meta_out["roundtrip_ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
