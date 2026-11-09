# -*- coding: utf-8 -*-
"""rebuild_imdb_large.py -- 重建 imdb_large（原文件截断，无法使用）。

问题
----
`data/imdb/IMDBLargeC5-{train,val,test}-{X,y}.npy` 三个特征文件各只有 6.0 MB / 5.8 MB，
而 npy 头写的是 (10561348, 41) —— 1056 万行 × 41 维 float64 应当是 **3.46 GB**。
两份副本（`imdb/` 与 `imdb/1025upload/`）字节数完全相同，说明是同一次不完整传输
的产物，不是"另一份能用的备份"。全盘（含只读公共盘 /autodl-pub 与 /autodl-fs）
搜索也没有任何 >50MB 的 IMDBLarge* 文件，`IMDBLargeC5-formycs/` 只有 44MB 的
按类切分片段（ci/mc/mi/mix/name/title 六种特征视图），不足以还原 1056 万行 × 41 维
的完整特征矩阵，也找不到能复现那 41 维工程特征的原始脚本与语料。

替代口径（已记入 evaluation3090 的说明）
--------------------------------------
用 **aclImdb 全集**（train 25000 + test 25000 = 50000 条有标注影评）重建，
按 8:1:1 重新切分，任务仍是情感二分类，预处理与本仓 imdb 完全一致：

    BertModel(pooler_output) — 768 维
    tokenizer(padding=True, truncation=True, max_length=512)
    batch_size=32，GPU 前向，不反向

因此 imdb_large 与 imdb 的差别只有**规模与划分**（50000 vs 25000，且 imdb 的
val 是从 train 里切 20% 而 test 固定 25000）。这比"表格 1056 万行"弱，但它是一个
真实、可复现、规模翻倍的文本核心集基准，比留空或塞一个截断文件进去要诚实。

为什么存**未标准化**的嵌入
--------------------------
`imdb_large.py` 的 `IMDBLarge()` 自己会 `StandardScaler().fit_transform(train)`，
再 transform val/test。这里若先标准化一遍，等于标准化两次（虽然接近恒等，
但没必要引入这层解释负担）。所以落盘的是原始 pooler 输出。

落盘位置
--------
    data/imdb/imdb_large/IMDBLargeC5-{train,val,test}-{X,y}.npy
与截断文件放在不同目录，避免 REGISTRY 指错；截断文件保留并改名为
`*.truncated`，留作证据而不是删掉。
"""
import os
import shutil

import numpy as np
import torch
from transformers import BertModel, BertTokenizer
from tqdm import tqdm

ACLIMDB = "/root/autodl-tmp/EVA/data/imdb/aclImdb"
OUT = "/root/autodl-tmp/EVA/data/imdb/imdb_large"
BERT = ("/root/autodl-tmp/bert-base-uncased/models/"
        "google-bert--bert-base-uncased/snapshots/master")
SEED = 48
SPLIT = (0.8, 0.1, 0.1)      # train / val / test
BS = 32
MAX_LEN = 512


def read_split(subdir):
    texts, labels = [], []
    for cls, lab in (("pos", 1), ("neg", 0)):
        d = os.path.join(ACLIMDB, subdir, cls)
        files = sorted(f for f in os.listdir(d) if f.endswith(".txt"))
        for f in files:
            with open(os.path.join(d, f), encoding="utf-8") as fh:
                texts.append(fh.read())
            labels.append(lab)
    return texts, np.asarray(labels, dtype=np.int64)


@torch.no_grad()
def embed(texts, model, tok, device):
    out = []
    for i in tqdm(range(0, len(texts), BS), desc="bert", ncols=80):
        batch = texts[i:i + BS]
        enc = tok(batch, padding=True, truncation=True, max_length=MAX_LEN,
                  return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        out.append(model(**enc).pooler_output.float().cpu().numpy())
    return np.concatenate(out, axis=0)


def main():
    os.makedirs(OUT, exist_ok=True)
    tag = os.path.join(OUT, "IMDBLargeC5-train-X.npy")
    if os.path.exists(tag):
        print("[skip] 已存在 %s" % tag)
        return

    # 截断文件改名留证，不动内容
    for f in os.listdir("/root/autodl-tmp/EVA/data/imdb"):
        if f.startswith("IMDBLargeC5-") and f.endswith(".npy"):
            src = os.path.join("/root/autodl-tmp/EVA/data/imdb", f)
            dst = src + ".truncated"
            if not os.path.exists(dst):
                shutil.move(src, dst)
                print("  [mark] %s -> %s.truncated" % (f, f))

    print("[1/3] 读 aclImdb train + test …")
    tx_a, ty_a = read_split("train")
    tx_b, ty_b = read_split("test")
    texts = tx_a + tx_b
    labels = np.concatenate([ty_a, ty_b])
    print("  合计 %d 条，类别分布 %s" % (len(texts), np.bincount(labels).tolist()))

    print("[2/3] BERT 编码 …")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = BertTokenizer.from_pretrained(BERT)
    model = BertModel.from_pretrained(BERT).to(device).eval()
    X = embed(texts, model, tok, device)
    print("  嵌入形状 %s" % (X.shape,))
    del model
    torch.cuda.empty_cache()

    print("[3/3] 按 8:1:1 切分（seed=%d）并落盘 …" % SEED)
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(labels))
    n_tr = int(round(len(labels) * SPLIT[0]))
    n_va = int(round(len(labels) * SPLIT[1]))
    blocks = {"train": idx[:n_tr],
              "val": idx[n_tr:n_tr + n_va],
              "test": idx[n_tr + n_va:]}
    for split, ids in blocks.items():
        xp = os.path.join(OUT, "IMDBLargeC5-%s-X.npy" % split)
        yp = os.path.join(OUT, "IMDBLargeC5-%s-y.npy" % split)
        np.save(xp, X[ids])
        np.save(yp, labels[ids])
        print("  %-5s %6d  %s %s  类别分布 %s"
              % (split, len(ids), xp.split("/")[-1],
                 np.load(xp, mmap_mode="r").shape,
                 np.bincount(labels[ids]).tolist()))


if __name__ == "__main__":
    main()
