# -*- coding: utf-8 -*-
"""fetch_imdb.py -- 从 hf-mirror 取 stanfordnlp/imdb，重建成 aclImdb 目录布局。

为什么要重建目录而不是直接读 parquet
----------------------------------
`mydatasets/imdb.py` / `imdb0.py` 都是按标准 aclImdb 布局读的：
    <data_path>/aclImdb/train/{pos,neg}/*.txt
    <data_path>/aclImdb/test/{pos,neg}/*.txt
且标签来自**目录名**（pos->1, neg->0）、文本来自**文件内容全文**。
重建时严格照这个语义写，因此重建后的数据集与"官方 tar.gz 解压"逐位等价：
同一个模型吃到的 (text, label) 序列完全一样，只是文件名的 id/rating 是我们造的。

为什么走 hf-mirror
------------------
ai.stanford.edu 直连被拒（原官方 aclImdb_v1.tar.gz 就在那儿），
huggingface.co 域名不可达，但 hf-mirror.com 的 resolve 端点可用（实测 206 + 206）。
API 的 HEAD 请求会被挡（返回 404/401），所以这里一律用 GET。

unsup 写进 train/unsup/
-----------------------
这是 aclImdb 官方的原始布局（train 下还有 unsup 子目录，50000 条无标注）。
imdb.py / imdb0.py 只扫 pos/neg，所以多出 unsup 不影响它们的数字；
而 imdb_large 要的就是"更大的文本量"，正好用上，不必再单独下载。
"""
import os
import subprocess
import sys

import numpy as np
import pyarrow.parquet as pq

MIRROR = "https://hf-mirror.com/datasets/stanfordnlp/imdb/resolve/main/plain_text"
OUT = "/root/autodl-tmp/EVA/data/imdb/aclImdb"
TMP = "/root/autodl-tmp/EVA/data/imdb/_parquet"

SPLITS = {
    "train":         ("train", "pos/neg"),
    "test":          ("test", "pos/neg"),
    "unsupervised":  ("train/unsup", "none"),
}


def fetch(fname, dst):
    if os.path.exists(dst) and os.path.getsize(dst) > 1_000_000:
        print("  [skip] %s 已存在 %d B" % (fname, os.path.getsize(dst)))
        return dst
    url = "%s/%s" % (MIRROR, fname)
    cmd = ["curl", "-L", "--fail", "--retry", "5", "--retry-delay", "3",
           "--max-time", "900", "-o", dst, url]
    print("  [get ] %s" % url)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("下载失败 %s\n%s" % (url, r.stderr[-500:]))
    print("  [ok  ] %d B" % os.path.getsize(dst))
    return dst


def write_split(path, subdir, drop_label):
    t = pq.read_table(path)
    df = t.to_pandas()
    cols = set(df.columns)
    if "text" not in cols or ("label" not in cols and not drop_label):
        raise RuntimeError("%s 列不对：%s" % (path, sorted(cols)))
    target = os.path.join(OUT, subdir)
    os.makedirs(target, exist_ok=True)
    n = {"pos": 0, "neg": 0, "unsup": 0}

    if drop_label:
        for i, txt in enumerate(df["text"].tolist()):
            # 官方文件名形如 0_0.txt，这里保持同样的 "<idx>_<rating>.txt" 形态
            with open(os.path.join(target, "%d_0.txt" % i), "w",
                      encoding="utf-8") as fh:
                fh.write(txt)
            n["unsup"] += 1
    else:
        for i, (txt, lab) in enumerate(zip(df["text"].tolist(),
                                           df["label"].tolist())):
            sub = "pos" if int(lab) == 1 else "neg"
            d = os.path.join(target, sub)
            os.makedirs(d, exist_ok=True)
            # rating 用 10 写进文件名：官方 test 段也是整 10/1，imdb.py 不解析它
            with open(os.path.join(d, "%d_%d.txt" % (i, 10 if sub == "pos" else 1)),
                      "w", encoding="utf-8") as fh:
                fh.write(txt)
            n[sub] += 1
    print("  [wrote] %-14s %s" % (subdir, n))
    return n


def main():
    os.makedirs(TMP, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    for split, (subdir, _) in SPLITS.items():
        fname = "%s-00000-of-00001.parquet" % split
        p = fetch(fname, os.path.join(TMP, fname))
        write_split(p, subdir, drop_label=(split == "unsupervised"))

    print("\n=== 结果 ===")
    for root, dirs, files in os.walk(OUT):
        if files:
            print("%-52s %d 个文件" % (root.replace(OUT, "aclImdb"), len(files)))


if __name__ == "__main__":
    main()
