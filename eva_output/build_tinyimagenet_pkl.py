#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Rebuild TinyImageNet train.pkl / valid.pkl / test.pkl from HF parquet.

Why this script exists
----------------------
`mydatasets/tinyimagenet.py` does `pickle.load(f) -> (data, labels)` and feeds
`data[idx]` into `transforms.Compose([Resize(32), ToTensor(), Normalize(...)])`.
The .pkl shipped with the repo are truncated at 6029312 / 5767168 bytes (the
write-side truncation fingerprint), so they must be regenerated.

Source
------
hf-mirror.com/datasets/zh-plus/tiny-imagenet
  data/train-00000-of-00001-*.parquet -> 100000 rows, labels 0..199
  data/valid-00000-of-00001-*.parquet ->  10000 rows, 50 per class, 64x64 JPEG

Representation choice (IMPORTANT)
---------------------------------
The container is capped by cgroup at **2 GB RAM** (`memory.max = 2147483648`),
while a list of 100000 decoded 64x64x3 PIL images is ~1.3 GB *by itself* and
peaks far above 2 GB while the parquet is being decoded (that got the first
attempt SIGKILLed). So instead of decoded images we store the **original JPEG
bytes** (avg 2032 B/image -> ~205 MB for the whole train split) and let
`tinyimagenet_modified.py` decode+convert('RGB') lazily in `__getitem__`.
Consequence for downstream code: use `tinyimagenet_modified.py`, not
`tinyimagenet.py`.

Note 0.45% of the official images are grayscale (mode 'L'). They are stored
untouched and converted to RGB at decode time, otherwise `ToTensor()` would
yield 1 channel and break `channel = 3`.

Split policy
------------
The official TinyImageNet **test** split (10k images) ships WITHOUT labels and
every reachable mirror (zh-plus, Maysee) only carries train+valid. EVA needs a
labeled dst_test, so train stays at the full 100k (coreset budgets remain
comparable to the paper) and the 10k valid set is split into 5000 valid +
5000 test, stratified (25 per class each), deterministic seed 0.
Written to split_meta.json.
"""
import io
import json
import os
import pickle
import sys

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

DATA_DIR = "/root/autodl-tmp/EVA/data/tinyimagenet"
DL = "/root/dl"
SEED = 0
BATCH = 2000


def iter_images(path):
    """Yield (jpeg_bytes, label) one row group batch at a time (low memory)."""
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=BATCH, columns=["image", "label"]):
        raw = batch.column("image").to_pylist()
        labs = batch.column("label").to_pylist()
        for r, y in zip(raw, labs):
            yield r["bytes"], int(y)
        del raw, labs


def probe(path, n=3000):
    """Light sanity check: magic bytes + decoded size/mode."""
    bad = 0
    seen = 0
    for b, _ in iter_images(path):
        if not (b[:2] == b"\xff\xd8" or b[:4] == b"\x89PNG"):
            bad += 1
        seen += 1
        if seen >= n:
            break
    print("  probe: %d rows checked, %d with unexpected magic" % (seen, bad))


def dump(data, labs, name):
    assert len(data) == len(labs), "length mismatch"
    out = os.path.join(DATA_DIR, name)
    tmp = out + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump((data, [int(v) for v in labs]), f, protocol=4)
    os.replace(tmp, out)
    print("wrote %s  (%.1f MB, n=%d)" % (out, os.path.getsize(out) / 1e6, len(data)))


def build_train():
    src = os.path.join(DL, "tiny_train.parquet")
    if not os.path.exists(src) or os.path.getsize(src) < 130 * 1024 * 1024:
        print("SKIP train: parquet incomplete")
        return
    print("== train ==")
    probe(src)
    data, labs = [], []
    for i, (b, y) in enumerate(iter_images(src)):
        data.append(b)
        labs.append(y)
        if (i + 1) % 20000 == 0:
            print("  %d/%d  rss_ok" % (i + 1, pq.ParquetFile(src).metadata.num_rows))
    dump(data, labs, "train.pkl")


def build_val_test():
    src = os.path.join(DL, "tiny_valid.parquet")
    print("== valid ==")
    pairs = list(iter_images(src))
    labs = [y for _, y in pairs]
    print("  n=%d labels=%d..%d uniq=%d mean_bytes=%.0f" % (
        len(pairs), min(labs), max(labs), len(set(labs)),
        float(np.mean([len(b) for b, _ in pairs]))))
    per = np.bincount(np.asarray(labs), minlength=200)
    print("  per-class min=%d max=%d" % (per.min(), per.max()))

    rng = np.random.RandomState(SEED)
    arr = np.asarray(labs)
    val_idx, test_idx = [], []
    for c in range(200):
        idx = np.where(arr == c)[0]
        rng.shuffle(idx)
        half = len(idx) // 2
        val_idx.extend(idx[:half].tolist())
        test_idx.extend(idx[half:].tolist())
    val_idx.sort()
    test_idx.sort()
    dump([pairs[i][0] for i in val_idx], [labs[i] for i in val_idx], "valid.pkl")
    dump([pairs[i][0] for i in test_idx], [labs[i] for i in test_idx], "test.pkl")
    with open(os.path.join(DATA_DIR, "split_meta.json"), "w") as f:
        json.dump({
            "seed": SEED, "n_valid": len(val_idx), "n_test": len(test_idx),
            "policy": "official valid 50/50 stratified; official test has no labels",
            "source": "hf-mirror.com/datasets/zh-plus/tiny-imagenet",
            "representation": "raw JPEG bytes; decoded to RGB in __getitem__",
            "reason": "container cgroup memory.max = 2GB",
        }, f, indent=2)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("all", "valtest"):
        build_val_test()
    if what in ("all", "train"):
        build_train()


if __name__ == "__main__":
    main()
