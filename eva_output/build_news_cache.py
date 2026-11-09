# -*- coding: utf-8 -*-
"""build_news_cache.py -- 手工构造 sklearn 的 20news-bydate.pkz 缓存。

为什么要这么做
--------------
sklearn 1.9.1 的 `_twenty_newsgroups._download_20newsgroups()` 里**没有**本地
归档回退分支：只要 `cache_path` 不存在，就无条件 `_fetch_remote(ARCHIVE, ...)` 去
figshare 拉包。本机到那个地址是 403，所以 `fetch_20newsgroups(data_home=...)`
一定失败 —— 哪怕 data_home 下已经躺着解压好的 20news-bydate-train。

那就把 sklearn 自己该写的缓存替它写好。`_download_20newsgroups` 的收尾是：

    cache = dict(train=load_files(train_path, encoding="latin1"),
                 test =load_files(test_path,  encoding="latin1"))
    with open(cache_path, "wb") as f:
        f.write(codecs.encode(pickle.dumps(cache), "zlib_codec"))

这里逐字复刻这三行，用的是同一份官方 tar.gz 解压结果、同一个 load_files、
同一个 encoding。因此写出来的 pkz 与"联网下载成功"得到的 pkz 语义等价
（唯一的差别是 pickle 的字节序，内容一致）。

解压目录用临时目录：脚本跑完就删，避免在 data 下留一份 200MB 的明文语料 ——
反正 sklearn 读完也只认 pkz，原 `_download_20newsgroups` 同样会
`shutil.rmtree(target_dir)`。
"""
import codecs
import os
import pickle
import shutil
import tarfile
import tempfile

from sklearn.datasets._twenty_newsgroups import load_files

DATA_HOME = "/root/autodl-tmp/EVA/data/news"
ARCHIVE = os.path.join(DATA_HOME, "20news-bydate.tar.gz")
CACHE = os.path.join(DATA_HOME, "20news-bydate.pkz")
TRAIN_FOLDER = "20news-bydate-train"
TEST_FOLDER = "20news-bydate-test"


def main():
    if os.path.exists(CACHE):
        print("[skip] 已存在 %s (%d B)" % (CACHE, os.path.getsize(CACHE)))
        return
    if not os.path.exists(ARCHIVE):
        raise SystemExit("找不到 %s" % ARCHIVE)

    tmp = tempfile.mkdtemp(prefix="news_")
    try:
        with tarfile.open(ARCHIVE, "r:gz") as fp:
            fp.extractall(path=tmp)
        cache = dict(
            train=load_files(os.path.join(tmp, TRAIN_FOLDER), encoding="latin1"),
            test=load_files(os.path.join(tmp, TEST_FOLDER), encoding="latin1"),
        )
        blob = codecs.encode(pickle.dumps(cache), "zlib_codec")
        with open(CACHE, "wb") as fh:
            fh.write(blob)
        print("[ok] 写入 %s (%d B)" % (CACHE, os.path.getsize(CACHE)))
        for split in ("train", "test"):
            b = cache[split]
            print("  %-5s n=%d  类别数=%d  data[0][:60]=%r"
                  % (split, len(b.data), len(b.target_names),
                     b.data[0][:60]))
            print("         target_names[:5]=%s" % list(b.target_names[:5]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
