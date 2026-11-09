#!/usr/bin/env python3
"""run_preprocess.py -- 逐个执行 mydatasets 下的数据预处理/加载脚本。

设计要点
--------
* 每个数据集在**独立子进程**里跑（由 run_preprocess_all.sh 循环调用 --only），
  这样单个数据集爆内存或崩溃不会带倒整轮，也避免 14 个数据集的对象累积在内存里。
* 用 importlib 按**文件路径**加载模块，而不是 `import mydatasets`：
  `mydatasets/__init__.py` 会连带 import imdb.py / news.py，而这两个需要
  transformers 与 CUDA，本机无卡会直接失败，从而拖死所有数据集。
* 只调用脚本里已有的加载函数，不复制其逻辑；需要修的脚本单独出 *_modified.py，
  在 SPECS 里换 module 名即可。
* 产物落在 raw 数据同一路径下（torchvision 自己解压 / 脚本自己写 npy、csv）。
"""
import argparse
import importlib.util
import inspect
import json
import os
import sys
import time
import traceback

DATA_ROOT = os.environ.get("EVA_DATA_ROOT", "/root/autodl-tmp/EVA/data")
MYDATASETS = os.environ.get(
    "EVA_MYDATASETS", "/root/chaichengliang/VLDB2027/EVA/EVA/mydatasets"
)

# key, module file, function name, data_path (相对 DATA_ROOT；"." 表示 DATA_ROOT 本身)
SPECS = [
    ("cifar10",      "cifar10.py",      "CIFAR10",      "cifar10"),
    ("cifar100",     "cifar100_modified.py",     "CIFAR100",     "cifar100"),
    ("mnist",        "mnist_modified.py",        "MNIST",        "mnist"),
    ("qmnist",       "qmnist_modified.py",       "QMNIST",       "qmnist"),
    ("fashionmnist", "fashionmnist_modified.py", "FashionMNIST", "fashionmnist"),
    ("svhn",         "svhn_modified.py",         "SVHN",         "SVHN"),
    ("number",       "number_modified.py",       "Number",       "."),
    ("covtype",      "covtype_modified.py",      "covtype",      "covtype"),
    ("HR",           "HR_modified.py",           "HR",           "HR"),
    ("House",        "House_modified.py",        "House",        "House"),
    ("bike",         "bike_modified.py",         "Bike",         "bike"),
    ("tinyimagenet", "tinyimagenet_modified.py", "TinyImageNet", "tinyimagenet"),
    ("IMDBLarge",    "imdb_large_modified.py",   "IMDBLarge",    "imdb"),
    ("mnist2mnistm", "mnist2mnistm_modified.py", "Mnist2MnistM", "mnist2mnistm"),
    ("imagenet",     "imagenet.py",     "ImageNet",     "imagenet"),
]

BY_KEY = {s[0]: s for s in SPECS}


def load_module(path):
    name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location("_pp_" + name, path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load %s" % path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def describe(info):
    """把 (channel, im_size, num_classes, names, mean, std, train, [val], test) 压成摘要。"""
    out = {}
    try:
        channel, im_size, num_classes, class_names, mean, std = info[:6]
        rest = info[6:]
        out.update(
            channel=channel,
            im_size=list(im_size) if hasattr(im_size, "__len__") else im_size,
            num_classes=num_classes,
            n_class_names=len(class_names) if hasattr(class_names, "__len__") else None,
            mean=[round(float(m), 4) for m in mean] if hasattr(mean, "__len__") else mean,
            std=[round(float(s), 4) for s in std] if hasattr(std, "__len__") else std,
            splits=[len(d) for d in rest],
        )
    except Exception as exc:  # 摘要失败不该掩盖主结果
        out["describe_error"] = repr(exc)
    return out


def run_one(key):
    if key not in BY_KEY:
        raise SystemExit("unknown dataset %r; choices: %s" % (key, ", ".join(BY_KEY)))
    _, fname, func_name, rel = BY_KEY[key]
    path = os.path.join(MYDATASETS, fname)
    data_path = DATA_ROOT if rel == "." else os.path.join(DATA_ROOT, rel)

    started = time.time()
    rec = {"dataset": key, "script": fname, "func": func_name, "data_path": data_path}
    try:
        if not os.path.exists(path):
            raise FileNotFoundError("script not found: %s" % path)
        mod = load_module(path)
        func = getattr(mod, func_name, None)
        if func is None:
            raise AttributeError("%s has no %s()" % (fname, func_name))
        params = inspect.signature(func).parameters
        kwargs = {}
        if "balance" in params:
            kwargs["balance"] = True
        info = func(data_path, **kwargs)
        rec["status"] = "ok"
        rec["elapsed_sec"] = round(time.time() - started, 1)
        rec["info"] = describe(info)
    except Exception as exc:
        rec["status"] = "failed"
        rec["elapsed_sec"] = round(time.time() - started, 1)
        rec["error"] = "%s: %s" % (type(exc).__name__, exc)
        rec["traceback_tail"] = traceback.format_exc().strip().splitlines()[-6:]
    print("RESULT " + json.dumps(rec, ensure_ascii=False), flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="只跑某一个数据集")
    ap.add_argument("--all", action="store_true", help="在当前进程内顺序跑全部（不推荐）")
    ap.add_argument("--list", action="store_true", help="列出可用 key")
    args = ap.parse_args()

    if args.list:
        for key, fname, func_name, rel in SPECS:
            print("%-14s %-18s %-14s %s" % (key, fname, func_name, rel))
        return

    if args.only:
        run_one(args.only)
        return

    if args.all:
        for key, *_ in SPECS:
            run_one(key)
        return

    ap.print_help()


if __name__ == "__main__":
    main()
