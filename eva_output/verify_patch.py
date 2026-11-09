"""补丁结果体检：精确统计关键符号出现次数 + 逐文件编译。

为什么不用 grep 拼一行：正则里的括号、管道、路径带空格都会让计数错位，
错位之后的「看起来正常」比报错更危险。这里全部按字面子串统计。
"""
import io
import os
import py_compile
import sys

EXP = "/root/autodl-tmp/EVA/baseline/dual-pruning-main/exp_cifar"
BASE = "/root/autodl-tmp/EVA/baseline"
OUT = "/root/eva_output"

CASES = [
    # (文件, 字面量子串, 期望次数)
    (OUT + "/eva_adapter.py", "ADAPTER_DATASETS = (", 1),
    (OUT + "/eva_adapter.py", "def _smoke_limit(", 1),
    (OUT + "/eva_adapter.py", "def _truncate(", 1),
    (OUT + "/eva_adapter.py", "_CACHE[_key] = meta", 1),

    (EXP + "/eva_names.py", "ADAPTER_DATASETS = list(", 1),
    (EXP + "/data.py", "from eva_names import ADAPTER_DATASETS", 1),
    (EXP + "/data.py", "def _eva_names():", 1),

    (EXP + "/train.py", "from eva_names import ALL_DATASETS, ADAPTER_DATASETS", 1),
    (EXP + "/train.py", "if args.dataset in ADAPTER_DATASETS:", 2),  # num_classes + 模型
    (EXP + "/train.py", "args.num_classes = int(_meta['num_classes'])", 1),
    (EXP + "/train.py", "choices=ALL_DATASETS", 1),

    (EXP + "/train_subset.py", "from eva_names import ALL_DATASETS, ADAPTER_DATASETS", 1),
    (EXP + "/train_subset.py", "from data_subset import", 1),
    (EXP + "/train_subset.py", "load_eva_sub(args, target_probs, score, mask)", 1),
    (EXP + "/train_subset.py", "choices=ALL_DATASETS", 1),

    (EXP + "/importance_evaluation.py", "from eva_names import ALL_DATASETS, ADAPTER_DATASETS", 1),
    (EXP + "/importance_evaluation.py", "choices=ALL_DATASETS", 1),
    (EXP + "/importance_evaluation.py", "elif args.dataset in ADAPTER_DATASETS:", 1),

    (EXP + "/data_subset.py", "def load_eva_sub(", 1),
    (EXP + "/data_subset.py", "beta_sampling(1 - args.subset_rate", 1),
    (EXP + "/data_subset.py", "raise NotImplementedError(\"EVA 数据集暂不支持 --sample beta\")", 0),

    (BASE + "/GM-Matching-main/run_classify_coreset.py",
     'model = _EVA.build_model(args.dataset, num_classes=_nc)', 1),
    (BASE + "/GM-Matching-main/run_classify_coreset.py",
     'get_model(model_name="CIFARResNet18"', 1),  # 仅剩原生 cifar10 分支

    (BASE + "/COPS-main/uncertainty_generation.py", "EVA.ADAPTER_DATASETS", 1),
    # withy/woy 各出现两次是正确的：一次在 cifar100 的分支排除条件里，
    # 一次在骨干构建的 EVA 分支里。
    (BASE + "/COPS-main/retraining_withy.py", "EVA.ADAPTER_DATASETS", 2),
    (BASE + "/COPS-main/retraining_withy.py", "arch=None, num_classes=class_n", 1),
    (BASE + "/COPS-main/retraining_woy.py", "EVA.ADAPTER_DATASETS", 2),
    (BASE + "/COPS-main/retraining_woy.py", "arch=None, num_classes=class_n", 1),
]

bad = 0
print("=" * 68)
for path, pat, want in CASES:
    if not os.path.exists(path):
        print("  [MISS] %-28s 文件不存在" % os.path.basename(path))
        bad += 1
        continue
    with io.open(path, "r", encoding="utf-8") as f:
        got = f.read().count(pat)
    ok = (got == want)
    if not ok:
        bad += 1
    print("  [%s] %-26s %-46s got=%d want=%d"
          % ("OK  " if ok else "BAD ", os.path.basename(path), pat[:46], got, want))

print("-" * 68)
print("编译校验:")
files = [OUT + "/eva_adapter.py"] + [
    os.path.join(EXP, n) for n in
    ("eva_names.py", "data.py", "train.py", "train_subset.py",
     "importance_evaluation.py", "data_subset.py")
] + [os.path.join(BASE, n) for n in
     ("GM-Matching-main/run_classify_coreset.py",
      "COPS-main/uncertainty_generation.py",
      "COPS-main/retraining_withy.py",
      "COPS-main/retraining_woy.py")]
for f in files:
    try:
        py_compile.compile(f, doraise=True)
        print("  [OK ] " + f.replace(BASE, "<base>").replace(OUT, "<out>"))
    except Exception as exc:
        bad += 1
        print("  [ERR] %s -> %r" % (f, exc))

print("-" * 68)
sys.path.insert(0, OUT)
import eva_adapter as E
print("ADAPTER_DATASETS = %r" % (E.ADAPTER_DATASETS,))
print("REGISTRY 覆盖检查:", all(d in E.REGISTRY for d in E.ADAPTER_DATASETS))
print("=" * 68)
print("BAD=%d" % bad)
sys.exit(1 if bad else 0)
