"""校验 main0.py / main1.py 的数据集登记是否可用。

只做静态解析 + 函数解析，不启动任何训练，因此几秒内可判定。

检查四件事：
  1. dataset_map 里确有 9 个目标数据集，且 argparse 的 choices 也包含它们
     （只有 map 没有 choices，命令行会在参数解析阶段就被拒）。
  2. 每个映射到的加载函数确实存在于对应模块（防止名字打错，直到运行时才炸）。
  3. 该函数来自 *_modified（不是 mydatasets/__init__ 暴露的原始版本）。
     原始 cifar100.py 把 num_classes 写死成 10，covtype.py 指向作者机器路径。
  4. 函数的第 2 个位置参数不是 permuted 之类的东西。
     main0/main1 用 dataset_func(args.data_path, args.balance_target) 位置传参，
     原始 MNIST 签名是 MNIST(data_path, permuted=False, ...)，
     --balance_target 1 会绑到 permuted 上**静默启用像素置换**。
"""
import ast
import importlib
import inspect
import io
import os
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

TARGETS = {
    "cifar100": "CIFAR100",
    "covtype": "covtype",
    "fashionmnist": "FashionMNIST",
    "HR": "HR",
    "mnist": "MNIST",
    "qmnist": "QMNIST",
    "SVHN": "SVHN",
    "tinyimagenet": "TinyImageNet",
    "mnist2mnistm": "Mnist2MnistM",
}

MODULE_OF = {
    "cifar100": "mydatasets.cifar100_modified",
    "covtype": "mydatasets.covtype_modified",
    "fashionmnist": "mydatasets.fashionmnist_modified",
    "HR": "mydatasets.HR_modified",
    "mnist": "mydatasets.mnist_modified",
    "qmnist": "mydatasets.qmnist_modified",
    "SVHN": "mydatasets.svhn_modified",
    "tinyimagenet": "mydatasets.tinyimagenet_modified",
    "mnist2mnistm": "mydatasets.mnist2mnistm_modified",
}

bad = 0


def ok(cond, msg, detail=""):
    global bad
    if not cond:
        bad += 1
    print("  [%s] %s %s" % ("OK  " if cond else "BAD ", msg, detail))


def read_map_and_choices(path):
    with io.open(path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), path)
    mp, choices = None, None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            tgt = node.targets[0]
            if isinstance(tgt, ast.Name) and tgt.id == "dataset_map":
                mp = {}
                for k, v in zip(node.value.keys, node.value.values):
                    if hasattr(k, "value") and isinstance(v, ast.Name):
                        mp[k.value] = v.id
            if isinstance(tgt, ast.Name) and tgt.id == "datasets":
                if isinstance(node.value, ast.List):
                    choices = [e.value for e in node.value.elts
                               if isinstance(e, ast.Constant)]
    return mp, choices


for main_name in ("main0.py", "main1.py"):
    path = os.path.join(ROOT, main_name)
    print("=" * 70)
    print(main_name)
    print("=" * 70)
    if not os.path.exists(path):
        ok(False, "文件存在", path)
        continue
    mp, choices = read_map_and_choices(path)
    ok(mp is not None, "解析出 dataset_map", "(共 %d 项)" % len(mp or {}))
    ok(choices is not None, "解析出 datasets(choices)", "(共 %d 项)" % len(choices or []))
    if mp is None or choices is None:
        continue

    for ds, expect_fn in TARGETS.items():
        in_map = ds in mp
        in_choices = ds in choices
        ok(in_map and in_choices,
           "%-13s 已登记" % ds,
           "map=%s choices=%s" % (in_map, in_choices))
        if not in_map:
            continue
        ok(mp[ds] == expect_fn,
           "%-13s 映射到 %s" % (ds, expect_fn),
           "实际 %s" % mp[ds])

        # 动态解析：确认模块里真的有这个函数，且来自 *_modified
        modname = MODULE_OF[ds]
        try:
            mod = importlib.import_module(modname)
        except Exception as exc:
            ok(False, "%-13s 导入 %s" % (ds, modname), repr(exc))
            continue
        fn = getattr(mod, expect_fn, None)
        ok(fn is not None and callable(fn),
           "%-13s %s.%s 可调用" % (ds, modname.split(".")[-1], expect_fn))
        if fn is None:
            continue

        src_mod = getattr(fn, "__module__", "") or ""
        ok("modified" in src_mod,
           "%-13s 来自 modified 实现" % ds, "__module__=%s" % src_mod)

        try:
            params = list(inspect.signature(fn).parameters)
        except (TypeError, ValueError):
            params = []
        # 第 2 个位置参数必须能安全接收 balance_target（1/0）
        second = params[1] if len(params) > 1 else None
        ok(second is not None and second not in ("permuted",),
           "%-13s 第2位置参数安全" % ds,
           "签名%s，第2参数=%r" % (tuple(params), second))

print("=" * 70)
print("BAD=%d" % bad)
sys.exit(1 if bad else 0)
