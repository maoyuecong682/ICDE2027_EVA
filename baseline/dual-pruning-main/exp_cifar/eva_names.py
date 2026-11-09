"""DUAL 的 EVA 数据集名，并修正 eva_adapter 的导入路径。

eva_adapter.py 的真身在 /root/eva_output/，仓库根目录只有软链；而 DUAL 的脚本在
exp_cifar/ 下启动，sys.path[0] 是 exp_cifar，直接 `import eva_adapter` 会 ImportError。
data.py 里 `_eva_names()` 原本用 `except Exception: return set()` 兜底，会把它吞成
空集，最终只在很深处报 "Dataset not supported"。故集中在这里，导入失败即显式抛错。
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), _HERE,
           os.environ.get("EVA_ADAPTER_DIR", "/root/eva_output")):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import eva_adapter as _EVA
except Exception as _exc:  # 不再静默
    raise RuntimeError(
        "eva_adapter 导入失败（真身应在 /root/eva_output/eva_adapter.py）: %r" % (_exc,)
    )

# 走适配层的九个数据集（含 cifar100，理由见 eva_adapter.ADAPTER_DATASETS 的注释）
ADAPTER_DATASETS = list(_EVA.ADAPTER_DATASETS)

# 保留 DUAL 原生分支的数据集：cifar10 与 adult 不在本次 9 数据集范围内，
# 且已有历史结果，不动。
# cifar10 / adult 已改为走适配层（见 patch_routing.py），原生分支只在
# 适配层不覆盖时才会命中，所以这里不再保留任何原生数据集。
NATIVE_DATASETS = []

# 给 argparse 的 choices 用
ALL_DATASETS = list(dict.fromkeys(NATIVE_DATASETS + ADAPTER_DATASETS))


def is_adapter_dataset(name):
    return name in ADAPTER_DATASETS
