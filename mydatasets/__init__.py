"""mydatasets 包的统一入口。

imdb / news / imdb_large 依赖 transformers（BERT 分词器）。它们的依赖与其余数据集
完全无关，但原来是**无条件**的 `from mydatasets.xxx import *`：

    一旦环境里没有 transformers，整个包 import 失败，
    连 cifar100 / mnist / covtype 都用不了。

表现是 `main0.py` 第一行 import 就崩，而错误信息指向 imdb/news ——
很容易误判成「数据集坏了」而不是「少装了一个包」。

这里改成缺依赖时只降级这三个名字：真正用到它们时才抛出带原始原因的 ImportError，
其余数据集完全不受影响。装了 transformers 的环境行为与原来完全一致。
"""
import importlib as _importlib


def _import_star(module_name):
    """等价于 `from mydatasets.<module_name> import *` 的公开名字集合。"""
    mod = _importlib.import_module("mydatasets." + module_name)
    return {k: v for k, v in vars(mod).items() if not k.startswith("_")}


def _guard(module_name, exported_name, hint="transformers"):
    """导入 module_name；失败时只把 exported_name 换成一个会明确报错的占位函数。"""
    try:
        globals().update(_import_star(module_name))
    except ImportError as exc:
        def _stub(*args, **kwargs):
            raise ImportError(
                "%s 数据集需要 %s，当前环境缺失：%r。安装方式：pip install %s"
                % (exported_name, hint, exc, hint)
            )
        _stub.__name__ = exported_name
        globals()[exported_name] = _stub


# ---- 不依赖 transformers 的数据集：正常导入 ------------------------------
from mydatasets.cifar10 import *
from mydatasets.adult import *
from mydatasets.covtype import *
from mydatasets.mnist import *
from mydatasets.cifar100 import *
from mydatasets.number import *
from mydatasets.mnist2mnistm import *
from mydatasets.HR import *
from mydatasets.House import *
from mydatasets.tinyimagenet import *
from mydatasets.bike import *

# ---- 可能缺依赖的数据集：缺了就只降级自己 ---------------------------------
_guard("imdb", "imdb")
_guard("news", "news")
_guard("imdb_large", "IMDBLarge")
