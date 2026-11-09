"""nets 包的统一入口。

vit / bert 依赖 timm / transformers，属于「可选骨干」。原来是无条件 import，
于是环境里缺 timm 时**整个 nets 包都导入失败**，`main0.py` 里的
`from nets import *` 直接崩，连 ResNet18 这种完全不需要 timm 的骨干也用不了。
错误信息还指向 vit.py，很容易误判成「代码坏了」。

这里改成：可选骨干缺依赖时只跳过它自己，并把原因记进 MISSING_BACKBONES。
装了 timm / transformers 的环境行为与原来完全一致。
"""
MISSING_BACKBONES = {}

from .alexnet import *
from .inceptionv3 import *
from .lenet import *
from .mlp import *
from .mobilenetv3 import *
from .resnet import *
from .vgg import *
from .wideresnet import *
from .proxy_mlp import *
from .text_mlp import *
from .re_mlp import *
from .cnn import *
from .lstm import *
from .tabnet import *
from .relstm import *

# ---- 可选骨干：缺依赖只跳过自己 -------------------------------------------
# 这三个文件各自依赖一个「本项目其它骨干完全不需要」的包，是可选的：
#   vit.py            -> timm
#   bert.py           -> transformers
#   tabtransformer.py -> tab_transformer_pytorch
for _name, _pkg in (("vit", "timm"),
                    ("bert", "transformers"),
                    ("tabtransformer", "tab_transformer_pytorch")):
    try:
        _mod = __import__("%s.%s" % (__name__, _name), fromlist=["*"])
        globals().update({k: v for k, v in vars(_mod).items()
                          if not k.startswith("_")})
    except ImportError as _exc:      # 通常是没装 _pkg
        MISSING_BACKBONES[_name] = repr(_exc)

if MISSING_BACKBONES:
    # 只提示一次，不阻断：主实验用的 ResNet18 / MLP 等不受影响
    import warnings as _w
    _w.warn("nets: 以下可选骨干因缺依赖被跳过 %r（如需使用请安装对应包）"
            % (MISSING_BACKBONES,))
