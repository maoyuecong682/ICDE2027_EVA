#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""eval_baselines.py —— 把三个 baseline 选出的核心集，用 EVA 的三套训练例程各评一次。

命名与列定义严格按 `/root/autodl-tmp/EVA/evaluation/命名和文件存放规则.md`：

    autodl-tmp/EVA/evaluation/<baseline 名字>/<baseline 名字>_result.csv

第一列 数据集
第 2-5  列  methods/random.py  的 Accuracy / Precision / Recall / F1
第 6-9  列  methods/full.py    的 Accuracy / Precision / Recall / F1
第 10-13 列 methods/kmeans.py  的 Accuracy / Precision / Recall / F1
第 14-15 列 MSE / MAE（仅回归任务，本仓库全部是分类 → 恒为空）

为什么是"三套例程"
------------------
三个 baseline 只负责**选核心集**，本身不产出可比指标（COPS 只 print 准确率、
GM 写自己的 csv、DUAL 只存 mask）。要让三者横向可比，必须用同一套训练/评估代码去评
同一个子集。仓库里恰好有三份 `train_model` 实现（random / full / kmeans），
彼此的 batch、patience、checkpoint 选择口径略有差异，于是各评一次、三组数字都填。

三者的口径差异（照实记录，不做统一，统一了就失去"三套"的意义）
* random.py: batch 256，WeightedCrossEntropyLoss，**每 epoch 在 test 上选最优**（偏乐观）
* full.py  : batch 128，CrossEntropy，在 **val** 上选最优 + 早停，最后报 test
* kmeans.py: batch 128，CrossEntropy，在 **val** 上选最优 + 早停，支持回归(MSE/MAE)

用法
----
    python eval_baselines.py --baseline COPS
    python eval_baselines.py --baseline GM_matching --datasets mnist covtype
    python eval_baselines.py --baseline DUAL --epochs 100

核心集索引从哪来（这一步是关键，不统一就不可比）
* COPS  : <output>/COPS/<ds>/weights_cops_wy/selected_indices.npy
          回退 weights_uniform/ 与 weights_cops_woy/（三种变体，用 --cops_variant 选）
* GM    : <output>/GM_matching/<ds>/embeds/<ds>/coreset_indices_rep*.npy（每次重复一份）
* DUAL  : 由 <output>/DUAL/<ds>/checkpoint/<ds>/42/generated_mask/dual_mask_T30.npy
          取分值最高的 round(fraction * n_train) 个 —— DUAL 只存 mask，不存索引
"""
import argparse
import csv
import glob
import importlib.util
import json
import os
import sys
import types

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

EVA_ROOT = os.environ.get("EVA_ROOT", "/root/chaichengliang/VLDB2027/EVA/EVA")
OUT_ROOT = os.environ.get("EVA_OUTPUT", "/root/autodl-tmp/EVA/output")
EVAL_ROOT = os.environ.get("EVA_EVAL", "/root/autodl-tmp/EVA/evaluation")
DATA_ROOT = os.environ.get("EVA_DATA_ROOT", "/root/autodl-tmp/EVA/data")

COLUMNS = [
    "dataset",
    "random_accuracy", "random_precision", "random_recall", "random_f1",
    "full_accuracy", "full_precision", "full_recall", "full_f1",
    "kmeans_accuracy", "kmeans_precision", "kmeans_recall", "kmeans_f1",
    "kmeans_mse", "kmeans_mae",
]


# --------------------------------------------------------------------------- #
# 复用 eval_runner 的 CPU 兼容层与 nets 桩
# --------------------------------------------------------------------------- #
def _load_eval_runner():
    path = os.path.join(EVA_ROOT, "evaluation", "eval_runner.py")
    spec = importlib.util.spec_from_file_location("_eva_eval_runner", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


ER = _load_eval_runner()
# eva_adapter.py 的正本在 /root/eva_output/（三个 baseline 仓库软链接它）。
# 评估脚本本身在 /root/autodl-tmp/EVA/evaluation/，不在同一目录，必须显式加进来。
ADAPTER_DIR = os.environ.get("EVA_ADAPTER_DIR", "/root/eva_output")
if ADAPTER_DIR not in sys.path:
    sys.path.insert(0, ADAPTER_DIR)
sys.path.insert(0, os.path.join(EVA_ROOT, "evaluation"))
import eva_adapter as EVA  # noqa: E402


# 当前正在评估的数据集的 meta。methods/*.py 通过 nets[arch](...) 建骨干时拿不到 meta，
# 表格数据会被错建成 ResNet18（13 维特征进 conv2d 直接崩）。三套例程串行执行，
# 每个例程入口把它设成本次评估的 meta，ResNet18Stub 读到后统一走 EVA.build_model。
_META_FOR_STUB = None


class ResNet18Stub(nn.Module):
    """补上 `nets` 桩里缺的图像骨干。

    `methods/*.py` 统一用 `nets.__dict__[arch](channel=, num_classes=, im_size=, pretrained=)`
    构造模型。真实 `nets/resnet.py` 依赖 timm，本容器没装，所以这里给一个同签名、
    与 eva_adapter 三个 baseline 共用的同一份 ResNet18。

    表格数据必须落在 TabMLP 上：HR/covtype 的样本是 (13,)/(54,) 的一维特征，若照
    ResNet18 建，第一个 conv2d 会收到 [128, 13] 并抛
        RuntimeError: Expected 3D (unbatched) or 4D (batched) input to conv2d
    而 methods/*.py 只把 channel/im_size 传进来，看不到 meta，无法自己判断形态。
    因此用 _META_FOR_STUB 把当前数据集的 meta 快照传进来 —— 三套例程是串行执行的，
    不存在并发覆盖。
    """

    def __init__(self, channel=3, num_classes=10, im_size=(32, 32),
                 pretrained=False, record_embedding=False, no_grad=False):
        super(ResNet18Stub, self).__init__()
        if pretrained:
            raise NotImplementedError("Pretrained models not available.")
        if _META_FOR_STUB is not None:
            # 统一骨干：图像 resnet18、表格 TabMLP，与 COPS / DUAL 完全同一份
            self.net = EVA.build_model(_META_FOR_STUB, arch=None,
                                       num_classes=int(num_classes))
            return
        size = im_size[0] if isinstance(im_size, (tuple, list)) else im_size
        self.net = EVA._resnet18(int(channel), int(num_classes), small_input=int(size) <= 64)

    def forward(self, x):
        return self.net(x)

    def get_last_layer(self):
        return self.net.fc


def _install_stubs():
    ER._install_stubs()
    nets = sys.modules["nets"]
    nets.ResNet18 = ResNet18Stub
    nets.ResNet18Stub = ResNet18Stub


# --------------------------------------------------------------------------- #
# 核心集索引解析
# --------------------------------------------------------------------------- #
def _cops_indices(ds, variant):
    base = os.path.join(OUT_ROOT, "COPS", ds)
    order = [variant]  # A requested variant must never silently become another method.
    for v in order:
        p = os.path.join(base, v, "selected_indices.npy")
        if os.path.exists(p):
            local = np.load(p)
            _, sampling, _, _ = EVA.cops_views(ds)
            if local.dtype.kind not in "iu" or local.size == 0 or local.min() < 0 or local.max() >= len(sampling):
                raise ValueError("Invalid COPS sampling-pool indices: " + p)
            return np.asarray(sampling.indices)[local], p
    return None, None


def _gm_indices(ds, rep=0):
    pats = [
        os.path.join(OUT_ROOT, "GM_matching", ds, "embeds", ds, "coreset_indices_rep*.npy"),
        os.path.join(OUT_ROOT, "GM_matching", ds, "embeds", "coreset_indices_rep*.npy"),
    ]
    for pat in pats:
        hits = sorted(glob.glob(pat))
        if hits:
            p = hits[min(rep, len(hits) - 1)]
            return np.load(p), p
    return None, None


def _dual_indices(ds, meta, fraction):
    base = os.path.join(OUT_ROOT, "DUAL", ds, "checkpoint", ds)
    cands = glob.glob(os.path.join(base, "**", "dual_mask_T30.npy"), recursive=True)
    if not cands:
        return None, None
    score_path = os.path.join(os.path.dirname(cands[0]), "dual_score_T30.npy")
    score = np.load(score_path).reshape(-1)
    n_train = len(meta["train"])

    # mask 的第 i 项必须对应 train 的第 i 个样本。长度不一致时这个对应关系不成立，
    # 取 top-k 会选到别的样本上，而且**不会报错**。最常见的成因是：DUAL 在
    # EVA_SMOKE_LIMIT 下跑（train 被截到 2000），评估时却没设（train 是 22120）——
    # 原先 k=min(k, len(score)) 会静默地把整张 mask 全选走。
    if score.size != n_train:
        print("[ERROR] DUAL/%s mask 长度 %d != train 长度 %d。"
              "请确认选核心集与评估用同一个 EVA_SMOKE_LIMIT（正式实验应为空）。mask=%s"
              % (ds, score.size, n_train, cands[0]), flush=True)
        return None, cands[0]

    k = max(1, int(round(fraction * n_train)))
    idx = np.argsort(-score)[:k]          # DUAL 的 mask 分值越高越该保留
    return np.sort(idx).astype(np.int64), cands[0]


# --------------------------------------------------------------------------- #
# 数据集包装（三套例程的 loader 期待不同元数）
# --------------------------------------------------------------------------- #
class _Plain(Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        x, y = self.ds[i]
        return x, y


class _Weighted(_Plain):
    def __getitem__(self, i):
        x, y = self.ds[i]
        return x, int(y), 1.0


# --------------------------------------------------------------------------- #
# 三套例程
# --------------------------------------------------------------------------- #
def _acc_of(model, ds, meta, bs=256):
    """[val-acc-2026-10-27] 在给定数据集上算 accuracy（只用于按 val 选 lr）。

    full.py 的 evaluate_model 是 full_selection 里的嵌套函数，模块级取不到；
    这里自己算一个标量就够了，不做完整指标。
    """
    device = next(model.parameters()).device
    model.eval()
    ok = tot = 0
    with torch.no_grad():
        for x, y in DataLoader(_Plain(ds), batch_size=bs, shuffle=False,
                               num_workers=_nw(meta)):
            x = x.to(device)
            pred = model(x).argmax(dim=1).cpu()
            ok += int((pred == y).sum().item())
            tot += int(y.numel())
    return ok / max(1, tot)


def _nw(meta):
    """DataLoader 的 worker 数：图像 4、表格 0、无 GPU 0。

    依据见 patch_workers.py 的模块注释（mnist 2.4x 加速 / covtype 2.6x 减速）。
    另外把上限压到 cpu_count-1，避免在核数很少的机器上开过多进程。
    """
    if not torch.cuda.is_available():
        return 0
    if meta.get("is_tabular"):
        return 0
    return max(0, min(4, (os.cpu_count() or 1) - 1))


def _bs(args, default, coreset=None):
    """批大小：未指定 --batch-size 时沿用各例程原来的默认值。

    为什么需要这个旋钮：coreset 往往只有一两百条，原来的
    `min(256, len(coreset))` 等于**把整个核心集当成一个批次**送进网络。
    在 2GB cgroup 下，ResNet18 对 200 张 28x28 图像的一次前向+反向，
    仅激活就要几百 MB，进程会在第一个 batch 就被 SIGKILL（退出码 137），
    日志停在 `Training model: 0%|`，看不出任何异常。
    小批次不改变训练语义（仍是同样的数据、同样的 epoch），只是慢一点。
    """
    bs = int(getattr(args, "batch_size", 0) or 0)
    if bs > 0:
        return max(1, bs if coreset is None else min(bs, max(1, len(coreset))))
    return default if coreset is None else min(default, max(1, len(coreset)))


def _tune_method(mod, args):
    """把例程内部的批大小/worker 数压到本次评估的口径。

    methods/{random,full,kmeans}.py 自己建 DataLoader，批大小是模块级常量
    （DATA_BATCH_SIZE / DATA_NUM_WORKERS，默认值等于原来的硬编码，由
    evaluation/patch_methods_batch.py 打进去）。只压外层 --batch-size 是不够的：
    实测 mnist/bs=16 时，训练 13 个 batch 全程平稳在 rss≈1.1 GB，一进
    random.py 自带的 `DataLoader(dst_test, batch_size=256, num_workers=4)`，
    rss 立刻 1106 → 1211 → SIGKILL。所以两个旋钮必须一起压。

    worker 只在「显式给了小批大小 + 没有 GPU」时才清零 —— 那是明确的小内存场景；
    默认路径（GPU 机器、未指定 --batch-size）保持原样。
    """
    bs = int(getattr(args, "batch_size", 0) or 0)
    if bs <= 0:
        return
    if hasattr(mod, "DATA_BATCH_SIZE"):
        mod.DATA_BATCH_SIZE = bs
    if hasattr(mod, "DATA_NUM_WORKERS") and not torch.cuda.is_available():
        mod.DATA_NUM_WORKERS = 0


def run_random_routine(coreset, dst_val, dst_test, meta, args, workdir):
    """random.py 例程。

    签名必须和其他两套一致（6 个参数），否则 ROUTINES 的统一调用会抛
    `takes 5 positional arguments but 6 were given`，random 三列永远是空的。
    dst_val 在这里不用：random.py 的训练循环是每 epoch 在 **test** 上选最优
    （见文件头注明的口径差异，这一条照实保留，不做统一）。
    """
    global _META_FOR_STUB
    _META_FOR_STUB = meta
    # [protocol-2026-10-27] 原先 del dst_val 并拿 dst_test 当监控集，
    # 等于用测试集做早停与选模。改为在 val 上监控。
    mod = ER.load_method_module("_m_random", os.path.join(EVA_ROOT, "methods", "random.py"),
                                num_workers=_nw(meta), verbose=False)
    _tune_method(mod, args)
    device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")
    model = EVA.build_model(meta, arch=None).to(device)
    loader = DataLoader(_Weighted(coreset), batch_size=_bs(args, 256, coreset),
                        shuffle=True, num_workers=_nw(meta))
    criterion = mod.WeightedCrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    best = os.path.join(workdir, "random.pth")
    mod.train_model(model, loader, criterion, optimizer, scheduler, args.epochs,
                    args.patience, device, best, _Plain(dst_val))
    # [protocol-2026-10-27] train_model 只落盘 best、不回载；不回载就等于报末轮。
    model.load_state_dict(torch.load(best, map_location=device))
    # 这里的 256 原来是硬编码的：--batch-size 只压了训练 loader，测试 loader 仍是 256。
    # 在 2GB cgroup 下这一点很致命 —— 训练到第一个 epoch 末，cgroup 已到 1.7/2.0 GB，
    # 再对 256 张 28x28 做一次 no_grad 前向（仅激活就要几百 MB）就被 SIGKILL，
    # 而 --batch-size 16 看起来"没有生效"。推理批大小不影响指标，统一走 _bs()。
    acc, prec, rec, f1 = mod.evaluate_model(
        model, DataLoader(_Plain(dst_test), batch_size=_bs(args, 256), shuffle=False,
                          num_workers=_nw(meta)), device)
    # [lr-select-2026-10-27] 供上游按 val 选 lr；val 不是 COLUMNS 的列，不会进 CSV。
    _v = mod.evaluate_model(
        model, DataLoader(_Plain(dst_val), batch_size=_bs(args, 256), shuffle=False,
                          num_workers=_nw(meta)), device)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "mse": "", "mae": "",
            "val": float(_v[0]), "lr": float(args.lr)}


def run_full_routine(coreset, dst_val, dst_test, meta, args, workdir):
    """full.py 的 train_model 是 full_selection 内部的嵌套函数，无法单独 import。

    但它不选核心集——直接训整份 dst_train。所以把 baseline 的核心集当作 dst_train 传进去，
    就等价于"用 full 例程评这个核心集"。指标从它写的日志里解析（与 eval_runner 一致）。
    """
    global _META_FOR_STUB
    _META_FOR_STUB = meta
    device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")
    mod = ER.load_method_module("_m_full", os.path.join(EVA_ROOT, "methods", "full.py"),
                                num_workers=_nw(meta), verbose=False)
    _tune_method(mod, args)
    m_args = ER.build_args(os.path.join(workdir, "full_out"), meta["name"], args.arch,
                           "full", args.fraction, 1, args.epochs, args.patience,
                           args.lr, 0)
    # 注：这里原来试图覆盖 m_args.batch_size，但 methods/*.py 全文都没有引用过
    # args.batch_size —— full.py 的 loader 是 `batch_size=128` 写死的，所以那几行
    # 是死代码，一直没生效。真正的旋钮是 _tune_method() 压的 DATA_BATCH_SIZE。
    ER._reset_root_logger()
    # [shuffle-fix-2026-10-27] full.py 的训练 loader 是 shuffle=False，而核心集
    # 常常按类排序（见文件头）。喂一个固定种子打乱的视图，等价于 shuffle=True。
    _perm = np.random.RandomState(48).permutation(len(coreset))
    _train_view = Subset(coreset, _perm.tolist())
    # [lr-select-2026-10-27] full_selection 结尾会 os.remove(best_model_path)。
    # 为了回载 best 自己算 val（它内部算的 val 只用于选模，不写日志），
    # 调用期间把 os.remove 换成空操作，算完再删。
    _rm, os.remove = os.remove, (lambda *_a, **_k: None)
    try:
        mod.full_selection(m_args, meta["channel"], meta["im_size"], meta["num_classes"],
                           meta["class_names"], meta["mean"], meta["std"],
                           _Plain(_train_view), _Plain(dst_val), _Plain(dst_test))
    finally:
        os.remove = _rm
    ER._reset_root_logger()
    got = ER.parse_metrics("full", ER.expected_log_path(m_args)) or {}
    val = ""
    _bp = (os.path.join(m_args.save_best_model_path, meta["name"],
                        str(m_args.selection_fraction), "full.pth"))
    if os.path.exists(_bp):
        try:
            # [full-arch-2026-10-27] 必须用 full.py 自己的模型构造函数，
            # 而不是 EVA.build_model：tabular 上 full.py 用的是 ResNet18，结构不同。
            _m = mod.nets.__dict__[m_args.model](
                channel=meta["channel"], num_classes=meta["num_classes"],
                im_size=meta["im_size"], pretrained=False).to(device)
            _m.load_state_dict(torch.load(_bp, map_location=device))
            val = float(_acc_of(_m, dst_val, meta, _bs(args, 128)))
        finally:
            try:
                _rm(_bp)
            except OSError:
                pass
    return {"accuracy": got.get("accuracy", ""), "precision": got.get("precision", ""),
            "recall": got.get("recall", ""), "f1": got.get("f1", ""), "mse": "", "mae": "",
            "val": val, "lr": float(args.lr)}


def run_kmeans_routine(coreset, dst_val, dst_test, meta, args, workdir):
    import time
    global _META_FOR_STUB
    _META_FOR_STUB = meta
    mod = ER.load_method_module("_m_kmeans", os.path.join(EVA_ROOT, "methods", "kmeans.py"),
                                num_workers=_nw(meta), verbose=False)
    _tune_method(mod, args)
    device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")
    # 回归（bike/House）：1 个输出 + MSELoss。评估时适配层没有 EVA_REG_BINS，
    # 所以 meta["num_classes"] 仍是 1，build_model 自然给出 1 维输出。
    reg = bool(meta.get("is_regression"))
    task = "regression" if reg else "classification"
    model = EVA.build_model(meta, arch=None).to(device)
    loader = DataLoader(_Plain(coreset), batch_size=_bs(args, 128, coreset),
                        shuffle=True, num_workers=_nw(meta))
    criterion = (nn.MSELoss() if reg else nn.CrossEntropyLoss()).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                weight_decay=5e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    best = os.path.join(workdir, "kmeans.pth")
    if device.type == 'cuda': torch.cuda.synchronize()
    train_started = time.perf_counter()
    train_info = mod.train_model(model, loader, criterion, optimizer, scheduler, args.epochs, device,
                    best, task, DataLoader(_Plain(dst_val), batch_size=_bs(args, 128),
                                           shuffle=False, num_workers=_nw(meta)),
                    args.patience)
    if device.type == 'cuda': torch.cuda.synchronize()
    train_sec = time.perf_counter() - train_started
    # [protocol-2026-10-27] 同上：回载 best 权重，与 full.py 的 full_selection 对齐。
    model.load_state_dict(torch.load(best, map_location=device))
    metrics_started = time.perf_counter()
    out = mod.evaluate_model(model, DataLoader(_Plain(dst_test), batch_size=_bs(args, 128),
                                               shuffle=False, num_workers=_nw(meta)),
                             device, task, return_details=True)
    if device.type == 'cuda': torch.cuda.synchronize()
    metrics_sec = time.perf_counter() - metrics_started
    # [lr-select-2026-10-27] 回归时 out[0] 是 MSE（越小越好），分类时是 acc。
    _v = mod.evaluate_model(model, DataLoader(_Plain(dst_val), batch_size=_bs(args, 128),
                                              shuffle=False, num_workers=_nw(meta)),
                            device, task)
    _val = float(_v[0])
    details = dict(train_info or {},train_sec=train_sec,test_predict_metrics_sec=metrics_sec)
    if reg:
        mse, mae = out[0], out[1]
        print("    [regress] MSE=%.4f MAE=%.4f" % (mse, mae), flush=True)
        return {"accuracy": "", "precision": "", "recall": "", "f1": "",
                "mse": float(mse), "mae": float(mae), "val": _val, "lr": float(args.lr), **details}
    if isinstance(out, dict):
        return dict(out, mse='',mae='',val=_val,lr=float(args.lr),**details)
    if len(out) == 4:
        acc, prec, rec, f1 = out
        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "mse": "", "mae": "",
                "val": _val, "lr": float(args.lr)}
    acc, prec, rec, f1 = out[:4]
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "mse": out[4], "mae": out[5], "val": _val, "lr": float(args.lr)}


ROUTINES = {"random": run_random_routine, "full": run_full_routine, "kmeans": run_kmeans_routine}


# --------------------------------------------------------------------------- #
# 内存前置检查
# --------------------------------------------------------------------------- #
def _mem_report():
    """打印 cgroup 用量，并在余量明显不够时把话说在前面。

    为什么不靠事后看日志：进程被 cgroup 配额杀掉时 `memory.events` 里
    `oom_kill` 可能是 0（走到的是另一条 kill 路径），而 Python 侧没有任何异常，
    日志就停在 `Training model: 0%|`，看起来像"卡住了"或"批大小不对"。
    实测这台机器：torch 导入 474 MB + 数据集全量 207 MB + 方法模块 114 MB
    + 参数/梯度/动量 135 MB ≈ 930 MB 就是地板；空闲余量不足 1.1 GB 时图像数据
    没有跑通的可能，与其试三次不如在这里说清楚。
    """
    try:
        with open("/sys/fs/cgroup/memory.current") as fh:
            cur = int(fh.read().strip())
        with open("/sys/fs/cgroup/memory.max") as fh:
            mx = fh.read().strip()
    except Exception:
        return
    if mx == "max":
        return
    try:
        mx = int(mx)
    except ValueError:
        return
    free = (mx - cur) / 1048576.0
    print("[mem] cgroup %d/%d MB，余量 %.0f MB（含其他进程与页缓存）"
          % (cur / 1048576, mx / 1048576, free), flush=True)
    if free < 1150:
        print("[mem][WARN] 余量不足 1.15 GB：图像数据集的评估大概率会在第一个 batch "
              "被 SIGKILL（退出码 137，日志停在 `Training model: 0%|`）。"
              "请改用单例程分进程（--routines <一个> + --resume）并给 --batch-size 16。",
              flush=True)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="用 EVA 三套例程评估 baseline 核心集")
    ap.add_argument("--baseline", required=True, choices=["COPS", "GM_matching", "DUAL"])
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--routines", nargs="*", default=["random", "full", "kmeans"])
    ap.add_argument("--cops_variant", default="weights_cops_wy")
    ap.add_argument("--gm_rep", type=int, default=0)
    ap.add_argument("--fraction", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=0,
                    help="批大小；0=沿用各例程默认（random 256 / full 128 / kmeans 128）。"
                         "小内存机器请显式给一个小值，例如 16")
    ap.add_argument("--patience", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--arch", default="ResNet18")
    ap.add_argument("--resume", action="store_true",
                    help="已有 CSV 存在时读入并只覆盖本次跑的例程列。"
                         "小内存机器上三套例程要分三个进程跑（每套一个干净进程），"
                         "不加这个开关后一个进程会把前一个进程写好的列全清空。")
    args = ap.parse_args()

    # 固定评估阶段的随机性。核心集索引本身是确定的，但评估模型的初始化、
    # DataLoader 的 shuffle、以及 methods/*.py 内部的随机源若不定种子，
    # 同一份核心集重跑会给出 ±0.01 的 MSE 差异（实测 bike/COPS：0.2614 vs 0.2694）——
    # 这种抖动和三个方法之间的真实差距同量级，会让排序变得不可信。
    import random as _rnd
    torch.manual_seed(48)
    np.random.seed(48)
    _rnd.seed(48)

    _install_stubs()
    _mem_report()
    datasets = args.datasets or ["cifar100", "covtype", "fashionmnist", "HR", "mnist",
                                 "qmnist", "SVHN", "tinyimagenet", "mnist2mnistm"]
    out_dir = os.path.join(EVAL_ROOT, args.baseline)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "%s_result.csv" % args.baseline)
    workdir = os.path.join(out_dir, "_work")
    os.makedirs(workdir, exist_ok=True)

    rows = []
    by_ds = {}
    if args.resume and os.path.exists(csv_path):
        with open(csv_path, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                row = {c: r.get(c, "") for c in COLUMNS}
                rows.append(row)
                by_ds[row["dataset"]] = row
        print("[resume] 读入已有 %d 行：%s"
              % (len(rows), ", ".join(r["dataset"] for r in rows)), flush=True)

    def _row_for(ds):
        """取该数据集的行：续跑时复用已有行（保留别的例程列），否则新建。"""
        if ds in by_ds:
            return by_ds[ds]
        row = {c: "" for c in COLUMNS}
        row["dataset"] = ds
        rows.append(row)
        by_ds[ds] = row
        return row

    for ds in datasets:
        if ds not in EVA.DATASETS:
            print("[skip] %s 不在适配层里" % ds, flush=True)
            continue
        meta = EVA.load(ds)
        if args.baseline == "COPS":
            idx, src = _cops_indices(ds, args.cops_variant)
        elif args.baseline == "GM_matching":
            idx, src = _gm_indices(ds, args.gm_rep)
        else:
            idx, src = _dual_indices(ds, meta, args.fraction)

        row = _row_for(ds)

        # 索引越界 = 核心集与评估看的不是同一份 train（多半是 EVA_SMOKE_LIMIT 不一致）。
        # 越界索引会让 EVA.select 抛 IndexError 或悄悄取到别的样本，必须挡在训练之前。
        if idx is not None:
            idx = np.asarray(idx).reshape(-1)
            n_train = len(meta["train"])
            if idx.size == 0 or int(idx.min()) < 0 or int(idx.max()) >= n_train:
                print("[ERROR] %s/%s 索引越界：范围 [%s, %s]，train 只有 %d 条；"
                      "核心集与评估应使用同一个 EVA_SMOKE_LIMIT。"
                      % (args.baseline, ds,
                         idx.min() if idx.size else "-", idx.max() if idx.size else "-",
                         n_train), flush=True)
                idx = None

        if idx is None:
            # src 有值说明「找到了文件但内容不可用」（上面已打 [ERROR]）；
            # src 为空才是真的没有产物。
            if src:
                print("[missing] %s/%s 索引文件存在但不可用，跳过（原因见上）"
                      % (args.baseline, ds), flush=True)
            else:
                print("[missing] %s/%s 没有核心集索引，跳过" % (args.baseline, ds), flush=True)
            continue

        print("[eval] %s/%s coreset=%d (%s)" % (args.baseline, ds, len(idx), src), flush=True)
        coreset = EVA.select(EVA._wrap(meta["train"], meta), idx)
        dst_val = EVA._wrap(meta["val"], meta)
        dst_test = EVA._wrap(meta["test"], meta)

        # [reg-gate-2026-10-27] 回归集只跑 kmeans：random/full 是分类配方，
        # int(y) 会给出非法标签，CUDA device-side assert 之后整个上下文作废。
        _routines = list(args.routines)
        if meta.get("is_regression"):
            _routines = [r for r in _routines if r == "kmeans"]
            if _routines != list(args.routines):
                print("    [reg] %s 是回归任务，跳过 %s，只跑 kmeans"
                      % (ds, [r for r in args.routines if r != "kmeans"]), flush=True)
        # [lr-select-2026-10-27] tabular 数据集按 val 在 {0.1, 0.01} 里选 lr；
        # 图像数据集沿用仓库的 CNN 配方（0.1），不翻倍评估时间。
        _lrs = [float(args.lr)]
        if meta.get("is_tabular") and abs(float(args.lr) - 0.1) < 1e-12:
            _lrs = [0.1, 0.01]
        _lr_used = {}
        for r in _routines:
            got = None
            vbest = None
            for _lr in _lrs:
                args.lr = _lr
                # [val-acc-2026-10-27] 单元格级置种：结果只取决于(数据集,例程,lr)，
                # 不受同一进程里前置例程消耗随机数的影响。
                torch.manual_seed(48)
                np.random.seed(48)
                _rnd.seed(48)
                try:
                    cand = ROUTINES[r](coreset, dst_val, dst_test, meta, args, workdir)
                except Exception:
                    import traceback
                    traceback.print_exc()
                    cand = {"accuracy": "", "precision": "", "recall": "", "f1": "",
                            "mse": "", "mae": "", "val": "", "lr": _lr}
                v = cand.get("val", "")
                if len(_lrs) > 1:
                    print("      [lr] %-6s lr=%-5g val=%s" % (
                        r, _lr, ("%.4f" % v) if isinstance(v, float) else "n/a"), flush=True)
                if got is None:
                    got, vbest = cand, v
                elif isinstance(v, float) and (vbest is None or (
                        (v < vbest) if meta.get("is_regression") else (v > vbest))):
                    got, vbest = cand, v
            args.lr = _lrs[0]
            _lr_used[r] = got.get("lr", _lrs[0])
            if len(_lrs) > 1:
                print("    [lr] %-6s 采用 lr=%g（val=%s）" % (
                    r, _lr_used[r],
                    ("%.4f" % vbest) if isinstance(vbest, float) else "n/a"), flush=True)
            for k in ("accuracy", "precision", "recall", "f1", "mse", "mae"):
                key = "%s_%s" % (r, k)
                if key not in COLUMNS:
                    # 命名规则只给 kmeans 留了 MSE/MAE 两列（回归用）。random/full
                    # 的 mse/mae 即使算出来也没有列位。若照写进 row，DictWriter 会抛
                    #   ValueError: dict contains fields not in fieldnames
                    # 然后整个 CSV 一行都写不出来 —— 前面几十分钟的训练全白跑。
                    continue
                v = got.get(k, "")
                row[key] = round(float(v), 4) if isinstance(v, (int, float)) else v
            print("    %-7s %s" % (r, got), flush=True)

        _write_csv(csv_path, rows, datasets)

    print("[done] %s" % csv_path, flush=True)


def _write_csv(csv_path, rows, order):
    """按固定数据集顺序落盘。续跑时新数据集是追加到末尾的，直接写会让行序随
    执行历史变化；科研表格按固定顺序读起来才不会每次 diff 都变。"""
    rank = {d: i for i, d in enumerate(order)}
    rows = sorted(rows, key=lambda r: rank.get(r.get("dataset"), len(rank)))
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows({c: r.get(c, "") for c in COLUMNS} for r in rows)
        fh.flush()
        os.fsync(fh.fileno())


if __name__ == "__main__":
    main()
