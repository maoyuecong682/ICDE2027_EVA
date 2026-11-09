"""三个 baseline 的「接缝级」冒烟：只验证数据/骨干/子集这条链是否通电。

为什么不用跑完整实验来验证「能不能在 CPU 上跑」：
  0.5 核配额下 mnist 全量一轮训练就要半小时，9 数据集 x 3 baseline 是几十天量级。
  真正容易出错、也最值得先验证的，是适配层与各仓库之间的接缝：
      * 数据集能否被各仓库的 loader 认出来（argparse choices / 分支分派）
      * 标签类型对不对（GM 用标签做 dict key，0 维 tensor 会退化成 N 个类）
      * 骨干能否吃下这份数据（单通道 / 表格 / 小分辨率）
      * 核心集索引能否被正确还原成子集
  这些都在几十秒内可判定，且与训练时长无关。

每个 baseline 必须独立进程运行：三个仓库都有 data.py / models，
同进程内 import 会互相覆盖 sys.modules。

用法：
    python smoke_seams.py --baseline cops|gm|dual --dataset mnist
"""
import argparse
import os
import sys
import types

BASE = "/root/autodl-tmp/EVA/baseline"
OUT = "/root/eva_output"
for _p in (OUT, BASE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch

torch.set_num_threads(1)

RESULT = []


def _enter(path):
    """切到某个仓库目录并把「真正的 import 根」放进 sys.path。

    只 chdir 是不够的：sys.path[0] 是本脚本所在的 /root/eva_output，
    各仓库的 data.py / models 在 chdir 之后依然 import 不到。
    """
    os.chdir(path)
    if path not in sys.path:
        sys.path.insert(0, path)
    return path


def check(name, fn):
    try:
        detail = fn()
        RESULT.append((True, name, detail))
        print("  [PASS] %-34s %s" % (name, detail))
    except Exception as exc:
        import traceback
        RESULT.append((False, name, repr(exc)))
        print("  [FAIL] %-34s %r" % (name, exc))
        traceback.print_exc()


def _fwd(model, ds, idx=(0,), bs=4):
    """从 ds 取 bs 个样本跑一次前向，返回输出形状。"""
    xs, ys = [], []
    for i in idx:
        x, y = ds[i]
        xs.append(x)
        ys.append(int(y))
    x = torch.stack(xs)
    y = torch.tensor(ys, dtype=torch.long)
    out = model(x)
    assert out.shape[0] == x.shape[0], "前向输出行数不符"
    assert out.shape[1] == int(model.out_dim) if hasattr(model, "out_dim") else True
    return "in=%s out=%s" % (tuple(x.shape), tuple(out.shape))


# --------------------------------------------------------------------------- #
# COPS
# --------------------------------------------------------------------------- #
def smoke_cops(ds_name):
    _enter(os.path.join(BASE, "COPS-main"))
    import eva_adapter as EVA

    def v_sizes():
        probe, sampling, test, meta = EVA.cops_views(ds_name)
        assert len(probe) > 0 and len(sampling) > 0 and len(test) > 0
        assert len(probe) + len(sampling) == len(EVA.Wrapped(meta["train"], meta)), \
            "probe + sampling 应恰好等于 train"
        a = set(np.asarray(probe.indices).tolist())
        b = set(np.asarray(sampling.indices).tolist())
        assert not (a & b), "probe 与 sampling 重叠，不确定性分数会泄漏"
        return "probe=%d sampling=%d test=%d 交集=0" % (len(probe), len(sampling), len(test))

    def v_model():
        probe, sampling, test, meta = EVA.cops_views(ds_name)
        # arch=None：与 uncertainty_generation.py 保持一致，交给适配层按形态选骨干。
        # 写死 resnet18 会让表格数据集（covtype/HR）在 conv2d 处直接崩。
        net = EVA.build_model(meta, arch=None, num_classes=meta["num_classes"])
        return _fwd(net, probe)

    check("cops_views 划分", v_sizes)
    check("骨干前向", v_model)


# --------------------------------------------------------------------------- #
# GM-Matching
# --------------------------------------------------------------------------- #
def smoke_gm(ds_name):
    _enter(os.path.join(BASE, "GM-Matching-main"))
    import eva_adapter as EVA
    from data import get_data

    state = {}

    def v_load():
        tr, nc = get_data(ds_name, "train")
        te, _ = get_data(ds_name, "test")
        state["tr"], state["te"], state["nc"] = tr, te, nc
        assert len(tr) > 0 and len(te) > 0
        return "train=%d test=%d n_classes=%d" % (len(tr), len(te), nc)

    def v_label_type():
        tr = state["tr"]
        _, y = tr[0]
        assert isinstance(y, int) and not isinstance(y, bool), \
            "标签必须是 python int（GM 用标签做 dict key），实际 %r" % (type(y),)
        # 抽样确认不是 0 维 tensor
        for i in (0, len(tr) // 2, len(tr) - 1):
            _, yy = tr[i]
            assert isinstance(yy, int), "第 %d 个样本的标签类型是 %r" % (i, type(yy))
        return "type(label)=int"

    def v_model():
        tr, nc = state["tr"], state["nc"]
        model = EVA.build_model(ds_name, num_classes=nc)
        return _fwd(model, tr)

    def v_coreset_index():
        # GM 选核后用 Subset(dataset, indices) 重训，这里验证索引能被正确取回
        tr = state["tr"]
        k = max(1, len(tr) // 10)
        idx = np.random.RandomState(0).choice(len(tr), k, replace=False)
        sub = torch.utils.data.Subset(tr, np.sort(idx))
        x, y = sub[0]
        assert x.shape == tr[0][0].shape
        return "sub=%d/%d 取回正常" % (len(sub), len(tr))

    check("get_data 加载", v_load)
    check("标签为 python int", v_label_type)
    check("骨干前向", v_model)
    check("核心集索引还原", v_coreset_index)


# --------------------------------------------------------------------------- #
# DUAL
# --------------------------------------------------------------------------- #
def smoke_dual(ds_name):
    _enter(os.path.join(BASE, "dual-pruning-main", "exp_cifar"))
    import eva_adapter as EVA
    from data import load_eva
    from data_subset import load_eva_sub

    args = types.SimpleNamespace(
        dataset=ds_name, data_path="./data", batch_size=16, workers=0,
        subset_rate=0.1, sample=None, arch="resnet18", gpu="0",
    )
    state = {}

    def v_loaders():
        tr, te = load_eva(args)
        state["tr"], state["te"] = tr, te
        n_tr = len(tr.dataset)
        assert n_tr > 0 and len(te.dataset) > 0
        return "train=%d test=%d" % (n_tr, len(te.dataset))

    def v_targets():
        # DUAL 的训练动力学按 .targets 的 [label, index] 取样本，格式必须对
        tr = state["tr"].dataset
        tg = tr.targets
        assert len(tg) == len(tr)
        assert isinstance(tg[0], (list, tuple)) and len(tg[0]) == 2, \
            ".targets 必须是 [[label, index], ...]，实际 %r" % (tg[0],)
        assert int(tg[0][0]) == int(tr.labels[0])
        return "targets[0]=%r 长度=%d" % (tg[0], len(tg))

    def v_model():
        tr = state["tr"].dataset
        net = EVA.build_model(ds_name, num_classes=EVA.load(ds_name)["num_classes"])
        x, y = None, None
        info = _fwd(net, tr)
        # 再做一次真实的反向，确认梯度能回传（表格分支尤其容易漏）
        xs = torch.stack([tr[i][0] for i in range(4)])
        ys = torch.tensor([int(tr[i][1]) for i in range(4)], dtype=torch.long)
        loss = torch.nn.CrossEntropyLoss()(net(xs), ys)
        loss.backward()
        g = sum(1 for p in net.parameters() if p.grad is not None)
        assert g > 0, "没有任何参数拿到梯度"
        return "%s 反向正常（%d 个参数有梯度）" % (info, g)

    def v_subset():
        # 伪造一个 DUAL 风格的 mask（argsort 排列），验证子集大小与 CIFAR 分支同口径
        n = len(state["tr"].dataset)
        fake_mask = np.argsort(np.random.RandomState(0).rand(n))
        tr_sub, te_sub = load_eva_sub(args, None, None, fake_mask)
        k = len(tr_sub.dataset)
        expect = int(args.subset_rate * n)
        assert abs(k - expect) <= 1, "子集大小 %d 与预期 %d 不符" % (k, expect)
        x, y = tr_sub.dataset[0]
        assert x.shape == state["tr"].dataset[0][0].shape
        return "子集=%d/%d (rate=%.2f) 取回正常" % (k, n, args.subset_rate)

    def v_beta():
        # 跑批脚本用的是 --sample beta（DUAL 官方示例同款），必须单独验一遍。
        # 它走 beta_sampling，需要 (T, N) 的 target_probs + argsort 排列的 mask，
        # 与 top-k 分支是完全不同的代码路径，不能靠上面那个检查覆盖。
        n = len(state["tr"].dataset)
        rng = np.random.RandomState(0)
        target_probs = rng.rand(30, n).astype(np.float32)   # 30 个 epoch 的动力学
        score = target_probs.mean(axis=0)
        mask = np.argsort(score)
        a2 = types.SimpleNamespace(**vars(args))
        a2.sample, a2.c_d = "beta", 4
        tr_sub, _ = load_eva_sub(a2, target_probs, score, mask)
        k = len(tr_sub.dataset)
        expect = int(a2.subset_rate * n)
        assert abs(k - expect) <= 1, "beta 子集大小 %d 与预期 %d 不符" % (k, expect)
        idx = np.asarray(tr_sub.dataset.indices)
        assert len(set(idx.tolist())) == k, "beta 子集有重复索引"
        return "beta 子集=%d/%d" % (k, n)

    check("load_eva 加载", v_loaders)
    check(".targets 为 [label, idx]", v_targets)
    check("骨干前向 + 反向", v_model)
    check("核心集 top-k 切分", v_subset)
    check("核心集 beta 切分", v_beta)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, choices=["cops", "gm", "dual"])
    ap.add_argument("--dataset", required=True)
    a = ap.parse_args()

    fn = {"cops": smoke_cops, "gm": smoke_gm, "dual": smoke_dual}[a.baseline]
    print("=" * 66)
    print("baseline=%-5s dataset=%-13s EVA_SMOKE_LIMIT=%s"
          % (a.baseline, a.dataset, os.environ.get("EVA_SMOKE_LIMIT", "(off)")))
    print("=" * 66)
    try:
        fn(a.dataset)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        RESULT.append((False, "整体加载", repr(exc)))

    ok = sum(1 for r in RESULT if r[0])
    bad = len(RESULT) - ok
    print("-" * 66)
    print("RESULT %s  pass=%d fail=%d" % ("OK" if bad == 0 else "FAIL", ok, bad))
    sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
