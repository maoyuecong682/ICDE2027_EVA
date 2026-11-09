#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run EVA's own reference methods (random / full / kmeans) on the adult dataset.

Design notes
------------
This runner is deliberately non-invasive: ``methods/random.py``, ``methods/full.py``
and ``methods/kmeans.py`` are parsed from source and executed **without being
modified on disk**.  Only two purely environmental rewrites are applied in memory:

1. The group of four ``print(f"torch.cuda...")`` lines is wrapped in a
   ``torch.cuda.is_available()`` guard, so the script can run on a CPU-only box.
   On a GPU machine the printed output is byte-identical to the original.
2. ``num_workers=4`` -> ``num_workers=_NW`` (``_NW`` defaults to 0).  Under a
   0.5-core cgroup quota, four dataloader workers only add context-switch cost.

Additionally ``torch.load(best_model_path)`` gains ``map_location=device``, which
is a no-op when the tensors were saved on the same device.

``mydatasets`` and ``nets`` are injected as stub modules because the real
packages pull in ``transformers`` (imdb.py) and ``timm`` (resnet/vit/...), neither
of which is installed in this container.  The stub ``nets`` exposes a ``TabNet``
that is a faithful copy of ``nets/tabnet.py`` (13 -> 128 -> 128 -> attn -> 128 -> 2),
which is the backbone ``all.sh`` selects for adult.

The dataset itself is *not* stubbed: ``mydatasets/adult_modified.py`` is loaded by
path, so ``dst_train / dst_val / dst_test`` are exactly what the EVA pipeline
would produce.
"""

import argparse
import importlib.util
import json
import logging
import os
import re
import sys
import time
import types

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

torch.set_num_threads(1)

EVA_ROOT = os.environ.get(
    "EVA_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
DEFAULT_DATA_PATH = "/root/autodl-tmp/EVA/data/adult"
DEFAULT_OUT_ROOT = "/root/autodl-tmp/EVA/output/evaluation"


# --------------------------------------------------------------------------- #
# nets stub: faithful copy of nets/tabnet.py without EmbeddingRecorder
# --------------------------------------------------------------------------- #
class TabNet(nn.Module):
    """Mirror of nets.tabnet.TabNet (same layer names, same forward order)."""

    def __init__(self, channel, num_classes, im_size, record_embedding: bool = False,
                 no_grad: bool = False, pretrained: bool = False):
        if pretrained:
            raise NotImplementedError("Pretrained models not available.")
        super(TabNet, self).__init__()

        if isinstance(im_size, (tuple, list)):
            input_features = channel * int(np.prod(im_size))
        else:
            input_features = im_size * channel if channel != 1 else im_size

        self.input_layer = nn.Linear(input_features, 128)
        self.shared_layer_1 = nn.Linear(128, 128)
        self.shared_layer_2 = nn.Linear(128, 128)

        self.attention_1 = nn.Linear(128, 1)
        self.attention_2 = nn.Linear(128, 1)

        self.decision_1 = nn.Linear(128, 128)
        self.decision_2 = nn.Linear(128, num_classes)

        self.fc = self.decision_2
        self.no_grad = no_grad

    def get_last_layer(self):
        return self.fc

    def forward(self, x):
        with torch.set_grad_enabled(not self.no_grad):
            out = torch.flatten(x, 1)
            out = F.relu(self.input_layer(out))
            out = F.relu(self.shared_layer_1(out))
            out = F.relu(self.shared_layer_2(out))

            attention_1 = torch.sigmoid(self.attention_1(out))
            attention_2 = torch.sigmoid(self.attention_2(out))
            out = out * attention_1 * attention_2

            out = F.relu(self.decision_1(out))
            out = self.fc(out)
        return out


def _install_stubs():
    """Register lightweight ``nets`` / ``mydatasets`` so the method modules import."""
    if "nets" not in sys.modules:
        nets_stub = types.ModuleType("nets")
        nets_stub.TabNet = TabNet
        nets_stub.__all__ = ["TabNet"]
        sys.modules["nets"] = nets_stub
    if "mydatasets" not in sys.modules:
        ds_stub = types.ModuleType("mydatasets")
        ds_stub.__all__ = []
        sys.modules["mydatasets"] = ds_stub


# --------------------------------------------------------------------------- #
# in-memory CPU compatibility rewrite
# --------------------------------------------------------------------------- #
_CUDA_PRINT_BLOCK = re.compile(
    r"[ \t]*print\(f\"CUDA_VISIBLE_DEVICES:.*?\n"
    r"[ \t]*print\(f\"torch\.cuda\.current_device\(\):.*?\n"
    r"[ \t]*print\(f\"torch\.cuda\.device_count\(\):.*?\n"
    r"[ \t]*print\(f\"torch\.cuda\.get_device_name\(0\):.*?\n",
    re.S,
)


def _cuda_guard_replacement(match):
    indent = re.match(r"[ \t]*", match.group(0)).group(0)
    body = [
        "if torch.cuda.is_available():",
        "    print(f\"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}\")",
        "    print(f\"torch.cuda.current_device(): {torch.cuda.current_device()}\")",
        "    print(f\"torch.cuda.device_count(): {torch.cuda.device_count()}\")",
        "    print(f\"torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}\")",
        "else:",
        "    print(\"torch.cuda.*: unavailable (CPU-only run)\")",
    ]
    return "".join(indent + line + "\n" for line in body)


def load_method_module(name, path, num_workers=0, verbose=True, extra_rewrites=None,
                       extra_globals=None):
    """Load a method file by path, applying the documented, minimal rewrites."""
    with open(path, "r", encoding="utf-8") as fh:
        src = fh.read()

    src, n_cuda = _CUDA_PRINT_BLOCK.subn(_cuda_guard_replacement, src)
    src, n_nw = re.subn(r"num_workers\s*=\s*4", "num_workers=_NW", src)
    src, n_load = re.subn(
        r"torch\.load\(best_model_path\)",
        "torch.load(best_model_path, map_location=device)",
        src,
    )

    extra_applied = []
    for pattern, repl in (extra_rewrites or []):
        src, cnt = re.subn(pattern, repl, src)
        extra_applied.append((pattern[:40], cnt))

    if verbose:
        print(
            f"[compat] {os.path.basename(path)}: cuda_print={n_cuda} "
            f"num_workers={n_nw} torch_load={n_load} NW={num_workers} "
            f"extra={extra_applied}",
            flush=True,
        )

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    module.__file__ = path
    module._NW = int(num_workers)
    for key, value in (extra_globals or {}).items():
        setattr(module, key, value)
    sys.modules[name] = module
    exec(compile(src, path, "exec"), module.__dict__)
    return module


def load_module_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# dataset wrapper: AdultsDataset returns numpy, the methods expect tensors
# --------------------------------------------------------------------------- #
class TensorizedDataset(Dataset):
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        x, y = self.base[index]
        x = torch.as_tensor(np.asarray(x), dtype=torch.float32)
        return x, int(y)


# --------------------------------------------------------------------------- #
# log parsing
# --------------------------------------------------------------------------- #
_METRIC_PATTERNS = {
    "random": re.compile(
        r"Evaluation Results - Accuracy: ([\d.]+), Precision: ([\d.]+), "
        r"Recall: ([\d.]+), F1 Score: ([\d.]+)"
    ),
    "full": re.compile(r"acc: ([\d.]+), precision: ([\d.]+), recall: ([\d.]+), f1: ([\d.]+)"),
    "kmeans": re.compile(
        r"Accuracy: ([\d.]+), Precision: ([\d.]+), Recall: ([\d.]+), F1: ([\d.]+)"
    ),
}


def parse_metrics(method, log_path):
    if not os.path.exists(log_path):
        return None
    with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    hits = _METRIC_PATTERNS[method].findall(text)
    if not hits:
        # fall back to the per-epoch trace so a crash still yields a datapoint
        ep = re.findall(r"Epoch (\d+)/\d+, Loss: ([\d.]+), Acc: ([\d.]+)", text)
        if ep:
            last = ep[-1]
            return {"accuracy": float(last[2]), "epochs_done": int(last[0]), "partial": True}
        return None
    acc, prec, rec, f1 = (float(v) for v in hits[-1])
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def _set_seed(seed):
    """Seed everything that affects training.

    Note: this does **not** change the coreset that random.py / kmeans.py pick --
    both hardcode their own selection seed (42 and random_state=0), so repeated
    runs with different --seed measure *training* variance only.
    """
    import random as _random
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_args(out_root, dataset, model, method, fraction, balance_target,
               epochs, patience, lr, workers):    return argparse.Namespace(
        save_log_path=os.path.join(out_root, "logs") + os.sep,
        save_best_model_path=os.path.join(out_root, "best_models") + os.sep,
        dataset=dataset,
        model=model,
        method=method,
        selection_fraction=fraction,
        balance_target=balance_target,
        proxy_model="none",
        selection_lr=lr,
        selection_momentum=0.9,
        selection_weight_decay=5e-4,
        selection_optimizer="SGD",
        selection_batch=128,
        epochs=epochs,
        patience=patience,
        workers=workers,
        task_type="classification",
        gpu=0,
        random_seed=0,
    )


def _reset_root_logger():
    """Detach handlers left over from a previous method.

    ``logging.basicConfig()`` silently does nothing when the root logger already
    has handlers, so without this the second and third method would keep writing
    into the *first* method's log file and their own logs would never exist.
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        try:
            handler.flush()
            handler.close()
        except Exception:
            pass
        root.removeHandler(handler)
    for name in list(logging.Logger.manager.loggerDict):
        logger = logging.getLogger(name)
        if not isinstance(logger, logging.PlaceHolder):
            logger.handlers.clear()
            logger.propagate = True


def expected_log_path(args):
    """The three files disagree on the directory layout -- reproduce each one."""
    if args.method == "random":
        return os.path.join(
            args.save_log_path, args.dataset, args.method,
            str(args.selection_fraction), f"{args.balance_target}.log",
        )
    return os.path.join(
        args.save_log_path, args.dataset, args.model,
        str(args.selection_fraction), f"{args.method}.log",
    )


def main():
    ap = argparse.ArgumentParser(description="EVA reference methods on adult (CPU-safe)")
    ap.add_argument("--data_path", default=DEFAULT_DATA_PATH)
    ap.add_argument("--out_root", default=DEFAULT_OUT_ROOT)
    ap.add_argument("--dataset", default="adult")
    ap.add_argument("--model", default="TabNet")
    ap.add_argument("--methods", default="full,random,kmeans")
    ap.add_argument("--fraction", type=float, default=0.1)
    ap.add_argument("--balance_target", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=100)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=None,
                    help="seed torch/numpy before each method (None = current, unseeded behaviour)")
    ap.add_argument("--step_budget", type=int, default=0,
                    help="if >0, derive epochs per method so every method spends this many "
                         "gradient updates (ignored for full)")
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    _install_stubs()

    # ---------------- dataset (real adult_modified.py, not a stub) -----------
    adult_path = os.path.join(EVA_ROOT, "mydatasets", "adult_modified.py")
    if not os.path.exists(adult_path):
        raise SystemExit(f"adult_modified.py not found at {adult_path}")
    print(f"[data] loading {adult_path}", flush=True)
    t0 = time.time()
    adult_mod = load_module_by_path("adult_modified", adult_path)
    (channel, im_size, num_classes, class_names, mean, std,
     dst_train, dst_val, dst_test) = adult_mod.adult(args.data_path, args.balance_target)
    print(
        f"[data] channel={channel} im_size={im_size} num_classes={num_classes} "
        f"train={len(dst_train)} val={len(dst_val)} test={len(dst_test)} "
        f"({time.time() - t0:.1f}s)",
        flush=True,
    )

    dst_train = TensorizedDataset(dst_train)
    dst_val = TensorizedDataset(dst_val)
    dst_test = TensorizedDataset(dst_test)

    # ---------------- methods ----------------------------------------------
    module_map = {
        "random": os.path.join(EVA_ROOT, "methods", "random.py"),
        "full": os.path.join(EVA_ROOT, "methods", "full.py"),
        "kmeans": os.path.join(EVA_ROOT, "methods", "kmeans.py"),
    }
    entry_map = {
        "random": "random_selection",
        "full": "full_selection",
        "kmeans": "kmeans_selection",
    }

    # ``full.py`` reuses the embedding-extraction DataLoader (shuffle=False) to
    # train. That is harmless on CIFAR, whose rows are in random order, but this
    # adult split is built as pd.concat([label==0, label==1]) -- i.e. sorted by
    # label. With shuffle=False the model sees 154 consecutive all-class-0
    # batches, collapses to "always predict 0" (test acc = 0.7638, exactly the
    # majority rate) and then gets a gradient shock from the class-1 tail.
    # random.py and kmeans.py already build their coreset loader with
    # shuffle=True, so only full.py needs the repair.
    extra_rewrites_map = {
        "full": [(
            r"train_model\(train_model_instance, train_loader,",
            "train_loader = DataLoader(dst_train, batch_size=128, shuffle=True)\n"
            "    train_model(train_model_instance, train_loader,",
        )],
    }

    # At very low coreset fractions random.py cannot train at all, for two reasons
    # that are both hardcoded and fine at 10% but fatal at 1%:
    #   * the coreset loader uses batch_size=256, so a 260-row coreset yields
    #     ceil(260/256) = 2 gradient steps per epoch;
    #   * train_model is called with patience=10, so it stops after ~11 epochs,
    #     i.e. ~22 steps in total, while the test accuracy sits exactly on the
    #     majority class (0.7638) the whole time.
    # The result is not "random selection is bad", it is "the model never left its
    # initialisation". Scale both so the step count stays comparable to the
    # 10% run; this only applies for fractions <= 5% so 10% numbers are untouched.
    if args.fraction <= 0.05:
        extra_rewrites_map["random"] = [
            (r"DataLoader\(coreset_dataset, batch_size=256, shuffle=True",
             "DataLoader(coreset_dataset, batch_size=32, shuffle=True"),
            (r"patience=10", "patience=_PATIENCE"),
        ]

    results = []
    for method in [m.strip() for m in args.methods.split(",") if m.strip()]:
        if method not in module_map:
            print(f"[skip] unknown method {method}", flush=True)
            continue

        print(f"\n{'=' * 70}\n[run] method={method} fraction={args.fraction}\n{'=' * 70}", flush=True)

        # Optimisation budget. Comparing methods at a fixed epoch count is
        # meaningless when batch sizes differ: at a 1% coreset (260 rows) EVA's
        # random.py with batch 32 takes 9 steps per epoch while COPS with batch
        # 256 takes 2. --step_budget fixes the number of gradient updates
        # instead, and derives the epoch count per method. full trains on the
        # whole training set and is left on its own schedule (it is a reference,
        # not a coreset method).
        ep, pat = args.epochs, args.patience
        if args.step_budget > 0 and method != "full":
            n_sel = max(1, int(round(len(dst_train) * args.fraction)))
            batch = 32 if method == "random" else 128
            steps_per_epoch = max(1, -(-n_sel // batch))
            ep = max(1, -(-args.step_budget // steps_per_epoch))
            pat = ep  # no early stopping: spending the whole budget is the point
            print(f"[budget] {method}: {n_sel} rows / batch {batch} = "
                  f"{steps_per_epoch} steps/epoch -> {ep} epochs "
                  f"= {ep * steps_per_epoch} steps", flush=True)

        margs = build_args(
            args.out_root, args.dataset, args.model, method, args.fraction,
            args.balance_target, ep, pat, args.lr, args.workers,
        )

        started = time.time()
        status, error = "ok", ""

        # start every method from a clean logging state and a clean log file
        _reset_root_logger()
        stale_log = expected_log_path(margs)
        if os.path.exists(stale_log):
            os.remove(stale_log)

        try:
            module = load_method_module(
                f"eva_method_{method}", module_map[method], num_workers=args.workers,
                extra_rewrites=extra_rewrites_map.get(method),
                extra_globals={"_PATIENCE": pat},
            )
            entry = getattr(module, entry_map[method])
            if args.seed is not None:
                _set_seed(args.seed)
            entry(margs, channel, im_size, num_classes, class_names, mean, std,
                  dst_train, dst_val, dst_test)
        except Exception as exc:  # keep going: one method failing must not kill the rest
            import traceback
            traceback.print_exc()
            status, error = "failed", f"{type(exc).__name__}: {exc}"

        elapsed = time.time() - started
        _reset_root_logger()  # flush + close so the log file is complete on disk
        log_path = expected_log_path(margs)
        metrics = parse_metrics(method, log_path) or {}

        row = {
            "method": method,
            "dataset": args.dataset,
            "model": args.model,
            "selection_fraction": args.fraction,
            "epochs": ep,
            "patience": pat,
            "lr": args.lr,
            "seed": args.seed,
            "status": status,
            "error": error,
            "elapsed_sec": round(elapsed, 1),
            "log": log_path,
            "metrics": metrics,
        }
        results.append(row)
        print(f"[done] {method}: status={status} {metrics} {elapsed:.1f}s", flush=True)

    # ---------------- summary ----------------------------------------------
    summary_json = os.path.join(args.out_root, "summary.json")
    with open(summary_json, "w", encoding="utf-8") as fh:
        json.dump({"data": {"channel": channel, "im_size": im_size,
                            "num_classes": num_classes,
                            "n_train": len(dst_train), "n_val": len(dst_val),
                            "n_test": len(dst_test)},
                   "results": results}, fh, indent=2, ensure_ascii=False)

    csv_path = os.path.join(args.out_root, "summary.csv")
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write("method,accuracy,precision,recall,f1,elapsed_sec,status,seed,fraction\n")
        for row in results:
            m = row["metrics"]
            fh.write(
                f"{row['method']},"
                f"{m.get('accuracy', '')},{m.get('precision', '')},"
                f"{m.get('recall', '')},{m.get('f1', '')},"
                f"{row['elapsed_sec']},{row['status']},"
                f"{'' if row['seed'] is None else row['seed']},{row['selection_fraction']}\n"
            )

    print("\n" + "=" * 70)
    print(f"{'method':<10}{'acc':>10}{'f1':>10}{'time(s)':>12}  status")
    for row in results:
        m = row["metrics"]
        print(f"{row['method']:<10}{m.get('accuracy', float('nan')):>10.4f}"
              f"{m.get('f1', float('nan')):>10.4f}{row['elapsed_sec']:>12.1f}  {row['status']}")
    print("=" * 70)
    print(f"summary -> {summary_json}\n           {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
