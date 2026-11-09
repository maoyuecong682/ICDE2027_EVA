import os
import argparse
from tqdm import tqdm
import numpy as np

import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.data import Subset

try:
    from pytorch_lightning import seed_everything
except ImportError:  # 环境未安装 pytorch_lightning 时的等价实现
    def seed_everything(seed, workers=False):
        import random as _random
        _random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return seed

from data import get_data
from models import get_model
from logger import CSVLogger
from corrupt import add_label_noise
from sampling.sample import run_coreset_select


def _eva_names():
    """EVA 适配层托管的 9 个数据集名单。

    名单的真理源在 eva_adapter.ADAPTER_DATASETS，本仓库只做查询、不复制一份，
    避免「同一个名单两处维护、改一处忘一处」。GM 仓库根目录下的 eva_adapter.py
    是指向 /root/eva_output/eva_adapter.py 的软链接，因此这里能直接 import。
    """
    import eva_adapter as _EVA
    return _EVA.ADAPTER_DATASETS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--noise_fraction", type=float, default=0)
    parser.add_argument("--log_filename", type=str, default="logs/cifar10/baseline/logs.csv")
    parser.add_argument("--coreset_size_dataset", type=float, default=None)
    parser.add_argument("--coreset_method", type=str, default="random")
    parser.add_argument("--num_repetitions", type=int, default=3)
    # Adult 表格数据相关
    parser.add_argument("--model", type=str, default="AdultTabNet")
    parser.add_argument("--input_features", type=int, default=13)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--embeds_dir", type=str, default="embeds",
                        help="嵌入缓存根目录（表格数据由 prepare_adult_embed.py 预生成）")
    return parser.parse_args()


def test(model, loader, device="cuda" if torch.cuda.is_available() else "cpu"):
    model.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0.
    total = 0.
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            pred = model(images)

        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels).sum().item()

    val_acc = correct / total
    model.train()
    return val_acc


def main(args):
    os.makedirs(os.path.dirname(args.log_filename), exist_ok=True)
    csv_logger = CSVLogger(
        fieldnames=['epoch', 'batch', 'train_acc', 'test_acc', 'train_loss'],
        filename=args.log_filename
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

    # load dataset
    # --------------------------------------------
    tr_dataset, n_classes = get_data(dataset_name=args.dataset, split="train")
    test_dataset, _ = get_data(dataset_name=args.dataset, split="test")

    # Add label noise
    # --------------------------------------------
    if args.noise_fraction > 0:
        print(f"Adding label noise with fraction {args.noise_fraction}")
        if args.noise_fraction > 1:
            raise ValueError("Noise fraction should be between 0 and 1.")
        # Apply noise to the dataset
        tr_dataset = add_label_noise(
            dataset=tr_dataset,
            noise_ratio=args.noise_fraction,
        )

    # Subset Selection
    # --------------------------------------------
    if args.noise_fraction > 0:
        embedding_file_path = os.path.join(args.embeds_dir, args.dataset + str(args.noise_fraction))
    else:
        # No noise, use the original dataset
        embedding_file_path = os.path.join(args.embeds_dir, args.dataset)

    os.makedirs(embedding_file_path, exist_ok=True)
    embedding_file = os.path.join(embedding_file_path, "embeddings.pt")
    subset_size = args.coreset_size_dataset

    total_samples = len(tr_dataset)
    print(f"Total samples in dataset: {total_samples}")

    # 支持用比例（<=1）指定核心集大小，便于和 EVA 的 selection_fraction 对齐
    if subset_size is not None and 0 < subset_size <= 1:
        subset_size = int(round(subset_size * total_samples))
        print(f"coreset_size_dataset given as fraction -> {subset_size} samples")

    if subset_size and subset_size < total_samples:
        subset_indices = run_coreset_select(
            dataset=tr_dataset,
            coreset_size=subset_size,
            coreset_method=args.coreset_method,
            embedding_file=embedding_file,
            embed_bs=512
        )

        tr_dataset = Subset(tr_dataset, subset_indices)

        # 核心集索引必须落盘：统一评估器要用它复现完全相同的子集。
        # GM 会重复 num_repetitions 次，每次重新选，故按出现顺序编号。
        import glob as _glob
        _k = len(_glob.glob(os.path.join(embedding_file_path, "coreset_indices_rep*.npy")))
        _p = os.path.join(embedding_file_path, "coreset_indices_rep%d.npy" % _k)
        np.save(_p, np.asarray(subset_indices, dtype=np.int64))
        print("[coreset] saved %d indices -> %s" % (len(subset_indices), _p))

    else:
        print(f"Using full dataset with {total_samples} samples.")

    # Training
    # --------------------------------------------
    if args.dataset == 'cifar10' and args.dataset not in _eva_names():
        # We will run ResNet18 on CIFAR10 --
        # Hyperparameters are taken from the paper:

        # Improved Regularization of Convolutional Neural Networks with Cutout
        # http://arxiv.org/abs/1708.04552
        # github: https://github.com/uoguelph-mlrg/Cutout

        model = get_model(model_name="CIFARResNet18", num_classes=10)

        optimizer = SGD(
            model.parameters(), 
            lr=0.1, 
            momentum=0.9, 
            nesterov=True,
            weight_decay=5e-4
        )
        criterion = nn.CrossEntropyLoss()
        scheduler = MultiStepLR(
            optimizer,
            milestones=[60, 120, 160],
            gamma=0.2
        )
        tr_dataloader = DataLoader(
            tr_dataset,
            batch_size=128,
            shuffle=True,
            pin_memory=True,
            num_workers=4
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=128,
            shuffle=False,
            pin_memory=True,
            num_workers=4
        )
        num_epochs = 200

    elif args.dataset == 'adult' and args.dataset not in _eva_names():
        # ---------------- Adult 表格数据 ----------------
        # 骨干与 EVA nets/tabnet.py 同构；优化器超参沿用 CIFAR 分支的协议
        model = get_model(
            model_name=args.model,
            num_classes=2,
            input_features=args.input_features,
        )

        optimizer = SGD(
            model.parameters(),
            lr=0.1,
            momentum=0.9,
            nesterov=True,
            weight_decay=5e-4
        )
        criterion = nn.CrossEntropyLoss()
        scheduler = MultiStepLR(
            optimizer,
            milestones=[60, 120, 160],
            gamma=0.2
        )
        tr_dataloader = DataLoader(
            tr_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
        num_epochs = args.epochs

    elif args.dataset in _eva_names():
        # ---------------- EVA 预处理好的 9 个数据集 ----------------
        # 优化器/调度器协议沿用 CIFAR 分支（SGD 0.1 / nesterov / milestones 60-120-160），
        # 骨干按数据形态选：图像 CIFARResNet18，表格 args.model。
        import eva_adapter as _EVA
        _m = _EVA.load(args.dataset)
        _nc = _m["num_classes"]
        # 骨干统一由 eva_adapter 决定：图像 resnet18（按通道数/分辨率自适应 stem），
        # 表格 TabMLP。GM 自带的 CIFARResNet18 首层写死 3 通道，MNIST 这类单通道
        # 数据会直接崩；而且三个 baseline 必须同构，核心集质量才可横向比较。
        model = _EVA.build_model(args.dataset, num_classes=_nc)

        optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9,
                        nesterov=True, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
        tr_dataloader = DataLoader(tr_dataset, batch_size=args.batch_size,
                                   shuffle=True, num_workers=args.num_workers)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                     shuffle=False, num_workers=args.num_workers)
        num_epochs = args.epochs

    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented.")

    model = model.to(device)
    best_acc = 0.

    for epoch in range(num_epochs):
        acc = 0.
        loss_avg = 0.
        correct = 0.
        total = 0.

        pbar = tqdm(tr_dataloader)

        for batch_idx, (images, labels) in enumerate(pbar):
            pbar.set_description(f"Epoch {epoch + 1}/{num_epochs}")
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            loss_avg += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            acc = correct / total
            current_loss = loss_avg / (batch_idx + 1)
            pbar.set_postfix(
                loss='%.3f' % current_loss,
                acc='%.3f' % acc
            )
            
            # Log every batch
            row = {
                'epoch': str(epoch),
                'batch': str(batch_idx),
                'train_acc': str(acc),
                'test_acc': '',  # Empty for batch updates
                'train_loss': str(current_loss)
            }
            csv_logger.writerow(row)

        scheduler.step()
        test_acc = test(model, test_dataloader)
        if test_acc > best_acc:
            best_acc = test_acc
            print("Best accuracy so far: %.3f" % best_acc)

        tqdm.write('test_acc: %.3f' % test_acc)
        row = {
            'epoch': str(epoch),
            'batch': 'end',
            'train_acc': str(acc),
            'test_acc': str(test_acc),
            'train_loss': str(loss_avg / len(tr_dataloader))
        }
        csv_logger.writerow(row)

    print("Finished Training")
    csv_logger.close()
    return best_acc


if __name__ == "__main__":
    args = parse_args()
    print(args)
    n_repetitions = args.num_repetitions
    best_acc = []
    for i in range(n_repetitions):
        print(f"Repetition {i + 1}/{n_repetitions}")
        seed_everything(i)
        acc = main(args)
        print(f"Seed/Best: {i}/{acc}")
        best_acc.append(acc)

    print("All repetitions finished.")
    print(f"Best accuracy so far: {best_acc}")
    print(f"Best accuracy over {n_repetitions} repetitions: {max(best_acc)}")
    mean_acc = 100 * np.mean(best_acc)
    std = 100 * np.std(best_acc)

    print(f"Mean accuracy: {mean_acc:.3f} ± {std:.3f}")



