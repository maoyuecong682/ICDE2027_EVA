from torchvision import datasets, transforms
import numpy as np
import torch


def _eva_names():
    """EVA 适配层支持的数据集名；适配层缺失时返回空集合，保持仓库可独立运行。"""
    try:
        import eva_adapter as _EVA
        return set(_EVA.DATASETS)
    except Exception:
        return set()


def get_data(
    dataset_name, 
    split="train",
    transform=None,
    download=True, 
    data_dir="./data"
):

    """
    Load datasets for either vision tasks or LLM benchmarks.

    Parameters:
        dataset_name (str):
        split (str):
        transform (callable, optional): A function/transform that takes in an image and returns a transformed version.
                                        Only applies for vision tasks.
        download (bool): Whether to download the dataset if it is not already present.
        data_dir (str): Directory where the data is (or will be) stored.

    Returns:
        dataset: A dataset object.
    """
    dataset = None

    if dataset_name == 'cifar10' and dataset_name not in _eva_names():
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2470, 0.2435, 0.2616]
        n_classes = 10

        if split == "train":
            if transform is None:
                # Default transforms for training data
                tr_transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),

                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                    
                    Cutout(n_holes=1, length=16)
                ])
            else:
                tr_transform = transform

            dataset = datasets.CIFAR10(
                root=data_dir,
                train=True,
                transform=tr_transform,
                download=download
            )

        elif split == "test":
            if transform is None:
                # Default transforms for test data
                te_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std)
                ])
            else:
                te_transform = transform

            dataset = datasets.CIFAR10(
                root=data_dir,
                train=False,
                transform=te_transform,
                download=download
            )
    elif dataset_name == 'adult' and dataset_name not in _eva_names():
        # ---------------- Adult 表格数据 ----------------
        # 数据来自 EVA 预处理产物 adult_processed.npz（z-score 归一化）
        from adult_adapter import load_adult_arrays, AdultDataset

        d = load_adult_arrays()
        n_classes = d["num_classes"]
        if split == "train":
            dataset = AdultDataset(d["train_x"], d["train_y"])
        else:
            dataset = AdultDataset(d["test_x"], d["test_y"])

    elif dataset_name in _eva_names():
        # ---------------- EVA 预处理好的 9 个数据集 ----------------
        # 与 COPS / DUAL 共用同一份适配层，因此 train/test 与 EVA 主实验逐位一致。
        # 注意：这里**不套 transform**。EVA 数据集吐出的张量已经 ToTensor+Normalize 过，
        # 再叠 GM 的 RandomCrop/Flip 会把归一化破坏成非物理区间。
        import eva_adapter as _EVA
        _m = _EVA.load(dataset_name)
        n_classes = _m["num_classes"]
        dataset = _EVA.Wrapped(_m["train"] if split == "train" else _m["test"], _m)

    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented.")

    return dataset, n_classes


class Cutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

