"""CIFAR100's original pixels/labels, memory-mapped in the original row order."""
import json
from pathlib import Path

import numpy as np
from torchvision.datasets import CIFAR100
from torchvision.datasets.vision import VisionDataset


class MemoryMappedCIFAR100(CIFAR100):
    def __init__(self,root,train=True,transform=None,target_transform=None,download=False):
        VisionDataset.__init__(self,root,transform=transform,target_transform=target_transform)
        self.train=train
        split='train' if train else 'test'
        cache=Path(root)/'original_pixels_memmap'
        if not (cache/'complete.json').exists():
            raise FileNotFoundError('CIFAR memmap cache missing; run matrix/prepare_cifar_cache.py first')
        self.data=np.load(cache/(split+'_pixels.npy'),mmap_mode='r')
        self.targets=np.load(cache/(split+'_labels.npy')).tolist()
        self._load_meta()

