"""MNIST-M original encoded image bytes without loading pandas/Arrow in workers."""
import io
import struct
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch


class MemoryMappedMNIST(Dataset):
    def __init__(self,root,train=True,transform=None,download=False):
        raw=Path(root)/'MNIST/raw'
        prefix='train' if train else 't10k'
        image_path=raw/(prefix+'-images-idx3-ubyte')
        label_path=raw/(prefix+'-labels-idx1-ubyte')
        with image_path.open('rb') as f:magic,n,h,w=struct.unpack('>4I',f.read(16))
        with label_path.open('rb') as f:label_magic,label_n=struct.unpack('>2I',f.read(8))
        if magic!=2051 or label_magic!=2049 or n!=label_n or (h,w)!=(28,28):
            raise ValueError('MNIST original IDX header mismatch')
        self.data=np.memmap(image_path,mode='r',dtype=np.uint8,offset=16,shape=(n,h,w))
        labels=np.memmap(label_path,mode='r',dtype=np.uint8,offset=8,shape=(n,))
        self.targets=torch.from_numpy(np.asarray(labels,dtype=np.int64).copy())
        self.transform=transform
    def __len__(self):return len(self.targets)
    def __getitem__(self,i):
        image=Image.fromarray(self.data[i],mode='L')
        return (self.transform(image) if self.transform else image),int(self.targets[i])


class MemoryMappedMNISTM(Dataset):
    def __init__(self,parquet_path,transform=None):
        cache=Path(parquet_path).parent/'original_encoded_memmap'
        if not (cache/'complete.json').exists():
            raise FileNotFoundError('MNIST-M mmap cache missing; run matrix/prepare_mnistm_cache.py')
        self.encoded=np.memmap(cache/'images.bin',mode='r',dtype=np.uint8)
        self.offsets=np.load(cache/'offsets.npy',mmap_mode='r')
        self.labels=np.load(cache/'labels.npy').tolist()
        self.transform=transform
    def __len__(self):return len(self.labels)
    def __getitem__(self,i):
        image=Image.open(io.BytesIO(self.encoded[self.offsets[i]:self.offsets[i+1]].tobytes()))
        return (self.transform(image) if self.transform else image),self.labels[i]
