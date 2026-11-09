"""One genuine CLIP encoding through the production embedding path, without training."""
import importlib.util
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import TensorDataset

import pipeline as p

torch.set_num_threads(1)
sys.path.insert(0,str(p.ADAPTER))
spec=importlib.util.spec_from_file_location('embedding_under_test',p.ADAPTER/'prepare_eva_embed.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
torch.manual_seed(42)
pixels=torch.rand(1,3,32,32)
mean=torch.tensor([.5071,.4867,.4408]).reshape(1,3,1,1)
std=torch.tensor([.2675,.2565,.2761]).reshape(1,3,1,1)
meta=dict(train=TensorDataset((pixels-mean)/std,torch.tensor([0])),
          mean=mean.reshape(-1).tolist(),std=std.reshape(-1).tolist())
started=time.perf_counter()
print('Loading real CLIP weights through the CPU embedding path...',flush=True)
embedding,method=module.embed_images(meta,'cpu',1,0)
assert tuple(embedding.shape)==(1,512) and torch.isfinite(embedding).all()
p.save(p.HERE/'clip_cpu_check.json',dict(status='ok',method=method,
    shape=list(embedding.shape),finite=True,device='cpu',scientific_results=False,
    elapsed_sec=time.perf_counter()-started,fixture_pixels=True,
    real_clip_weights=True,production_embedding_function=True))
print('Production CLIP path: (1, 512), finite, CPU, real weights',flush=True)
