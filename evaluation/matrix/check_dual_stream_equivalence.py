"""Compare streamed DUAL scoring against original stacked math on real dynamics."""
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import pipeline as p
from test_pipeline import functions

torch.set_num_threads(1)
source=p.BASE/'dual-pruning-main/exp_cifar/importance_evaluation.py'
scope=functions(source,{'dual','stream_target_probs'},dict(os=os,np=np,torch=torch,F=F))
old=p.HERE/'smoke_runs/705d8aaf5045'
images=p.HERE/'native_remaining_runs/2abf0bcff6b1'
mnist=p.HERE/'native_mnist_completion/674ff16bbd6f'
adapter=p.load_adapter();rows=[]
for dataset,root,limit in [('adult',old,100),('House',old,100),('cifar100',images,200),('news',images,200),('mnist2mnistm',mnist,20)]:
    os.environ['EVA_SMOKE_LIMIT']=str(limit)
    if dataset=='House':os.environ['EVA_REG_BINS']='10'
    else:os.environ.pop('EVA_REG_BINS',None)
    meta=adapter.load(dataset)
    labels=torch.as_tensor(adapter.Wrapped(meta['train'],meta).labels,dtype=torch.long)
    directory=root/'prepare/DUAL'/dataset
    dynamics=directory/'checkpoint'/dataset/'42/npy'
    canonical=[]
    for epoch in range(30):
        logits=torch.from_numpy(np.load(dynamics/f'{epoch}_Output.npy'))
        indices=torch.from_numpy(np.load(dynamics/f'{epoch}_Index.npy')).long()
        canonical.append(torch.zeros_like(logits).index_add(0,indices,logits))
    expected=F.softmax(torch.stack(canonical),dim=2).gather(2,labels[None,:,None].expand(30,-1,-1)).squeeze(2)
    streamed=scope['stream_target_probs'](dynamics,labels,30,chunk_size=7)
    assert torch.equal(streamed,expected),(dataset,'target probabilities differ')
    np.testing.assert_array_equal(streamed.numpy(),np.load(directory/'mask/target_probs.npy'))
    score,mask=scope['dual'](streamed)
    original_score,original_mask=scope['dual'](expected)
    assert torch.equal(score,original_score)
    np.testing.assert_array_equal(mask,original_mask)
    np.testing.assert_array_equal(score,np.load(directory/'mask/dual_score_T30.npy'))
    np.testing.assert_array_equal(mask,np.load(directory/'mask/dual_mask_T30.npy'))
    rows.append(dict(dataset=dataset,n_samples=len(labels),epochs=30,
                     probabilities_bit_exact=True,scores_bit_exact=True,masks_bit_exact=True))
    print(dataset,'original/streamed probabilities, scores and masks are bit-exact',flush=True)
p.save(p.HERE/'dual_stream_equivalence.json',dict(status='passed',checks=rows,scientific_results=False))
