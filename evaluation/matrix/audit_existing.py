"""Read-only evidence checks. Only exact raw-feature equivalence qualifies for reuse."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

import pipeline as p


def audit(dataset):
    old=Path('/root/autodl-tmp/eva_output')
    if dataset in ('cifar100','mnist2mnistm'):
        paths=[('COPS',old/'COPS'/dataset/'cops_npy/train_UN_2_raw.npy'),
               ('GM_matching',old/'GM_matching'/dataset/'embeds'/dataset/'embeddings.pt')]
        paths += [('DUAL',path) for path in (old/'DUAL'/dataset).glob('**/dual_mask_T30.npy')]
        return dict(dataset=dataset,fingerprint=None,artifacts=[dict(baseline=bl,path=str(path),reuse=False,
            reason=('Historical image embedding preprocessing cannot be certified; regenerate with corrected inverse normalization'
                    if bl=='GM_matching' else 'Historical preparation provenance is insufficient; not imported'))
            for bl,path in paths if path.exists()])
    adapter=p.load_adapter()
    if dataset=='House':
        __import__('os').environ['EVA_REG_BINS']='10'
    if dataset=='news':
        # news.py returns these existing normalized caches directly. Do not
        # import the BERT architecture merely to audit their numeric contents.
        cache=Path('/root/autodl-tmp/EVA/data/embedding')
        class CachedNews:
            def __init__(self,split):
                self.features=np.load(cache/('balance_news_'+split+'_embeddings.npy'),mmap_mode='r')
                self.labels=np.load(cache/('balance_news_'+split+'_labels.npy'))
            def __len__(self):return len(self.labels)
            def __getitem__(self,i):return self.features[i],int(self.labels[i])
        meta=dict(train=CachedNews('train'),val=CachedNews('val'),test=CachedNews('test'),
                  num_classes=20,is_regression=False,is_tabular=True)
    else:
        meta=adapter.load(dataset)
    n=len(meta['train'])
    records=[]
    cops=old/'COPS'/dataset/'cops_npy'
    if (cops/'train_UN_2_raw.npy').exists():
        labs=(adapter.Wrapped(meta['train'],meta).labels if meta['is_regression']
              else adapter._labels_of(meta['train']))
        _,pool=adapter.split_train_probe(labs)
        logits=np.load(cops/'train_UN_2_raw.npy',mmap_mode='r')
        labels=np.load(cops/'train_Y_2.npy')
        matches=logits.ndim==3 and logits.shape[1:]==(len(pool),meta['num_classes']) and np.array_equal(labels,np.asarray(labs)[pool])
        records.append(dict(baseline='COPS',path=str(cops),shape=list(logits.shape),
            labels_and_shape_match=bool(matches),reuse=False,
            reason='Matching labels alone do not establish feature preprocessing, model configuration or checkpoint provenance'))
    embeds=old/'GM_matching'/dataset/'embeds'/dataset/'embeddings.pt'
    if embeds.exists():
        entry=dict(baseline='GM_matching',path=str(embeds),reuse=False)
        if meta['is_tabular']:
            matrix=torch.load(embeds,map_location='cpu',mmap=True,weights_only=True)
            exact=matrix.ndim==2 and matrix.shape[0]==n
            if exact:
                for start in range(0,n,512):
                    expected=torch.stack([torch.as_tensor(meta['train'][i][0],dtype=torch.float32).reshape(-1) for i in range(start,min(n,start+512))])
                    if not torch.equal(matrix[start:start+len(expected)],expected):
                        exact=False
                        break
            entry.update(reuse=bool(exact),reason='All rows equal current deterministic raw features exactly' if exact else 'Features differ from current training features; reject')
            if exact:
                entry['sha256']=hashlib.sha256(embeds.read_bytes()).hexdigest()
        else:
            entry['reason']='Historical image embedding preprocessing cannot be certified; regenerate with corrected inverse normalization'
        records.append(entry)
    masks=list((old/'DUAL'/dataset).glob('**/dual_mask_T30.npy'))
    for mask in masks:
        order=np.load(mask)
        permutation=len(order)==n and np.array_equal(np.sort(order),np.arange(n))
        records.append(dict(baseline='DUAL',path=str(mask),valid_permutation=bool(permutation),reuse=False,
                            reason='Mask alone does not establish training dynamics, preprocessing and beta sampling provenance'))
    return dict(dataset=dataset,fingerprint=p.fingerprint(meta,adapter),artifacts=records)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset')
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    p.save(args.out,audit(args.dataset))
