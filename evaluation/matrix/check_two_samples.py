"""Two real samples per split: production model, evaluator, and image encoder."""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pipeline as p


def check(dataset,out):
    os.environ.pop('EVA_SMOKE_LIMIT',None)
    import numpy as np
    import torch
    adapter=p.load_adapter()
    torch.manual_seed(42);np.random.seed(42)
    meta=adapter.load(dataset)
    tiny=dict(meta)
    for split in ('train','val','test'):
        tiny[split]=adapter.select(adapter.Wrapped(meta[split],meta),[0,1])
    dest=out/dataset;dest.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    encoder=None
    if not meta['is_tabular']:
        sys.path.insert(0,str(p.ADAPTER))
        from prepare_eva_embed import embed_images
        embedding,method=embed_images(tiny,'cpu',2,0)
        assert tuple(embedding.shape)==(2,512) and torch.isfinite(embedding).all()
        torch.save(embedding,dest/'real_clip_embeddings.pt')
        encoder=dict(method=method,shape=list(embedding.shape),finite=True,
                     checkpoint_sha256=tiny['_clip_checkpoint_sha256'])
        del embedding
        gc.collect()
    sys.path.insert(0,str(p.HERE.parent))
    import eval_baselines as ev
    ev._install_stubs()
    torch.manual_seed(42);np.random.seed(42)
    result=ev.run_kmeans_routine(tiny['train'],tiny['val'],tiny['test'],tiny,
           SimpleNamespace(epochs=1,patience=1,lr=.01 if meta['is_tabular'] else .1,batch_size=2),str(dest))
    for metric in ('accuracy','precision','recall','f1'):
        assert np.isfinite(result[metric]) and 0<=result[metric]<=1
    assert (dest/'kmeans.pth').exists()
    result.update(dataset=dataset,status='ok',samples_per_split=2,seed=42,
                  scientific_results=False,production_backbone=True,production_shared_evaluator=True,
                  real_image_encoder=encoder,source_indices=[0,1],
                  elapsed_sec=time.perf_counter()-started,
                  scope='real data/model/encoder/training/checkpoint/metrics integration; native selector matrix checked separately')
    p.save(dest/'check.json',result)
    print(dataset,'two-sample production pipeline passed',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset')
    parser.add_argument('--out',default=str(p.HERE/'two_sample_checks'))
    args=parser.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if args.dataset:
        check(args.dataset,out)
        return
    rows=[]
    for ds in ('cifar100','news','mnist2mnistm'):
        p.release_idle_input_pages(p.HERE)
        with (out/(ds+'.log')).open('w') as log:
            subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--dataset',ds,'--out',str(out)],
                           env=p.env(),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
        rows.append(json.loads((out/ds/'check.json').read_text()))
        print(ds,'2 samples/split passed',flush=True)
    p.save(out/'report.json',dict(status='passed',datasets=3,samples_per_split=2,
                                scientific_results=False,checks=rows))


if __name__=='__main__':main()
