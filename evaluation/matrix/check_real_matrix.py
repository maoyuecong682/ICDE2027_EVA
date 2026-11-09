"""CPU integration checks on real data. Fixture logits/image embeddings are NEVER results."""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

import pipeline as p


def check_dataset(dataset,out):
    import os
    os.environ['EVA_SMOKE_LIMIT']='1000'
    if dataset=='House':os.environ['EVA_REG_BINS']='10'
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,mean_absolute_error,mean_squared_error
    from scipy.stats import beta
    from test_pipeline import functions
    adapter=p.load_adapter()
    meta=adapter.load(dataset)
    train=adapter.Wrapped(meta['train'],meta)
    n=len(train)
    labels=train.labels
    classes=meta['num_classes']
    _,pool=adapter.split_train_probe(labels)
    rng=np.random.RandomState(42)
    # Fixture ensemble logits follow the real sampling-pool order.
    logits=rng.normal(size=(len(pool),2,classes))
    metrics={}
    scripts={v:p.BASE/('COPS-main/retraining_withy.py' if v=='wy' else 'COPS-main/retraining_woy.py') for v in ('wy','woy')}
    for v,script in scripts.items():
        ns=functions(script,{'cal_un_wy','cal_un_woy','_sample_probs'},dict(np=np))
        onehot=np.eye(classes)[labels[pool]]
        metrics[v]=np.asarray([ns['cal_un_wy'](logits[i],onehot[i]) if v=='wy' else ns['cal_un_woy'](logits[i]) for i in range(len(pool))])
    # DUAL native scoring and beta selection, distinct scores and permutations.
    d=p.BASE/'dual-pruning-main/exp_cifar'
    dual=functions(d/'importance_evaluation.py',{'dual'},dict(np=np,torch=torch))
    beta_ns=functions(d/'data_subset.py',{'beta_sampling'},dict(np=np,torch=torch,beta=beta))
    probs=torch.tensor(rng.uniform(.05,.95,(30,n)),dtype=torch.float32)
    score,mask=dual['dual'](probs)
    score=np.asarray(score);mask=np.asarray(mask)
    sys.path.insert(0,str(p.BASE/'GM-Matching-main'))
    from sampling.sample import run_coreset_select
    # Tabular embeddings are actual native features. Images use a labeled fixture.
    embedding=torch.stack([train[i][0].reshape(-1) if meta['is_tabular'] else train[i][0].mean(dim=(1,2)) for i in range(n)])
    embedding_path=out/(dataset+'_fixture_embedding.pt')
    torch.save(embedding,embedding_path)
    # Prove the production backbone accepts real batches, including backward.
    original=adapter.build_model(meta,arch=None)
    xb,yb=next(iter(DataLoader(train,batch_size=2)))
    loss=nn.CrossEntropyLoss()(original(xb),yb)
    loss.backward()
    del original,loss,xb,yb
    # Continuous labels for the real regression training/evaluation path.
    os.environ.pop('EVA_REG_BINS',None)
    raw=adapter.load(dataset)
    train_raw=adapter.Wrapped(raw['train'],raw)
    val=adapter.Wrapped(raw['val'],raw)
    test=adapter.Wrapped(raw['test'],raw)
    ns=functions(p.EVA_ROOT/'methods/kmeans.py',{'train_model','evaluate_model'},locals()|dict(tqdm=lambda x,**kw:x))
    rows=[]
    config=json.loads((p.HERE/'config.json').read_text())
    for case in [c for c in p.cases(config) if c['dataset']==dataset]:
        begin=time.perf_counter()
        np.random.seed(42);torch.manual_seed(42)
        fraction=case['fraction'];wanted=round(fraction*n)
        if case['baseline']=='COPS':
            v=case['variant'];script=scripts[v]
            score_cops=metrics[v]
            scope=functions(script,{'_sample_probs'},dict(np=np))
            scope.update(args=SimpleNamespace(test_mode='oracle_sampling_cut',sample_number=max(1,round(wanted/classes)),constant_1=float(np.median(score_cops))),
                         class_n=classes,train_metric=score_cops,train_lab=labels[pool])
            source=script.read_text();start=source.index("if args.test_mode == 'random':")
            exec(compile(source[start:source.index('\nif IS_EVA:',start)],str(script),'exec'),scope)
            idx=pool[np.asarray(scope['train_ind'])]
        elif case['baseline']=='GM_matching':
            with contextlib.redirect_stdout(io.StringIO()):
                idx=run_coreset_select(train,str(embedding_path),'gm_matching',wanted).numpy()
        else:
            idx=beta_ns['beta_sampling'](1-fraction,4,np.asarray(probs),mask,score)
        idx=p.valid_indices(idx,n)
        dest=p.case_dir(out,case);dest.mkdir(parents=True,exist_ok=True)
        np.save(dest/'indices.npy',idx)
        if meta['is_tabular']:
            model=adapter.build_model(raw,arch=None)
        else:
            model=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(meta['channel'],classes))
        reg=raw['is_regression'];task='regression' if reg else 'classification'
        loader=DataLoader(adapter.select(train_raw,idx),batch_size=16,shuffle=True)
        optimizer=torch.optim.SGD(model.parameters(),lr=.01)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=1)
        checkpoint=str(dest/'best.pt')
        train_info=ns['train_model'](model,loader,nn.MSELoss() if reg else nn.CrossEntropyLoss(),
            optimizer,scheduler,1,'cpu',checkpoint,task,DataLoader(val,batch_size=32),1)
        model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
        got=ns['evaluate_model'](model,DataLoader(test,batch_size=32),'cpu',task,return_details=True)
        if reg:
            base=raw['train']
            while not hasattr(base,'label_scaler'):base=base.base
            scale=float(base.label_scaler.scale_[0])
            got=dict(mse=float(got[0])*scale**2,mae=float(got[1])*scale)
        result=dict(**case,**got,**train_info,status='ok',scientific_results=False,seed=42,
                    n_train=n,n_selected=len(idx),requested_count=wanted,shortfall=max(0,wanted-len(idx)),
                    excess_count=max(0,len(idx)-wanted),
                    actual_fraction=len(idx)/n,elapsed_sec=time.perf_counter()-begin,
                    fixture_preparation=True,fixture_image_model=not meta['is_tabular'],
                    production_backbone_forward_backward_passed=True)
        p.save(dest/'pipeline_check.json',result)
        rows.append(result)
    p.save(out/(dataset+'_checks.json'),rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset')
    parser.add_argument('--out',default=str(p.HERE/'real_matrix_checks'))
    args=parser.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if args.dataset:
        check_dataset(args.dataset,out)
        return
    config=json.loads((p.HERE/'config.json').read_text())
    for ds in config['datasets']:
        with (out/(ds+'.log')).open('w') as f:
            subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--dataset',ds,'--out',str(out)],
                           stdout=f,stderr=subprocess.STDOUT,env=p.env(),check=True,timeout=600)
        print(ds,'16/16 checked',flush=True)
    rows=sum([json.loads((out/(ds+'_checks.json')).read_text()) for ds in config['datasets']],[])
    assert len(rows)==80 and all(r['status']=='ok' and not r['scientific_results'] for r in rows)
    p.save(out/'report.json',dict(status='ok',family_configurations=60,result_configurations=80,
                               scientific_results=False,rows=rows))


if __name__=='__main__':main()
