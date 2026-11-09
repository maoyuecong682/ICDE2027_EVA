"""Functional checks using native selectors on fixtures; never scientific results."""
import ast
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import warnings

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

import pipeline as p

torch.set_num_threads(1)


def functions(path, names, namespace):
    tree=ast.parse(Path(path).read_text())
    selected=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names]
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),namespace)
    return namespace


class PipelineTests(unittest.TestCase):
    def test_matrix_and_paths(self):
        config=json.loads((p.HERE/'config.json').read_text())
        matrix=p.cases(config)
        self.assertEqual(len(matrix),80)
        self.assertEqual(len({(c['dataset'],c['baseline'],c['fraction']) for c in matrix}),60)
        self.assertEqual(len({str(p.case_dir(Path('/fixture'),c)) for c in matrix}),80)

    def test_native_selection_all_80_cases(self):
        """Run actual native sampling blocks, GM implementation, and DUAL beta."""
        sys.path.insert(0,str(p.BASE/'GM-Matching-main'))
        from sampling.sample import run_coreset_select
        d=p.BASE/'dual-pruning-main/exp_cifar'
        from scipy.stats import beta
        dual=functions(d/'data_subset.py',{'beta_sampling'},dict(np=np,torch=torch,beta=beta))
        config=json.loads((p.HERE/'config.json').read_text())
        rng=np.random.RandomState(7)
        count=0
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            for ds,ncls in [('cifar100',100),('adult',2),('news',20),('mnist2mnistm',10),('House',10)]:
                n=1000
                labels=np.arange(n)%ncls
                data=TensorDataset(torch.tensor(rng.randn(n,4),dtype=torch.float32),torch.tensor(labels))
                class IntegerLabels:
                    def __len__(self): return n
                    def __getitem__(self,i): return data[i][0],int(labels[i])
                embeddings=Path(tmp)/(ds+'.pt')
                torch.save(data.tensors[0],embeddings)
                pool=np.arange(int(.8*n))  # Eight equally sized class cycles.
                for fraction in config['fractions']:
                    wanted=round(fraction*n)
                    for variant in ('wy','woy'):
                        script=p.BASE/('COPS-main/retraining_withy.py' if variant=='wy' else 'COPS-main/retraining_woy.py')
                        ns=functions(script,{'_sample_probs'},dict(np=np))
                        ns.update(args=SimpleNamespace(test_mode='oracle_sampling_cut',sample_number=round(wanted/ncls),constant_1=.8),
                                  train_metric=rng.uniform(.1,1,len(pool)),train_lab=labels[pool],class_n=ncls)
                        source=script.read_text()
                        block=source[source.index("if args.test_mode == 'random':"):source.index('\nif IS_EVA:',source.index("if args.test_mode == 'random':"))]
                        exec(compile(block,str(script),'exec'),ns)
                        local=np.asarray(ns['train_ind'])
                        global_idx=p.valid_indices(pool[local],n)
                        self.assertEqual(len(global_idx),wanted)
                        count+=1
                    idx=run_coreset_select(IntegerLabels(),str(embeddings),'gm_matching',wanted).numpy()
                    self.assertEqual(len(p.valid_indices(idx,n)),wanted)
                    count+=1
                    probs=rng.uniform(.1,.9,(30,n))
                    score=rng.uniform(.1,1,n)
                    idx=dual['beta_sampling'](1-fraction,4,probs,np.argsort(score),score)
                    self.assertLessEqual(abs(len(p.valid_indices(idx,n))-wanted),1)
                    count+=1
        self.assertEqual(count,80)

    def test_cops_indices_are_mapped_and_no_variant_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)/'COPS/adult/weights_cops_wy'
            base.mkdir(parents=True)
            np.save(base/'selected_indices.npy',np.array([2,0]))
            class Pool:
                indices=np.array([10,20,30])
                def __len__(self):return len(self.indices)
            ns=functions(p.HERE.parent/'eval_baselines.py',{'_cops_indices'},
                         dict(os=__import__('os'),np=np,OUT_ROOT=tmp,
                              EVA=SimpleNamespace(cops_views=lambda ds:(None,Pool(),None,None))))
            idx,_=ns['_cops_indices']('adult','weights_cops_wy')
            np.testing.assert_array_equal(idx,[30,10])
            self.assertEqual(ns['_cops_indices']('adult','weights_cops_woy'),(None,None))

    def test_dual_score_is_not_mask(self):
        import glob
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)/'DUAL/adult/checkpoint/adult/42/generated_mask'
            base.mkdir(parents=True)
            np.save(base/'dual_mask_T30.npy',np.array([1,3,0,2]))
            np.save(base/'dual_score_T30.npy',np.array([.7,.1,.9,.3]))
            ns=functions(p.HERE.parent/'eval_baselines.py',{'_dual_indices'},dict(np=np,os=__import__('os'),glob=glob,OUT_ROOT=tmp))
            idx,_=ns['_dual_indices']('adult',dict(train=range(4)),.5)
            np.testing.assert_array_equal(idx,[0,2])

    def test_streamed_dual_probabilities_and_scores_are_bit_exact(self):
        import os
        import torch.nn.functional as F
        path=p.BASE/'dual-pruning-main/exp_cifar/importance_evaluation.py'
        scope=functions(path,{'stream_target_probs','dual'},dict(os=os,np=np,torch=torch,F=F))
        rng=np.random.RandomState(42)
        n,classes,epochs=23,5,30
        labels=torch.tensor(rng.randint(classes,size=n))
        canonical=[]
        with tempfile.TemporaryDirectory() as tmp:
            for epoch in range(epochs):
                indices=rng.permutation(n)
                logits=rng.randn(n,classes).astype(np.float32)
                np.save(Path(tmp)/f'{epoch}_Index.npy',indices)
                np.save(Path(tmp)/f'{epoch}_Output.npy',logits)
                canonical.append(torch.zeros_like(torch.from_numpy(logits)).index_add(0,torch.tensor(indices),torch.from_numpy(logits)))
            expected=F.softmax(torch.stack(canonical),dim=2).gather(2,labels[None,:,None].expand(epochs,-1,-1)).squeeze(2)
            actual=scope['stream_target_probs'](tmp,labels,epochs,chunk_size=7)
            self.assertTrue(torch.equal(actual,expected))
            expected_score,expected_mask=scope['dual'](expected)
            score,mask=scope['dual'](actual)
            self.assertTrue(torch.equal(score,expected_score))
            np.testing.assert_array_equal(mask,expected_mask)
            np.save(Path(tmp)/'0_Index.npy',np.zeros(n,dtype=np.int64))
            with self.assertRaisesRegex(ValueError,'every sample exactly once'):
                scope['stream_target_probs'](tmp,labels,epochs)

    def test_regression_loss_singleton_metrics_and_float_labels(self):
        from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,mean_absolute_error,mean_squared_error
        import torch.nn as nn
        ns=functions(p.EVA_ROOT/'methods/kmeans.py',{'evaluate_model','train_model'},locals()|dict(torch=torch,tqdm=lambda x,**kw:x))
        x=torch.tensor([[.25],[1.75],[2.5]])
        loader=DataLoader(TensorDataset(x,x.clone()),batch_size=2)
        mse,mae=ns['evaluate_model'](nn.Identity(),loader,'cpu','regression')
        self.assertEqual((mse,mae),(0.,0.))
        plain=functions(p.HERE.parent/'eval_baselines.py',{'_Plain'},dict(Dataset=torch.utils.data.Dataset))['_Plain']
        self.assertEqual(float(plain(TensorDataset(x,x))[0][1]),.25)
        with tempfile.TemporaryDirectory() as tmp, warnings.catch_warnings():
            warnings.simplefilter('error',UserWarning)
            model=nn.Linear(1,1)
            optimizer=torch.optim.SGD(model.parameters(),lr=.01)
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=2)
            ns['train_model'](model,loader,nn.MSELoss(),optimizer,scheduler,2,'cpu',str(Path(tmp)/'best.pt'),
                              'regression',loader,2)
            self.assertTrue((Path(tmp)/'best.pt').exists())

    def test_invalid_selections_fail(self):
        for invalid in ([],[1,1],[-1],[5],[1.1]):
            with self.assertRaises(ValueError):p.valid_indices(np.asarray(invalid),5)

    def test_clip_receives_denormalized_pixels(self):
        import time
        from unittest.mock import patch
        class Encoder:
            visual=SimpleNamespace(output_dim=3)
            text_projection=torch.empty(3,3)
            checkpoint_sha256='unit-fixture'
            def eval(self):return self
            def to(self,device):return self
            def encode_image(self,x):return x.mean(dim=(2,3))
        encoder=Encoder()
        clip=SimpleNamespace(load=lambda *a,**kw:(encoder,None))
        mean=torch.tensor([.5,.4,.3]).reshape(3,1,1)
        std=torch.tensor([.2,.3,.4]).reshape(3,1,1)
        raw=torch.tensor([0.,.5,1.]).reshape(3,1,1)
        sample=(raw-mean)/std
        meta=dict(train=TensorDataset(sample.unsqueeze(0),torch.tensor([0])),
                  mean=mean.reshape(-1).tolist(),std=std.reshape(-1).tolist())
        ns=functions(p.ADAPTER/'prepare_eva_embed.py',{'embed_images'},dict(np=np,torch=torch,time=time,to_clip_batch=lambda x:x))
        with patch.dict(sys.modules,{'clip_visual_cpu':SimpleNamespace(load_visual=lambda:encoder)}),contextlib.redirect_stdout(io.StringIO()):
            embedding,_=ns['embed_images'](meta,'cpu',1,0)
        torch.testing.assert_close(embedding,raw.reshape(1,3))


if __name__=='__main__':
    unittest.main(verbosity=2)
