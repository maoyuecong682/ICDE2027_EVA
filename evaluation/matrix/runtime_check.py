"""Fail-fast checks before the single-GPU matrix is launched. No training."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import py_compile
import sys

import pipeline as p


def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-cuda',action='store_true')
    parser.add_argument('--out',default=str(p.HERE/'runtime_readiness.json'))
    args=parser.parse_args()
    config=json.loads((p.HERE/'config.json').read_text())
    report=dict(status='checking',errors=[],versions={},cuda=None,seed=config['seed'],
                expected_family_configurations=60,expected_results=80,training_started=False)
    if set(config['datasets'])!={'cifar100','adult','news','mnist2mnistm','House'} or config['fractions']!=[.1,.3,.5,.7] or config['cops_variants']!=['wy','woy']:
        report['errors'].append('Formal configuration does not contain the agreed 60 family configurations / 80 results')
    if config.get('smoke_limit') or not config.get('scientific_results',True):
        report['errors'].append('Formal configuration contains smoke-test settings')
    packages={'torch':'torch','torchvision':'torchvision','numpy':'numpy','pandas':'pandas',
              'scipy':'scipy','scikit-learn':'sklearn','Pillow':'PIL','tqdm':'tqdm',
              'matplotlib':'matplotlib','ftfy':'ftfy','regex':'regex','clip':'clip'}
    for distribution,module in packages.items():
        if importlib.util.find_spec(module) is None:report['errors'].append('Missing module: '+module)
        try:report['versions'][distribution]=importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:report['versions'][distribution]='installed from source'
    files=[p.BASE/'COPS-main/retraining_withy.py',p.BASE/'COPS-main/retraining_woy.py',
           p.BASE/'GM-Matching-main/sampling/algos.py',p.BASE/'GM-Matching-main/sampling/sample.py',
           p.BASE/'dual-pruning-main/exp_cifar/importance_evaluation.py',
           p.BASE/'dual-pruning-main/exp_cifar/data_subset.py',p.HERE/'sequence_worker.py',
           p.ADAPTER/'eva_adapter.py',p.ADAPTER/'clip_visual_cpu.py',p.EVA_ROOT/'methods/kmeans.py']
    for path in files:
        if not path.exists():report['errors'].append('Missing source: '+str(path))
        else:
            try:py_compile.compile(str(path),doraise=True)
            except py_compile.PyCompileError as e:report['errors'].append(str(e))
    data=Path('/root/autodl-tmp/EVA/data')
    required=[data/'cifar100/original_pixels_memmap/complete.json',data/'mnistm/original_encoded_memmap/complete.json',
              data/'adult/adult_train_split.csv',data/'adult/adult_val_split.csv',data/'adult/adult_test_split.csv',
              data/'House/train.csv',data/'House/test.csv',data/'mnistm/mnistm_test.parquet',
              Path('/root/.cache/clip/ViT-B-32.pt')]
    required += [data/'mnist2mnistm/MNIST/raw'/name for name in ('train-images-idx3-ubyte','train-labels-idx1-ubyte')]
    required += [data/'embedding'/('balance_news_'+split+'_'+part+'.npy')
                 for split in ('train','val','test') for part in ('embeddings','labels')]
    required += [data/'embedding'/('balance_news_'+part+'.npy') for part in ('mean','std')]
    for path in required:
        if not path.exists():report['errors'].append('Missing input: '+str(path))
    cache=data/'cifar100/original_pixels_memmap/complete.json'
    if cache.exists():
        manifest=json.loads(cache.read_text())
        for split,entry in manifest['splits'].items():
            for suffix,field in [('pixels','pixels_sha256'),('labels','labels_sha256')]:
                path=cache.parent/(split+'_'+suffix+'.npy')
                if not path.exists() or sha256(path)!=entry[field]:report['errors'].append('CIFAR cache integrity: '+str(path))
    mnistm=data/'mnistm/original_encoded_memmap/complete.json'
    if mnistm.exists():
        manifest=json.loads(mnistm.read_text())
        for filename,field in [('images.bin','blob_sha256'),('labels.npy','labels_sha256'),('offsets.npy','offsets_sha256')]:
            path=mnistm.parent/filename
            if not path.exists() or sha256(path)!=manifest[field]:report['errors'].append('MNIST-M cache integrity: '+str(path))
    clip=Path('/root/.cache/clip/ViT-B-32.pt')
    if clip.exists():
        report['clip_checkpoint_sha256']=sha256(clip)
        if report['clip_checkpoint_sha256']!='40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af':
            report['errors'].append('CLIP checkpoint SHA256 does not match official ViT-B/32')
    try:
        p.release_idle_input_pages(p.HERE)
        import torch
        torch.set_num_threads(1)
        report['cuda']=torch.cuda.is_available()
        report['torch_cuda_version']=torch.version.cuda
        if report['cuda']:
            report['gpu']=torch.cuda.get_device_name(0)
            report['gpu_total_memory_bytes']=torch.cuda.get_device_properties(0).total_memory
            x=torch.arange(16,device='cuda',dtype=torch.float32)
            assert float((x*x).sum())==1240.
            convolution=torch.nn.Conv2d(3,4,3).cuda()
            convolution(torch.randn(2,3,8,8,device='cuda')).square().mean().backward()
            torch.cuda.synchronize()
            report['cuda_operation_passed']=True
            report['cudnn_convolution_backward_passed']=True
        elif args.require_cuda:report['errors'].append('CUDA GPU is not available; no formal training started')
    except Exception as e:report['errors'].append('Torch/CUDA check: '+repr(e))
    report['status']='failed' if report['errors'] else ('ready' if report['cuda'] else 'cpu_prerequisites_passed_waiting_for_cuda')
    p.save(args.out,report)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
    raise SystemExit(1 if report['errors'] else 0)


if __name__=='__main__':main()
