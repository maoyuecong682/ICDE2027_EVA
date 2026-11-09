"""Isolated 3 x 4 x 5 experiment DAG. No historical artifacts are reused implicitly."""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
BASE = Path('/root/autodl-tmp/EVA/baseline')
EVA_ROOT = Path(os.environ.get('EVA_ROOT', '/root/chaichengliang/VLDB2027/EVA/EVA'))
ADAPTER = Path('/root/autodl-tmp/eva_output')


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def cases(config):
    return [dict(dataset=ds, fraction=f, baseline=bl, variant=v)
            for ds in config['datasets'] for f in config['fractions']
            for bl, variants in [('COPS', config['cops_variants']),
                                 ('GM_matching', ['gm_matching']), ('DUAL', ['beta'])]
            for v in variants]


def case_dir(root, case):
    return root / case['baseline'] / case['dataset'] / ('fraction_%s' % case['fraction']) / case['variant']


def env():
    result = dict(os.environ)
    result.update(PYTHONPATH=str(ADAPTER), EVA_ROOT=str(EVA_ROOT),
                  OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', WANDB_MODE='offline')
    result.pop('EVA_SMOKE_LIMIT', None)
    result.pop('EVA_REG_BINS', None)
    return result


def release_idle_input_pages(root):
    # CPU-only verification has a 2 GiB cgroup, including filesystem cache.
    # Evict only our own cold input/checkpoint pages, preserving all file data.
    if not hasattr(os,'posix_fadvise'):return
    try:
        limit=Path('/sys/fs/cgroup/memory.max').read_text().strip()
        current=int(Path('/sys/fs/cgroup/memory.current').read_text())
    except (OSError,ValueError):return
    if not limit.isdigit() or int(limit)>4*1024**3:return
    if current<int(limit)*.6:return
    data=Path('/root/autodl-tmp/EVA/data/cifar100')
    files=[Path('/root/.cache/clip/ViT-B-32.pt'),data/'cifar-100-python/train',data/'cifar-100-python/test',data/'cifar-100-python.tar.gz']
    files+=list((data/'original_pixels_memmap').glob('*.npy'))
    files+=list(root.glob('**/*.pth'))+list(root.glob('**/*.pt'))+list(root.glob('**/*.pth.tar'))
    for path in files:
        if path.is_file():
            with path.open('rb') as f:os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED)


def native(script, args, log, extra_env=None):
    child_env = dict(os.environ)
    child_env.update(PYTHONPATH=str(ADAPTER), EVA_ROOT=str(EVA_ROOT))
    child_env.update(extra_env or {})
    with Path(log).open('w') as f:
        subprocess.run([sys.executable, '-u', str(script), *map(str, args)],
                       cwd=script.parent, env=child_env, stdout=f, stderr=subprocess.STDOUT, check=True)


def load_adapter():
    sys.path.insert(0, str(ADAPTER))
    import eva_adapter as adapter
    adapter.set_threads(1)
    return adapter


def valid_indices(idx, n):
    import numpy as np
    idx = np.asarray(idx)
    if idx.ndim != 1 or idx.dtype.kind not in 'iu' or not len(idx):
        raise ValueError('Selection must be a nonempty integer vector')
    if idx.min() < 0 or idx.max() >= n or len(np.unique(idx)) != len(idx):
        raise ValueError('Selection contains duplicates or out-of-range indices')
    return idx.astype(np.int64)


def fingerprint(meta, adapter):
    import numpy as np
    h = hashlib.sha256()
    for split in ('train', 'val', 'test'):
        ds = meta[split]
        h.update(str(len(ds)).encode())
        labels = adapter._targets_float(ds) if meta['is_regression'] else adapter._labels_of(ds)
        h.update(np.asarray(labels).tobytes())
        # Include deterministic feature sentinels; RNG is restored afterwards.
        import torch
        state = torch.get_rng_state()
        for i in sorted(set((0, len(ds)//2, len(ds)-1))):
            torch.manual_seed(48)
            h.update(np.asarray(ds[i][0]).tobytes())
        torch.set_rng_state(state)
    return h.hexdigest()


def inspect_dataset(ds):
    import numpy as np
    import torch
    adapter = load_adapter()
    meta = adapter.load(ds)
    info = dict(dataset=ds, counts={s: len(meta[s]) for s in ('train','val','test')},
                classes=meta['num_classes'], is_regression=meta['is_regression'],
                is_tabular=meta['is_tabular'], fingerprint=fingerprint(meta, adapter))
    info['selection_classes']=meta['num_classes']
    if meta['is_regression']:
        base=meta['train']
        while hasattr(base,'base'):base=base.base
        edges=np.unique(np.quantile(adapter._targets_float(base),np.linspace(0,1,11)[1:-1]))
        info['selection_classes']=len(edges)+1
    for s in ('train','val','test'):
        x, y = adapter.Wrapped(meta[s], meta)[0]
        if not torch.isfinite(x).all() or not np.isfinite(np.asarray(y)).all():
            raise ValueError('Nonfinite sample: ' + s)
    if not meta['is_regression']:
        labels = adapter.Wrapped(meta['train'], meta).labels
        _, counts = np.unique(labels, return_counts=True)
        _, pool = adapter.split_train_probe(labels)
        pool_labels = labels[pool]
        info['native_quota_counts'] = []
        for f in (.1,.3,.5,.7):
            quota = max(1, round(f*len(labels)/len(counts)))
            info['native_quota_counts'].append(dict(fraction=f,
                requested=round(f*len(labels)),
                cops_wy=sum(min(quota, int((pool_labels==c).sum())) for c in range(len(counts))),
                gm=sum(max(1,min(int(round(f*len(labels)))//len(counts), int(n))) for n in counts)))
    info['cuda'] = torch.cuda.is_available()
    info['gpu'] = torch.cuda.get_device_name(0) if info['cuda'] else None
    return info


def execute_sequence(path,steps,result_path,preset,started,selection=None):
    save(path,dict(steps=steps,result_path=str(result_path),preset=preset,started=started,selection=selection))
    os.execvpe(sys.executable,[sys.executable,'-u',str(HERE/'sequence_worker.py'),str(path)],dict(os.environ))


def prepare(config, root, ds, bl, result_path=None, started=None):
    # Build commands from the already validated preflight, before importing Torch.
    meta=json.loads((root/'preflight'/(ds+'.json')).read_text())
    p=root/'prepare'/bl/ds
    p.mkdir(parents=True,exist_ok=True)
    seed=config['seed'];steps=[]
    def queue(script,args,log):
        steps.append(dict(script=str(script),args=list(map(str,args)),log=str(log)))
    if bl=='COPS':
        queue(BASE/'COPS-main/uncertainty_generation.py',
              ['--dataset',ds,'--model','resnet18','--model_num',config['cops_models'],
               '--num_classes',meta.get('selection_classes',meta['classes']),
               '--epochs',config['cops_prepare_epochs'],'--batch_size',config['batch_size'],
               '--lr',.001,'--weight_folder',p/'weights','--npy_folder',p/'logits','--selection-only'],p/'native.log')
        queue(ADAPTER/'calibrate_adult_metric.py',['--npy_folder',p/'logits','--out',p/'metric_stats.json'],p/'calibrate.log')
    elif bl=='GM_matching':
        arguments=['--dataset',ds,'--out_dir',p/'embeds','--batch_size',config['batch_size'],'--force']
        cached=ADAPTER/'GM_matching'/ds/'embeds'/ds/'embeddings.pt'
        if meta['is_tabular'] and cached.exists() and not config.get('smoke_limit'):
            arguments += ['--reuse-source',cached]
        queue(ADAPTER/'prepare_eva_embed.py',arguments,p/'native.log')
    else:
        d=BASE/'dual-pruning-main/exp_cifar'
        train_arguments=['--dataset',ds,'--data_path','dummy','--arch','resnet18',
              '--epochs',config['dual_prepare_epochs'],'--batch-size',config['batch_size'],
              '--dynamics-epochs',30,
              '--manualSeed',seed,'--workers',0 if meta['is_tabular'] or not meta['cuda'] else 4,
              '--dynamics','--save_path',p/'checkpoint']
        if not config.get('scientific_results',True):
            train_arguments += ['--max-update-batches',config.get('smoke_max_update_batches',1)]
        queue(d/'train.py',train_arguments,p/'native.log')
        queue(d/'importance_evaluation.py',['--dataset',ds,'--num_epochs',config['dual_prepare_epochs'],
              '--dynamics_path',str(p/'checkpoint'/ds/str(seed)/'npy')+'/',
              '--save_path',p/'mask','--dual-only'],p/'importance.log')
    preset=dict(dataset=ds,baseline=bl,fingerprint=meta['fingerprint'],cuda=meta['cuda'],
                seed=seed,reused_artifact=False,scientific_results=config.get('scientific_results',True))
    if bl=='DUAL':
        preset.update(dynamics_epochs=min(30,config['dual_prepare_epochs']),
                      learning_rate_horizon_epochs=config['dual_prepare_epochs'],
                      max_update_batches=0 if config.get('scientific_results',True) else config.get('smoke_max_update_batches',1))
    if ds=='House':os.environ['EVA_REG_BINS']=str(config['regression_bins'])
    execute_sequence(p/'native_sequence.json',steps,result_path,preset,started)


def select_cops(config,root,case,result_path,started):
    ds=case['dataset'];variant=case['variant'];fraction=case['fraction']
    meta=json.loads((root/'preflight'/(ds+'.json')).read_text())
    prep=root/'prepare/COPS'/ds
    info=json.loads((prep/'complete.json').read_text())
    if info['fingerprint']!=meta['fingerprint']:raise ValueError('COPS preflight and preparation differ')
    dest=case_dir(root,case);dest.mkdir(parents=True,exist_ok=True)
    n=meta['counts']['train'];wanted=round(n*fraction)
    per_class=max(1,round(wanted/meta.get('selection_classes',meta['classes'])))
    clip=json.loads((prep/'metric_stats.json').read_text())['constant_1']
    import math
    if not math.isfinite(clip) or clip<=0:raise ValueError('Degenerate COPS uncertainty threshold')
    script=BASE/('COPS-main/retraining_withy.py' if variant=='wy' else 'COPS-main/retraining_woy.py')
    arguments=['--dataset',ds,'--test_mode','oracle_sampling_cut','--constant_1',clip,
               '--sample_number',per_class,'--seed',config['seed'],'--un_folder',prep/'logits',
               '--weight_folder',dest,'--select-only']
    if ds=='House':os.environ['EVA_REG_BINS']=str(config['regression_bins'])
    execute_sequence(dest/'native_sequence.json',
        [dict(script=str(script),args=list(map(str,arguments)),log=str(dest/'native.log'))],result_path,
        dict(**case,fingerprint=info['fingerprint'],scientific_results=config.get('scientific_results',True)),
        started,selection=dict(source=str(dest/'selected_train_indices.npy'),destination=str(dest/'indices.npy'),n_train=n,wanted=wanted))


def select(config, root, case, result_path=None, started=None):
    if case['baseline']=='COPS':return select_cops(config,root,case,result_path,started)
    import numpy as np
    import torch
    adapter = load_adapter()
    ds, bl, variant, fraction = (case[k] for k in ('dataset','baseline','variant','fraction'))
    if ds == 'House':
        os.environ['EVA_REG_BINS'] = str(config['regression_bins'])
    meta = adapter.load(ds)
    prep = root/'prepare'/bl/ds
    prep_info = json.loads((prep/'complete.json').read_text())
    if fingerprint(meta,adapter) != prep_info['fingerprint']:
        raise ValueError('Preparation and selection datasets differ')
    dest = case_dir(root,case)
    dest.mkdir(parents=True,exist_ok=True)
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    n = len(meta['train'])
    wanted = round(n*fraction)
    if bl == 'GM_matching':
        sys.path.insert(0,str(BASE/'GM-Matching-main'))
        from sampling.sample import run_coreset_select
        idx = run_coreset_select(adapter.Wrapped(meta['train'],meta),
              str(prep/'embeds/embeddings.pt'),'gm_matching',wanted).numpy()
    else:
        sys.path.insert(0,str(BASE/'dual-pruning-main/exp_cifar'))
        from data_subset import beta_sampling
        mask = np.load(prep/'mask/dual_mask_T30.npy')
        score = np.load(prep/'mask/dual_score_T30.npy')
        probs = np.load(prep/'mask/target_probs.npy')
        idx = beta_sampling(1-fraction,4,probs,mask,score)
    idx = valid_indices(idx,n)
    np.save(dest/'indices.npy',idx)
    return dict(**case,n_train=n,n_selected=len(idx),requested_count=wanted,
                actual_fraction=len(idx)/n,shortfall=max(0,wanted-len(idx)),
                excess_count=max(0,len(idx)-wanted),
                fingerprint=prep_info['fingerprint'],index_sha256=hashlib.sha256(idx.tobytes()).hexdigest())


def evaluate(config, root, case):
    import numpy as np
    import torch
    adapter = load_adapter()
    os.environ.pop('EVA_REG_BINS',None)
    meta = adapter.load(case['dataset'])
    dest = case_dir(root,case)
    selection = json.loads((dest/'selection.json').read_text())
    if fingerprint(meta,adapter) != selection['fingerprint']:
        raise ValueError('Selection and evaluation datasets differ')
    idx = valid_indices(np.load(dest/'indices.npy'),len(meta['train']))
    from types import SimpleNamespace
    sys.path.insert(0,str(HERE.parent))
    import eval_baselines as ev
    ev._install_stubs()
    args = SimpleNamespace(epochs=config['eval_epochs'],patience=config['patience'],
             lr=.01 if meta['is_tabular'] else .1,batch_size=config['batch_size'])
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    import random
    random.seed(config['seed'])
    train = adapter.select(adapter.Wrapped(meta['train'],meta),idx)
    val = adapter.Wrapped(meta['val'],meta)
    test = adapter.Wrapped(meta['test'],meta)
    result = ev.run_kmeans_routine(train,val,test,meta,args,str(dest))
    required = ('mse','mae') if meta['is_regression'] else ('accuracy','precision','recall','f1')
    if not all(isinstance(result.get(k),(float,int)) and np.isfinite(result[k]) for k in required):
        raise ValueError('Missing or nonfinite metrics')
    if meta['is_regression']:
        base = meta['train']
        while not hasattr(base,'label_scaler') and hasattr(base,'base'):
            base = base.base
        scaler = base.label_scaler
        scale = float(scaler.scale_[0])
        result.update(mse_standardized=result['mse'],mae_standardized=result['mae'],
                      mse=result['mse']*scale*scale,mae=result['mae']*scale)
    result.update(case)
    result.update(n_train=selection['n_train'],requested_count=selection['requested_count'],
                  shortfall=selection['shortfall'],excess_count=selection.get('excess_count',0),
                  n_selected=selection['n_selected'],actual_fraction=selection['actual_fraction'],
                  metric_average='weighted' if not meta['is_regression'] else None,
                  target_units='original' if meta['is_regression'] else None,
                  gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                  seed=config['seed'],hit_epoch_cap=not result.get('stopped_early',False))
    return result


def worker(args,config,root):
    started = time.perf_counter()
    if args.worker == 'inspect':
        result = inspect_dataset(args.dataset)
    elif args.worker == 'prepare':
        result = prepare(config,root,args.dataset,args.baseline,args.result,started)
    else:
        case = dict(dataset=args.dataset,baseline=args.baseline,variant=args.variant,fraction=args.fraction)
        result = select(config,root,case,args.result,started) if args.worker=='select' else evaluate(config,root,case)
    result.update(status='ok',elapsed_sec=time.perf_counter()-started,
                  scientific_results=config.get('scientific_results',True))
    save(args.result,result)


def launch(args,config,root,kind,result,**kw):
    result = Path(result)
    if kind!='inspect' and result.exists() and json.loads(result.read_text()).get('status')=='ok':
        return True
    result.parent.mkdir(parents=True,exist_ok=True)
    cmd = [sys.executable,'-u',str(Path(__file__).resolve()),'--config',str(Path(args.config).resolve()),
           '--out',str(root),'--worker',kind,'--result',str(result)]
    for k,v in kw.items():
        cmd += ['--'+k,str(v)]
    start = time.perf_counter()
    release_idle_input_pages(root)
    child_env=env()
    if config.get('smoke_limit'):
        child_env['EVA_SMOKE_LIMIT']=str(config['smoke_limit'])
    with result.with_suffix('.log').open('w') as f:
        try:
            proc = subprocess.Popen(cmd,env=child_env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            proc.wait(timeout=args.timeout)
            ok = proc.returncode==0 and result.exists()
        except subprocess.TimeoutExpired:
            # Native preparation may have a child process: stop the entire stage.
            os.killpg(proc.pid,signal.SIGKILL)
            proc.wait()
            proc=None
            ok=False
    if not ok:
        save(result,dict(status='failed',stage=kind,returncode=proc.returncode if proc else 'timeout',
                         elapsed_sec=time.perf_counter()-start,**kw))
    print(kind,kw,'ok' if ok else 'FAILED',flush=True)
    return ok


def recorded_case_timings(root,matrix,config):
    """One full preparation charge plus selection/evaluation wall time per case."""
    rows=[]
    for case in matrix:
        directory=case_dir(root,case)
        paths={'prepare_sec':root/'prepare'/case['baseline']/case['dataset']/'complete.json',
               'selection_sec':directory/'selection.json',
               'train_eval_sec':directory/'result.json'}
        row=dict(**case,status='ok',total_sec=None,
                 scientific_results=config.get('scientific_results',True),
                 seed=config['seed'],gpu=None,reused_preparation_artifact=False,
                 source_files={field:str(path.relative_to(root)) for field,path in paths.items()})
        for field,path in paths.items():
            row[field]=None
            if not path.exists():
                if row['status']=='ok':row['status']='missing'
                continue
            stage=json.loads(path.read_text())
            if stage.get('status')!='ok':
                row['status']='failed'
                continue
            for key in ('dataset','baseline'):
                if stage.get(key)!=case[key]:raise ValueError('Timing stage identity mismatch: '+str(path))
            if field!='prepare_sec':
                for key in ('variant','fraction'):
                    if stage.get(key)!=case[key]:raise ValueError('Timing case identity mismatch: '+str(path))
            seconds=stage.get('elapsed_sec')
            if isinstance(seconds,bool) or not isinstance(seconds,(int,float)) or not math.isfinite(seconds) or seconds<0:
                raise ValueError('Missing/invalid measured elapsed_sec: '+str(path))
            row[field]=float(seconds)
            if field=='prepare_sec':row['reused_preparation_artifact']=stage.get('reused_artifact',False)
            if field=='train_eval_sec':row['gpu']=stage.get('gpu')
        if row['status']=='ok':
            row['total_sec']=sum(row[field] for field in paths)
        rows.append(row)
    return rows


def write_case_timings(root,matrix,config):
    rows=recorded_case_timings(root,matrix,config)
    save(root/'timing.json',dict(schema_version=2,unit='seconds',
        definition='total_sec = prepare_sec + selection_sec + train_eval_sec',
        measurement='Sum of recorded stage wall times; not separately measured cold-start runs',
        preparation_accounting='Charge the full shared preparation measurement to each configuration; do not divide it across fractions or COPS variants',
        excludes='Matrix-wide preflight/orchestration outside stage timers and historical cache generation',
        expected_results=len(matrix),ok_results=sum(row['status']=='ok' for row in rows),
        scientific_results=config.get('scientific_results',True),results=rows))
    identity=['dataset','baseline','variant','fraction']
    for filename,columns in [('timing.csv',identity+['total_sec']),
        ('timing_breakdown.csv',identity+['prepare_sec','selection_sec','train_eval_sec','total_sec',
                                        'reused_preparation_artifact','gpu','seed','status'])]:
        path=root/filename;tmp=path.with_suffix(path.suffix+'.tmp')
        with tmp.open('w') as output:
            writer=csv.DictWriter(output,columns);writer.writeheader()
            for row in rows:
                writer.writerow({column:(format(row[column],'.6f') if column.endswith('_sec') and row[column] is not None else row[column]) for column in columns})
        tmp.replace(path)
    return rows


def summarize(root,matrix,config,invocation_started=None):
    records=[]
    for case in matrix:
        result=case_dir(root,case)/'result.json'
        record=json.loads(result.read_text()) if result.exists() else dict(**case,status='missing')
        record['scientific_results']=config.get('scientific_results',True)
        d=case_dir(root,case)
        for label,path in [('prepare',root/'prepare'/case['baseline']/case['dataset']/'complete.json'),
                           ('selection',d/'selection.json')]:
            if path.exists():
                stage=json.loads(path.read_text())
                record[label+'_sec']=stage.get('elapsed_sec')
                if label=='prepare':
                    record['reused_preparation_artifact']=stage.get('reused_artifact',False)
                if label=='selection':
                    for key in ('n_train','n_selected','requested_count','actual_fraction'):
                        record[key]=stage[key]
                    record['shortfall']=max(0,stage['requested_count']-stage['n_selected'])
                    record['excess_count']=max(0,stage['n_selected']-stage['requested_count'])
        record['train_eval_sec']=record.pop('elapsed_sec',None)
        if record['status']=='ok':
            record['total_sec']=sum(record[k] for k in ('prepare_sec','selection_sec','train_eval_sec'))
            record['independent_total_sec']=record['total_sec']
        records.append(record)
    save(root/'summary.json',records)
    preparation_sec=sum(json.loads(path.read_text()).get('elapsed_sec',0)
                        for path in (root/'prepare').glob('*/*/complete.json'))
    save(root/'matrix_timing.json',dict(invocation_wall_sec=(time.perf_counter()-invocation_started if invocation_started is not None else None),
        preparation_sec=preparation_sec,
        selection_sec=sum(r.get('selection_sec') or 0 for r in records),
        train_eval_sec=sum(r.get('train_eval_sec') or 0 for r in records),
        expected_results=len(matrix),ok_results=sum(r['status']=='ok' for r in records),
        scientific_results=config.get('scientific_results',True)))
    write_case_timings(root,matrix,config)
    columns=sorted(set().union(*(r.keys() for r in records)))
    with (root/'summary.csv').open('w') as f:
        w=csv.DictWriter(f,columns);w.writeheader();w.writerows(records)
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default=str(HERE/'config.json'))
    p.add_argument('--out',default=str(HERE/'runs'))
    p.add_argument('--mode',choices=['plan','preflight','smoke','run'],default='plan')
    p.add_argument('--datasets',nargs='+')
    p.add_argument('--timeout',type=int,default=7200)
    p.add_argument('--smoke-limit',type=int,default=200)
    p.add_argument('--smoke-batch-size',type=int,default=8)
    p.add_argument('--worker',choices=['inspect','prepare','select','evaluate'])
    p.add_argument('--dataset')
    p.add_argument('--baseline')
    p.add_argument('--variant')
    p.add_argument('--fraction',type=float)
    p.add_argument('--result')
    args=p.parse_args()
    config=json.loads(Path(args.config).read_text())
    invocation_started=time.perf_counter()
    root=Path(args.out).resolve()
    if args.worker:
        worker(args,config,root)
        return
    if args.datasets:
        if not set(args.datasets)<=set(config['datasets']):
            raise SystemExit('Unknown datasets')
        config['datasets']=args.datasets
    if args.mode=='smoke':
        config.update(smoke_limit=args.smoke_limit,cops_models=2,cops_prepare_epochs=1,
                      dual_prepare_epochs=30,eval_epochs=1,patience=1,batch_size=args.smoke_batch_size,
                      smoke_max_update_batches=1,
                      scientific_results=False)
    if args.mode=='run' and (config.get('smoke_limit') or not config.get('scientific_results',True)):
        raise SystemExit('Formal run cannot use smoke limits or a non-scientific configuration')
    sources=[Path(__file__),HERE.parent/'eval_baselines.py',ADAPTER/'eva_adapter.py',
             ADAPTER/'prepare_eva_embed.py',ADAPTER/'clip_visual_cpu.py',HERE/'sequence_worker.py',ADAPTER/'calibrate_adult_metric.py',
             EVA_ROOT/'methods/kmeans.py',EVA_ROOT/'mydatasets/House.py',EVA_ROOT/'mydatasets/news.py']
    sources += list((BASE/'COPS-main').glob('*.py'))
    sources += list((BASE/'GM-Matching-main/sampling').glob('*.py'))
    sources += list((BASE/'dual-pruning-main/exp_cifar').glob('*.py'))
    sources += list((EVA_ROOT/'mydatasets').glob('*.py'))
    sources += [EVA_ROOT/'evaluation/eval_runner.py']
    source_hashes={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    digest=hashlib.sha256(json.dumps([config,source_hashes],sort_keys=True).encode()).hexdigest()[:12]
    root=root/digest
    save(root/'effective_config.json',config)
    args.config=str(root/'effective_config.json')
    matrix=cases(config)
    save(root/'manifest.json',dict(config=config,source_hashes=source_hashes,
                                 family_configurations=len(config['datasets'])*len(config['fractions'])*3,
                                 result_configurations=len(matrix),cases=matrix))
    print('Run root:',root,'; cases:',len(matrix),flush=True)
    if args.mode=='plan':
        return
    all_ok=True
    for ds in config['datasets']:
        all_ok=launch(args,config,root,'inspect',root/'preflight'/(ds+'.json'),dataset=ds) and all_ok
    if args.mode=='preflight':
        raise SystemExit(0 if all_ok else 1)
    pending=[k for k in ('evaluation','regression_policy','quota_policy') if config[k]=='pending']
    if not config['variants_confirmed']:
        pending.append('variants_confirmed')
    if pending:
        raise SystemExit('Pending user decisions: '+', '.join(pending))
    if config['evaluation']!='unified' or config['quota_policy']!='native':
        raise SystemExit('This runner currently supports unified evaluation and native quotas only')
    if config['regression_policy']!='binned':
        raise SystemExit('House selection requires an explicitly selected regression policy')
    info=[json.loads((root/'preflight'/(ds+'.json')).read_text()) for ds in config['datasets']]
    if not all_ok:
        raise SystemExit('Dataset preflight failed; inspect preflight/*.json and *.log')
    if args.mode!='smoke' and not all(r.get('cuda') for r in info):
        raise SystemExit('Formal full run requires an available CUDA GPU')
    for ds in config['datasets']:
        for bl in ('COPS','GM_matching','DUAL'):
            if not launch(args,config,root,'prepare',root/'prepare'/bl/ds/'complete.json',dataset=ds,baseline=bl):
                all_ok=False
                continue
            for case in [c for c in matrix if c['dataset']==ds and c['baseline']==bl]:
                d=case_dir(root,case)
                ok=launch(args,config,root,'select',d/'selection.json',**case)
                if ok:
                    ok=launch(args,config,root,'evaluate',d/'result.json',**case)
                all_ok=ok and all_ok
    summarize(root,matrix,config,invocation_started)
    raise SystemExit(0 if all_ok else 1)


if __name__=='__main__':
    main()
