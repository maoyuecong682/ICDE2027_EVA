"""Require all 80 real-data native end-to-end cases, not fixture-only coverage."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

import pipeline as p


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-root',action='append',required=True)
    parser.add_argument('--out',default=str(p.HERE/'native_acceptance.json'))
    parser.add_argument('--two-sample-root',default=str(p.HERE/'two_sample_checks'))
    args=parser.parse_args()
    expected=p.cases(json.loads((p.HERE/'config.json').read_text()))
    key=lambda case:tuple(case[k] for k in ('dataset','baseline','variant','fraction'))
    records={};preparations=[];runs=[]
    for name in args.native_root:
        root=Path(name);config=json.loads((root/'effective_config.json').read_text())
        assert config['scientific_results'] is False and config['seed']==42
        runs.append(dict(root=str(root),verification_config=config))
        for ds in config['datasets']:
            preflight=json.loads((root/'preflight'/(ds+'.json')).read_text())
            assert preflight['status']=='ok'
            n=preflight['counts']['train']
            for baseline in ('COPS','GM_matching','DUAL'):
                directory=root/'prepare'/baseline/ds
                if not (directory/'complete.json').exists():continue
                prep=json.loads((directory/'complete.json').read_text())
                assert prep['status']=='ok' and prep['fingerprint']==preflight['fingerprint']
                if baseline=='GM_matching':
                    meta=json.loads((directory/'embeds/embed_meta.json').read_text())
                    assert meta['roundtrip_ok'] and meta['n_rows']==n
                    if ds in ('cifar100','mnist2mnistm'):
                        assert meta['method']=='clip-vit-b32'
                        assert meta['checkpoint_sha256']=='40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af'
                    else:assert meta['method']=='raw-features(standardized-by-mydatasets)'
                elif baseline=='DUAL':
                    dynamics=directory/'checkpoint'/ds/str(config['seed'])/'npy'
                    for epoch in range(30):
                        indices=np.load(dynamics/f'{epoch}_Index.npy')
                        np.testing.assert_array_equal(np.sort(indices),np.arange(n))
                        output=np.load(dynamics/f'{epoch}_Output.npy',mmap_mode='r')
                        assert output.shape[0]==n and np.isfinite(output).all()
                    score=np.load(directory/'mask/dual_score_T30.npy')
                    assert score.shape==(n,) and np.isfinite(score).all()
                else:
                    logits=np.load(directory/'logits/train_UN_2_raw.npy',mmap_mode='r')
                    assert logits.shape[0]==config['cops_models'] and np.isfinite(logits).all()
                preparations.append(dict(dataset=ds,baseline=baseline,status='ok',root=str(directory)))
        for case in p.cases(config):
            dest=p.case_dir(root,case)
            if not (dest/'result.json').exists():continue
            result=json.loads((dest/'result.json').read_text())
            selection=json.loads((dest/'selection.json').read_text())
            assert result['status']==selection['status']=='ok'
            preflight=json.loads((root/'preflight'/(case['dataset']+'.json')).read_text())
            preparation=json.loads((root/'prepare'/case['baseline']/case['dataset']/'complete.json').read_text())
            assert selection['fingerprint']==preparation['fingerprint']==preflight['fingerprint']
            assert selection['n_train']==preflight['counts']['train']
            assert result.get('scientific_results',config['scientific_results']) is False
            idx=p.valid_indices(np.load(dest/'indices.npy'),selection['n_train'])
            assert hashlib.sha256(idx.tobytes()).hexdigest()==selection['index_sha256']
            assert len(idx)==result['n_selected']==selection['n_selected']
            metrics=('mae','mse') if case['dataset']=='House' else ('accuracy','precision','recall','f1','precision_macro','recall_macro','f1_macro')
            for metric in metrics:
                assert math.isfinite(result[metric]) and result[metric]>=0
                if case['dataset']!='House':assert result[metric]<=1
            if case['dataset']=='House':assert result['target_units']=='original'
            for field in ('train_sec','test_predict_metrics_sec','elapsed_sec'):
                assert math.isfinite(result[field]) and result[field]>=0
            assert (dest/'kmeans.pth').exists()
            assert key(case) not in records,'Duplicate case across verification runs'
            records[key(case)]=dict(**case,status='ok',result_path=str(dest/'result.json'),
                                    preparation_path=str(root/'prepare'/case['baseline']/case['dataset']/'complete.json'),
                                    n_train=selection['n_train'],n_selected=len(idx),
                                    requested_count=selection['requested_count'],actual_fraction=len(idx)/selection['n_train'],
                                    shortfall=max(0,selection['requested_count']-len(idx)),
                                    excess_count=max(0,len(idx)-selection['requested_count']),
                                    prepare_sec=preparation['elapsed_sec'],selection_sec=selection['elapsed_sec'],
                                    train_eval_sec=result['elapsed_sec'],train_sec=result['train_sec'],
                                    test_predict_metrics_sec=result['test_predict_metrics_sec'],
                                    seed=config['seed'],epochs_completed=result['epochs_completed'],
                                    scientific_results=False,**{metric:result[metric] for metric in metrics})
    native_expected={key(case) for case in expected}
    assert set(records)==native_expected,'Completed native evidence is missing a configuration'
    unique_preparations={(entry['dataset'],entry['baseline']) for entry in preparations}
    assert len(unique_preparations)==15,'All five datasets require all three native preparation families'
    two_sample=json.loads((Path(args.two_sample_root)/'report.json').read_text())
    assert two_sample['status']=='passed' and two_sample['samples_per_split']==2
    assert {row['dataset'] for row in two_sample['checks']}=={'cifar100','news','mnist2mnistm'}
    for row in two_sample['checks']:
        assert row['status']=='ok' and row['samples_per_split']==2 and row['production_backbone']
        assert row['scientific_results'] is False and row['epochs_completed']==1
    report=dict(status='passed',native_end_to_end_results=len(records),family_configurations=60,
                by_dataset=dict(Counter(row['dataset'] for row in records.values())),
                preparation_stages=len(unique_preparations),audited_preparation_artifacts=len(preparations),
                all_dynamics_cover_every_sample_in_30_epochs=True,
                image_embeddings_use_real_official_clip=True,scientific_results=False,
                formal_full_training_started=False,two_sample_checks=two_sample,
                verification_scope='All 80 cases: real-data native preparation and selection, production downstream training, validation-selected checkpoint, finite metrics and timing. Small-sample CPU checks only; CUDA performance not measured.',
                records=list(records.values()),preparations=preparations,runs=runs)
    p.save(args.out,report)
    csv_path=Path(args.out).with_suffix('.csv')
    columns=sorted(set().union(*(row.keys() for row in records.values())))
    with csv_path.open('w') as output:
        writer=csv.DictWriter(output,columns);writer.writeheader();writer.writerows(records.values())
    print(json.dumps({k:v for k,v in report.items() if k not in ('records','preparations','runs')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
