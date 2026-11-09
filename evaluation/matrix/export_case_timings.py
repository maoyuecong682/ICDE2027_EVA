"""Rebuild per-configuration timing outputs from existing raw stage records. No training."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

import pipeline as p


def export(root):
    root=Path(root).resolve()
    config=json.loads((root/'effective_config.json').read_text());matrix=p.cases(config)
    paths=set()
    for case in matrix:
        directory=p.case_dir(root,case)
        paths.update([root/'prepare'/case['baseline']/case['dataset']/'complete.json',
                      directory/'selection.json',directory/'result.json'])
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    before={str(path.relative_to(root)):digest(path) for path in sorted(paths) if path.exists()}
    # Validate every source before changing any generated outputs.
    rows=p.recorded_case_timings(root,matrix,config)
    complete=all(row['status']=='ok' for row in rows)
    legacy=root/'timing.json';aggregate=root/'matrix_timing.json'
    if legacy.exists() and not aggregate.exists():
        previous=json.loads(legacy.read_text())
        if 'invocation_wall_sec' in previous:aggregate.write_bytes(legacy.read_bytes())
    rows=p.write_case_timings(root,matrix,config)
    after={str(path.relative_to(root)):digest(path) for path in sorted(paths) if path.exists()}
    if before!=after:raise RuntimeError('Raw timing records changed during export')
    audit=dict(status='complete' if complete else 'incomplete',
        exported_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        expected_results=len(matrix),reconstructed_results=sum(row['status']=='ok' for row in rows),
        unique_preparation_records=len({row['source_files']['prepare_sec'] for row in rows}),
        selection_records=sum(row['selection_sec'] is not None for row in rows),
        train_eval_records=sum(row['train_eval_sec'] is not None for row in rows),
        training_rerun=False,raw_records_unchanged=True,
        source_sha256=before,
        decision='All configuration totals can be reconstructed without rerunning experiments' if complete else 'Some configurations lack successful measured stage records; inspect timing.json before rerunning only affected cases',
        timing_csv=str(root/'timing.csv'),timing_json=str(root/'timing.json'))
    p.save(root/'timing_reconstruction_audit.json',audit)
    return audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True)
    args=parser.parse_args()
    report=export(args.root)
    print(json.dumps({key:value for key,value in report.items() if key!='source_sha256'},ensure_ascii=False,indent=2))
    raise SystemExit(0 if report['status']=='complete' else 1)


if __name__=='__main__':main()
