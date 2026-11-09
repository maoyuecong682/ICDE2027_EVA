"""Show progress without importing Torch or touching the running experiment."""
import argparse
import json
import os
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('--out',default=str(Path(__file__).resolve().parent/'production_runs'))
args=parser.parse_args();root=Path(args.out)
manifests=sorted(root.glob('*/manifest.json'),key=lambda path:path.stat().st_mtime)
state={'running':False,'completed':0,'failed':0,'expected':80}
pid_file=root/'launcher.pid'
if pid_file.exists():
    pid=int(pid_file.read_text().strip());state['pid']=pid
    try:os.kill(pid,0);state['running']=True
    except ProcessLookupError:pass
if manifests:
    run=manifests[-1].parent;state['run_root']=str(run)
    manifest=json.loads(manifests[-1].read_text());state['expected']=manifest['result_configurations']
    for path in run.glob('*/*/fraction_*/*/result.json'):
        result=json.loads(path.read_text());state['completed']+=result.get('status')=='ok';state['failed']+=result.get('status')=='failed'
    state['preparation_stages_completed']=sum(json.loads(path.read_text()).get('status')=='ok' for path in run.glob('prepare/*/*/complete.json'))
    state['failed_stages']=[str(path.relative_to(run)) for pattern in
        ('preflight/*.json','prepare/*/*/complete.json','*/*/fraction_*/*/selection.json')
        for path in run.glob(pattern) if json.loads(path.read_text()).get('status')=='failed']
    state['pending_results']=state['expected']-state['completed']-state['failed']
if (root/'latest_log.txt').exists():state['log']=(root/'latest_log.txt').read_text().strip()
print(json.dumps(state,ensure_ascii=False,indent=2))
