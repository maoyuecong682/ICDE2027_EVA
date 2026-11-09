"""Small supervisor: native training runs after the Torch parent has been exec'd away."""
import hashlib
import json
from pathlib import Path
import sys
import time

import pipeline as p

job=json.loads(Path(sys.argv[1]).read_text())
print('Native supervisor RSS:',next(line for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('VmRSS:')),flush=True)
for step in job['steps']:
    p.native(Path(step['script']),step['args'],step['log'])
result=job['preset']
if result.get('baseline')=='GM_matching':
    meta_path=Path(job['result_path']).parent/'embeds/embed_meta.json'
    if not meta_path.exists():raise ValueError('Embedding metadata missing after preparation')
    meta=json.loads(meta_path.read_text())
    if not meta.get('roundtrip_ok'):raise ValueError('Embedding roundtrip validation failed')
    if meta.get('method') not in ('clip-vit-b32','raw-features(standardized-by-mydatasets)'):
        raise ValueError('Matrix preparation rejects fixture/unknown embeddings')
    if meta.get('n_rows')!=meta.get('adapter_n_train'):
        raise ValueError('Matrix embedding rows do not match the training set')
    if meta['method']=='clip-vit-b32' and meta.get('checkpoint_sha256')!='40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af':
        raise ValueError('Matrix requires the official CLIP checkpoint')
    result['reused_artifact']=bool(meta.get('reused_source'))
selection=job.get('selection')
if selection:
    import numpy as np
    idx=p.valid_indices(np.load(selection['source']),selection['n_train'])
    np.save(selection['destination'],idx)
    result.update(n_train=selection['n_train'],n_selected=len(idx),requested_count=selection['wanted'],
                  actual_fraction=len(idx)/selection['n_train'],shortfall=max(0,selection['wanted']-len(idx)),
                  excess_count=max(0,len(idx)-selection['wanted']),index_sha256=hashlib.sha256(idx.tobytes()).hexdigest())
result.update(status='ok',elapsed_sec=time.perf_counter()-job['started'])
p.save(job['result_path'],result)
