"""Store original MNIST-M encoded images in one mmap-able blob, without Torch."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT=Path('/root/autodl-tmp/EVA/data/mnistm')
CACHE=ROOT/'original_encoded_memmap'


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    CACHE.mkdir(exist_ok=True)
    source=ROOT/'mnistm_test.parquet'
    table=pq.read_table(source,columns=['image','label'])
    images=table['image'].to_pylist();labels=np.asarray(table['label'].to_pylist(),dtype=np.int64)
    offsets=[0];source_hash=hashlib.sha256()
    temporary=CACHE/'images.tmp.bin'
    with temporary.open('wb') as f:
        for entry in images:
            image=entry['bytes'];f.write(image);source_hash.update(image);offsets.append(offsets[-1]+len(image))
    temporary.replace(CACHE/'images.bin')
    np.save(CACHE/'offsets.npy',np.asarray(offsets,dtype=np.int64))
    np.save(CACHE/'labels.npy',labels)
    assert source_hash.hexdigest()==digest(CACHE/'images.bin')
    assert len(offsets)==len(labels)+1
    record=dict(n=len(labels),source_sha256=digest(source),blob_sha256=source_hash.hexdigest(),
                labels_sha256=digest(CACHE/'labels.npy'),offsets_sha256=digest(CACHE/'offsets.npy'),
                representation='Exact original encoded image bytes and labels in parquet row order')
    (CACHE/'complete.json').write_text(json.dumps(record,indent=2))
    print(len(labels),'original MNIST-M images and labels verified',flush=True)


if __name__=='__main__':main()
