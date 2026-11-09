"""Convert trusted CIFAR100 pickle arrays to .npy without importing Torch."""
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np

ROOT=Path('/root/autodl-tmp/EVA/data/cifar100')
CACHE=ROOT/'original_pixels_memmap'
EXPECTED={'train':'16019d7e3df5f24257cddd939b257f8d','test':'f0ef6b0ae62326f3e7ffdfab6717acfc'}


def digest(path,kind='sha256'):
    h=hashlib.new(kind)
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    CACHE.mkdir(exist_ok=True)
    record={'representation':'exact uint8 NHWC original CIFAR100 rows; no augmentation or normalization', 'splits':{}}
    for split in ('train','test'):
        source=ROOT/'cifar-100-python'/split
        if digest(source,'md5')!=EXPECTED[split]:raise ValueError('Official CIFAR100 MD5 mismatch: '+split)
        with source.open('rb') as f:payload=pickle.load(f,encoding='latin1')
        pixels=payload['data'].reshape(-1,3,32,32).transpose(0,2,3,1)
        temporary=CACHE/(split+'_pixels.tmp.npy')
        dest=np.lib.format.open_memmap(temporary,mode='w+',dtype=np.uint8,shape=pixels.shape)
        for start in range(0,len(pixels),256):
            dest[start:start+256]=pixels[start:start+256]
        dest.flush()
        assert np.array_equal(dest,pixels)
        del dest
        output=CACHE/(split+'_pixels.npy')
        temporary.replace(output)
        labels=np.asarray(payload['fine_labels'],dtype=np.int64)
        np.save(CACHE/(split+'_labels.npy'),labels)
        record['splits'][split]=dict(n=len(labels),source_md5=EXPECTED[split],pixels_sha256=digest(output),
                                    labels_sha256=digest(CACHE/(split+'_labels.npy')))
        del pixels,payload
        print(split,record['splits'][split]['n'],'exact rows verified',flush=True)
    temporary=CACHE/'complete.tmp.json'
    temporary.write_text(json.dumps(record,indent=2))
    temporary.replace(CACHE/'complete.json')


if __name__=='__main__':main()
