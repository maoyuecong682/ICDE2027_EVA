"""Load CLIP's original image tower without materializing the unused text tower.

The local official TorchScript archive stores plain tensor records. Map those
records, reconstruct its parameter tree without compiling TorchScript graphs,
then load the image weights strictly into OpenAI CLIP's VisionTransformer.
"""
import collections
import ast
import hashlib
import io
import importlib.util
import mmap
import pickle
import struct
import zipfile
from pathlib import Path

import torch
from torch import nn


class _ArchiveObject:
    pass


class _ArchiveUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        if module.startswith('__torch__.'):
            return _ArchiveObject
        if module=='torch._utils' and name=='_rebuild_tensor_v2':
            return torch._utils._rebuild_tensor_v2
        if module=='torch' and name in ('FloatStorage','HalfStorage','LongStorage','IntStorage','BoolStorage','ByteStorage'):
            return getattr(torch,name)
        if module=='collections' and name=='OrderedDict':
            return collections.OrderedDict
        raise pickle.UnpicklingError('Unexpected CLIP archive object: '+module+'.'+name)


def _state(node,prefix=''):
    for key,value in vars(node).items():
        name=prefix+key
        if torch.is_tensor(value):
            yield name,value
        elif isinstance(value,_ArchiveObject):
            yield from _state(value,name+'.')


def load_visual(checkpoint=None):
    def progress(message):
        import os
        if os.environ.get('CLIP_LOAD_DEBUG'):
            rss=next(line.strip() for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('VmRSS:'))
            print(message,rss,flush=True)
    # Import the official model definition directly, avoiding the unused text
    # tokenizer and TorchScript graph loader imported by clip.__init__.
    clip_dir=Path(importlib.util.find_spec('clip').origin).parent
    spec=importlib.util.spec_from_file_location('_official_clip_image_model',clip_dir/'model.py')
    definition=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(definition)
    VisionTransformer=definition.VisionTransformer
    tree=ast.parse((clip_dir/'clip.py').read_text())
    _MODELS=next(ast.literal_eval(node.value) for node in tree.body
        if isinstance(node,ast.Assign) and any(isinstance(target,ast.Name) and target.id=='_MODELS' for target in node.targets))
    progress('imports')
    checkpoint=Path(checkpoint or Path.home()/'.cache/clip/ViT-B-32.pt')
    expected=_MODELS['ViT-B/32'].split('/')[-2]
    digest=hashlib.sha256()
    with checkpoint.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
    if digest.hexdigest()!=expected:
        raise ValueError('CLIP checkpoint does not match the official ViT-B/32 SHA256')
    progress('hash verified')
    source=checkpoint.open('rb')
    import os
    if hasattr(os,'posix_fadvise'):
        os.posix_fadvise(source.fileno(),0,0,os.POSIX_FADV_DONTNEED)
    mapped=mmap.mmap(source.fileno(),0,access=mmap.ACCESS_COPY)
    tensors={}
    with zipfile.ZipFile(checkpoint) as archive:
        data_name=next(name for name in archive.namelist() if name.endswith('/data.pkl'))
        prefix=data_name[:-len('data.pkl')]
        def storage(saved):
            kind,storage_type,key,location,count=saved
            if kind!='storage':raise pickle.UnpicklingError('Expected tensor storage')
            if key not in tensors:
                info=archive.getinfo(prefix+'data/'+key)
                if info.compress_type!=zipfile.ZIP_STORED:
                    raise ValueError('Expected uncompressed official CLIP tensor records')
                header=struct.unpack_from('<4s5H3I2H',mapped,info.header_offset)
                offset=info.header_offset+30+header[-2]+header[-1]
                dtype={'FloatStorage':torch.float32,'HalfStorage':torch.float16,
                       'LongStorage':torch.int64,'IntStorage':torch.int32,
                       'BoolStorage':torch.bool,'ByteStorage':torch.uint8}[storage_type.__name__]
                tensor=torch.frombuffer(memoryview(mapped)[offset:offset+info.file_size],dtype=dtype,count=count)
                tensors[key]=tensor._typed_storage()
            return tensors[key]
        unpickler=_ArchiveUnpickler(io.BytesIO(archive.read(data_name)))
        unpickler.persistent_load=storage
        obj=unpickler.load()
    progress('archive unpickled')
    weights=dict(_state(obj.visual))
    progress('image weights collected')
    width=weights['conv1.weight'].shape[0]
    patch=weights['conv1.weight'].shape[-1]
    grid=int((weights['positional_embedding'].shape[0]-1)**.5)
    layers=len({key.split('.')[2] for key in weights if key.startswith('transformer.resblocks.')})
    output=weights['proj'].shape[1]
    # Avoid meta-device initialization: its symbolic-shape imports themselves
    # consume substantial memory in a small CPU worker.
    previous_dtype=torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float16)
        visual=VisionTransformer(grid*patch,patch,width,layers,width//64,output)
    finally:
        torch.set_default_dtype(previous_dtype)
    progress('vision architecture initialized')
    visual.load_state_dict(weights,strict=True,assign=True)
    progress('original parameter dtypes assigned')
    del weights,obj,unpickler
    tensors.clear()
    import gc
    gc.collect()
    visual=visual.eval()
    progress('image tower ready in original checkpoint precision')
    # Keep the original checkpoint dtypes (the same precision as CUDA CLIP).
    # Only image-tower pages are touched during inference; text weights stay cold.
    class ImageEncoder(nn.Module):
        def __init__(self,visual):super().__init__();self.visual=visual
        def encode_image(self,x):return self.visual(x.to(self.visual.conv1.weight.dtype))
    encoder=ImageEncoder(visual).eval()
    encoder._checkpoint_mapping=mapped
    encoder._checkpoint_file=source
    encoder.checkpoint_sha256=digest.hexdigest()
    encoder.visual_parameter_count=sum(t.numel() for t in visual.parameters())
    return encoder
