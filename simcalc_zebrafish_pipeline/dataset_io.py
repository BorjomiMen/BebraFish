"""Portable grayscale dataset preparation; explicit calibration, native geometry."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def write_json(path,value):
    path=Path(path)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


class ConstantFrames:
    ndim=3
    def __init__(self,shape,value,dtype):
        self.shape,self.value,self.dtype=tuple(shape),value,np.dtype(dtype)
    def __len__(self):
        return self.shape[0]
    def __getitem__(self,index):
        if np.isscalar(index):
            return np.full(self.shape[1:],self.value,dtype=self.dtype)
        ids=np.arange(len(self))[index]
        return np.full((len(ids),*self.shape[1:]),self.value,dtype=self.dtype)


class DirectoryFrames:
    ndim=3
    def __init__(self,path):
        self.files=sorted(p for p in Path(path).iterdir() if p.suffix.lower() in {'.png','.tif','.tiff'})
        if not self.files:
            raise ValueError(f'No grayscale PNG/TIFF frames: {path}')
        first=self._read(0)
        self.shape=(len(self.files),*first.shape)
        self.dtype=first.dtype
    def _read(self,index):
        from PIL import Image
        path=self.files[index]
        if path.suffix.lower() in {'.tif','.tiff'}:
            import tifffile
            image=tifffile.imread(path)
        else:
            with Image.open(path) as handle:
                image=np.asarray(handle).copy()
        if image.ndim!=2:
            raise ValueError(f'Each directory file must contain one grayscale 2D image: {path}')
        return image
    def __len__(self):
        return self.shape[0]
    def __getitem__(self,index):
        if np.isscalar(index):
            image=self._read(int(index))
            if image.shape!=self.shape[1:]:
                raise ValueError('Frame geometry changes within the directory')
            return image
        return np.stack([self[int(i)] for i in np.arange(len(self))[index]])


def open_array(path,key='',axes='NHW'):
    path=Path(path)
    owner=None
    if path.is_dir():
        return DirectoryFrames(path),owner,''
    suffix=path.suffix.lower()
    if suffix=='.npz':
        owner=np.load(path,allow_pickle=False)
        if not key:
            key=next((k for k in ['synthetic_images','images','frames'] if k in owner.files),'')
        if key not in owner.files:
            raise ValueError(f'Select --images-key from {owner.files}')
        array=owner[key]
    elif suffix=='.npy':
        array=np.load(path,mmap_mode='r',allow_pickle=False)
    elif suffix in {'.tif','.tiff'}:
        import tifffile
        try:
            array=tifffile.memmap(path)
        except ValueError:
            array=tifffile.imread(path)
    elif suffix=='.mat':
        if not key:
            raise ValueError('MAT input requires --images-key and explicit --axes')
        import scipy.io
        try:
            array=scipy.io.loadmat(path,variable_names=[key])[key]
        except NotImplementedError as error:
            raise ValueError('MAT v7.3/HDF5: export the selected array to NPY first, with explicit axis order') from error
    else:
        raise ValueError('Use NPZ, NPY, TIFF, a grayscale image directory, or MATLAB <=v7.2')
    if array.ndim==2:
        array=array[None]
    elif array.ndim==3 and axes=='HWN':
        array=np.moveaxis(array,-1,0)
    if array.ndim!=3 or not len(array) or min(array.shape[1:])<1:
        raise ValueError('Expected nonempty grayscale [N,H,W] array')
    if array.dtype.kind not in 'biuf':
        raise ValueError('Object/complex/image arrays are unsupported')
    return array,owner,key


def calibrated_uint16(frame,mode,black=None,white=None):
    pixels=np.asarray(frame,dtype=np.float64)
    if not np.isfinite(pixels).all():
        raise ValueError('Image contains NaN/Inf')
    if mode=='adu':
        if pixels.min()<0 or pixels.max()>65535:
            raise ValueError('ADU input must be within [0,65535]; calibrate/offset signed data explicitly')
    else:
        if black is None or white is None or not 0<=black<white<=65535:
            raise ValueError('unit/display inputs require --adu-black and --adu-white in [0,65535]')
        denominator={'unit':1.,'display-u8':255.,'display-u16':65535.}[mode]
        if pixels.min()<0 or pixels.max()>denominator:
            raise ValueError(f'Input outside declared {mode} range')
        pixels=black+pixels/denominator*(white-black)
    return np.rint(pixels).astype(np.uint16)


def load_prepared(folder):
    folder=Path(folder)
    metadata=json.loads((folder/'dataset.json').read_text(encoding='utf-8'))
    if metadata.get('format')!='microscopy-addon-dataset-v1':
        raise ValueError('Unknown prepared dataset format')
    arrays={}
    for key,entry in metadata['files'].items():
        name=entry['file']
        if Path(name).name!=name:
            raise ValueError('Prepared array path must be a local filename')
        path=folder/name
        if file_hash(path)!=entry['sha256']:
            raise ValueError(f'Prepared dataset checksum mismatch: {name}')
        arrays[key]=np.load(path,mmap_mode='r',allow_pickle=False)
    identity=hashlib.sha256(json.dumps(metadata,sort_keys=True).encode()).hexdigest()
    return arrays,metadata,identity


def prepare_main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--role',choices=['synthetic','real'],default='synthetic')
    p.add_argument('--images-key',default='')
    p.add_argument('--labels-key',default='')
    p.add_argument('--tissue-key',default='')
    p.add_argument('--labels',default='')
    p.add_argument('--tissue',default='')
    p.add_argument('--axes',choices=['NHW','HWN'],default='NHW')
    p.add_argument('--intensity',choices=['adu','unit','display-u8','display-u16'],required=True)
    p.add_argument('--adu-black',type=float)
    p.add_argument('--adu-white',type=float)
    p.add_argument('--split',help='Optional NPY: -1 excluded, 0 train, 1 val, 2 test')
    p.add_argument('--groups',help='Optional NPY recording/fish/profile group IDs')
    args=p.parse_args(argv)
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError('Prepared output must be empty')
    images,owner,image_key=open_array(args.input,args.images_key,args.axes)
    n,h,w=images.shape
    metadata={}
    if owner is not None and 'metadata_json' in owner.files:
        metadata=json.loads(str(owner['metadata_json']))
    labels=tissue=None
    label_key=args.labels_key or (next((k for k in ['synthetic_labels','labels','instance_masks'] if owner is not None and k in owner.files),''))
    tissue_key=args.tissue_key or (next((k for k in ['synthetic_tissue_masks','tissue_masks'] if owner is not None and k in owner.files),''))
    if args.labels:
        labels,_,_=open_array(args.labels,args.labels_key,args.axes)
    elif label_key:
        labels,_,_=open_array(args.input,label_key,args.axes)
    if args.tissue:
        tissue,_,_=open_array(args.tissue,args.tissue_key,args.axes)
    elif tissue_key:
        tissue,_,_=open_array(args.input,tissue_key,args.axes)
    for name,value in [('labels',labels),('tissue',tissue)]:
        if value is not None and tuple(value.shape)!=tuple(images.shape):
            raise ValueError(f'{name} shape {value.shape} differs from images {images.shape}; no broadcasting/resizing is performed')
    target=np.lib.format.open_memmap(out/'synthetic_images.npy',mode='w+',dtype=np.uint16,shape=images.shape)
    label_target=None
    if labels is not None:
        if labels.dtype.kind not in 'iu':
            raise ValueError('Instance IDs must be integer, with 0 background')
        label_target=np.lib.format.open_memmap(out/'synthetic_labels.npy',mode='w+',dtype=labels.dtype,shape=labels.shape)
    tissue_target=np.lib.format.open_memmap(out/'synthetic_tissue_masks.npy',mode='w+',dtype=bool,shape=tissue.shape) if tissue is not None else None
    for i in range(n):
        target[i]=calibrated_uint16(images[i],args.intensity,args.adu_black,args.adu_white)
        if label_target is not None:
            if labels[i].min()<0:
                raise ValueError('Instance IDs cannot be negative')
            label_target[i]=labels[i]
        if tissue_target is not None:
            tissue_target[i]=tissue[i]!=0
    target.flush()
    if label_target is not None:
        label_target.flush()
    if tissue_target is not None:
        tissue_target.flush()
    excluded={image_key,label_key,tissue_key,'metadata_json'}
    if owner is not None:
        for key in owner.files:
            if key in excluded:
                continue
            value=owner[key]
            if value.dtype.kind=='O' or (value.ndim and len(value)!=n):
                raise ValueError(f'Unsupported extra array: {key}')
            if Path(key).name!=key:
                raise ValueError('NPZ keys must be flat filenames')
            np.save(out/(key+'.npy'),value,allow_pickle=False)
    for name,path in [('split',args.split),('groups',args.groups)]:
        if path:
            values=np.load(path,allow_pickle=False)
            if values.shape!=(n,):
                raise ValueError(f'{name}: expected [{n}]')
            np.save(out/(name+'.npy'),values,allow_pickle=False)
    files={}
    for path in sorted(out.glob('*.npy')):
        array=np.load(path,mmap_mode='r',allow_pickle=False)
        files[path.stem]=dict(file=path.name,sha256=file_hash(path),shape=list(array.shape),dtype=str(array.dtype))
    metadata.update(format='microscopy-addon-dataset-v1',role=args.role,shape=[n,h,w],
        labels_available=labels is not None,tissue_mask_available=tissue is not None,
        intensity_mode='adu_uint16',input_intensity_mode=args.intensity,
        calibration=dict(adu_black=args.adu_black,adu_white=args.adu_white,quantization='round to uint16; no per-frame stretch'),
        preparation_source=dict(name=Path(args.input).name,images_key=image_key,axes=args.axes),files=files)
    write_json(out/'dataset.json',metadata)
    print(f'Prepared {args.role}: {n} native {h}x{w} frames -> {out}',flush=True)
    return out


if __name__=='__main__':
    prepare_main()
