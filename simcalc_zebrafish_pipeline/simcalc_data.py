"""SimCalc ADU adapter, one-time NPZ extraction and provenance-aware real split."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import zipfile

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def json_write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def load_simcalc(path, cache='/content/simcalc_cache'):
    """Extract .npy members once, then mmap; never decompress per minibatch."""
    path = Path(path).resolve()
    if path.is_dir():
        from dataset_io import load_prepared
        arrays,metadata,identity=load_prepared(path)
        if metadata.get('role')!='synthetic':
            raise ValueError('Synthetic input must be prepared with --role synthetic')
        arrays['metadata_json']=np.asarray(json.dumps(metadata))
        return validate_synthetic(arrays,metadata,identity,path,path)
    source_hash = digest(path)
    folder = Path(cache) / source_hash[:20]
    folder.mkdir(parents=True, exist_ok=True)
    receipt = folder / 'extraction.json'
    complete = json.loads(receipt.read_text()) if receipt.exists() else {}
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(set(names)) != len(names) or any(Path(n).name != n or not n.endswith('.npy') for n in names):
            raise ValueError('Expected flat NPZ containing only unique .npy members')
        for info in archive.infolist():
            target = folder / info.filename
            if complete.get('sha256') == source_hash and target.exists() and target.stat().st_size == info.file_size:
                continue
            print(f'Extract SimCalc once: {info.filename}', flush=True)
            temporary = target.with_name(target.name + '.tmp')
            with archive.open(info) as src, temporary.open('wb') as dst:
                shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
            temporary.replace(target)
    json_write(receipt, dict(sha256=source_hash, source_name=path.name))
    data = {Path(n).stem: np.load(folder / n, mmap_mode='r', allow_pickle=False) for n in names}
    if 'metadata_json' not in data:
        raise ValueError('NPZ needs metadata_json; use pipeline.py prepare for generic simulator output')
    metadata = json.loads(str(data['metadata_json']))
    return validate_synthetic(data,metadata,source_hash,path,folder)


def validate_synthetic(data,metadata,source_hash,path,folder):
    from dataset_io import ConstantFrames
    if 'synthetic_images' not in data:
        raise ValueError('Expected synthetic_images; use pipeline.py prepare --images-key for other names')
    images = data['synthetic_images']
    if images.ndim != 3 or not len(images) or min(images.shape[1:])<1 or images.dtype != np.uint16:
        raise ValueError('Expected uint16 [N,H,W] native ADU frames; use pipeline.py prepare for calibration')
    labels_available='synthetic_labels' in data
    tissue_available='synthetic_tissue_masks' in data
    source_profiles_available='synthetic_profile_source_frame_indices' in data
    data.setdefault('synthetic_labels',ConstantFrames(images.shape,0,np.uint16))
    data.setdefault('synthetic_tissue_masks',ConstantFrames(images.shape,True,bool))
    for key in ['synthetic_labels', 'synthetic_tissue_masks']:
        if data[key].shape != images.shape:
            raise ValueError(f'{key}: shape mismatch')
    if data['synthetic_labels'].dtype.kind not in 'iu' or data['synthetic_tissue_masks'].dtype != np.bool_:
        raise ValueError('Instance IDs must be integer; tissue masks must be bool')
    if data['synthetic_labels'].dtype.kind=='i' and np.min(data['synthetic_labels'])<0:
        raise ValueError('Instance IDs cannot be negative')
    for key, value in data.items():
        if value.ndim and value.shape[0] != len(images):
            raise ValueError(f'{key}: first dimension must equal image count')
    data.setdefault('synthetic_profile_source_frame_indices',data.get('groups',np.arange(len(images))))
    data.update(source_sha256=source_hash, source_path=str(path), metadata=metadata, cache_path=str(folder))
    data.update(labels_available=labels_available,tissue_available=tissue_available,
                source_profiles_available=source_profiles_available)
    return data


class ADUFrames:
    dtype = np.dtype('uint16')
    ndim = 3

    def __init__(self, raw, black, white):
        self.raw, self.black, self.white = raw, float(black), float(white)
        self.shape = raw.shape
        if not np.isfinite([black, white]).all() or white <= black:
            raise ValueError('Invalid shared normalization')

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, index):
        unit = np.clip((self.raw[index].astype(np.float32) - self.black) / (self.white - self.black), 0, 1)
        return np.rint(unit * 65535).astype(np.uint16)


def aligned_split(count, metadata, profile_sources, gap=32):
    """Temporal holdouts minus template/profile sources and their neighbours.

    Generator validation-source neighbourhoods are excluded from prior training
    too. Preview groups in synthetic data are not an independent fish holdout.
    """
    if count < 20 or gap < 0:
        raise ValueError('Need count>=20 and gap>=0')
    groups = np.minimum(np.arange(count, dtype=np.int64) * 10 // count, 9)
    split = np.zeros(count, dtype=np.int8)
    split[groups == 4], split[groups == 8] = 1, 2
    boundaries = np.flatnonzero(split[1:] != split[:-1]) + 1
    for boundary in boundaries:
        split[max(0, boundary-gap):min(count, boundary+gap)] = -1
    template = metadata.get('template_metadata', {})
    if 'training_source_indices' not in template or 'validation_source_indices' not in template:
        raise ValueError('SimCalc provenance must declare template training/validation source indices')
    training = np.asarray(template['training_source_indices'], dtype=np.int64)
    validation = np.asarray(template['validation_source_indices'], dtype=np.int64)
    profiles = np.asarray(profile_sources, dtype=np.int64)
    if np.intersect1d(training, validation).size:
        raise ValueError('Template train/validation source overlap')
    used = np.unique(np.concatenate([training, profiles]))
    all_sources = np.unique(np.concatenate([used, validation]))
    if len(all_sources) and (all_sources.min() < 0 or all_sources.max() >= count):
        raise ValueError('Source frame IDs outside the real recording')
    if np.intersect1d(profiles, validation).size:
        raise ValueError('Synthetic profiles used reserved generator-validation sources; regenerate or audit input')
    for index in used:
        region = slice(max(0, int(index)-gap), min(count, int(index)+gap+1))
        view = split[region]
        view[(view == 1) | (view == 2)] = -1
    for index in validation:
        split[max(0, int(index)-gap):min(count, int(index)+gap+1)] = -1
    if any(not np.any(split == c) for c in [0, 1, 2]):
        raise ValueError('Provenance exclusions emptied a split; reduce gap or supply a different recording')
    report = dict(method='10 time blocks val=4/test=8; embargo; exclude template/profile neighbourhoods from evaluation',
                  template_training_source_ids=training.tolist(), template_validation_source_ids=validation.tolist(),
                  profile_source_ids=np.unique(profiles).tolist(), gap_frames=gap,
                  counts={name: int(np.sum(split == code)) for name, code in [('train',0),('val',1),('test',2),('excluded',-1)]},
                  used_source_overlap_val=int(np.isin(used, np.flatnonzero(split == 1)).sum()),
                  used_source_overlap_test=int(np.isin(used, np.flatnonzero(split == 2)).sum()),
                  reserved_generator_val_overlap_train=int(np.isin(validation, np.flatnonzero(split == 0)).sum()),
                  limitation='Same fish and recording-derived anatomy; not independent-fish validation')
    return split, groups, report


def synthetic_view(data, normalization):
    profiles = np.asarray(data['synthetic_profile_source_frame_indices'])
    unique = np.unique(profiles)
    # Stable held-out PREVIEW groups only; the prior is never trained on synthetic.
    chosen = unique[::10]
    split = np.isin(profiles, chosen).astype(np.int8)
    return dict(images=ADUFrames(data['synthetic_images'], normalization['black'], normalization['white']),
                split=split, groups=profiles, metadata_json=np.asarray(json.dumps(data['metadata'])),
                simcalc=data)


def load_domains(args):
    from zenodo_full_data import (download_verified, RawTiffFrames, train_percentiles,
                                 RECORD, FILENAME, TOTAL_BYTES, MD5)
    synthetic = load_simcalc(args.synthetic_npz, args.synthetic_cache)
    if getattr(args,'real_input',None):
        return load_custom_real(args,synthetic)
    path = download_verified(args.zenodo_cache)
    raw = RawTiffFrames(path)
    if raw.shape != (9800, 400, 485):
        raise ValueError(f'Unexpected real shape {raw.shape}')
    if tuple(raw.shape[1:])!=tuple(synthetic['synthetic_images'].shape[1:]):
        raise ValueError('Real/synthetic native geometry differs; supply a matching --real-input. No resizing is performed.')
    template=synthetic['metadata'].get('template_metadata',{})
    if {'training_source_indices','validation_source_indices'}<=template.keys():
        profiles=synthetic['synthetic_profile_source_frame_indices'] if synthetic['source_profiles_available'] else []
        split, groups, audit = aligned_split(len(raw), synthetic['metadata'],profiles,args.temporal_gap)
    else:
        from zenodo_full_data import temporal_split
        split,groups=temporal_split(len(raw),args.temporal_gap)
        audit=dict(method='temporal blocks; simulator template/profile provenance not declared',
                   gap_frames=args.temporal_gap,
                   counts={name:int(np.sum(split==code)) for name,code in [('train',0),('val',1),('test',2),('excluded',-1)]},
                   used_source_overlap_val=None,used_source_overlap_test=None,
                   limitation='Unknown simulator/real source overlap; one-recording temporal holdout, not independent fish')
    black, white, sampled = train_percentiles(raw, np.flatnonzero(split == 0), args.norm_samples)
    normalization = dict(black=black, white=white, sample_frames=sampled, pixel_stride=4,
                         method='one train-only real p1/p99.8 ADU mapping for BOTH real and SimCalc')
    metadata = dict(source=dict(record=RECORD, filename=FILENAME, bytes=TOTAL_BYTES, md5=MD5),
                    normalization=normalization, split_audit=audit,
                    split_sha256=hashlib.sha256(split.tobytes()).hexdigest(), synthetic_sha256=synthetic['source_sha256'])
    fingerprint = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    real = dict(images=ADUFrames(raw, black, white), split=split, groups=groups,
                metadata_json=np.asarray(json.dumps(metadata)), fingerprint=fingerprint)
    json_write(Path(args.zenodo_cache) / 'simcalc_aligned_manifest.json', metadata)
    print('Real split:', audit['counts'], '\nShared ADU normalization:', normalization, flush=True)
    return real, synthetic_view(synthetic, normalization)


def load_custom_real(args,synthetic):
    from dataset_io import load_prepared,open_array,file_hash
    from zenodo_full_data import temporal_split,train_percentiles
    path=Path(args.real_input).resolve()
    explicit_groups=None
    if path.is_dir() and (path/'dataset.json').is_file():
        arrays,prepared,identity=load_prepared(path)
        if prepared.get('role')!='real':
            raise ValueError('Custom real dataset must be prepared with --role real')
        raw=arrays['synthetic_images']
        split=arrays.get('split')
        explicit_groups=arrays.get('groups')
    else:
        raw,_,_=open_array(path,getattr(args,'real_images_key',''))
        split=None
        identity=file_hash(path) if path.is_file() else hashlib.sha256(json.dumps(
            [(p.name,file_hash(p)) for p in raw.files]).encode()).hexdigest()
    if raw.dtype!=np.uint16:
        raise ValueError('Custom real data must be uint16 ADU; use pipeline.py prepare --role real')
    if tuple(raw.shape[1:])!=tuple(synthetic['synthetic_images'].shape[1:]):
        raise ValueError('Real and synthetic native H,W must match; no resize/crop is performed')
    if getattr(args,'real_split_file',None):
        split=np.load(args.real_split_file,allow_pickle=False)
    if split is None:
        split,groups=temporal_split(len(raw),args.temporal_gap)
        split_method='10 temporal blocks with guard; source/template overlap not audited'
    else:
        split=np.asarray(split)
        if split.shape!=(len(raw),) or split.dtype.kind not in 'iu' or not np.isin(split,[-1,0,1,2]).all():
            raise ValueError('Explicit split: integer [N] with -1/0/1/2 codes')
        if any(not np.any(split==code) for code in [0,1]):
            raise ValueError('Custom real input needs nonempty train and validation')
        groups=np.asarray(explicit_groups) if explicit_groups is not None else np.arange(len(raw))
        split_method='explicit user-supplied split; independence depends on source provenance'
        if explicit_groups is not None:
            for a,b in [(0,1),(0,2),(1,2)]:
                if np.intersect1d(groups[split==a],groups[split==b]).size:
                    raise ValueError('Provided real groups overlap train/validation/test')
    black,white,sampled=train_percentiles(raw,np.flatnonzero(split==0),args.norm_samples)
    normalization=dict(black=black,white=white,sample_frames=sampled,pixel_stride=4,
                       method='train-only shared real p1/p99.8 ADU mapping')
    audit=dict(method=split_method,gap_frames=args.temporal_gap,
               counts={name:int(np.sum(split==code)) for name,code in [('train',0),('val',1),('test',2),('excluded',-1)]},
               used_source_overlap_val=None,used_source_overlap_test=None,
               limitation='Supply independent recordings and audited splits; simulator provenance is not inferred')
    metadata=dict(source=dict(kind='custom-real',name=path.name,sha256=identity),normalization=normalization,
                  split_audit=audit,split_sha256=hashlib.sha256(split.tobytes()).hexdigest(),
                  synthetic_sha256=synthetic['source_sha256'])
    fingerprint=hashlib.sha256(json.dumps(metadata,sort_keys=True).encode()).hexdigest()
    real=dict(images=ADUFrames(raw,black,white),split=split,groups=groups,
              metadata_json=np.asarray(json.dumps(metadata)),fingerprint=fingerprint)
    print('Custom real split:',audit['counts'],'; shared normalization:',normalization,flush=True)
    return real,synthetic_view(synthetic,normalization)
