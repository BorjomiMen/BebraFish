"""Dataset adapters for folders, TIFF stacks, NPY/NPZ and NAOMi/SimCalc."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import tifffile

EXTENSIONS = {'.png', '.tif', '.tiff'}

def json_write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def plane(path):
    path = Path(path)
    value = tifffile.imread(path) if path.suffix.lower() in {'.tif', '.tiff'} else np.array(Image.open(path))
    if value.ndim != 2 or min(value.shape) < 16 or not np.isfinite(value).all():
        raise ValueError(f'Expected finite grayscale HxW image >=16x16: {path}, {value.shape}')
    return value

def source_frames(source, image_key=None):
    source = Path(source).expanduser().resolve()
    if source.is_dir():
        files = sorted(p for p in source.rglob('*') if p.suffix.lower() in EXTENSIONS)
        if not files:
            raise FileNotFoundError(f'No PNG/TIFF images: {source}')
        return len(files), lambda i: plane(files[i]), [p.relative_to(source).as_posix() for p in files], None
    if source.suffix.lower() == '.npz':
        with np.load(source, allow_pickle=False) as z:
            key = image_key or next((k for k in ('synthetic_images', 'images') if k in z), None)
            if key is None:
                raise ValueError('NPZ requires images/synthetic_images or --image-key')
            images = z[key]
            groups = z['synthetic_profile_indices'].astype(str).tolist() if 'synthetic_profile_indices' in z else None
    elif source.suffix.lower() == '.npy':
        images = np.load(source, mmap_mode='r', allow_pickle=False)
        groups = None
    elif source.suffix.lower() in {'.tif', '.tiff'}:
        try:
            images = tifffile.memmap(source, mode='r')
        except ValueError:
            images = tifffile.imread(source)
        groups = None
    else:
        raise ValueError(f'Unsupported source: {source}')
    if images.ndim == 2:
        images = images[None]
    if images.ndim != 3:
        raise ValueError(f'Expected NxHxW grayscale array: {images.shape}')
    names = [f'frame_{i:06d}' for i in range(len(images))]
    return len(images), lambda i: np.asarray(images[i]), names, groups

def split_groups(groups, val=.15, test=.15, seed=42):
    unique = np.unique(np.asarray(groups, dtype=str))
    if len(unique) < 5:
        raise ValueError('Need at least 5 independent groups; supply groups.csv or another split method')
    order = np.random.default_rng(seed).permutation(unique)
    nv = max(1, int(round(len(order) * val)))
    nt = max(1, int(round(len(order) * test)))
    if nv + nt >= len(order):
        raise ValueError('Validation/test fractions leave no training groups')
    result = np.zeros(len(groups), dtype=np.int8)
    result[np.isin(groups, order[:nv])] = 1
    result[np.isin(groups, order[nv:nv+nt])] = 2
    return result

def prepare_diffusion(source, output, image_key=None, method='auto', groups_csv=None,
                      val=.15, test=.15, seed=42, gap=32):
    if not 0 < val < 1 or not 0 < test < 1 or val+test >= 1:
        raise ValueError('Need val>0, test>0 and val+test<1')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    count, get, names, source_groups = source_frames(source, image_key)
    first = get(0)
    if first.ndim != 2 or min(first.shape) < 16:
        raise ValueError('Expected grayscale images >=16x16')
    groups = np.asarray(source_groups if source_groups is not None else names, dtype=str)
    explicit_split = None
    if groups_csv:
        with Path(groups_csv).open(encoding='utf-8-sig', newline='') as f:
            table = {r['name']: r for r in csv.DictReader(f)}
        if set(names) != set(table):
            raise ValueError('groups.csv names must exactly match source image/frame names')
        groups = np.asarray([table[n]['group'] for n in names], dtype=str)
        if all(table[n].get('split') for n in names):
            codes = {'train': 0, 'val': 1, 'test': 2, 'exclude': -1}
            explicit_split = np.asarray([codes[table[n]['split']] for n in names], dtype=np.int8)
    if method == 'auto':
        method = 'groups' if groups_csv or source_groups is not None or Path(source).is_dir() else 'temporal'
    if explicit_split is not None:
        split = explicit_split
    elif method == 'temporal':
        from .zenodo import temporal_split
        split, temporal_groups = temporal_split(count, gap)
        groups = temporal_groups.astype(str)
    else:
        split = split_groups(groups, val, test, seed)
    owners = {}
    for group, code in zip(groups, split):
        if code < 0:
            continue
        if group in owners and owners[group] != int(code):
            raise ValueError(f'Group crosses split boundaries: {group}')
        owners[group] = int(code)
    if any(not np.any(split == code) for code in (0, 1, 2)):
        raise ValueError('Train, validation and test must be nonempty')
    images = np.lib.format.open_memmap(output/'images.npy', mode='w+', dtype=first.dtype,
                                     shape=(count, *first.shape))
    content_hash = hashlib.sha256()
    seen = {}
    for i in range(count):
        value = get(i)
        if value.shape != first.shape or value.dtype != first.dtype or not np.isfinite(value).all():
            raise ValueError(f'Frame {i}: inconsistent shape/dtype or nonfinite pixels')
        image_hash = hashlib.sha256(value.tobytes()).hexdigest()
        if image_hash in seen and seen[image_hash] != int(split[i]) and min(seen[image_hash], split[i]) >= 0:
            raise ValueError(f'Identical image in different splits: {names[i]}')
        seen[image_hash] = int(split[i])
        images[i] = value
        content_hash.update(value.tobytes())
    images.flush()
    ids = np.flatnonzero(split == 0)
    selected = ids[np.linspace(0, len(ids)-1, min(512, len(ids))).astype(int)]
    pixels = np.concatenate([np.asarray(images[i])[::4, ::4].astype(np.float32).ravel() for i in selected])
    black, white = np.percentile(pixels, [1, 99.8]).tolist()
    if not np.isfinite([black, white]).all() or white <= black:
        raise ValueError('Empty train intensity range')
    np.save(output/'split.npy', split)
    np.save(output/'groups.npy', groups)
    np.save(output/'names.npy', np.asarray(names, dtype=str))
    metadata = dict(format='zebrafish-image-dataset-v1', source=dict(path=str(Path(source).resolve()),
                    kind=Path(source).suffix or 'folder', image_key=image_key), shape=list(images.shape),
                    raw_dtype=str(images.dtype), images_sha256=content_hash.hexdigest(),
                    normalization=dict(method='train-only shared p1/p99.8', black=black, white=white,
                                       sample_frames=len(selected), pixel_stride=4),
                    split_method=method, split_seed=seed,
                    groups_sha256=hashlib.sha256(groups.tobytes()).hexdigest(),
                    names_sha256=hashlib.sha256(np.asarray(names,dtype=str).tobytes()).hexdigest(),
                    split_counts={name:int((split==code).sum()) for name,code in [('train',0),('val',1),('test',2),('excluded',-1)]},
                    split_sha256=hashlib.sha256(split.tobytes()).hexdigest())
    fingerprint = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    metadata['fingerprint'] = fingerprint
    json_write(output/'dataset.json', metadata)
    del images
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return output

class PreparedFrames:
    def __init__(self, folder, normalization):
        self.raw = np.load(Path(folder)/'images.npy', mmap_mode='r', allow_pickle=False)
        self.shape = self.raw.shape
        self.black, self.white = normalization['black'], normalization['white']
    def __len__(self):
        return len(self.raw)
    def __getitem__(self, index):
        unit = np.clip((self.raw[index].astype(np.float32)-self.black)/(self.white-self.black), 0, 1)
        return np.rint(unit*65535).astype(np.uint16)
    def close(self):
        if hasattr(self.raw, '_mmap'):
            self.raw._mmap.close()

def load_prepared(folder):
    folder = Path(folder)
    metadata = json.loads((folder/'dataset.json').read_text(encoding='utf-8'))
    if metadata.get('format') != 'zebrafish-image-dataset-v1':
        raise ValueError('Run prepare-diffusion first')
    result = {key:np.load(folder/f'{key}.npy', allow_pickle=False) for key in ('split','groups','names')}
    result.update(images=PreparedFrames(folder, metadata['normalization']),
                  metadata_json=np.asarray(json.dumps(metadata)), fingerprint=metadata['fingerprint'])
    if list(result['images'].shape) != metadata['shape'] or len(result['split']) != len(result['images']):
        raise ValueError('Prepared dataset shape changed')
    # Hash raw pixels in bounded blocks, also detecting an edited prepared dataset.
    actual = hashlib.sha256()
    for i in range(len(result['images'])):
        actual.update(np.asarray(result['images'].raw[i]).tobytes())
    if actual.hexdigest() != metadata['images_sha256'] or hashlib.sha256(result['split'].tobytes()).hexdigest() != metadata['split_sha256']:
        raise ValueError('Prepared dataset images/split changed; prepare it again')
    if any(hashlib.sha256(result[key].tobytes()).hexdigest() != metadata[key+'_sha256'] for key in ('groups','names')):
        raise ValueError('Prepared dataset names/groups changed; prepare it again')
    return result

def segmentation_records(folder):
    folder = Path(folder).resolve()
    with (folder/'pairs.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    result = []
    for r in rows:
        image, mask = (folder/r['image']).resolve(), (folder/r['mask']).resolve()
        if not image.is_relative_to(folder) or not mask.is_relative_to(folder) or not image.is_file() or not mask.is_file():
            raise ValueError(f'Invalid pair paths: {r}')
        group = r.get('group') or r.get('seed')
        if not group:
            raise ValueError('pairs.csv requires group or seed')
        result.append(dict(name=r.get('name') or image.name, image=str(image), mask=str(mask), group=str(group)))
    if not result or len({r['name'] for r in result}) != len(result):
        raise ValueError('No pairs or duplicate pair names')
    return result, dict(root=str(folder), csv_rows=len(rows), pairs=len(result), missing_pairs=[])

def prepare_segmentation(source, output, image_key=None, mask_key=None):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output/'images').mkdir(); (output/'masks').mkdir()
    if source.suffix.lower() == '.npz':
        with np.load(source, allow_pickle=False) as z:
            ik = image_key or ('synthetic_images' if 'synthetic_images' in z else 'images')
            mk = mask_key or next((k for k in ('synthetic_labels','masks','labels') if k in z), None)
            if mk is None:
                raise ValueError('NPZ requires labels/masks or --mask-key')
            images, masks = z[ik], z[mk]
            if images.ndim != 3 or masks.shape != images.shape:
                raise ValueError('Expected matching NxHxW image and mask arrays')
            groups = z['synthetic_seeds'].astype(str).tolist() if 'synthetic_seeds' in z else [str(i) for i in range(len(images))]
        pairs = ((f'frame_{i:06d}', images[i], masks[i], groups[i]) for i in range(len(images)))
    elif (source/'pairs.csv').is_file():
        rows, _ = segmentation_records(source)
        pairs = ((Path(r['name']).stem, plane(r['image']), plane(r['mask']), r['group']) for r in rows)
    elif (source/'masks_png_400x485'/'correspondence.csv').is_file():
        mask_root = source/'masks_png_400x485'
        with (mask_root/'correspondence.csv').open(encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f))
        rows = [r for r in rows if (source/'images'/r['image']).is_file() and (mask_root/r['semantic_mask']).is_file()]
        pairs = ((Path(r['image']).stem, plane(source/'images'/r['image']), plane(mask_root/r['semantic_mask']), r['seed']) for r in rows)
    else:
        image_files = sorted(p for p in (source/'images').glob('*') if p.suffix.lower() in EXTENSIONS)
        masks = {p.stem:p for p in (source/'masks').glob('*') if p.suffix.lower() in EXTENSIONS}
        if not image_files or any(p.stem not in masks for p in image_files):
            raise ValueError('Expected images/ and masks/ with matching file stems')
        pairs = ((p.stem, plane(p), plane(masks[p.stem]), p.stem) for p in image_files)
    count = 0
    with (output/'pairs.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['name','image','mask','group']); writer.writeheader()
        used = set()
        for name, image, mask, group in pairs:
            if name in used or image.ndim != 2 or mask.shape != image.shape or not np.isfinite(image).all() or not np.isfinite(mask).all():
                raise ValueError(f'Invalid segmentation pair: {name}')
            used.add(name)
            image_name, mask_name = f'images/{name}.tif', f'masks/{name}.png'
            tifffile.imwrite(output/image_name, image)
            Image.fromarray((mask > 0).astype(np.uint8)*255).save(output/mask_name)
            writer.writerow(dict(name=f'{name}.tif', image=image_name, mask=mask_name, group=group))
            count += 1
    json_write(output/'source.json', dict(source=str(source.resolve()), count=count,
               mask_rule='foreground = source mask > 0', image_key=image_key, mask_key=mask_key))
    if not count:
        raise ValueError('No pairs')
    print(f'Prepared {count} segmentation pairs: {output}')
    return output
