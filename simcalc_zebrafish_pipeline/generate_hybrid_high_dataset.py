"""PyCharm Run: prepare 1000 highSNR hybrid heads in one NPZ, with resume.

Generation uses the approved mild model. Committed batches are checkpoints;
final NPY members are streamed into NPZ without stacking the whole dataset.
"""
from copy import deepcopy
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import time
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from hybrid_high_model import (CELL_DEFAULTS, WARP_DEFAULTS, DETECTOR_DEFAULTS,
                               TEMPLATE_KEYS, simulate_hybrid_high)
from prepare_hybrid_high_template import (TEMPLATE_PATH, SEGMENTATION_PARAMETERS,
                                          prepare_template, sha256)

ROOT = Path(__file__).resolve().parent

# PyCharm: открой этот файл и нажми Run. Полный набор готовится только при Run.
IMAGE_COUNT = 1000
RANDOM_SEED = 20261026
OUTPUT_DIR = ROOT / 'reports/hybrid_highSNR_dataset_1000_mild'
CHECKPOINT_SIZE = 10
CELL_PARAMETERS = deepcopy(CELL_DEFAULTS)
WARP_PARAMETERS = {**WARP_DEFAULTS, 'strength': 1., 'elastic_rms_px': 3.}
DETECTOR_PARAMETERS = deepcopy(DETECTOR_DEFAULTS)

SOURCE_FILES = (
    'generate_hybrid_high_dataset.py', 'hybrid_high_model.py',
    'prepare_hybrid_high_template.py', 'simcalc_core.py',
    'vendor/SimCalc/simcalc/simcalc.py', 'vendor/SimCalc/simcalc/simcalclib.py',
)
# Canonical keys: one copy of each large array; every image is highSNR.
ARRAYS = {
    'synthetic_images': (np.dtype('uint16'), 'image'),
    'synthetic_labels': (np.dtype('uint16'), 'image'),
    'synthetic_tissue_masks': (np.dtype('bool'), 'image'),
    'synthetic_seeds': (np.dtype('uint32'), ()),
    'synthetic_cell_counts': (np.dtype('int32'), ()),
    'synthetic_profile_indices': (np.dtype('int32'), ()),
    'synthetic_profile_source_frame_indices': (np.dtype('int32'), ()),
    'warp_rotation_degrees': (np.dtype('float32'), ()),
    'warp_scales_yx': (np.dtype('float32'), (2,)),
    'warp_translation_yx_px': (np.dtype('float32'), (2,)),
    'warp_jacobian_min_max': (np.dtype('float32'), (2,)),
}


def _write_json(path, value):
    temporary = path.with_suffix('.partial.json')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def load_template(path):
    with np.load(path, allow_pickle=False) as z:
        template = {key: z[key] for key in TEMPLATE_KEYS}
        profiles = {key: z[key] for key in ('percentiles', 'highSNR_quantiles')}
        metadata = json.loads(z['metadata_json'].item())
    if metadata['preparation_code_sha256'] != sha256(ROOT / 'prepare_hybrid_high_template.py'):
        raise ValueError('Template extraction code changed. Rebuild using prepare_hybrid_high_template.py first.')
    if metadata['statistics']['segmentation_parameters'] != SEGMENTATION_PARAMETERS:
        raise ValueError('Template extraction settings changed. Rebuild the template first.')
    if len(profiles['highSNR_quantiles']) != len(metadata['training_source_indices']):
        raise ValueError('Template intensity profiles do not match their source frame indices')
    return template, profiles, metadata


def _batch_shape(length, tail, shape):
    return (length, *(shape if tail == 'image' else tail))


def _validate_part(path, start, stop, seed, signature, shape):
    with np.load(path, allow_pickle=False) as z:
        if z['run_signature'].item() != signature:
            raise ValueError(f'Checkpoint belongs to a different run: {path}')
        for key, (dtype, tail) in ARRAYS.items():
            a = z[key]
            if a.shape != _batch_shape(stop-start, tail, shape) or a.dtype != dtype:
                raise ValueError(f'Invalid checkpoint array: {path.name}/{key}')
        np.testing.assert_array_equal(z['synthetic_seeds'], np.arange(seed+start, seed+stop, dtype=np.uint32))
        if np.any((z['synthetic_labels'] > 0) & ~z['synthetic_tissue_masks']):
            raise ValueError(f'Labels outside tissue: {path}')


def _write_small_member(archive, name, value):
    with archive.open(name + '.npy', 'w', force_zip64=True) as member:
        np.lib.format.write_array(member, np.asarray(value), allow_pickle=False)


def pack_dataset(path, parts, count, shape, metadata):
    """One valid NPZ, streamed in checkpoint-sized blocks (including Zip64)."""
    temporary = path.with_suffix('.partial.npz')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED,
                         compresslevel=4, allowZip64=True) as archive:
        for key, (dtype, tail) in ARRAYS.items():
            with archive.open(key + '.npy', 'w', force_zip64=True) as member:
                np.lib.format.write_array_header_2_0(member, {
                    'descr': np.lib.format.dtype_to_descr(dtype), 'fortran_order': False,
                    'shape': _batch_shape(count, tail, shape),
                })
                written = 0
                for part in parts:
                    with np.load(part, allow_pickle=False) as z:
                        a = z[key]
                        member.write(a.tobytes(order='C'))
                        written += len(a)
                if written != count:
                    raise ValueError(f'Incomplete dataset member: {key}')
            print(f'Packing NPZ: {key}', flush=True)
        _write_small_member(archive, 'normalization_offset_highSNR', metadata['normalization']['offset'])
        _write_small_member(archive, 'normalization_scale_highSNR', metadata['normalization']['scale'])
        _write_small_member(archive, 'metadata_json', json.dumps(metadata, ensure_ascii=False))
    # CRC validation streams decompression instead of loading a 1000-image array.
    with zipfile.ZipFile(temporary) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f'NPZ checksum validation failed: {bad}')
    validate_archive(temporary, count, shape, metadata['run_signature'])
    temporary.replace(path)


def validate_archive(path, count, shape, signature):
    """Read only headers and small metadata; never allocate the image stack."""
    with zipfile.ZipFile(path) as archive:
        for key, (dtype, tail) in ARRAYS.items():
            with archive.open(key + '.npy') as member:
                version = np.lib.format.read_magic(member)
                if version != (2, 0):
                    raise ValueError('Unexpected streamed NPY format')
                actual_shape, fortran_order, actual_dtype = np.lib.format.read_array_header_2_0(member)
                if actual_shape != _batch_shape(count, tail, shape) or actual_dtype != dtype or fortran_order:
                    raise ValueError(f'Invalid dataset header: {key}')
    with np.load(path, allow_pickle=False) as z:
        metadata = json.loads(z['metadata_json'].item())
        if metadata['run_signature'] != signature:
            raise ValueError('Output NPZ belongs to a different configuration')
        expected = metadata['sampling_seed'] + np.arange(count, dtype=np.uint32)
        np.testing.assert_array_equal(z['synthetic_seeds'], expected)


def save_previews(output, parts, count, shape, limits):
    selected = np.unique(np.rint(np.linspace(0, count-1, min(10, count))).astype(int))
    examples, offset = [], 0
    for path in parts:
        with np.load(path, allow_pickle=False) as z:
            length = len(z['synthetic_seeds'])
            local = selected[(selected >= offset) & (selected < offset+length)] - offset
            if len(local):
                images, labels = z['synthetic_images'], z['synthetic_labels']
                for i in local:
                    examples.append((offset+int(i), int(z['synthetic_seeds'][i]), images[i].copy(), labels[i] > 0))
            offset += length
    columns = min(5, len(examples)); rows = int(np.ceil(len(examples)/columns))
    for index, name, title, display in (
        (2, 'preview_highSNR.png', 'Выборка из готового highSNR датасета · мягкие деформации', limits),
        (3, 'preview_masks.png', 'Маски тех же синтетических изображений', (0, 1)),
    ):
        fig, axes = plt.subplots(rows, columns, squeeze=False,
                                  figsize=(4*columns, 3.7*rows), constrained_layout=True)
        for ax in axes.flat: ax.axis('off')
        for ax, example in zip(axes.flat, examples):
            ax.imshow(example[index], cmap='gray', vmin=display[0], vmax=display[1], interpolation='nearest')
            ax.set_title(f'№ {example[0]+1}/{count} · seed {example[1]}', fontsize=10)
        fig.suptitle(title); fig.savefig(output/name, dpi=140); plt.close(fig)


def export_dataset(*, template_path, output_dir, image_count, random_seed,
                   checkpoint_size=10, cell_parameters=None, warp_parameters=None,
                   detector_parameters=None):
    if not isinstance(image_count, int) or image_count < 1:
        raise ValueError('image_count must be a positive integer')
    if not isinstance(checkpoint_size, int) or checkpoint_size < 1:
        raise ValueError('checkpoint_size must be a positive integer')
    if not isinstance(random_seed, int) or random_seed < 0 or random_seed+image_count+10000 >= 2**32:
        raise ValueError('Seed range must fit the SimCalc uint32 RNG')
    cell = {**CELL_DEFAULTS, **(cell_parameters or {})}
    warp = {**WARP_DEFAULTS, **(warp_parameters or {})}
    detector = {**DETECTOR_DEFAULTS, **(detector_parameters or {})}
    template_path, output_dir = Path(template_path), Path(output_dir)
    template, profiles, template_metadata = load_template(template_path)
    shape = tuple(template['head_mask'].shape)
    config = {
        'format_version': 1, 'image_count': image_count, 'shape_yx': list(shape),
        'sampling_seed': random_seed, 'checkpoint_size': checkpoint_size,
        'cell_parameters': cell, 'warp_parameters': warp, 'detector_parameters': detector,
        'template_sha256': sha256(template_path),
        'source_code_sha256': {name: sha256(ROOT/name) for name in SOURCE_FILES},
    }
    signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    # OS releases this lock after a crash/Stop. An old lock file cannot block resume.
    with (output_dir/'generation.lock').open('a') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('This dataset is already being generated in another process') from None
        config_path = output_dir/'run_config.json'
        if config_path.exists():
            previous = json.loads(config_path.read_text())
            if previous.get('run_signature') != signature:
                raise ValueError('This output directory has different settings/code/template. '
                                 'Choose a new OUTPUT_DIR to create another dataset.')
        else:
            if any(entry.name != 'generation.lock' for entry in output_dir.iterdir()):
                raise ValueError('Output directory is not empty; choose a new OUTPUT_DIR')
            _write_json(config_path, {**config, 'run_signature': signature})
        snapshot = output_dir/'template_snapshot.npz'
        if not snapshot.exists():
            temporary = snapshot.with_suffix('.partial.npz')
            shutil.copyfile(template_path, temporary); temporary.replace(snapshot)
        if sha256(snapshot) != config['template_sha256']:
            raise ValueError('Saved template snapshot checksum mismatch')
        archive_path = output_dir/f'hybrid_highSNR_{image_count}.npz'
        if archive_path.exists():
            validate_archive(archive_path, image_count, shape, signature)
            print(f'Dataset already complete: {archive_path}', flush=True)
            return archive_path
        parts_dir = output_dir/'_parts'; parts_dir.mkdir(exist_ok=True)
        parts = []; started = time.monotonic()
        for start in range(0, image_count, checkpoint_size):
            stop = min(start+checkpoint_size, image_count)
            path = parts_dir/f'part_{start:06d}_{stop:06d}.npz'
            if path.exists():
                _validate_part(path, start, stop, random_seed, signature, shape)
                parts.append(path)
                print(f'Resume: verified {stop}/{image_count}', flush=True)
                continue
            batch = {key: np.empty(_batch_shape(stop-start, tail, shape), dtype=dtype)
                     for key, (dtype, tail) in ARRAYS.items()}
            for index in range(start, stop):
                seed = random_seed+index; j = index-start
                r = simulate_hybrid_high(template, profiles, seed, cell, warp, detector)
                if not np.isfinite(r['highSNR']).all() or not np.isfinite(r['latent_fluorescence']).all():
                    raise ValueError(f'Nonfinite image at index {index}')
                forbidden = (~r['head_mask']) | r['eye_mask'] | r['channel_mask'] | (r['tissue'] < .5)
                if np.any((r['labels'] > 0) & forbidden):
                    raise ValueError(f'Invalid neuron masks at index {index}')
                batch['synthetic_images'][j] = np.rint(r['highSNR']).astype(np.uint16)
                batch['synthetic_labels'][j] = r['labels']
                batch['synthetic_tissue_masks'][j] = r['tissue_mask']
                batch['synthetic_seeds'][j] = seed
                batch['synthetic_cell_counts'][j] = r['neuron_count']
                batch['synthetic_profile_indices'][j] = r['profile_index']
                batch['synthetic_profile_source_frame_indices'][j] = template_metadata['training_source_indices'][r['profile_index']]
                wm = r['warp_metadata']
                batch['warp_rotation_degrees'][j] = wm['rotation_degrees']
                batch['warp_scales_yx'][j] = wm['scale_yx']
                batch['warp_translation_yx_px'][j] = wm['translation_yx_px']
                batch['warp_jacobian_min_max'][j] = [wm['inverse_jacobian_min'], wm['inverse_jacobian_max']]
                print(f'HighSNR {index+1}/{image_count}: {r["neuron_count"]} cells; '
                      f'{time.monotonic()-started:.0f} s this run', flush=True)
                del r
            temporary = path.with_suffix('.partial.npz')
            np.savez_compressed(temporary, **batch, run_signature=np.asarray(signature))
            temporary.replace(path)
            _validate_part(path, start, stop, random_seed, signature, shape)
            parts.append(path); del batch
            _write_json(output_dir/'progress.json', {'phase': 'generating', 'completed_images': stop,
                         'requested_images': image_count, 'run_signature': signature})
        qi = int(np.argmin(abs(profiles['percentiles']-99.5)))
        limits = [template_metadata['statistics']['floor'],
                  float(np.median(profiles['highSNR_quantiles'][:, qi]))]
        metadata = {
            **config, 'run_signature': signature,
            'model': 'Hybrid raster template + new static SimCalc neurons; highSNR only; mild deformations',
            'template_metadata': template_metadata,
            'normalization': {'offset': limits[0], 'scale': 32767-limits[0],
                              'formula': 'clip((image - offset) / scale, 0, 1)'},
            'array_axes': 'N,Y,X; neuron IDs are local to each image',
            'calibration': 'HighSNR training-frame quantile matching',
            'contents': 'Synthetic images and masks only. No real images or other detector conditions.',
            'limitations': ['All anatomy is a deformation of the same recording-derived template',
                            'Random image splits do not establish generalization to independent fish',
                            'Extracted tissue masks are approximate; single-owner neuron labels lose overlaps'],
        }
        save_previews(output_dir, parts, image_count, shape, limits)
        _write_json(output_dir/'progress.json', {'phase': 'packing_npz', 'completed_images': image_count,
                     'requested_images': image_count, 'run_signature': signature})
        pack_dataset(archive_path, parts, image_count, shape, metadata)
        _write_json(output_dir/'progress.json', {'phase': 'complete', 'completed_images': image_count,
                     'requested_images': image_count, 'archive': archive_path.name,
                     'archive_sha256': sha256(archive_path), 'run_signature': signature})
        (output_dir/'README.md').write_text(f'''# HighSNR dataset: {image_count} synthetic images

Готовый файл: `{archive_path.name}`. Достаточно забрать этот NPZ для обучения.
Основные массивы: `synthetic_images` uint16 ({image_count}, {shape[0]}, {shape[1]}),
`synthetic_labels` uint16 той же формы, `synthetic_tissue_masks` bool.
Есть seed каждого изображения, количество клеток, номера яркостных профилей,
исходные номера кадров профилей, параметры деформаций и JSON метаданных.
0 в labels — фон; положительное значение — ID клетки внутри данного изображения.
Изображения содержат только синтетические highSNR полные кадры; реальных кадров нет.

```python
import numpy as np
with np.load('{archive_path.name}', allow_pickle=False) as data:
    images = data['synthetic_images']
    masks = data['synthetic_labels']
    offset = float(data['normalization_offset_highSNR'])
    scale = float(data['normalization_scale_highSNR'])
    image0 = np.clip((images[0].astype(np.float32) - offset) / scale, 0, 1)
```

Сырые изображения не нормализованы. Маски нормализовать нельзя.
NPZ сжат: при чтении большого массива NumPy распаковывает весь этот массив.
Для одного массива изображений при {image_count} кадрах нужно
примерно {image_count*shape[0]*shape[1]*2/1e6:.0f} MB памяти без нормализации.
`preview_highSNR.png` и `preview_masks.png` показывают выборочные примеры.

`_parts/` — контрольные порции для продолжения после Stop/ошибки и повторной упаковки.
Они сохранены отдельно от итогового NPZ; для обучения их забирать не нужно.
Повторный Run с теми же параметрами продолжает незаконченный набор или сообщает,
что он уже готов. Параметры/код/шаблон менять при продолжении нельзя.
Для другого набора выбери новый OUTPUT_DIR и RANDOM_SEED.
Извлечённый шаблон сохранён как template_snapshot.npz, настройки — run_config.json.
Все головы основаны на одной записи; новые seed не означают новых независимых рыб.
''')
        print(f'Saved complete dataset: {archive_path}', flush=True)
        return archive_path


def main():
    if not TEMPLATE_PATH.exists():
        prepare_template()
    return export_dataset(template_path=TEMPLATE_PATH, output_dir=OUTPUT_DIR,
        image_count=IMAGE_COUNT, random_seed=RANDOM_SEED, checkpoint_size=CHECKPOINT_SIZE,
        cell_parameters=CELL_PARAMETERS, warp_parameters=WARP_PARAMETERS,
        detector_parameters=DETECTOR_PARAMETERS)


if __name__ == '__main__':
    main()
