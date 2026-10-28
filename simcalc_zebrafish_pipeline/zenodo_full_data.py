"""Verified Zenodo download to Colab /content; lazy full-frame TIFF reader.

Only highSNR is a target domain. lowSNR is the same recording, not a second fish.
No persistent dataset is written to Google Drive. No all-frame float array in RAM.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

import numpy as np
import tifffile

RECORD = 'https://zenodo.org/records/6293696'
FILENAME = '01_ZebrafishMul_GCaMP6s_485x400x9800_highSNR.tif'
URL = f'{RECORD}/files/{FILENAME}?download=1'
TOTAL_BYTES = 3859298816
MD5 = '06bb336e85d4890ed86528015c1ce958'
DEFAULT_CACHE = ('/content/zenodo_zebrafish' if os.name != 'nt'
                 else str(Path(__file__).resolve().parent / 'tmp' / 'zenodo_zebrafish'))


def file_digest(path, algorithm='sha256'):
    digest = hashlib.new(algorithm)
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def download_verified(cache_dir, *, url=URL, filename=FILENAME,
                      total_bytes=TOTAL_BYTES, expected_md5=MD5, retries=4):
    """Resume partial transfers, then verify the entire published file checksum."""
    folder = Path(cache_dir).resolve()
    if str(folder).replace('\\', '/').startswith('/content/drive/') or str(folder) == '/content/drive':
        raise ValueError('Use a temporary Colab folder such as /content/zenodo_zebrafish, not Google Drive')
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / filename
    receipt = folder / (filename + '.verified.json')
    if target.exists():
        stat = target.stat()
        if stat.st_size != total_bytes:
            raise ValueError(f'Unexpected size of {target}; keep it separately and retry in a fresh cache folder')
        if receipt.exists():
            saved = json.loads(receipt.read_text(encoding='utf-8'))
            if (saved.get('md5') == expected_md5 and saved.get('bytes') == stat.st_size
                    and saved.get('mtime_ns') == stat.st_mtime_ns):
                print(f'Verified temporary cache: {target}', flush=True)
                return target
        print('Verifying existing TIFF MD5...', flush=True)
        if file_digest(target, 'md5') != expected_md5:
            raise ValueError(f'MD5 mismatch: {target}; use a fresh temporary cache folder')
    else:
        partial = target.with_name(target.name + '.part')
        if partial.exists() and partial.stat().st_size > total_bytes:
            raise ValueError(f'Oversized partial file: {partial}')
        for attempt in range(retries):
            start = partial.stat().st_size if partial.exists() else 0
            if start == total_bytes:
                break
            headers = {'User-Agent': 'zebrafish-ddpm-research/1.0', 'Accept-Encoding': 'identity'}
            if start:
                headers['Range'] = f'bytes={start}-'
            request = urllib.request.Request(url, headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=90) as response:
                    status = response.status
                    if start and status == 200:
                        # Range ignored: restart rather than append duplicate bytes.
                        print('Server ignored resume Range; restarting temporary download.', flush=True)
                        start = 0
                    elif status == 206:
                        expected = f'bytes {start}-{total_bytes - 1}/{total_bytes}'
                        if response.headers.get('Content-Range') != expected:
                            raise ValueError(f'Unexpected Content-Range: {response.headers.get("Content-Range")}')
                    elif status != 200 or start:
                        raise ValueError(f'Unexpected download HTTP status {status}')
                    content_length = response.headers.get('Content-Length')
                    if content_length is not None and int(content_length) != total_bytes - start:
                        raise ValueError('Unexpected source length; refusing a different/truncated response')
                    done, last_report, started = start, time.monotonic(), time.monotonic()
                    print(f'Zenodo -> {partial}: {start / 1e9:.2f}/{total_bytes / 1e9:.2f} GB', flush=True)
                    with partial.open('ab' if start else 'wb') as stream:
                        while True:
                            chunk = response.read(8 * 1024 * 1024)
                            if not chunk:
                                break
                            if done + len(chunk) > total_bytes:
                                raise ValueError('Source exceeds the published file size')
                            stream.write(chunk)
                            done += len(chunk)
                            if time.monotonic() - last_report > 4:
                                rate = (done - start) / max(time.monotonic() - started, 1e-6) / 1e6
                                print(f'Download {100 * done / total_bytes:.1f}% ({rate:.1f} MB/s)', flush=True)
                                last_report = time.monotonic()
                if partial.stat().st_size != total_bytes:
                    raise OSError('Download interrupted; the next attempt will resume the partial file')
                break
            except (OSError, ValueError) as error:
                if attempt + 1 == retries:
                    raise RuntimeError(f'Zenodo download failed; partial file retained at {partial}') from error
                print(f'Download retry {attempt + 1}/{retries - 1}: {error}', flush=True)
                time.sleep(min(2 ** attempt, 8))
        if not partial.exists() or partial.stat().st_size != total_bytes:
            raise ValueError('Incomplete TIFF')
        print('Download complete; checking full-file MD5...', flush=True)
        if file_digest(partial, 'md5') != expected_md5:
            raise ValueError(f'MD5 mismatch; do not train on {partial}. Retry with a fresh cache folder')
        partial.replace(target)
    stat = target.stat()
    write_json(receipt, dict(url=url, bytes=stat.st_size, md5=expected_md5,
                            mtime_ns=stat.st_mtime_ns, full_source_md5_verified=True))
    print(f'Full TIFF verified: {target}', flush=True)
    return target


class RawTiffFrames:
    """Memory-mapped uncompressed grayscale pages, including inter-page TIFF tags."""
    def __init__(self, path):
        self.path = Path(path)
        with tifffile.TiffFile(path) as tif:
            self.shape = (len(tif.pages), *tif.pages[0].shape)
            if len(self.shape) != 3:
                raise ValueError('Expected a stack of 2D grayscale frames')
            self.dtype = np.dtype(tif.pages[0].dtype).newbyteorder(tif.byteorder)
            if self.dtype.kind not in 'iu' or self.dtype.itemsize != 2:
                raise ValueError('Expected a 16-bit integer TIFF')
            frame_bytes = int(np.prod(self.shape[1:])) * self.dtype.itemsize
            offsets = []
            for page in tif.pages:
                if (tuple(page.shape) != self.shape[1:] or np.dtype(page.dtype).kind != self.dtype.kind
                        or np.dtype(page.dtype).itemsize != self.dtype.itemsize):
                    raise ValueError('Inconsistent TIFF page shape/dtype')
                if int(page.compression) != 1:
                    raise ValueError('This lazy reader requires the uncompressed Zenodo TIFF')
                if page.is_tiled:
                    raise ValueError('Tiled TIFF layout is not supported by this reader')
                positions = np.asarray(page.dataoffsets, dtype=np.int64)
                counts = np.asarray(page.databytecounts, dtype=np.int64)
                if (not len(positions) or counts.sum() < frame_bytes
                        or not np.array_equal(np.diff(positions), counts[:-1])):
                    raise ValueError('Noncontiguous pixels within a TIFF page')
                offsets.append(int(positions[0]))
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self._mapped = np.memmap(path, mode='r', dtype=np.uint8)
        if self.offsets[-1] + frame_bytes > len(self._mapped):
            raise ValueError('TIFF pixel range exceeds the file')
        strides = np.diff(self.offsets)
        self._stack = None
        if len(strides) and np.all(strides == strides[0]):
            self._stack = np.ndarray(self.shape, dtype=self.dtype, buffer=self._mapped,
                                     offset=int(self.offsets[0]), strides=(int(strides[0]),
                                     self.shape[2] * self.dtype.itemsize, self.dtype.itemsize))

    def __len__(self):
        return self.shape[0]

    def close(self):
        self._stack = None
        self._mapped._mmap.close()

    def __getitem__(self, index):
        if self._stack is not None:
            return self._stack[index]
        if np.isscalar(index):
            return np.ndarray(self.shape[1:], dtype=self.dtype, buffer=self._mapped,
                              offset=int(self.offsets[index]))
        indices = np.arange(len(self))[index]
        return np.stack([self[int(i)] for i in indices])


class NormalizedTiffFrames:
    dtype = np.dtype('uint16')
    ndim = 3

    def __init__(self, raw, black, white):
        self.raw, self.black, self.white = raw, float(black), float(white)
        self.shape = raw.shape

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, index):
        pixels = self.raw[index].astype(np.float32)
        pixels = np.clip((pixels - self.black) / (self.white - self.black), 0, 1)
        return np.rint(pixels * 65535).astype(np.uint16)


def temporal_split(count, gap=32):
    """10 contiguous blocks: block4 validation, block8 test; embargo boundaries.

    Frames within gap of a train/heldout boundary are not used in training or
    evaluation. This reduces adjacent-frame leakage, not shared-fish anatomy.
    """
    if count < 20 or gap < 0:
        raise ValueError('Need at least 20 frames and nonnegative temporal gap')
    groups = np.minimum(np.arange(count, dtype=np.int64) * 10 // count, 9)
    split = np.zeros(count, dtype=np.int8)
    split[groups == 4] = 1
    split[groups == 8] = 2
    original = split.copy()
    boundaries = np.flatnonzero(original[1:] != original[:-1]) + 1
    for boundary in boundaries:
        split[max(0, boundary - gap):min(count, boundary + gap)] = -1
    if any(not np.any(split == code) for code in [0, 1, 2]):
        raise ValueError('Temporal gap is too large for this recording')
    return split, groups


def train_percentiles(raw, indices, sample_frames=512):
    """Deterministic train-only histogram sample, one shared mapping for all frames."""
    selected = indices[np.linspace(0, len(indices) - 1, min(sample_frames, len(indices))).astype(int)]
    histogram = np.zeros(65536, dtype=np.int64)
    signed = raw.dtype.kind == 'i'
    offset = 32768 if signed else 0
    for index in selected:
        values = raw[int(index)][::4, ::4].astype(np.int32).ravel() + offset
        histogram += np.bincount(values, minlength=65536)
    cumulative = np.cumsum(histogram)
    black = int(np.searchsorted(cumulative, cumulative[-1] * 0.01)) - offset
    white = int(np.searchsorted(cumulative, cumulative[-1] * 0.998)) - offset
    if white <= black:
        raise ValueError('Train-only intensity percentile range is empty')
    return black, white, len(selected)


def prepare_tiff_dataset(path, *, source_identity, gap=32, norm_samples=512,
                         expected_shape=(9800, 400, 485)):
    raw = RawTiffFrames(path)
    if expected_shape is not None and raw.shape != expected_shape:
        raise ValueError(f'Unexpected source shape {raw.shape}; expected {expected_shape}')
    split, groups = temporal_split(len(raw), gap)
    black, white, sampled = train_percentiles(raw, np.flatnonzero(split == 0), norm_samples)
    metadata = dict(source=source_identity, shape=list(raw.shape), raw_dtype=str(raw.dtype),
                    normalization=dict(method='train-only shared p1/p99.8 histogram sample',
                                       black=black, white=white, sample_frames=sampled, pixel_stride=4),
                    split_method='10 contiguous time blocks; val=4, test=8, others=train',
                    temporal_gap_frames_each_side=gap,
                    split_counts={name: int(np.sum(split == code)) for name, code in
                                  [('train', 0), ('val', 1), ('test', 2), ('excluded_gap', -1)]},
                    split_sha256=hashlib.sha256(split.tobytes()).hexdigest(),
                    recording_limit='One recording; held-out time is not an independent fish test')
    fingerprint = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    return dict(images=NormalizedTiffFrames(raw, black, white), split=split, groups=groups,
                names=np.asarray([f'zenodo_frame_{i:05d}' for i in range(len(raw))]),
                metadata_json=np.asarray(json.dumps(metadata)), fingerprint=fingerprint)


def load_zenodo_full(cache_dir=DEFAULT_CACHE, gap=32, norm_samples=512):
    path = download_verified(cache_dir)
    data = prepare_tiff_dataset(path, source_identity=dict(record=RECORD, filename=FILENAME,
                               bytes=TOTAL_BYTES, md5=MD5, full_source_md5_verified=True),
                               gap=gap, norm_samples=norm_samples)
    metadata = json.loads(str(data['metadata_json']))
    write_json(Path(cache_dir) / 'dataset_manifest.json', metadata)
    print(f'Zenodo highSNR: {len(data["images"])} full frames; '
          f'splits={metadata["split_counts"]}; lazy TIFF, not loaded into RAM', flush=True)
    print(f'Real shared normalization: {metadata["normalization"]}', flush=True)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', default=DEFAULT_CACHE)
    parser.add_argument('--gap', type=int, default=32)
    args = parser.parse_args()
    load_zenodo_full(args.cache_dir, gap=args.gap)
