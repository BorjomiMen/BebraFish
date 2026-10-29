#!/usr/bin/env python3
"""Image-derived zebrafish density prior -> static NAOMi frames and nucleus masks."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import hashlib
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
CODE = ROOT / "simulator" / "code"
ZENODO_REFERENCE_CELLS = 12_000
# Positive pixels of density_base.png prepared from the Zenodo TIFF at 0.875 um/px.
ZENODO_REFERENCE_TISSUE_AREA_UM2 = 104_437.375


def cells_from_tissue_area(area_um2: float) -> int:
    """Scale the calibrated NAOMi count by estimated physical tissue area."""
    if not np.isfinite(area_um2) or area_um2 <= 0:
        raise ValueError("Tissue area must be positive and finite")
    return max(1, int(round(ZENODO_REFERENCE_CELLS *
                            area_um2 / ZENODO_REFERENCE_TISSUE_AREA_UM2)))

def save_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def blur(array: np.ndarray, sigma: float) -> np.ndarray:
    pad = int(np.ceil(3 * sigma))
    padded = np.pad(array.astype(np.float32), pad, mode="reflect")
    fy = np.fft.fftfreq(padded.shape[0])[:, None]
    fx = np.fft.fftfreq(padded.shape[1])[None, :]
    kernel = np.exp(-2 * np.pi**2 * sigma**2 * (fy * fy + fx * fx))
    result = np.fft.ifft2(np.fft.fft2(padded) * kernel).real
    h, w = array.shape
    return result[pad:pad + h, pad:pad + w].astype(np.float32)


def maximum5(array: np.ndarray) -> np.ndarray:
    padded = np.pad(array, 2, mode="edge")
    h, w = array.shape
    return np.maximum.reduce([padded[dy:dy + h, dx:dx + w]
                              for dy in range(5) for dx in range(5)])


def grayscale(image: Image.Image) -> np.ndarray:
    """Preserve the intensity scale when possible; accept color and grayscale images."""
    if image.mode == "P":
        image = image.convert("RGB")
    pixels = np.asarray(image)
    if pixels.ndim == 2:
        result = pixels.astype(np.float32)
    elif pixels.ndim == 3 and pixels.shape[2] >= 3:
        rgb = pixels[..., :3].astype(np.float32)
        result = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    elif pixels.ndim == 3 and pixels.shape[2] == 1:
        result = pixels[..., 0].astype(np.float32)
    else:
        raise ValueError(f"Unsupported image array shape: {pixels.shape}")
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def native_shape(shape: tuple[int, int], pixel_size_um: float) -> tuple[int, int]:
    # Preserve the exact original NAOMi raster for the existing Zenodo profile.
    if shape == (400, 485) and abs(pixel_size_um - 0.875) < 1e-9:
        return (350, 423)
    return tuple(max(8, int(round(n * pixel_size_um))) for n in shape)


def prepare(input_path: Path, output: Path, frame_count: int,
            band_strength: float | None, pixel_size_um: float) -> None:
    with Image.open(input_path) as image:
        width, height = image.size
        if min(width, height) < 16:
            raise ValueError("The image must be at least 16 pixels along each axis")
        if width * height > 12_000_000:
            raise ValueError("Image is too large for this in-memory NAOMi profile; crop it first")
        n_frames = getattr(image, "n_frames", 1)
        indices = np.unique(np.linspace(0, n_frames - 1, min(frame_count, n_frames), dtype=int))
        frames = []
        for index in indices:
            image.seek(int(index))
            frames.append(grayscale(image))
    reference = np.median(np.stack(frames), axis=0).astype(np.float32)
    h, w = reference.shape
    is_zenodo = (h, w) == (400, 485) and reference.max() > 255
    scale = 0.875 / pixel_size_um
    if is_zenodo:
        # Keep the original, already compared Zenodo recipe byte-for-byte.
        tissue = blur(reference, 8) > np.percentile(reference, 25) + 1000
        spot = blur(reference, 1) - blur(reference, 3)
        candidates = (spot == maximum5(spot)) & (spot > np.percentile(spot[tissue], 83)) & tissue
        candidates &= ~(maximum5((reference >= 32760).astype(np.uint8)) > 0)
        local_sigma, broad_sigma = 5.0, 14.0
        default_band = 0.70
        method = "zenodo_calibrated"
    else:
        # Bright spots are only a rough proxy for cell locations in a new image.
        p1, p25, p90, p99 = np.percentile(reference, [1, 25, 90, 99])
        dynamic = max(float(p99 - p1), 1e-6)
        tissue = blur(reference, max(1.0, 8 * scale)) > p25 + 0.15 * (p90 - p25)
        if tissue.mean() < 0.01:
            tissue = np.ones((h, w), dtype=bool)
        spot = blur(reference, max(0.5, scale)) - blur(reference, max(1.5, 3 * scale))
        response_floor = max(float(np.percentile(spot[tissue], 83)), 0.05 * dynamic)
        candidates = (spot == maximum5(spot)) & (spot > response_floor) & tissue
        local_sigma, broad_sigma = max(1.0, 5 * scale), max(2.0, 14 * scale)
        default_band = 0.0
        method = "adaptive_brightness_proxy"
    if not tissue.any():
        tissue = np.ones((h, w), dtype=bool)
    if candidates.any():
        local = blur(candidates.astype(np.float32), local_sigma)
        broad = blur(candidates.astype(np.float32), broad_sigma)
        combined = 0.85 * local + 0.15 * broad
        density = np.where(tissue, combined + 0.005 * combined.max(), 0).astype(np.float32)
    else:
        # A uniform or very dim image still yields a valid, explicitly uncertain prior.
        density = tissue.astype(np.float32)
    applied_band = default_band if band_strength is None else band_strength
    if applied_band:
        if (h, w) != (400, 485):
            raise ValueError("The row-217 band belongs to the 400x485 Zenodo field; use --band-strength 0")
        y, x = np.indices((h, w))
        center_y = 217 + 1.5 * np.sin(2 * np.pi * x / 310)
        band = np.exp(-0.5 * ((y - center_y) / 4.5) ** 2)
        density *= 1 - applied_band * band
    density /= max(float(density.max()), 1e-9)
    output.mkdir(parents=True, exist_ok=True)
    density_png = np.round(density * 65535).astype(np.uint16)
    save_png(output / "density_base.png", density_png)
    tissue_pixels = int(np.count_nonzero(density_png))
    tissue_area_um2 = tissue_pixels * pixel_size_um**2
    suggested_cells = cells_from_tissue_area(tissue_area_um2)
    metadata = {"input": str(input_path.resolve()), "input_frames": n_frames,
                "sampled_frames": indices.tolist(), "shape_yx": [h, w],
                "native_shape_yx": list(native_shape((h, w), pixel_size_um)),
                "pixel_size_um": pixel_size_um, "candidate_spots": int(candidates.sum()),
                "tissue_pixels": tissue_pixels, "tissue_area_um2": tissue_area_um2,
                "suggested_cells_by_area": suggested_cells,
                "band_strength": applied_band, "method": method,
                "note": "Heuristic placement prior, not measured cell density or real neuron masks."}
    (output / "density_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Density map: {output / 'density_base.png'}; {h}x{w}; "
          f"tissue area: {tissue_area_um2:.1f} um^2; area estimate: {suggested_cells} cells")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def density_variant(base: np.ndarray, seed: int) -> np.ndarray:
    tissue = base > 0
    rng = np.random.default_rng(seed)
    h, w = base.shape
    field = blur(rng.standard_normal((h, w)).astype(np.float32), 38)
    field = (field - field[tissue].mean()) / max(field[tissue].std(), 1e-6)
    field = np.clip(field, -2, 2)
    result = base * np.exp(0.12 * field)
    result[~tissue] = 0
    result /= result.max()
    return np.round(np.clip(result, 0, 1) * 65535).astype(np.uint16)


def export_masks(native_file: Path, destination: Path, shape: tuple[int, int],
                 native: tuple[int, int]) -> int:
    records = np.fromfile(native_file, dtype="<u2")
    if records.size == 0 or records.size % 3:
        raise ValueError(f"Invalid footprint triples: {native_file}")
    records = records.reshape(-1, 3)
    ids = records[:, 0].astype(np.uint32)
    h, w = shape
    native_h, native_w = native
    source = (records[:, 1].astype(np.int32) - 1) * native_w + records[:, 2].astype(np.int32) - 1
    if source.min() < 0 or source.max() >= native_h * native_w:
        raise ValueError("Footprint coordinates outside native grid")
    sy = np.minimum(((np.arange(h) + 0.5) * native_h / h).astype(int), native_h - 1)
    sx = np.minimum(((np.arange(w) + 0.5) * native_w / w).astype(int), native_w - 1)
    target_source = (sy[:, None] * native_w + sx[None, :]).ravel()
    order = np.argsort(target_source)
    sorted_source = target_source[order]
    left = np.searchsorted(sorted_source, source, side="left")
    right = np.searchsorted(sorted_source, source, side="right")
    lengths = right - left
    starts = np.cumsum(lengths, dtype=np.int64) - lengths
    expanded_ids = np.repeat(ids, lengths)
    target_ix = np.repeat(left, lengths) + np.arange(expanded_ids.size) - np.repeat(starts, lengths)
    pixels = order[target_ix].astype(np.uint32)
    pairs = np.unique(np.stack([expanded_ids, pixels], axis=1), axis=0)
    expanded_ids, pixels = pairs[:, 0], pairs[:, 1]
    counts = np.bincount(expanded_ids, minlength=int(ids.max()) + 1)
    offsets = np.r_[0, np.cumsum(counts[1:], dtype=np.uint32)]
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / "all_instance_footprints.npz", offsets=offsets,
                        pixel_index=pixels, shape_yx=np.array([h, w], dtype=np.uint32))
    save_png(destination / "semantic_mask.png",
             ((np.bincount(pixels, minlength=h * w).reshape(h, w) > 0) * 255).astype(np.uint8))
    # A PNG can hold one ID per pixel. Overlaps remain complete in the NPZ above.
    label = np.zeros(h * w, dtype=np.uint16)
    order_by_pixel = np.argsort(pixels, kind="stable")
    sorted_pixels = pixels[order_by_pixel]
    first = np.r_[True, sorted_pixels[1:] != sorted_pixels[:-1]]
    label[sorted_pixels[first]] = expanded_ids[order_by_pixel[first]].astype(np.uint16)
    save_png(destination / "instance_ids.png", label.reshape(h, w))
    return int(ids.max())


def build_mex() -> None:
    compiler = shutil.which("mkoctfile")
    if not compiler:
        raise RuntimeError("mkoctfile is missing; install GNU Octave development tools")
    for name in ("sample_dense_locations_exact", "array_SubSubTest"):
        source = CODE / "MEX" / f"{name}.cpp"
        subprocess.run([compiler, "--mex", "-O3", "-o", str(source.with_suffix("")), str(source)], check=True)
        print(f"Built {name}")


def generate_one(seed: int, output: Path, base: np.ndarray, args: argparse.Namespace) -> None:
    final = output / "samples" / f"seed{seed}"
    expected = [final / "noisy.png", final / "clean.png", final / "instance_ids.png",
                final / "semantic_mask.png", final / "all_instance_footprints.npz"]
    density_hash = file_sha256(output / "density_base.png")
    density_meta = json.loads((output / "density_metadata.json").read_text())
    h, w = base.shape
    native = tuple(density_meta["native_shape_yx"])
    pixel_size_um = density_meta["pixel_size_um"]
    metadata_path = final / "metadata.json"
    if any(path.exists() for path in expected):
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text())
            same = (metadata.get("requested_cells") == args.cells and
                    metadata.get("nucleus_radius_um") == args.nucleus_radius and
                    metadata.get("density_sha256") == density_hash and
                    metadata.get("pixel_size_um") == pixel_size_um)
            if same and all(path.is_file() and path.stat().st_size > 0 for path in expected):
                print(f"seed {seed}: already complete", flush=True)
                return
            if not same:
                raise RuntimeError(f"Seed {seed} already exists with different parameters/density; use another output directory")
    run = output / "raw" / f"seed{seed}"
    run.mkdir(parents=True, exist_ok=True)
    # A failed older run may have anatomy cached for another input map.
    for stale in run.iterdir():
        if stale.is_file():
            stale.unlink()
    prior = output / "priors" / f"seed{seed}.png"
    save_png(prior, density_variant(base, seed))
    env = os.environ.copy()
    env.update({"NAOMI_SEED": str(seed), "NAOMI_CELL_COUNT": str(args.cells),
                "NAOMI_NUC_RAD": str(args.nucleus_radius), "NAOMI_DEPTH_UM": "10",
                "NAOMI_EXC_NA": "0.6", "NAOMI_OBJ_NA": "0.8",
                "NAOMI_HALO_SCALE": "0.75", "NAOMI_LOCAL_BG_LEVEL": "10",
                "NAOMI_PRIOR_PNG": str(prior), "NAOMI_OUT_ROOT": str(output / "raw"),
                "NAOMI_RUN_DIR": str(run), "NAOMI_EXPORT_DATASET": "1",
                "NAOMI_PIXEL_SIZE_UM": str(pixel_size_um),
                "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    log = run / "octave.log"
    with log.open("w") as stream:
        result = subprocess.run(["octave", "--quiet", "--eval",
                                 f"run('{str(CODE / 'run_naomi_density_image.m').replace(chr(39), chr(39)*2)}')"],
                                env=env, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"NAOMi failed for seed {seed}; see {log}")
    final.mkdir(parents=True, exist_ok=True)
    for src, dst in (("naomi_noisy.png", "noisy.png"),
                     ("naomi_clean.png", "clean.png")):
        shutil.copy2(run / src, final / dst)
    max_labeled_id = export_masks(run / "nucleus_footprints_native_u16.bin", final,
                                  (h, w), native)
    centers_file = run / "naomi_centers_yx_px.csv"
    placed = sum(1 for _ in centers_file.open())
    if placed != args.cells or max_labeled_id > placed:
        raise RuntimeError(f"Expected {args.cells} cells, got {placed} centers and max mask ID {max_labeled_id}")
    shutil.copy2(centers_file, final / "centers_yx_px.csv")
    metadata_path.write_text(json.dumps({"seed": seed, "requested_cells": args.cells,
        "placed_cells": placed, "nucleus_radius_um": args.nucleus_radius,
        "cell_count_mode": "area_auto" if args.auto_cells else "explicit",
        "density_sha256": density_hash, "pixel_size_um": pixel_size_um,
        "image_shape_yx": [h, w], "native_shape_yx": list(native)}, indent=2) + "\n")
    # Raw caches and float arrays are reproducible from the seed; keep only the log.
    for path in run.iterdir():
        if path.is_file() and path.name != "octave.log":
            path.unlink()
    print(f"seed {seed}: {placed} cells; {final}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "prepare", "generate", "all"))
    parser.add_argument("--input", type=Path, help="input PNG, TIFF stack, or any Pillow-readable image")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    parser.add_argument("--frame-count", type=int, default=64)
    parser.add_argument("--band-strength", type=float, default=None,
                        help="Zenodo 400x485 corridor strength; automatic 0.70 for that profile, 0 otherwise")
    parser.add_argument("--pixel-size-um", type=float, default=0.875,
                        help="physical pixel size in micrometers; 0.576 for the original NDA frames")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--start-seed", type=int, default=20263000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--cells", default="auto",
                        help="'auto' scales 12000 Zenodo cells by physical tissue area; or enter an integer")
    parser.add_argument("--nucleus-radius", type=float, default=1.6)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.auto_cells = args.cells == "auto"
    if not args.auto_cells:
        try:
            args.cells = int(args.cells)
        except ValueError:
            parser.error("cells must be 'auto' or a positive integer")
        if not 1 <= args.cells <= 65535:
            parser.error("cells must be 1..65535 because masks store uint16 IDs")
    if args.action == "build":
        build_mex()
        return
    if args.frame_count < 1 or args.count < 1 or args.workers < 1:
        parser.error("frame-count, count and workers must be positive")
    if args.band_strength is not None and not (0 <= args.band_strength <= 1):
        parser.error("band-strength must be 0..1")
    if not (0 < args.nucleus_radius < 3.8 and 0 < args.pixel_size_um <= 2):
        parser.error("nucleus-radius must be 0..3.8 and pixel-size-um must be 0..2")
    if args.action in ("prepare", "all"):
        if args.input is None:
            parser.error("--input is required for prepare/all")
        prepare(args.input, args.output, args.frame_count, args.band_strength,
                args.pixel_size_um)
    if args.action in ("generate", "all"):
        path = args.output / "density_base.png"
        if not path.is_file():
            raise FileNotFoundError(f"Run prepare first: {path}")
        if not shutil.which("octave"):
            raise RuntimeError("octave is not in PATH")
        for name in ("sample_dense_locations_exact", "array_SubSubTest"):
            if not list((CODE / "MEX").glob(name + ".mex*")):
                raise RuntimeError("Compile MEX first: python pipeline.py build")
        base = np.asarray(Image.open(path), dtype=np.float32) / 65535
        if args.auto_cells:
            density_meta = json.loads((args.output / "density_metadata.json").read_text())
            area_um2 = int(np.count_nonzero(base)) * density_meta["pixel_size_um"]**2
            args.cells = cells_from_tissue_area(area_um2)
            print(f"Auto cells: {args.cells} = 12000 * {area_um2:.1f} / "
                  f"{ZENODO_REFERENCE_TISSUE_AREA_UM2:.1f} um^2", flush=True)
        if args.cells > 65535:
            parser.error("automatic cell count exceeds 65535; crop the input field")
        seeds = range(args.start_seed, args.start_seed + args.count)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(generate_one, seed, args.output, base, args): seed for seed in seeds}
            for future in as_completed(futures):
                future.result()

if __name__ == "__main__":
    main()
