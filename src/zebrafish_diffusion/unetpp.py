#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""U-Net++ semantic segmentation: paired grayscale images and binary masks.
Training, validation, checkpointing and prediction; native image geometry is preserved."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:
    raise SystemExit("Установите зависимости: python -m pip install torch numpy pillow") from exc


FORMAT = "naomi-unetpp-fullframe-v1"
SHAPE = (400, 485)  # строго H,W, не W,H
DEFAULT_DATA = "data/segmentation"
NORMALIZATION = {"low_percentile": 0.5, "high_percentile": 99.5}
THRESHOLDS = np.linspace(0.1, 0.9, 17).tolist()


@dataclass(frozen=True)
class Pair:
    name: str
    image: Path
    mask: Path
    group: str


def save_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def save_checkpoint(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Один seed фиксирует split и генераторы; CUDA не обещает битовую идентичность.
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def read_plane(path):
    with Image.open(path) as image:
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError(f"Нужен один полный 2D-кадр, получен стек: {path}")
        array = np.array(image)
    if array.ndim != 2:
        raise ValueError(f"Ожидался grayscale 2D, получено {array.shape}: {path}")
    if not np.isfinite(array).all():
        raise ValueError(f"NaN/Inf в файле: {path}")
    return array


def normalize_image(array, settings=NORMALIZATION):
    array = array.astype(np.float32, copy=False)
    low, high = np.percentile(array, [settings["low_percentile"], settings["high_percentile"]])
    if high <= low:
        low, high = float(array.min()), float(array.max())
    if high <= low:
        return np.zeros_like(array)
    # uint16 НЕ преобразуется через PIL.convert('L'): это обрезало бы диапазон.
    return np.clip((array - low) / (high - low), 0, 1).astype(np.float32)


def discover_pairs(root):
    root = Path(root).expanduser().resolve()
    if (root / "pairs.csv").is_file():
        from .data import segmentation_records
        rows, report = segmentation_records(root)
        return [Pair(r["name"], Path(r["image"]), Path(r["mask"]), r["group"]) for r in rows], report
    image_dir = root / "images"
    mask_dir = root / "masks_png_400x485"
    manifest = mask_dir / "correspondence.csv"
    if not manifest.is_file() or not image_dir.is_dir():
        raise FileNotFoundError(f"Нужны {image_dir} и {manifest}; задайте --data.")
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not {"image", "semantic_mask", "seed"}.issubset(rows[0]):
        raise ValueError("В correspondence.csv нужны image, semantic_mask, seed.")
    pairs, missing, seen, seen_images = [], [], set(), set()
    for row in rows:
        name = row["image"]
        if name in seen:
            raise ValueError(f"Повтор image в correspondence.csv: {name}")
        seen.add(name)
        image = (image_dir / name).resolve()
        mask = (mask_dir / row["semantic_mask"]).resolve()
        if not image.is_relative_to(image_dir) or not mask.is_relative_to(mask_dir):
            raise ValueError(f"Путь выходит за папку данных: {name}")
        if not image.is_file() or not mask.is_file():
            missing.append({"image": name, "image_exists": image.is_file(),
                            "mask_exists": mask.is_file()})
            continue
        if image in seen_images:
            raise ValueError(f"Повтор пути изображения: {image}")
        seen_images.add(image)
        group = row["seed"].strip()
        if not group:
            raise ValueError(f"Не задан seed для {name}")
        pairs.append(Pair(name, image, mask, group))
    unlisted = sorted(p.name for p in image_dir.glob("*.png") if p.name not in seen)
    if unlisted:
        raise ValueError(f"PNG без записи в correspondence.csv: {unlisted[:10]}")
    if not pairs:
        raise ValueError("Не найдено ни одной существующей пары image + semantic_mask.")
    print(f"Данные: {len(pairs)} пар; отсутствующих пар из CSV: {len(missing)}", flush=True)
    if missing:
        print("Пропущены незавершённые пары: " + ", ".join(x["image"] for x in missing), flush=True)
    return pairs, {"root": str(root), "csv_rows": len(rows), "pairs": len(pairs),
                   "missing_pairs": missing}


def audit_pairs(pairs, report):
    records = {}
    print("Проверка размеров, масок и контрольных сумм всех полных кадров...", flush=True)
    for pair in pairs:
        image, mask = read_plane(pair.image), read_plane(pair.mask)
        if image.shape != SHAPE or mask.shape != SHAPE:
            raise ValueError(f"{pair.name}: image={image.shape}, mask={mask.shape}, нужно {SHAPE}.")
        if float(image.max()) <= float(image.min()):
            raise ValueError(f"Постоянное изображение: {pair.name}")
        values = set(np.unique(mask).tolist())
        if values not in ({0, 1}, {0, 255}):
            raise ValueError(f"Нужна непустая binary semantic_mask, получено {values}: {pair.mask}")
        records[pair.name] = {
            "seed": pair.group, "foreground_pixels": int(np.count_nonzero(mask)),
            "pixels": int(mask.size), "dtype": str(image.dtype),
            "image_sha256": hashlib.sha256(pair.image.read_bytes()).hexdigest(),
            "mask_sha256": hashlib.sha256(pair.mask.read_bytes()).hexdigest(),
        }
    report.update({"shape_hw": list(SHAPE), "records": records})
    print(f"Проверено {len(pairs)} пар: каждый кадр {SHAPE[0]}x{SHAPE[1]}, без resize/crop.", flush=True)
    return report


def make_split(pairs, seed, val_fraction, test_fraction):
    groups = sorted({p.group for p in pairs})
    if len(groups) < 5:
        raise ValueError("Для train/val/test нужно хотя бы 5 независимых seed.")
    rng = random.Random(seed)
    rng.shuffle(groups)
    n_val = max(1, round(len(groups) * val_fraction))
    n_test = max(1, round(len(groups) * test_fraction))
    if n_val + n_test >= len(groups):
        raise ValueError("Слишком большие val_fraction/test_fraction.")
    val_groups = set(groups[:n_val])
    test_groups = set(groups[n_val:n_val + n_test])
    result = {"train": [], "val": [], "test": []}
    for pair in pairs:
        key = "val" if pair.group in val_groups else "test" if pair.group in test_groups else "train"
        result[key].append(pair.name)
    return result


def validate_split(split, pairs, records):
    names = {p.name for p in pairs}
    flattened = [name for key in ("train", "val", "test") for name in split[key]]
    if set(flattened) != names or len(set(flattened)) != len(flattened):
        raise ValueError("Split не соответствует данным или содержит пересечения.")
    if any(not split[key] for key in ("train", "val", "test")):
        raise ValueError("Пустая часть split.")
    group_owners, hash_owners = {}, {}
    for key, items in split.items():
        for name in items:
            for value, owners in ((records[name]["seed"], group_owners),
                                  (records[name]["image_sha256"], hash_owners)):
                if value in owners and owners[value] != key:
                    raise ValueError("Утечка: один seed или дубликат изображения оказался в разных split.")
                owners[value] = key


class FullFrameDataset(Dataset):
    def __init__(self, pairs, augment=False, normalization=NORMALIZATION):
        self.pairs = pairs
        self.augment = augment
        self.normalization = normalization

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        pair = self.pairs[index]
        image = normalize_image(read_plane(pair.image), self.normalization)
        mask = (read_plane(pair.mask) > 0).astype(np.float32)
        if image.shape != SHAPE or mask.shape != SHAPE:
            raise ValueError(f"Данные изменились после проверки: {pair.name}")
        if self.augment:
            # Геометрия одинакова для image и mask; только полные кадры.
            if random.random() < 0.5:
                image, mask = image[:, ::-1], mask[:, ::-1]
            if random.random() < 0.5:
                image, mask = image[::-1, :], mask[::-1, :]
            # Мягкая фотометрия: маска не изменяется.
            if random.random() < 0.5:
                image = np.clip(image, 0, 1) ** random.uniform(0.9, 1.1)
            if random.random() < 0.5:
                image = image * random.uniform(0.9, 1.1) + random.uniform(-0.025, 0.025)
            if random.random() < 0.3:
                image = image + np.random.normal(0, random.uniform(0.002, 0.012), image.shape)
            image = np.clip(image, 0, 1)
        return (torch.from_numpy(np.ascontiguousarray(image[None], dtype=np.float32)),
                torch.from_numpy(np.ascontiguousarray(mask[None], dtype=np.float32)), pair.name)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        groups = math.gcd(8, out_channels)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetPlusPlus(nn.Module):
    def __init__(self, base_channels=32):
        super().__init__()
        channels = [base_channels * 2**i for i in range(5)]
        self.nodes = nn.ModuleDict()
        for i in range(5):
            self.nodes[f"{i}_0"] = ConvBlock(1 if i == 0 else channels[i - 1], channels[i])
        for j in range(1, 5):
            for i in range(5 - j):
                self.nodes[f"{i}_{j}"] = ConvBlock(j * channels[i] + channels[i + 1], channels[i])
        self.heads = nn.ModuleList([nn.Conv2d(channels[0], 1, 1) for _ in range(4)])

    def forward(self, image):
        height, width = image.shape[-2:]
        image = F.pad(image, (0, (-width) % 16, 0, (-height) % 16), mode="replicate")
        nodes = {}
        for i in range(5):
            value = image if i == 0 else F.max_pool2d(nodes[i - 1, 0], 2)
            nodes[i, 0] = self.nodes[f"{i}_0"](value)
        for j in range(1, 5):
            for i in range(5 - j):
                up = F.interpolate(nodes[i + 1, j - 1], size=nodes[i, 0].shape[-2:],
                                   mode="bilinear", align_corners=False)
                nodes[i, j] = self.nodes[f"{i}_{j}"](
                    torch.cat([nodes[i, k] for k in range(j)] + [up], dim=1))
        return [head(nodes[0, j + 1])[..., :height, :width] for j, head in enumerate(self.heads)]


def segmentation_loss(outputs, target, pos_weight):
    losses = []
    for logits in outputs:
        logits = logits.float()
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
        probability = logits.sigmoid()
        intersection = (probability * target).sum(dim=(1, 2, 3))
        denominator = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice_loss = 1 - ((2 * intersection + 1) / (denominator + 1)).mean()
        losses.append(0.5 * bce + 0.5 * dice_loss)
    return torch.stack(losses).mean()


def predict_probability(outputs):
    # Одинаковое правило при validation, test и predict.
    return torch.stack([logits.float().sigmoid() for logits in outputs]).mean(dim=0)


def autocast_context(device, enabled):
    return torch.amp.autocast("cuda", dtype=torch.float16) if enabled else nullcontext()


def metrics_from_counts(tp, fp, fn, tn):
    tp, fp, fn, tn = map(float, (tp, fp, fn, tn))
    return {
        "dice": (2 * tp) / max(2 * tp + fp + fn, 1),
        "iou": tp / max(tp + fp + fn, 1),
        "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


@torch.inference_mode()
def evaluate(model, loader, device, amp, pos_weight, thresholds):
    model.eval()
    counts = torch.zeros((len(thresholds), 4), dtype=torch.int64, device=device)
    threshold_tensor = torch.tensor(thresholds, device=device).view(-1, 1, 1, 1, 1)
    loss_sum, number = 0.0, 0
    for image, target, _ in loader:
        image, target = image.to(device, non_blocking=True), target.to(device, non_blocking=True)
        with autocast_context(device, amp):
            outputs = model(image)
        loss_sum += float(segmentation_loss(outputs, target, pos_weight)) * image.size(0)
        number += image.size(0)
        prediction = predict_probability(outputs).unsqueeze(0) >= threshold_tensor
        truth = target.bool().unsqueeze(0)
        dims = (1, 2, 3, 4)
        counts[:, 0] += (prediction & truth).sum(dims)
        counts[:, 1] += (prediction & ~truth).sum(dims)
        counts[:, 2] += (~prediction & truth).sum(dims)
        counts[:, 3] += (~prediction & ~truth).sum(dims)
    rows = []
    for threshold, values in zip(thresholds, counts.cpu().tolist()):
        row = metrics_from_counts(*values)
        row.update(threshold=float(threshold), loss=loss_sum / max(number, 1))
        rows.append(row)
    return rows


def train_epoch(model, loader, optimizer, scaler, device, amp, pos_weight, accumulation, clip):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss, number = 0.0, 0
    for step, (image, target, _) in enumerate(loader):
        image, target = image.to(device, non_blocking=True), target.to(device, non_blocking=True)
        with autocast_context(device, amp):
            outputs = model(image)
        loss = segmentation_loss(outputs, target, pos_weight)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Loss стал NaN/Inf. Проверьте данные или отключите AMP через --no-amp.")
        # Неполная последняя группа получает правильный масштаб градиента.
        group_start = (step // accumulation) * accumulation
        group_end = min(group_start + accumulation, len(loader))
        group_samples = min(group_end * loader.batch_size, len(loader.dataset)) - group_start * loader.batch_size
        scaler.scale(loss * (image.size(0) / group_samples)).backward()
        if (step + 1) % accumulation == 0 or step + 1 == len(loader):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        total_loss += float(loss.detach()) * image.size(0)
        number += image.size(0)
        if (step + 1) % 25 == 0 or step + 1 == len(loader):
            print(f"  train batch {step + 1}/{len(loader)} loss={total_loss / number:.5f}", flush=True)
    return total_loss / number


def overlay(image, mask, color):
    rgb = np.repeat((image[..., None] * 255).round().astype(np.uint8), 3, axis=2)
    rgb[mask] = (0.55 * rgb[mask] + 0.45 * np.asarray(color)).astype(np.uint8)
    return rgb


def save_preview(path, image, truth, probability, threshold):
    h, w = image.shape
    panels = [np.repeat((image[..., None] * 255).round().astype(np.uint8), 3, axis=2),
              overlay(image, truth, (255, 80, 80)),
              overlay(image, probability >= threshold, (60, 255, 90)),
              np.repeat((probability[..., None] * 255).round().astype(np.uint8), 3, axis=2)]
    canvas = Image.new("RGB", (w * len(panels), h + 24), "black")
    draw = ImageDraw.Draw(canvas)
    for i, (panel, label) in enumerate(zip(panels, ("Input", "Ground truth", "Prediction", "Probability"))):
        canvas.paste(Image.fromarray(panel), (i * w, 24))
        draw.text((i * w + 5, 5), label, fill="white")
    canvas.save(path)


@torch.inference_mode()
def export_test_predictions(model, pairs, out, device, amp, normalization, threshold):
    out.mkdir(parents=True, exist_ok=True)
    model.eval()
    for pair in pairs:
        image = normalize_image(read_plane(pair.image), normalization)
        tensor = torch.from_numpy(image[None, None]).to(device)
        with autocast_context(device, amp):
            outputs = model(tensor)
        probability = predict_probability(outputs)[0, 0].cpu().numpy()
        Image.fromarray(((probability >= threshold) * 255).astype(np.uint8)).save(out / pair.name)
        save_preview(out / (Path(pair.name).stem + "_preview.jpg"), image,
                     read_plane(pair.mask) > 0, probability, threshold)


def choose_device(requested):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if requested == "auto" else torch.device(requested)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Поддерживаются --device auto, cpu, cuda или cuda:0.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA недоступна. Установите CUDA-сборку PyTorch или задайте --device cpu.")
    print(f"PyTorch {torch.__version__}; устройство {device}", flush=True)
    if device.type == "cuda":
        print(torch.cuda.get_device_name(device), flush=True)
    else:
        print("CPU: полное обучение может идти долго; используйте NVIDIA GPU для основного запуска.", flush=True)
    return device


def load_checkpoint(path):
    state = torch.load(Path(path), map_location="cpu", weights_only=True)
    if state.get("format") != FORMAT:
        raise ValueError("Это не checkpoint данного U-Net++.")
    return state


def write_history(path, row):
    exists = path.is_file()
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_train(args):
    seed_everything(args.seed)
    global SHAPE
    pairs, report = discover_pairs(args.data)
    SHAPE = tuple(args.image_size) if args.image_size else tuple(read_plane(pairs[0].image).shape)
    report = audit_pairs(pairs, report)
    resume = load_checkpoint(args.resume) if args.resume else None
    if resume:
        if args.base_channels != resume["base_channels"]:
            raise ValueError(f"Для resume задайте --base-channels {resume['base_channels']}.")
        if args.seed != resume["seed"]:
            raise ValueError(f"Для resume задайте --seed {resume['seed']}.")
        split = resume["split"]
        if report["records"] != resume["data_records"]:
            raise ValueError("Данные или маски изменились с момента checkpoint. Начните новый запуск.")
        run_dir = Path(args.resume).resolve().parent
        if not (run_dir / "best.pt").is_file():
            raise FileNotFoundError("Для resume рядом с last.pt нужен best.pt этого запуска.")
    else:
        split = make_split(pairs, args.seed, args.val_fraction, args.test_fraction)
        run_dir = Path(args.output).resolve() / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir.mkdir(parents=True, exist_ok=False)
    validate_split(split, pairs, report["records"])
    save_json(run_dir / "data_audit.json", report)
    save_json(run_dir / "split.json", split)
    config = vars(args).copy()
    config.update(format=FORMAT, native_shape_hw=list(SHAPE), normalization=NORMALIZATION,
                  probability_rule="mean_of_four_sigmoid_heads", split_sizes={k: len(v) for k, v in split.items()})
    save_json(run_dir / ("resume_config.json" if resume else "config.json"), config)
    print(f"Train/val/test: {len(split['train'])}/{len(split['val'])}/{len(split['test'])}", flush=True)
    print(f"Результаты: {run_dir}", flush=True)
    lookup = {pair.name: pair for pair in pairs}
    subsets = {key: [lookup[name] for name in names] for key, names in split.items()}
    device = choose_device(args.device)
    amp = device.type == "cuda" and not args.no_amp
    model = UNetPlusPlus(args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=8, min_lr=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    records = report["records"]
    foreground = sum(records[name]["foreground_pixels"] for name in split["train"])
    pixels = sum(records[name]["pixels"] for name in split["train"])
    weight = float(np.clip((pixels - foreground) / max(foreground, 1), 0.5, 4))
    pos_weight = torch.tensor(weight, dtype=torch.float32, device=device)
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {}
    for key in ("train", "val", "test"):
        loaders[key] = DataLoader(
            FullFrameDataset(subsets[key], augment=key == "train"),
            batch_size=args.batch_size, shuffle=key == "train", num_workers=args.workers,
            pin_memory=device.type == "cuda", worker_init_fn=seed_worker, generator=generator,
            drop_last=False,
        )
    start_epoch, best_score, bad_epochs = 1, -1.0, 0
    if resume:
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        scaler.load_state_dict(resume["scaler"])
        start_epoch = resume["epoch"] + 1
        best_score, bad_epochs = resume["best_score"], resume["bad_epochs"]
        random.setstate(resume["random_state"])
        torch.set_rng_state(resume["torch_rng"])
        # Сохраняем numpy-состояние примитивами, совместимыми с weights_only=True.
        ns = resume["numpy_rng"]
        np.random.set_state((ns[0], np.asarray(ns[1], dtype=np.uint32), ns[2], ns[3], ns[4]))
        generator.set_state(resume["loader_rng"])
        if device.type == "cuda" and resume["cuda_rng"]:
            if len(resume["cuda_rng"]) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all(resume["cuda_rng"])
        print(f"Продолжение с эпохи {start_epoch}.", flush=True)
    print(f"Параметров: {sum(p.numel() for p in model.parameters()):,}; BCE pos_weight={weight:.3f}", flush=True)
    for epoch in range(start_epoch, args.epochs + 1):
        # Эпоха задаёт независимый и восстанавливаемый порядок/аугментации.
        seed_everything(args.seed + epoch)
        generator.manual_seed(args.seed + epoch)
        begin = time.perf_counter()
        train_loss = train_epoch(model, loaders["train"], optimizer, scaler, device, amp,
                                 pos_weight, args.accumulation, args.clip_grad)
        validation = evaluate(model, loaders["val"], device, amp, pos_weight, [0.5])[0]
        score = validation["dice"]
        scheduler.step(score)
        improved = score > best_score + args.min_delta
        if improved:
            best_score, bad_epochs = score, 0
        else:
            bad_epochs += 1
        ns = np.random.get_state()
        state = {
            "format": FORMAT, "epoch": epoch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(), "base_channels": args.base_channels,
            "seed": args.seed, "split": split, "data_records": report["records"],
            "normalization": NORMALIZATION, "native_shape_hw": list(SHAPE),
            "pos_weight": weight, "threshold": 0.5, "best_score": best_score,
            "bad_epochs": bad_epochs, "validation": validation,
            "random_state": random.getstate(), "torch_rng": torch.get_rng_state(),
            "numpy_rng": [ns[0], ns[1].tolist(), ns[2], ns[3], ns[4]],
            "loader_rng": generator.get_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        }
        save_checkpoint(run_dir / "last.pt", state)
        if improved:
            save_checkpoint(run_dir / "best.pt", state)
            export_test_predictions(model, subsets["val"][:args.preview_count],
                                    run_dir / "val_previews", device, amp, NORMALIZATION, 0.5)
        seconds = time.perf_counter() - begin
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": validation["loss"],
               "val_dice": score, "val_iou": validation["iou"], "val_precision": validation["precision"],
               "val_recall": validation["recall"], "lr": optimizer.param_groups[0]["lr"], "seconds": seconds}
        write_history(run_dir / "history.csv", row)
        print(f"Epoch {epoch:03d}/{args.epochs}: train={train_loss:.5f} val={validation['loss']:.5f} "
              f"Dice={score:.4f} IoU={validation['iou']:.4f} {seconds:.1f}s best={best_score:.4f}", flush=True)
        if bad_epochs >= args.patience:
            print(f"Early stopping: {bad_epochs} эпох без улучшения Dice.", flush=True)
            break
    # Порог выбираем ТОЛЬКО по validation лучшей модели. Test не участвует в выборе.
    best = load_checkpoint(run_dir / "best.pt")
    model.load_state_dict(best["model"])
    threshold_scores = evaluate(model, loaders["val"], device, amp, pos_weight, THRESHOLDS)
    selected = max(threshold_scores, key=lambda row: (row["dice"], -abs(row["threshold"] - 0.5)))
    threshold = selected["threshold"]
    best["threshold"] = threshold
    best["threshold_validation"] = selected
    save_checkpoint(run_dir / "best.pt", best)
    test = evaluate(model, loaders["test"], device, amp, pos_weight, [threshold])[0]
    save_json(run_dir / "metrics.json", {"best_epoch": best["epoch"], "validation": selected,
              "validation_threshold_search": threshold_scores, "test": test,
              "test_images": len(subsets["test"]), "domain": "synthetic_NAOMi",
              "metrics": "micro pixel-level; four-head mean probability; native resolution"})
    export_test_predictions(model, subsets["test"][:args.preview_count], run_dir / "test_previews",
                            device, amp, NORMALIZATION, threshold)
    print(f"ГОТОВО. best.pt: {run_dir / 'best.pt'}", flush=True)
    print(f"Test (NAOMi): Dice={test['dice']:.4f}, IoU={test['iou']:.4f}, "
          f"precision={test['precision']:.4f}, recall={test['recall']:.4f}, threshold={threshold:.2f}", flush=True)


@torch.inference_mode()
def run_predict(args):
    if not args.checkpoint or not args.input:
        raise ValueError("Для predict нужны --checkpoint best.pt и --input файл_или_папка.")
    state = load_checkpoint(args.checkpoint)
    device = choose_device(args.device)
    amp = device.type == "cuda" and not args.no_amp
    model = UNetPlusPlus(state["base_channels"]).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    source = Path(args.input).resolve()
    if source.is_file():
        files = [source]
    else:
        files = sorted(p for p in source.iterdir() if p.suffix.lower() in {".png", ".tif", ".tiff"})
    if not files:
        raise FileNotFoundError(f"Нет изображений в {source}")
    out = Path(args.prediction_output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    threshold = state["threshold"] if args.threshold is None else args.threshold
    stems = set()
    for path in files:
        if path.stem in stems:
            raise ValueError(f"Одинаковые имена без расширения: {path.stem}")
        stems.add(path.stem)
        image = normalize_image(read_plane(path), state["normalization"])
        if min(image.shape) < 16:
            raise ValueError(f"Слишком маленький кадр {image.shape}: {path}")
        tensor = torch.from_numpy(image[None, None]).to(device)
        with autocast_context(device, amp):
            outputs = model(tensor)
        probability = predict_probability(outputs)[0, 0].cpu().numpy()
        mask = probability >= threshold
        Image.fromarray((mask * 255).astype(np.uint8)).save(out / (path.stem + "_mask.png"))
        Image.fromarray((probability * 65535).round().astype(np.uint16)).save(out / (path.stem + "_probability.png"))
        Image.fromarray(overlay(image, mask, (60, 255, 90))).save(out / (path.stem + "_overlay.png"))
        print(f"{path.name}: {image.shape[0]}x{image.shape[1]} -> mask, probability, overlay", flush=True)
    save_json(out / "prediction_info.json", {"checkpoint": str(Path(args.checkpoint).resolve()),
              "threshold": threshold, "normalization": state["normalization"],
              "files": [str(p) for p in files], "output": "semantic neuron/background"})


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("train", "check", "predict"), default="train")
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--image-size", type=int, nargs=2, default=None, metavar=("H", "W"))
    parser.add_argument("--output", default="unetpp_naomi_runs")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0, help="0 безопасно для Windows/Colab; можно поставить 2-4.")
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--resume", default="")
    parser.add_argument("--preview-count", type=int, default=4)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--input", default="")
    parser.add_argument("--prediction-output", default="unetpp_predictions")
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.base_channels, args.accumulation, args.threads, args.patience) < 1:
        parser.error("epochs, batch-size, base-channels, accumulation, threads, patience должны быть >=1.")
    if args.workers < 0 or args.preview_count < 0 or args.lr <= 0 or args.clip_grad <= 0:
        parser.error("Неверные workers, preview-count, lr или clip-grad.")
    if args.min_delta < 0 or args.weight_decay < 0:
        parser.error("min-delta и weight-decay должны быть >=0.")
    if not 0 < args.val_fraction < 1 or not 0 < args.test_fraction < 1 or args.val_fraction + args.test_fraction >= 1:
        parser.error("val/test fractions должны быть >0, а их сумма <1.")
    if args.threshold is not None and not 0 < args.threshold < 1:
        parser.error("threshold должен быть между 0 и 1.")
    return args


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    if args.mode == "check":
        global SHAPE
        pairs, report = discover_pairs(args.data)
        SHAPE = tuple(args.image_size) if args.image_size else tuple(read_plane(pairs[0].image).shape)
        report = audit_pairs(pairs, report)
        split = make_split(pairs, args.seed, args.val_fraction, args.test_fraction)
        validate_split(split, pairs, report["records"])
        out = Path(args.output).resolve()
        out.mkdir(parents=True, exist_ok=True)
        save_json(out / "check_data_audit.json", report)
        print("Train/val/test:", {key: len(value) for key, value in split.items()}, flush=True)
        return
    if args.mode == "predict":
        run_predict(args)
    else:
        run_train(args)


if __name__ == "__main__":
    try:
        main()
    except torch.OutOfMemoryError as exc:
        raise SystemExit("Не хватает памяти. Повторите с --batch-size 1 --base-channels 16; "
                         "--accumulation 4 сохраняет больший эффективный batch без crop/resize.") from exc
