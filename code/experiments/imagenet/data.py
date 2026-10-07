"""Seeded clean, severity-5 ImageNet-C, and final-frame ImageNet-P subsets."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
import random
import tarfile
from typing import Callable

import numpy as np
from PIL import Image
import torch

from .config import ImageNetConfig


@dataclass(frozen=True)
class EvaluationData:
    name: str
    images: torch.Tensor
    labels: torch.Tensor
    source_indices: np.ndarray
    metadata: dict[str, object]


@dataclass(frozen=True)
class ArchiveItem:
    archive: Path
    member: str
    source: str
    wnid: str


def _class_map(cfg: ImageNetConfig) -> dict[str, int]:
    from torchvision.datasets import ImageNet

    return ImageNet(cfg.data_root / "imagenet", split="val").wnid_to_idx


def _processor(cfg: ImageNetConfig):
    from transformers import AutoImageProcessor

    return AutoImageProcessor.from_pretrained(cfg.model_id)


def _encode(
    images: list[Image.Image], cfg: ImageNetConfig, device: torch.device
) -> torch.Tensor:
    processor = _processor(cfg)
    chunks: list[torch.Tensor] = []
    for start in range(0, len(images), 32):
        batch = processor(images=images[start : start + 32], return_tensors="pt")
        chunks.append(batch["pixel_values"])
    return torch.cat(chunks).to(device)


def _reservoir_items(
    archives: list[Path],
    count: int,
    seed: int,
    parser: Callable[[tarfile.TarInfo], tuple[str, str] | None],
) -> tuple[list[ArchiveItem], list[str]]:
    """Retain a bounded seeded reservoir per corruption/perturbation source."""
    rng = random.Random(seed)
    reservoirs: dict[str, list[ArchiveItem]] = defaultdict(list)
    seen: dict[str, int] = defaultdict(int)
    for archive in archives:
        print(f"Indexing {archive}...", flush=True)
        # ImageNet-C's official group archives are gzip-compressed despite
        # their historical ``.tar`` suffix; ImageNet-P is stored as plain tar.
        # Automatic compression detection supports both layouts.
        with tarfile.open(archive, "r:*") as handle:
            for member in handle:
                parsed = parser(member)
                if parsed is None:
                    continue
                source, wnid = parsed
                seen[source] += 1
                item = ArchiveItem(archive, member.name, source, wnid)
                bucket = reservoirs[source]
                if len(bucket) < count:
                    bucket.append(item)
                else:
                    replacement = rng.randrange(seen[source])
                    if replacement < count:
                        bucket[replacement] = item
    sources = sorted(reservoirs)
    if not sources:
        raise RuntimeError(f"no matching samples found in {archives}")
    pairs = np.random.default_rng(seed).permutation(len(sources) * count)[:count]
    selected = [reservoirs[sources[index // count]][index % count] for index in pairs]
    return selected, sources


def _read_archive_images(
    items: list[ArchiveItem], *, videos: bool
) -> list[Image.Image]:
    def decode(payload: bytes, item: ArchiveItem) -> Image.Image:
        if videos:
            import av

            last_frame = None
            with av.open(BytesIO(payload)) as container:
                for frame in container.decode(video=0):
                    last_frame = frame
            if last_frame is None:
                raise ValueError(f"video contains no frames: {item.member}")
            return last_frame.to_image().convert("RGB")
        image = Image.open(BytesIO(payload)).convert("RGB")
        image.load()
        return image

    output: list[Image.Image | None] = [None] * len(items)
    grouped: dict[Path, list[tuple[int, ArchiveItem]]] = defaultdict(list)
    for position, item in enumerate(items):
        grouped[item.archive].append((position, item))
    for archive, values in grouped.items():
        with archive.open("rb") as source:
            compressed = source.read(2) == b"\x1f\x8b"
        if not compressed:
            # Plain ImageNet-P tars are seekable: index their headers once,
            # then jump directly to the selected videos.
            with tarfile.open(archive, "r:") as handle:
                members = {member.name: member for member in handle.getmembers()}
                for position, item in values:
                    member = members.get(item.member)
                    fileobj = handle.extractfile(member) if member is not None else None
                    if fileobj is None:
                        raise FileNotFoundError(f"missing {item.member} in {archive}")
                    output[position] = decode(fileobj.read(), item)
            continue

        # Gzip-compressed ImageNet-C tars are not efficiently seekable, so
        # read each one sequentially instead of repeatedly decompressing it.
        wanted = {item.member: (position, item) for position, item in values}
        remaining = len(wanted)
        with tarfile.open(archive, "r:*") as handle:
            for member in handle:
                match = wanted.get(member.name)
                if match is None:
                    continue
                position, item = match
                fileobj = handle.extractfile(member)
                if fileobj is None:
                    raise FileNotFoundError(f"missing {item.member} in {archive}")
                output[position] = decode(fileobj.read(), item)
                remaining -= 1
                if remaining == 0:
                    break
    if any(image is None for image in output):
        raise RuntimeError("failed to decode one or more archive members")
    return [image for image in output if image is not None]


def clean_subset(
    cfg: ImageNetConfig,
    count: int,
    seed: int,
    device: torch.device,
    *,
    offset: int = 0,
) -> EvaluationData:
    from torchvision.datasets import ImageNet

    dataset = ImageNet(cfg.data_root / "imagenet", split="val")
    indices = np.random.default_rng(seed).permutation(len(dataset))[offset : offset + count]
    images: list[Image.Image] = []
    labels: list[int] = []
    for index in indices:
        image, label = dataset[int(index)]
        images.append(image.convert("RGB"))
        labels.append(label)
    return EvaluationData(
        "clean",
        _encode(images, cfg, device),
        torch.as_tensor(labels, device=device),
        indices,
        {"split": "validation", "selection": "seeded permutation", "offset": offset},
    )


def imagenet_c_subset(
    cfg: ImageNetConfig, count: int, seed: int, device: torch.device
) -> EvaluationData:
    archives = sorted((cfg.data_root / "imagenet" / "downloads").glob("*.tar"))
    if len(archives) < 4:
        raise FileNotFoundError("expected the four official ImageNet-C group archives")

    def parse(member: tarfile.TarInfo) -> tuple[str, str] | None:
        parts = PurePosixPath(member.name).parts
        if member.isfile() and len(parts) == 4 and parts[1] == "5":
            return parts[0], parts[2]
        return None

    items, sources = _reservoir_items(archives, count, seed + 100, parse)
    mapping = _class_map(cfg)
    images = _read_archive_images(items, videos=False)
    labels = [mapping[item.wnid] for item in items]
    return EvaluationData(
        "imagenet_c",
        _encode(images, cfg, device),
        torch.as_tensor(labels, device=device),
        np.asarray([item.member for item in items]),
        {
            "severity": 5,
            "corruptions": sources,
            "selection": "seeded sample across corruption types",
            "storage": "official group tar archives",
        },
    )


def imagenet_p_subset(
    cfg: ImageNetConfig, count: int, seed: int, device: torch.device
) -> EvaluationData:
    archives = sorted((cfg.data_root / "ImageNet-P" / "downloads").glob("*.tar"))
    if len(archives) < 4:
        raise FileNotFoundError("expected the four official ImageNet-P group archives")

    def parse(member: tarfile.TarInfo) -> tuple[str, str] | None:
        parts = PurePosixPath(member.name).parts
        if member.isfile() and len(parts) == 3 and parts[-1].lower().endswith(".mp4"):
            return parts[0], parts[1]
        return None

    items, sources = _reservoir_items(archives, count, seed + 200, parse)
    mapping = _class_map(cfg)
    images = _read_archive_images(items, videos=True)
    labels = [mapping[item.wnid] for item in items]
    return EvaluationData(
        "imagenet_p",
        _encode(images, cfg, device),
        torch.as_tensor(labels, device=device),
        np.asarray([item.member for item in items]),
        {
            "frame": "last decoded video frame",
            "perturbations": sources,
            "selection": "seeded sample across perturbation types",
            "storage": "official group tar archives",
        },
    )


def evaluation_dataset(
    name: str,
    cfg: ImageNetConfig,
    count: int,
    seed: int,
    device: torch.device,
) -> EvaluationData:
    if name == "clean":
        return clean_subset(cfg, count, seed, device)
    if name == "imagenet_c":
        return imagenet_c_subset(cfg, count, seed, device)
    if name == "imagenet_p":
        return imagenet_p_subset(cfg, count, seed, device)
    raise ValueError(f"unknown evaluation dataset: {name}")
