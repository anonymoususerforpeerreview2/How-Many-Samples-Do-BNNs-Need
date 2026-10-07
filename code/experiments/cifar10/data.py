"""Clean and shifted CIFAR-10 evaluation datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from bnn_demo_lib.cifar100 import CIFAR10_MEAN, CIFAR10_STD


@dataclass(frozen=True)
class EvaluationData:
    name: str
    images: torch.Tensor
    labels: torch.Tensor
    source_indices: np.ndarray
    metadata: dict[str, object]


def _normalize(images: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(np.asarray(images).copy()).permute(0, 3, 1, 2).float() / 255.0
    mean = torch.tensor(CIFAR10_MEAN, dtype=tensor.dtype)[None, :, None, None]
    std = torch.tensor(CIFAR10_STD, dtype=tensor.dtype)[None, :, None, None]
    return ((tensor - mean) / std).to(device)


def _raw_split(data_root: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    root = data_root / "cifar-10-batches-py"
    names = ["test_batch"] if split == "test" else [f"data_batch_{i}" for i in range(1, 6)]
    image_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    for name in names:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"missing local CIFAR-10 split: {path}")
        with path.open("rb") as handle:
            payload = pickle.load(handle, encoding="latin1")
        image_parts.append(np.asarray(payload["data"], dtype=np.uint8))
        label_parts.append(np.asarray(payload["labels"], dtype=np.int64))
    images = np.concatenate(image_parts).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return images, np.concatenate(label_parts)


def clean_subset(
    data_root: Path,
    count: int,
    seed: int,
    device: torch.device,
    *,
    offset: int = 0,
) -> EvaluationData:
    images, labels = _raw_split(data_root, "test")
    indices = np.random.default_rng(seed).permutation(len(images))[offset : offset + count]
    return EvaluationData(
        "clean",
        _normalize(images[indices], device),
        torch.as_tensor(labels[indices], device=device),
        indices,
        {"split": "test", "selection": "seeded permutation", "offset": offset},
    )


class Cifar10LocalDataset(Dataset):
    def __init__(self, data_root: Path, split: str = "train") -> None:
        self.images, self.labels = _raw_split(data_root, split)
        self.mean = torch.tensor(CIFAR10_MEAN)[:, None, None]
        self.std = torch.tensor(CIFAR10_STD)[:, None, None]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = torch.from_numpy(self.images[index].copy()).permute(2, 0, 1).float() / 255.0
        return (image - self.mean) / self.std, int(self.labels[index])


def _benchmark_arrays(root: Path) -> list[Path]:
    direct = sorted(path for path in root.glob("*.npy") if path.name != "labels.npy")
    return direct or sorted(path for path in root.rglob("*.npy") if path.name != "labels.npy")


def _balanced_pairs(num_sources: int, per_source: int, count: int, seed: int) -> np.ndarray:
    selected = np.random.default_rng(seed).permutation(num_sources * per_source)[:count]
    return np.column_stack((selected // per_source, selected % per_source))


def cifar10_c_subset(
    data_root: Path, count: int, seed: int, device: torch.device
) -> EvaluationData:
    root = data_root / "CIFAR-10-C"
    files = _benchmark_arrays(root)
    if not files:
        raise FileNotFoundError(f"no CIFAR-10-C arrays found in {root}")
    pairs = _balanced_pairs(len(files), 10_000, count, seed + 100)
    images = np.empty((count, 32, 32, 3), dtype=np.uint8)
    labels_all = np.load(root / "labels.npy", mmap_mode="r")
    labels = np.empty(count, dtype=np.int64)
    for source_index in np.unique(pairs[:, 0]):
        positions = np.flatnonzero(pairs[:, 0] == source_index)
        base_indices = pairs[positions, 1]
        array = np.load(files[int(source_index)], mmap_mode="r")
        images[positions] = array[40_000 + base_indices]
        label_offset = 40_000 if len(labels_all) >= 50_000 else 0
        labels[positions] = labels_all[label_offset + base_indices]
    return EvaluationData(
        "cifar10_c",
        _normalize(images, device),
        torch.as_tensor(labels, device=device),
        pairs,
        {
            "severity": 5,
            "corruptions": [path.stem for path in files],
            "selection": "seeded sample across corruption types",
        },
    )


def _last_frame(array: np.ndarray, indices: np.ndarray) -> np.ndarray:
    if array.ndim == 5:
        if array.shape[0] == 10_000:
            return np.asarray(array[indices, -1])
        if array.shape[1] == 10_000:
            return np.asarray(array[-1, indices])
    if array.ndim == 4 and array.shape[0] > 10_000 and array.shape[0] % 10_000 == 0:
        return np.asarray(array.reshape(10_000, -1, *array.shape[1:])[indices, -1])
    raise ValueError(f"unsupported CIFAR-10-P array shape: {array.shape}")


def cifar10_p_subset(
    data_root: Path, count: int, seed: int, device: torch.device
) -> EvaluationData:
    root = data_root / "CIFAR-10-P"
    files = _benchmark_arrays(root)
    if not files:
        raise FileNotFoundError(f"no CIFAR-10-P arrays found in {root}")
    pairs = _balanced_pairs(len(files), 10_000, count, seed + 200)
    images = np.empty((count, 32, 32, 3), dtype=np.uint8)
    _, clean_labels = _raw_split(data_root, "test")
    labels = np.empty(count, dtype=np.int64)
    for source_index in np.unique(pairs[:, 0]):
        positions = np.flatnonzero(pairs[:, 0] == source_index)
        base_indices = pairs[positions, 1]
        images[positions] = _last_frame(
            np.load(files[int(source_index)], mmap_mode="r"), base_indices
        )
        labels[positions] = clean_labels[base_indices]
    return EvaluationData(
        "cifar10_p",
        _normalize(images, device),
        torch.as_tensor(labels, device=device),
        pairs,
        {
            "frame": "last (maximum sequence displacement)",
            "perturbations": [path.stem for path in files],
            "selection": "seeded sample across perturbation types",
        },
    )


def evaluation_dataset(
    name: str, data_root: Path, count: int, seed: int, device: torch.device
) -> EvaluationData:
    if name == "clean":
        return clean_subset(data_root, count, seed, device)
    if name == "cifar10_c":
        return cifar10_c_subset(data_root, count, seed, device)
    if name == "cifar10_p":
        return cifar10_p_subset(data_root, count, seed, device)
    raise ValueError(f"unknown evaluation dataset: {name}")
