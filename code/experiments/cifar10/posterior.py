"""CIFAR-10 all-parameter diagonal-GGN Laplace posterior."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
import types
from typing import Any

import torch
from backpack import extend
from laplace import Laplace
from laplace.curvature import BackPackGGN
from torch.nn.utils import vector_to_parameters
from torch.utils.data import DataLoader, Subset

from bnn_demo_lib.cifar100 import load_cifar10_resnet
from experiments.cifar100.io import cpu_state_dict

from .config import Cifar10Config
from .data import Cifar10LocalDataset, EvaluationData


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_model(device: torch.device) -> torch.nn.Module:
    model = load_cifar10_resnet(device)

    def basic_block_forward(block: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = block.conv1(x)
        x = block.bn1(x)
        x = block.drop_block(x)
        x = block.act1(x)
        x = block.aa(x)
        x = block.conv2(x)
        x = block.bn2(x)
        if block.se is not None:
            x = block.se(x)
        if block.drop_path is not None:
            x = block.drop_path(x)
        if block.downsample is not None:
            shortcut = block.downsample(shortcut)
        return block.act2(x + shortcut)

    for module in model.modules():
        if isinstance(module, torch.nn.ReLU):
            module.inplace = False
        if module.__class__.__name__ == "BasicBlock" and all(
            hasattr(module, name)
            for name in ("conv1", "bn1", "conv2", "bn2", "act2")
        ):
            module.forward = types.MethodType(basic_block_forward, module)
    return extend(model, use_converter=True)


def new_laplace(model: torch.nn.Module) -> Laplace:
    return Laplace(
        model,
        "classification",
        subset_of_weights="all",
        hessian_structure="diag",
        backend=BackPackGGN,
    )


def load_laplace_cache(
    model: torch.nn.Module, path: Path, device: torch.device
) -> tuple[Laplace, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if "laplace_state" not in payload:
        raise ValueError(f"not a full-network Laplace cache: {path}")
    approximation = new_laplace(model)
    approximation.load_state_dict(payload["laplace_state"])
    return approximation, payload


def fit_or_load_curvature(
    cfg: Cifar10Config, model: torch.nn.Module, device: torch.device
) -> tuple[Laplace, dict[str, Any]]:
    source = cfg.posterior_cache if cfg.posterior_cache.exists() else cfg.source_cache
    if source is not None and source.exists():
        print(f"Loading curvature cache: {source}", flush=True)
        return load_laplace_cache(model, source, device)
    approximation = new_laplace(model)
    dataset = Cifar10LocalDataset(cfg.data_root, "train")
    if cfg.fit_max_images:
        dataset = Subset(dataset, range(min(cfg.fit_max_images, len(dataset))))
    loader = DataLoader(
        dataset,
        batch_size=cfg.fit_batch_size,
        shuffle=False,
        num_workers=cfg.fit_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=cfg.fit_workers > 0,
    )
    print(f"Fitting diagonal GGN on {len(dataset)} training images...", flush=True)
    sync(device)
    start = perf_counter()
    approximation.fit(loader)
    sync(device)
    return approximation, {
        "fit_timing": {"curvature_seconds": perf_counter() - start},
        "fit_images": len(dataset),
        "torch_version": torch.__version__,
    }


@torch.inference_mode()
def predictive_stream(
    model: torch.nn.Module,
    approximation: Laplace,
    images: torch.Tensor,
    num_samples: int,
    batch_size: int,
    seed: int,
    *,
    keep_samples: bool,
) -> tuple[torch.Tensor | None, torch.Tensor, float]:
    device = images.device
    generator = torch.Generator(device=device).manual_seed(seed)
    samples = (
        torch.empty(num_samples, len(images), 10, dtype=torch.float32)
        if keep_samples
        else None
    )
    running_sum = torch.zeros(len(images), 10, device=device)
    sync(device)
    start = perf_counter()
    try:
        for sample_index in range(num_samples):
            parameter_sample = approximation.sample(1, generator=generator)[0]
            vector_to_parameters(parameter_sample, approximation.params)
            for batch_start in range(0, len(images), batch_size):
                probabilities = model(images[batch_start : batch_start + batch_size]).softmax(dim=1)
                batch_end = batch_start + len(probabilities)
                running_sum[batch_start:batch_end] += probabilities
                if samples is not None:
                    samples[sample_index, batch_start:batch_end] = probabilities.cpu()
            if (sample_index + 1) % 25 == 0 or sample_index + 1 == num_samples:
                print(f"  posterior predictions: {sample_index + 1}/{num_samples}", flush=True)
    finally:
        vector_to_parameters(approximation.mean, approximation.params)
    sync(device)
    return samples, (running_sum / num_samples).cpu(), perf_counter() - start


def calibrate_prior(
    cfg: Cifar10Config,
    model: torch.nn.Module,
    approximation: Laplace,
    calibration: EvaluationData,
) -> dict[str, Any]:
    rows = []
    label_indices = torch.arange(len(calibration.labels))
    for candidate in cfg.prior_candidates:
        approximation.prior_precision = torch.as_tensor(candidate, device=calibration.images.device)
        _, mean, seconds = predictive_stream(
            model,
            approximation,
            calibration.images,
            cfg.calibration_samples,
            cfg.inference_batch_size,
            cfg.seed + 5,
            keep_samples=False,
        )
        probabilities = mean[label_indices, calibration.labels.cpu()].clamp_min(1e-12)
        row = {
            "prior_precision": candidate,
            "nll": float(-probabilities.log().mean()),
            "accuracy": float((mean.argmax(dim=1) == calibration.labels.cpu()).float().mean()),
            "mean_max_probability": float(mean.max(dim=1).values.mean()),
            "seconds": seconds,
        }
        rows.append(row)
        print(f"Prior candidate: {row}", flush=True)
    selected = min(rows, key=lambda row: row["nll"])
    approximation.prior_precision = torch.as_tensor(
        selected["prior_precision"], device=calibration.images.device
    )
    return {"candidates": rows, "selected": selected}


def cache_payload(
    approximation: Laplace,
    inherited: dict[str, Any],
    prior: dict[str, Any],
) -> dict[str, Any]:
    return {
        "laplace_state": cpu_state_dict(approximation.state_dict()),
        "fit_timing": inherited.get("fit_timing", {}),
        "fit_images": inherited.get("fit_images"),
        "torch_version": inherited.get("torch_version", torch.__version__),
        "prior_calibration": prior,
    }
