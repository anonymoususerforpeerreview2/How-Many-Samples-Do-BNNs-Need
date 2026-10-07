"""Hugging Face ViT with dropout-only stochastic evaluation."""

from __future__ import annotations

from contextlib import nullcontext
from time import perf_counter
from typing import Iterator

import torch
from torch import nn

from .config import ImageNetConfig


class LogitsModel(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=images).logits


def load_model(
    cfg: ImageNetConfig,
    device: torch.device,
    *,
    data_parallel: bool = False,
) -> nn.Module:
    from transformers import AutoModelForImageClassification

    raw = AutoModelForImageClassification.from_pretrained(cfg.model_id)
    model: nn.Module = LogitsModel(raw).to(device).eval()
    if data_parallel and device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    return model


def configure_mc_dropout(model: nn.Module, probability: float) -> int:
    """Keep the network in eval mode while activating only dropout modules."""
    model.eval()
    count = 0
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = probability
            module.train()
            count += 1
    if count == 0:
        raise RuntimeError("the selected ViT exposes no torch.nn.Dropout modules")
    return count


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _chunks(total: int, size: int) -> Iterator[tuple[int, int]]:
    for start in range(0, total, size):
        yield start, min(size, total - start)


@torch.inference_mode()
def predictive_stream(
    cfg: ImageNetConfig,
    model: nn.Module,
    images: torch.Tensor,
    num_samples: int,
    seed: int,
    *,
    keep_samples: bool,
) -> tuple[torch.Tensor | None, torch.Tensor, float]:
    device = images.device
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    samples = (
        torch.empty(num_samples, len(images), 1_000, dtype=torch.float32)
        if keep_samples
        else None
    )
    running_sum = torch.zeros(len(images), 1_000, device=device)
    sync(device)
    start_time = perf_counter()
    amp = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if cfg.mixed_precision and device.type == "cuda"
        else nullcontext()
    )
    with amp:
        for sample_start, sample_count in _chunks(num_samples, cfg.mc_sample_chunk):
            for batch_start in range(0, len(images), cfg.inference_batch_size):
                batch = images[batch_start : batch_start + cfg.inference_batch_size]
                expanded = (
                    batch.unsqueeze(0)
                    .expand(sample_count, *batch.shape)
                    .reshape(sample_count * len(batch), *batch.shape[1:])
                )
                probabilities = model(expanded).softmax(dim=1).float()
                probabilities = probabilities.reshape(sample_count, len(batch), 1_000)
                batch_end = batch_start + len(batch)
                running_sum[batch_start:batch_end] += probabilities.sum(dim=0)
                if samples is not None:
                    samples[
                        sample_start : sample_start + sample_count,
                        batch_start:batch_end,
                    ] = probabilities.cpu()
            completed = sample_start + sample_count
            if completed % 20 == 0 or completed == num_samples:
                print(f"  MC-dropout predictions: {completed}/{num_samples}", flush=True)
    sync(device)
    return samples, (running_sum / num_samples).cpu(), perf_counter() - start_time
