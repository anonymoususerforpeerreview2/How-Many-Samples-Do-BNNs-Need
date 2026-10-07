"""Stage 1: fit/load the full-network diagonal GGN and calibrate its prior."""

from __future__ import annotations

import json

import numpy as np
import torch

from .config import Cifar100Config
from .data import clean_subset
from .io import save_json
from .posterior import cache_payload, calibrate_prior, fit_or_load_curvature, load_model


def main() -> None:
    cfg = Cifar100Config.from_environment()
    cfg.fit_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    model = load_model(device)
    approximation, inherited = fit_or_load_curvature(cfg, model, device)
    calibration = clean_subset(
        cfg.data_root,
        cfg.calibration_size,
        cfg.seed,
        device,
        offset=cfg.num_inputs,
    )
    prior = calibrate_prior(cfg, model, approximation, calibration)
    torch.save(
        cache_payload(approximation, inherited, prior),
        cfg.posterior_cache,
    )
    summary = {
        "stage": "fit_laplace",
        "config": cfg.as_json(),
        "posterior": {
            "subset_of_weights": "all",
            "hessian_structure": "diag",
            "curvature": "GGN",
            "prior_precision": prior["selected"]["prior_precision"],
        },
        "fit_timing": inherited.get("fit_timing", {}),
        "prior_calibration": prior,
    }
    save_json(cfg.fit_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
