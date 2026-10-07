"""Download the official CIFAR-10-P artifact when absent."""

from __future__ import annotations

import shutil

import gdown

from .config import Cifar10Config


CIFAR10_P_FOLDER = "https://drive.google.com/drive/folders/1dY1_zeRyjMKdfmMbQ_uK8O1i0MVI9UbZ"


def main() -> None:
    cfg = Cifar10Config.from_environment()
    root = cfg.data_root / "CIFAR-10-P"
    existing = sorted(root.rglob("*.npy")) if root.exists() else []
    if existing:
        print(f"CIFAR-10-P already contains {len(existing)} arrays in {root}")
        return
    root.mkdir(parents=True, exist_ok=True)
    print(f"Downloading official CIFAR-10-P folder to {root}", flush=True)
    gdown.download_folder(url=CIFAR10_P_FOLDER, output=str(root), quiet=False)
    nested = root / "CIFAR-10-P"
    if nested.is_dir():
        for path in nested.iterdir():
            destination = root / path.name
            if not destination.exists():
                shutil.move(str(path), destination)
        nested.rmdir()
    arrays = sorted(root.rglob("*.npy"))
    if not arrays:
        raise RuntimeError(f"CIFAR-10-P download produced no arrays in {root}")
    print(f"Prepared {len(arrays)} CIFAR-10-P arrays", flush=True)


if __name__ == "__main__":
    main()
