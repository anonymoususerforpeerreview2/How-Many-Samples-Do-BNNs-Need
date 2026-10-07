"""Download the official CIFAR-100-P artifact when it is absent."""

from __future__ import annotations

from pathlib import Path
import shutil

import gdown

from .config import Cifar100Config


CIFAR100_P_FOLDER = "https://drive.google.com/drive/folders/1Tpssw4Vn6X_4hmIW8KK5LzkKCZpAaae8"


def _arrays(root: Path) -> list[Path]:
    direct = sorted(path for path in root.glob("*.npy") if path.name != "labels.npy")
    if direct:
        return direct
    return sorted(path for path in root.rglob("*.npy") if path.name != "labels.npy")


def main() -> None:
    cfg = Cifar100Config.from_environment()
    root = cfg.data_root / "CIFAR-100-P"
    existing = _arrays(root) if root.exists() else []
    if existing:
        print(f"CIFAR-100-P already contains {len(existing)} arrays in {root}")
        return

    root.mkdir(parents=True, exist_ok=True)
    print(f"Downloading official CIFAR-100-P folder to {root}", flush=True)
    gdown.download_folder(
        url=CIFAR100_P_FOLDER,
        output=str(root),
        quiet=False,
        use_cookies=False,
        remaining_ok=True,
    )
    for archive in root.rglob("*"):
        if archive.is_file() and any(
            str(archive).lower().endswith(suffix)
            for suffix in (".zip", ".tar", ".tar.gz", ".tgz")
        ):
            destination = archive.parent / archive.name.split(".")[0]
            destination.mkdir(exist_ok=True)
            print(f"Extracting {archive} to {destination}", flush=True)
            shutil.unpack_archive(archive, destination)
    arrays = _arrays(root)
    if not arrays:
        raise RuntimeError(f"download completed but no .npy arrays were found under {root}")
    print(f"Prepared {len(arrays)} CIFAR-100-P arrays", flush=True)


if __name__ == "__main__":
    main()
