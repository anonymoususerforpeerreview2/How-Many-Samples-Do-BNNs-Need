"""Download and unpack the public CIFAR data used by the paper."""

from __future__ import annotations

import hashlib
from pathlib import Path
import tarfile
import urllib.request

from experiments.cifar10.prepare_data import main as prepare_cifar10_p
from experiments.cifar100.prepare_data import main as prepare_cifar100_p


ROOT = Path(__file__).resolve().parents[1] / "data" / "datasets"
ARCHIVES = (
    ("cifar-10-python.tar.gz", "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
     "cifar-10-batches-py", "c58f30108f718f92721af3b95e74349a"),
    ("cifar-100-python.tar.gz", "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
     "cifar-100-python", "eb9058c3a382ffc7106e4002c42a8d85"),
    ("CIFAR-10-C.tar", "https://zenodo.org/records/2535967/files/CIFAR-10-C.tar",
     "CIFAR-10-C", None),
    ("CIFAR-100-C.tar", "https://zenodo.org/records/3555552/files/CIFAR-100-C.tar",
     "CIFAR-100-C", "11f0ed0f1191edbf9fa23466ae6021d3"),
)


def _download(url: str, path: Path) -> None:
    partial = path.with_name(path.name + ".partial")
    print(f"Downloading {url} to {path}", flush=True)
    with urllib.request.urlopen(url) as response, partial.open("wb") as target:
        while block := response.read(8 * 1024 * 1024):
            target.write(block)
    partial.replace(path)


def _verify(path: Path, expected: str | None) -> None:
    if expected is None:
        return
    digest = hashlib.md5()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise RuntimeError(f"checksum mismatch: {path}")


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    for name, url, folder, checksum in ARCHIVES:
        destination = ROOT / folder
        if destination.is_dir() and any(destination.iterdir()):
            print(f"Using {destination}", flush=True)
            continue
        archive = ROOT / name
        if not archive.is_file():
            _download(url, archive)
        _verify(archive, checksum)
        with tarfile.open(archive) as source:
            for member in source.getmembers():
                target = (ROOT / member.name).resolve()
                if not target.is_relative_to(ROOT.resolve()):
                    raise ValueError(f"unsafe archive member: {member.name}")
            source.extractall(ROOT)
        if not destination.is_dir():
            raise FileNotFoundError(f"archive did not create {destination}")
        print(f"Prepared {destination}", flush=True)
    prepare_cifar10_p()
    prepare_cifar100_p()


if __name__ == "__main__":
    main()
