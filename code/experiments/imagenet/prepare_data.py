"""Verify ImageNet/ImageNet-C and fetch the official ImageNet-P archives."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import urllib.request

from .config import ImageNetConfig


ZENODO_RECORD = "https://zenodo.org/api/records/3565846"
EXPECTED_MD5 = {
    "blur.tar": "869b2d2fb1604d1baa4316b5ecc9fdea",
    "digital.tar": "8c1f73e4912812788e2fdb637cd55372",
    "noise.tar": "619b37b5139ce764f77ce3bed3aee837",
    "weather.tar": "a597ff6db8c6fb0dbcd6eb12ed620002",
}


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> None:
    partial = destination.with_suffix(destination.suffix + ".partial")
    start = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url)
    if start:
        request.add_header("Range", f"bytes={start}-")
    mode = "ab" if start else "wb"
    with urllib.request.urlopen(request) as response, partial.open(mode) as output:
        while True:
            block = response.read(8 * 1024 * 1024)
            if not block:
                break
            output.write(block)
    partial.replace(destination)


def main() -> None:
    cfg = ImageNetConfig.from_environment()
    imagenet = cfg.data_root / "imagenet"
    required = (
        imagenet / "ILSVRC2012_img_val.tar",
        imagenet / "ILSVRC2012_devkit_t12.tar.gz",
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"missing licensed ImageNet artifact: {path}")
    c_archives = sorted((imagenet / "downloads").glob("*.tar"))
    if len(c_archives) < 4:
        raise FileNotFoundError("the four ImageNet-C group archives are required")
    destination = cfg.data_root / "ImageNet-P" / "downloads"
    destination.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(ZENODO_RECORD) as response:
        record = json.load(response)
    urls = {item["key"]: item["links"]["self"] for item in record["files"]}
    for name, expected in EXPECTED_MD5.items():
        path = destination / name
        if not path.is_file():
            print(f"Downloading {name} to {path}", flush=True)
            _download(urls[name], path)
        actual = _md5(path)
        if actual != expected:
            raise RuntimeError(f"checksum mismatch for {path}: {actual} != {expected}")
        print(f"Verified {name} ({path.stat().st_size} bytes)", flush=True)


if __name__ == "__main__":
    main()
