from io import BytesIO
from pathlib import Path
import tarfile

from PIL import Image
import torch

from experiments.imagenet.config import ImageNetConfig
from experiments.imagenet.data import ArchiveItem, _read_archive_images
from experiments.imagenet.model import configure_mc_dropout


def test_imagenet_matches_sampling_protocol() -> None:
    cfg = ImageNetConfig(run_dir=Path("run"), data_root=Path("data"))
    assert cfg.datasets == ("clean", "imagenet_c", "imagenet_p")
    assert cfg.dropout_probability == 0.10
    assert cfg.reference_samples == 1_000
    assert cfg.evaluation_samples == 500


def test_only_dropout_is_reactivated() -> None:
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 2), torch.nn.Dropout(0.0), torch.nn.BatchNorm1d(2)
    )
    count = configure_mc_dropout(model, 0.1)
    assert count == 1
    assert model[1].training and model[1].p == 0.1
    assert not model[0].training
    assert not model[2].training


def test_gzip_tar_with_tar_suffix_is_decoded(tmp_path: Path) -> None:
    payload = BytesIO()
    Image.new("RGB", (2, 2), (12, 34, 56)).save(payload, format="JPEG")
    archive = tmp_path / "blur.tar"
    with tarfile.open(archive, "w:gz") as handle:
        info = tarfile.TarInfo("defocus_blur/5/n00000001/example.JPEG")
        info.size = len(payload.getvalue())
        handle.addfile(info, BytesIO(payload.getvalue()))
    images = _read_archive_images(
        [ArchiveItem(archive, info.name, "defocus_blur", "n00000001")],
        videos=False,
    )
    assert len(images) == 1
    assert images[0].size == (2, 2)
