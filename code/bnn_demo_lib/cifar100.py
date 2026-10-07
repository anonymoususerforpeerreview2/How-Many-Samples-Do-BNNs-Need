"""Small CIFAR-100 utilities shared by the paper experiment and its plotter."""

from dataclasses import dataclass
from io import BytesIO
from typing import Dict, Tuple

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms


CIFAR_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR_STD = (0.2675, 0.2565, 0.2761)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


@dataclass(frozen=True)
class Cifar100Result:
    """Metrics and arrays needed to draw the paper's CIFAR figure."""

    summary: Dict[str, float]
    fixed_budgets: np.ndarray
    fixed_agreement: np.ndarray
    fixed_accuracy: np.ndarray
    stop_times: np.ndarray
    reference_margins: np.ndarray


class Cifar100Parquet(Dataset):
    """CIFAR-100 split from the public Hugging Face parquet artifact."""

    def __init__(self, split: str) -> None:
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download

        parquet_path = hf_hub_download(
            "uoft-cs/cifar100",
            f"cifar100/{split}-00000-of-00001.parquet",
            repo_type="dataset",
        )
        table = pq.read_table(parquet_path, columns=["img", "fine_label"])
        self.images = table["img"]
        self.labels = table["fine_label"].to_numpy()
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)]
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        image_bytes = self.images[index].as_py()["bytes"]
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        return self.transform(image), int(self.labels[index])


def load_cifar_resnet(device: torch.device) -> nn.Module:
    """Load the public ResNet-18 checkpoint used in the paper."""

    import timm
    from huggingface_hub import hf_hub_download

    model = timm.create_model("resnet18", num_classes=100)
    model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    checkpoint = hf_hub_download(
        "edadaltocg/resnet18_cifar100",
        "pytorch_model.bin",
    )
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    return model.to(device).eval()


def load_cifar10_resnet(device: torch.device) -> nn.Module:
    """Load the public 94.98%-accuracy CIFAR-10 ResNet-18 checkpoint."""

    import timm
    from huggingface_hub import hf_hub_download

    model = timm.create_model("resnet18", num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    checkpoint = hf_hub_download(
        "edadaltocg/resnet18_cifar10",
        "pytorch_model.bin",
    )
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    return model.to(device).eval()
