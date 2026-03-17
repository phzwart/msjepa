from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from .utils import to_2tuple

SUPPORTED_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".npy",
    ".npz",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, image: Tensor) -> Tensor:
        if torch.rand(1).item() < self.p:
            return torch.flip(image, dims=(-1,))
        return image


class FolderImageDataset(Dataset[Tensor]):
    def __init__(
        self,
        root: str | Path,
        image_size: int | tuple[int, int],
        in_channels: int,
        augmentations: Sequence[Callable[[Tensor], Tensor]] | None = None,
    ) -> None:
        self.root = Path(root)
        self.image_size = to_2tuple(image_size)
        self.in_channels = in_channels
        self.augmentations = list(augmentations or [])
        self.samples = sorted(
            path for path in self.root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        if not self.samples:
            raise ValueError(f"No supported image files were found in {self.root}.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tensor:
        tensor = self._load_sample(self.samples[index])
        for augmentation in self.augmentations:
            tensor = augmentation(tensor)
        return tensor

    def _load_sample(self, path: Path) -> Tensor:
        if path.suffix.lower() == ".npy":
            array = np.load(path)
        elif path.suffix.lower() == ".npz":
            npz_file = np.load(path)
            keys = list(npz_file.keys())
            if not keys:
                raise ValueError(f"No arrays were found in {path}.")
            array = npz_file[keys[0]]
        else:
            with Image.open(path) as image:
                array = np.asarray(image)

        return self._to_tensor(array)

    def _to_tensor(self, array: np.ndarray) -> Tensor:
        if array.ndim == 2:
            array = array[..., None]
        if array.ndim != 3:
            raise ValueError(f"Expected a 2D or 3D image array, got shape {array.shape}.")

        if np.issubdtype(array.dtype, np.integer):
            max_value = np.iinfo(array.dtype).max
            array = array.astype(np.float32) / float(max_value)
        else:
            array = array.astype(np.float32)

        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        tensor = self._match_channels(tensor)
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=self.image_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        return tensor

    def _match_channels(self, tensor: Tensor) -> Tensor:
        channels = tensor.shape[0]
        if channels == self.in_channels:
            return tensor
        if channels == 1 and self.in_channels > 1:
            return tensor.repeat(self.in_channels, 1, 1)
        if channels > self.in_channels:
            return tensor[: self.in_channels]

        repeats = math.ceil(self.in_channels / channels)
        return tensor.repeat(repeats, 1, 1)[: self.in_channels]


def build_augmentations(train: bool) -> list[Callable[[Tensor], Tensor]]:
    return [RandomHorizontalFlip(p=0.5)] if train else []


def build_dataloader(
    root: str | Path,
    image_size: int | tuple[int, int],
    in_channels: int,
    batch_size: int,
    train: bool,
    num_workers: int = 0,
) -> DataLoader[Tensor]:
    dataset = FolderImageDataset(
        root=root,
        image_size=image_size,
        in_channels=in_channels,
        augmentations=build_augmentations(train),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        drop_last=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
