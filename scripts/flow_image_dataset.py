"""
PyTorch Dataset for the flow-to-image grids produced by
7_flow_to_image_encoding.py.

Design (see the project's Notion Work log, Section 13, and README):

    - Grids are stored SMALL on disk (grid_size x grid_size, uint8) rather
      than pre-rendered at ResNet18's 224x224 input resolution -- doing that
      ahead of time would have meant ~2000x more disk space for no benefit
      (confirmed on the actual run: 73.6 MB stored vs. an estimated 168.9 GB
      if pre-rendered). The upscale happens HERE, lazily, once per sample,
      in __getitem__.

    - Upscaling uses NEAREST-NEIGHBOR interpolation only (cv2.INTER_NEAREST),
      never bilinear/bicubic. Smooth interpolation would invent pixel values
      *between* features that have no real spatial relationship to each
      other (unlike a real photo, where neighboring pixels are physically
      related), and would blur the boundaries between grid cells -- which
      matters later for Grad-CAM, where a heatmap region needs to map
      cleanly back to one original feature.

    - The grayscale grid is replicated to 3 channels, since ResNet18 (and
      its ImageNet-pretrained weights) expect RGB input.

    - Pixel values are scaled to [0, 1] and then normalized with ImageNet's
      mean/std. This is standard practice specifically because the model
      uses ImageNet-pretrained weights (transfer learning, per the project
      pitch): those weights were learned under that normalization, so
      matching it -- even though the input here is a feature grid, not a
      photo -- gives the pretrained filters a distribution close to what
      they were trained on.

Usage (from a training script):
    from flow_image_dataset import FlowImageDataset
    train_ds = FlowImageDataset(train_grids, train_labels, cnn_input_size=224)
    loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4)
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# Standard ImageNet normalization statistics -- required for consistency
# with the pretrained ResNet18 weights used in the training script.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


class FlowImageDataset(Dataset):
    """Wraps a (N, grid_size, grid_size) uint8 array of flow-image grids
    and a (N,) array of integer labels. Each sample is upscaled to
    `cnn_input_size` x `cnn_input_size` and normalized on the fly."""

    def __init__(
        self,
        grids: np.ndarray,
        labels: np.ndarray,
        cnn_input_size: int = 224,
    ) -> None:
        if len(grids) != len(labels):
            raise ValueError(
                f"grids and labels must have the same length, got "
                f"{len(grids)} and {len(labels)}"
            )
        self.grids = grids
        self.labels = labels
        self.cnn_input_size = cnn_input_size

    def __len__(self) -> int:
        return len(self.grids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        grid = self.grids[idx]  # (grid_size, grid_size), uint8

        # Nearest-neighbor upscale only -- see module docstring for why.
        upscaled = cv2.resize(
            grid,
            (self.cnn_input_size, self.cnn_input_size),
            interpolation=cv2.INTER_NEAREST,
        )

        img = upscaled.astype(np.float32) / 255.0  # -> [0, 1]
        img = np.repeat(img[np.newaxis, :, :], 3, axis=0)  # grayscale -> 3ch
        img = (img - IMAGENET_MEAN) / IMAGENET_STD

        label = int(self.labels[idx])
        return torch.from_numpy(img), label
