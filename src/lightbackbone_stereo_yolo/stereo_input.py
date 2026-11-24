"""Rectified stereo input geometry and one-time CPU preprocessing."""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from PIL import Image
import torch


@dataclass(frozen=True)
class StereoInputGeometry:
    original_hw: tuple[int, int]
    crop_top_px: int = 0

    def __post_init__(self):
        height, width = self.original_hw
        if type(height) is not int or type(width) is not int or min(height, width) < 1:
            raise ValueError(f"expected positive integer image dimensions, actual {self.original_hw}")
        if type(self.crop_top_px) is not int or not 0 <= self.crop_top_px < height:
            raise ValueError(f"expected integer crop_top_px in [0, {height}), actual {self.crop_top_px}")

    @property
    def cropped_hw(self):
        return (self.original_hw[0] - self.crop_top_px, self.original_hw[1])

    @property
    def cropped_nchw(self):
        return (1, 3, *self.cropped_hw)

    @property
    def padded_hw(self):
        return tuple(size + (-size) % 32 for size in self.cropped_hw)

    def validate_prepared(self, left, right=None):
        actual = (tuple(left.shape), tuple(right.shape)) if right is not None else (tuple(left.shape),)
        if any(shape != self.cropped_nchw for shape in actual):
            raise ValueError(f"expected prepared NCHW {self.cropped_nchw} for each view, actual {actual}")

    def as_dict(self):
        return {"original_hw": self.original_hw, "cropped_hw": self.cropped_hw,
                "crop_top_px": self.crop_top_px, "encoder_padded_hw": self.padded_hw}


def normalize_rgb(image, mean, std):
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean_tensor = tensor.new_tensor(mean)[:, None, None]
    std_tensor = tensor.new_tensor(std)[:, None, None]
    return (tensor - mean_tensor) / std_tensor


def prepare_rectified_images(left, right, normalization, crop_top_px=0, *, timings=None):
    """Crop paired pixel rows before RGB conversion and normalization."""
    if left.size != right.size:
        raise ValueError(f"expected equal rectified image sizes, actual left {left.size}, right {right.size}")
    width, height = left.size
    geometry = StereoInputGeometry((height, width), crop_top_px)
    start = perf_counter()
    if crop_top_px:
        bounds = (0, crop_top_px, width, height)
        left, right = left.crop(bounds), right.crop(bounds)
    cropped = perf_counter()
    l, r = (normalize_rgb(image, normalization["mean"], normalization["std"])[None] for image in (left, right))
    if timings is not None:
        timings.update(crop_ms=(cropped - start) * 1000, normalization_ms=(perf_counter() - cropped) * 1000)
    return l, r, geometry


def load_rectified_pair(left_path, right_path, normalization, crop_top_px=0, *, timings=None):
    start = perf_counter()
    with Image.open(left_path) as left, Image.open(right_path) as right:
        left.load()
        right.load()
        if timings is not None:
            timings["read_ms"] = (perf_counter() - start) * 1000
        return prepare_rectified_images(left, right, normalization, crop_top_px, timings=timings)
