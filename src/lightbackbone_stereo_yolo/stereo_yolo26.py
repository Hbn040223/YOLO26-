"""YOLO26-inspired stereo detector blocks that reuse LightStereo features."""
from __future__ import annotations

from copy import deepcopy

import torch
from torch import Tensor, nn


class StereoFusion(nn.Module):
    """Fuse one already-extracted left/right feature pair without an encoder call."""

    def __init__(self, channels: int = 96) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive")
        self.project = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
        )

    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        if left.shape != right.shape:
            raise ValueError("left and right stereo features must have identical shape")
        return self.project(torch.cat((left, right), dim=1))


class SPPFLite(nn.Module):
    """Shape-preserving deepest-level context block."""

    def __init__(self, channels: int = 96) -> None:
        super().__init__()
        if channels < 2 or channels % 2:
            raise ValueError("SPPFLite requires a positive even channel count")
        hidden = channels // 2
        self.reduce = nn.Sequential(nn.Conv2d(channels, hidden, kernel_size=1), nn.SiLU(inplace=True))
        self.pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.expand = nn.Sequential(nn.Conv2d(hidden * 4, channels, kernel_size=1), nn.SiLU(inplace=True))

    def forward(self, feature: Tensor) -> Tensor:
        first = self.reduce(feature)
        second = self.pool(first)
        third = self.pool(second)
        fourth = self.pool(third)
        return self.expand(torch.cat((first, second, third, fourth), dim=1))



class _PanBlock(nn.Module):
    """Fuse one detector level with the next finer level after stride-2 downsampling."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(channels * 2, channels * 2, kernel_size=3, padding=1, groups=channels * 2)
        self.pointwise = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, current: Tensor, lower: Tensor) -> Tensor:
        if current.shape != lower.shape:
            raise ValueError("PAN inputs must have identical shape after downsampling")
        return self.activation(self.pointwise(self.activation(self.depthwise(torch.cat((current, lower), dim=1)))))


class StereoYOLO26PanNeck(nn.Module):
    """Independent stereo fusion followed by a bottom-up-only PAN.

    The LightStereo encoder already supplies the top-down FPN. This module adds
    only N3 -> N4 -> N5 localization feedback and never consumes P2.
    """

    def __init__(self, channels: int = 96) -> None:
        super().__init__()
        self.fusion_p3 = StereoFusion(channels)
        self.fusion_p4 = StereoFusion(channels)
        self.fusion_c5 = StereoFusion(channels)
        self.sppf = SPPFLite(channels)
        self.down_n3 = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
        self.down_n4 = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
        self.pan_n4 = _PanBlock(channels)
        self.pan_n5 = _PanBlock(channels)

    def forward(self, pyramid: object) -> tuple[Tensor, Tensor, Tensor]:
        f3 = self.fusion_p3(pyramid.p3_left, pyramid.p3_right)
        f4 = self.fusion_p4(pyramid.p4_left, pyramid.p4_right)
        f5 = self.sppf(self.fusion_c5(pyramid.c5_left, pyramid.c5_right))
        n3 = f3
        n4 = self.pan_n4(f4, self.down_n3(n3))
        n5 = self.pan_n5(f5, self.down_n4(n4))
        return n3, n4, n5


from .detection import DenseStereoPredictions


class _DepthwiseSeparable(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, feature: Tensor) -> Tensor:
        return self.activation(self.pointwise(self.activation(self.depthwise(feature))))


class _DecoupledPredictionLevel(nn.Module):
    def __init__(self, channels: int, num_classes: int) -> None:
        super().__init__()
        self.stem = _DepthwiseSeparable(channels)
        self.objectness = nn.Sequential(_DepthwiseSeparable(channels), nn.Conv2d(channels, 1, kernel_size=1))
        self.classes = nn.Sequential(_DepthwiseSeparable(channels), nn.Conv2d(channels, num_classes, kernel_size=1))
        self.boxes = nn.Sequential(_DepthwiseSeparable(channels), nn.Conv2d(channels, 6, kernel_size=1))

    def forward(self, feature: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        stem = self.stem(feature)
        return self.objectness(stem), self.classes(stem), self.boxes(stem)


class StereoYOLO26PanHead(nn.Module):
    """PAN-only stereo neck plus independent P3/P4/P5 direct prediction branches."""

    def __init__(self, channels: int, num_classes: int, *, one_to_one: bool = False) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError("num_classes must be positive")
        self.neck = StereoYOLO26PanNeck(channels)
        self.levels = nn.ModuleList([
            _DecoupledPredictionLevel(channels, num_classes),
            _DecoupledPredictionLevel(channels, num_classes),
            _DecoupledPredictionLevel(channels, num_classes),
        ])

        self.one_to_one_levels = deepcopy(self.levels) if one_to_one else None

    @property
    def one_to_one_enabled(self) -> bool:
        return self.one_to_one_levels is not None

    @staticmethod
    def _predict(features: tuple[Tensor, Tensor, Tensor], levels: nn.ModuleList) -> DenseStereoPredictions:
        objectness, classes, boxes = [], [], []
        for feature, branch in zip(features, levels):
            obj, cls, box = branch(feature)
            objectness.append(obj.flatten(1))
            classes.append(cls.flatten(2).transpose(1, 2))
            boxes.append(box.flatten(2).transpose(1, 2))
        return DenseStereoPredictions(torch.cat(objectness, dim=1), torch.cat(classes, dim=1), torch.cat(boxes, dim=1))

    def forward(self, pyramid: object) -> DenseStereoPredictions:
        levels = self.one_to_one_levels if self.one_to_one_enabled else self.levels
        return self._predict(self.neck(pyramid), levels)

    def forward_training(self, pyramid: object) -> tuple[DenseStereoPredictions, DenseStereoPredictions]:
        if self.one_to_one_levels is None:
            raise RuntimeError("one-to-one training requires a one-to-one head")
        features = self.neck(pyramid)
        many = self._predict(features, self.levels)
        one = self._predict(tuple(feature.detach() for feature in features), self.one_to_one_levels)
        return many, one


def initialize_one_to_one_weights(state: dict[str, Tensor]) -> dict[str, Tensor]:
    """Seed a new sparse branch from a legacy detector while preserving strict loading."""
    sparse_prefix = "detector.one_to_one_levels."
    if any(name.startswith(sparse_prefix) for name in state):
        return state
    dense_prefix = "detector.levels."
    return {
        **state,
        **{sparse_prefix + name[len(dense_prefix):]: weight
           for name, weight in state.items() if name.startswith(dense_prefix)},
    }
