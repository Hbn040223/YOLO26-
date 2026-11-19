"""Native-size local 2D regularization for residual disparity cost volumes."""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class AttentionPyramid:
    p2: Tensor
    p3: Tensor
    p4: Tensor


def _groups(channels: int) -> int:
    return next(group for group in (8, 4, 2, 1) if channels % group == 0)


class LocalInvertedResidual(nn.Module):
    """GroupNorm MobileNetV2-style block for one native-size RoI bucket."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        if stride not in (1, 2):
            raise ValueError("LocalInvertedResidual stride must be one or two")
        expanded = in_channels * 2
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, expanded, 1, bias=False), nn.GroupNorm(_groups(expanded), expanded), nn.SiLU()
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(expanded, expanded, 3, stride=stride, padding=1, groups=expanded, bias=False),
            nn.GroupNorm(_groups(expanded), expanded), nn.SiLU(),
        )
        self.project = nn.Sequential(
            nn.Conv2d(expanded, out_channels, 1, bias=False), nn.GroupNorm(_groups(out_channels), out_channels)
        )
        self.skip = nn.Identity() if stride == 1 and in_channels == out_channels else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False), nn.GroupNorm(_groups(out_channels), out_channels)
        )
        self.activation = nn.SiLU()

    def forward(self, features: Tensor) -> Tensor:
        return self.activation(self.project(self.depthwise(self.expand(features))) + self.skip(features))


class GatedContext(nn.Module):
    """Learn how strongly left-view semantic context conditions each scale."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.value = nn.Conv2d(in_channels, out_channels, 1)
        self.gate = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, features: Tensor) -> Tensor:
        return torch.sigmoid(self.gate(features)) * self.value(features)


class LocalAggregation(nn.Module):
    """Two-downsample/two-upsample local U-Net; inputs must have no batch padding."""

    def __init__(self, dlocal: int, p2_channels: int, p3_channels: int, p4_channels: int, hidden_channels: int = 32) -> None:
        super().__init__()
        if hidden_channels < 1:
            raise ValueError("hidden_channels must be positive")
        channels0, channels1, channels2 = hidden_channels, hidden_channels * 2, hidden_channels * 3
        self.stem = nn.Sequential(
            nn.Conv2d(dlocal, channels0, 3, padding=1, bias=False), nn.GroupNorm(_groups(channels0), channels0), nn.SiLU()
        )
        self.p2_context = GatedContext(p2_channels, channels0)
        self.p3_context = GatedContext(p3_channels, channels1)
        self.p4_context = GatedContext(p4_channels, channels2)
        self.stage0 = nn.Sequential(LocalInvertedResidual(channels0, channels0), LocalInvertedResidual(channels0, channels0))
        self.down1 = LocalInvertedResidual(channels0, channels1, stride=2)
        self.stage1 = nn.Sequential(LocalInvertedResidual(channels1, channels1), LocalInvertedResidual(channels1, channels1))
        self.down2 = LocalInvertedResidual(channels1, channels2, stride=2)
        self.stage2 = nn.Sequential(LocalInvertedResidual(channels2, channels2), LocalInvertedResidual(channels2, channels2))
        self.up1 = nn.Conv2d(channels2, channels1, 1)
        self.decode1 = LocalInvertedResidual(channels1, channels1)
        self.up0 = nn.Conv2d(channels1, channels0, 1)
        self.decode0 = LocalInvertedResidual(channels0, channels0)
        self.output = nn.Conv2d(channels0, dlocal, 1)

    @staticmethod
    def _mask(features: Tensor, valid: Tensor) -> Tensor:
        return features * valid.to(dtype=features.dtype)

    @staticmethod
    def _match(context: Tensor, reference: Tensor) -> Tensor:
        if context.shape[-2:] == reference.shape[-2:]:
            return context
        return F.interpolate(context, size=reference.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, cost: Tensor, attention: AttentionPyramid, spatial_valid: Tensor) -> Tensor:
        if cost.ndim != 4 or spatial_valid.shape != (cost.shape[0], 1, cost.shape[-2], cost.shape[-1]):
            raise ValueError("cost and spatial_valid must be [N,D,H,W] and [N,1,H,W]")
        mask0 = spatial_valid.bool()
        stage0 = self._mask(self.stem(cost) + self._match(self.p2_context(attention.p2), cost), mask0)
        stage0 = self._mask(self.stage0(stage0), mask0)
        mask1 = F.max_pool2d(mask0.float(), kernel_size=2, stride=2, ceil_mode=True).bool()
        stage1 = self._mask(self.down1(stage0), mask1)
        stage1 = self._mask(stage1 + self._match(self.p3_context(attention.p3), stage1), mask1)
        stage1 = self._mask(self.stage1(stage1), mask1)
        mask2 = F.max_pool2d(mask1.float(), kernel_size=2, stride=2, ceil_mode=True).bool()
        stage2 = self._mask(self.down2(stage1), mask2)
        stage2 = self._mask(stage2 + self._match(self.p4_context(attention.p4), stage2), mask2)
        stage2 = self._mask(self.stage2(stage2), mask2)
        decoded1 = self.up1(F.interpolate(stage2, size=stage1.shape[-2:], mode="bilinear", align_corners=False))
        decoded1 = self._mask(self.decode1(decoded1 + stage1), mask1)
        decoded0 = self.up0(F.interpolate(decoded1, size=stage0.shape[-2:], mode="bilinear", align_corners=False))
        decoded0 = self._mask(self.decode0(decoded0 + stage0), mask0)
        return self._mask(self.output(decoded0), mask0)
