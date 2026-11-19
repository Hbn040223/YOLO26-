"""Lightweight left-view instance-mask branch."""
from __future__ import annotations
import torch
from torch import Tensor, nn
from torch.nn import functional as F


class LightInstanceMaskHead(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 32) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 1),
        )

    def forward(self, left_roi: Tensor) -> Tensor:
        return F.interpolate(self.layers(left_roi), size=(28, 28), mode="bilinear", align_corners=False)


def mask_loss(mask_logits: Tensor, target_mask: Tensor, valid: Tensor) -> Tensor:
    per_pixel = F.binary_cross_entropy_with_logits(mask_logits, target_mask, reduction="none")
    weighted = per_pixel * valid.to(per_pixel.dtype)
    return weighted.sum() / valid.sum().clamp_min(1)
