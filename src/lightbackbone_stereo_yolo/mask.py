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



class ContextInstanceMaskHead(nn.Module):
    """Fuse a wider view of the existing left crop with its spatial detail."""
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.detail = self._block(in_channels, 24)
        self.half_grid = self._block(24, 32, 2)
        self.quarter_grid = self._block(32, 48, 2)
        self.context = self._block(48, 48)
        self.decode_half = self._block(48 + 32, 32)
        self.decode_full = self._block(32 + 24, 24)
        self.logits = nn.Conv2d(24, 1, 1)

    @staticmethod
    def _block(in_channels: int, out_channels: int, stride: int = 1) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, left_roi: Tensor) -> Tensor:
        detail = self.detail(F.avg_pool2d(left_roi, 2))
        half = self.half_grid(detail)
        coarse = self.context(self.quarter_grid(half))
        coarse = F.interpolate(coarse, size=half.shape[-2:], mode="bilinear", align_corners=False)
        decoded = self.decode_half(torch.cat((coarse, half), dim=1))
        decoded = F.interpolate(decoded, size=detail.shape[-2:], mode="bilinear", align_corners=False)
        return self.logits(self.decode_full(torch.cat((decoded, detail), dim=1)))


def mask_loss(mask_logits: Tensor, target_mask: Tensor, valid: Tensor, *,
              pos_weight: Tensor | None = None) -> Tensor:
    per_pixel = F.binary_cross_entropy_with_logits(
        mask_logits, target_mask, reduction="none", pos_weight=pos_weight,
    )
    weighted = per_pixel * valid.to(per_pixel.dtype)
    return weighted.sum() / valid.sum().clamp_min(1)



def foreground_disparity_validity(
    mask_logits: Tensor, mask_validity: Tensor, disparity: Tensor,
    disparity_validity: Tensor, supported: Tensor,
) -> Tensor:
    """Return instance foreground validity on the deployed disparity grid."""
    probabilities = F.interpolate(mask_logits.sigmoid(), size=disparity.shape[-2:],
                                  mode="bilinear", align_corners=False)[:, 0]
    spatial = F.interpolate(mask_validity.float(), size=disparity.shape[-2:], mode="nearest")[:, 0].bool()
    return (supported[:, None, None] & (probabilities > .50)
            & spatial & disparity_validity & torch.isfinite(disparity))
