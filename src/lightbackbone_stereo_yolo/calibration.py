"""Shared, RoI-bin calibration primitives for local instance disparity."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class GlobalDmax:
    """One batchable local-search range and the bin-unit quantities that produced it."""

    roi_residual_q_bins: float
    margin_roi_global_bins: float
    dmax_roi: int


def derive_global_dmax(
    residual_bins: Tensor,
    roi_bin_per_image_px: Tensor,
    *,
    quantile: float,
    margin_img_px: float,
) -> GlobalDmax:
    """Choose one Dmax directly in RoI-bin units for a batchable cost volume."""
    if residual_bins.numel() == 0 or roi_bin_per_image_px.numel() == 0:
        raise ValueError("residuals and RoI scales must be nonempty")
    if not 0 < quantile <= 1 or margin_img_px < 0:
        raise ValueError("quantile or margin is invalid")
    residual_q = float(torch.quantile(residual_bins.abs().flatten(), quantile).item())
    global_margin = float((roi_bin_per_image_px * margin_img_px).max().item())
    return GlobalDmax(residual_q, global_margin, int(torch.ceil(torch.tensor(residual_q + global_margin)).item()))
