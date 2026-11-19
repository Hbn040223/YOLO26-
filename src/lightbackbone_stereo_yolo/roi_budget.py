"""Select complete stereo pairs after NMS, before any RoI work."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import torch
from torch import Tensor

from .detection import PairDetections


@dataclass(frozen=True)
class RectifiedStereoCalibration:
    bf_px_m: float
    cx_delta_px: float


def coarse_depth_m(boxes6: Tensor, calibration: RectifiedStereoCalibration | None) -> Tensor:
    depth = boxes6.new_full((boxes6.shape[0],), float("nan"))
    if (calibration is None or not isfinite(calibration.bf_px_m)
            or calibration.bf_px_m <= 0 or not isfinite(calibration.cx_delta_px)):
        return depth
    x1l, y1, x2l, y2, x1r, x2r = boxes6.unbind(dim=1)
    corrected = ((x1l - x1r) + (x2l - x2r)) * .5 - calibration.cx_delta_px
    valid = (
        torch.isfinite(boxes6).all(dim=1) & torch.isfinite(corrected) & (corrected > 0)
        & (x2l > x1l) & (x2r > x1r) & (y2 > y1)
    )
    return torch.where(valid, calibration.bf_px_m / corrected, depth)


def select_roi_pair_indices(
    pairs: PairDetections, *, max_pairs: int | None, near_quota: int = 0,
    calibration: RectifiedStereoCalibration | None = None,
    near_depth_m: float = 10.,
) -> Tensor:
    """Return row indices in the original post-NMS order."""
    count = pairs.indices.numel()
    limit = count if max_pairs is None else max_pairs
    if limit < 0 or near_quota < 0 or (max_pairs is None and near_quota) or near_quota > limit:
        raise ValueError("expected a nonnegative cap and a near quota within that cap")
    if not isfinite(near_depth_m) or near_depth_m <= 0:
        raise ValueError("near depth threshold must be positive and finite")
    rows = torch.arange(count, device=pairs.boxes6.device)
    if limit >= count or near_quota == 0 or calibration is None:
        return rows[:limit]

    prefix = limit - near_quota
    near = coarse_depth_m(pairs.boxes6[prefix:], calibration) < near_depth_m
    reserved = rows[prefix:][near][:near_quota]
    kept = rows < prefix
    kept[reserved] = True
    kept[rows[~kept][:near_quota - reserved.numel()]] = True
    return rows[kept]


def select_roi_pairs(
    pairs: PairDetections, *, max_pairs: int | None, near_quota: int = 0,
    calibration: RectifiedStereoCalibration | None = None,
    near_depth_m: float = 10.,
) -> PairDetections:
    rows = select_roi_pair_indices(
        pairs, max_pairs=max_pairs, near_quota=near_quota,
        calibration=calibration, near_depth_m=near_depth_m,
    )
    if rows.numel() == pairs.indices.numel():
        return pairs
    return PairDetections(
        pairs.boxes6[rows], pairs.scores[rows], pairs.indices[rows],
        None if pairs.classes is None else pairs.classes[rows],
    )
