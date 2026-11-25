"""Residual disparity matching restricted to paired P2 RoIs."""
from __future__ import annotations
from contextlib import nullcontext
from dataclasses import dataclass
import torch
from torch.profiler import record_function
from torch import Tensor, nn
from torch.nn import functional as F
from .roi import CropMetadata
from .local_aggregation import AttentionPyramid, LocalAggregation


@dataclass(frozen=True)
class ResidualSearch:
    dmax_roi: int


@dataclass(frozen=True)
class LocalCostVolume:
    cost: Tensor
    valid_bins: Tensor
    spatial_valid: Tensor


class LearnableTemperature(nn.Module):
    """Positive correlation-logit temperature, initialized to 0.1."""

    def __init__(self, initial: float = 0.1) -> None:
        super().__init__()
        if initial <= 0:
            raise ValueError("temperature initial value must be positive")
        self.raw = nn.Parameter(torch.log(torch.expm1(torch.tensor(initial))))

    def forward(self) -> Tensor:
        return F.softplus(self.raw) + torch.finfo(self.raw.dtype).eps


@dataclass(frozen=True)
class RoiDisparity:
    residual: Tensor
    absolute: Tensor
    valid: Tensor
    peak: Tensor
    entropy: Tensor


class RegularizedIDispHead(nn.Module):
    amp_aggregation = False
    """Default RoI-only iDisp: cosine cost, local 2D regularization, soft-argmax."""

    def __init__(self, channels: int, dmax_roi: int, hidden_channels: int = 32) -> None:
        super().__init__()
        self.search = ResidualSearch(dmax_roi)
        self.temperature = LearnableTemperature()
        self.aggregation = LocalAggregation(2 * dmax_roi + 1, channels, channels, channels, hidden_channels)

    def _aggregate_by_native_size(
        self, cost: Tensor, attention: AttentionPyramid, metadata: CropMetadata, spatial_valid: Tensor
    ) -> Tensor:
        """Keep GroupNorm and decoder interpolation independent of batch padding."""
        buckets: dict[tuple[int, int], list[int]] = {}
        for index, (height, width) in enumerate(zip(metadata.h_roi.tolist(), metadata.w_roi.tolist())):
            buckets.setdefault((int(height), int(width)), []).append(index)
        refined = cost.new_zeros(cost.shape)
        full_height, full_width = cost.shape[-2:]
        for (height, width), indices in buckets.items():
            selected = torch.tensor(indices, device=cost.device, dtype=torch.long)
            height3, width3 = (height + 1) // 2, (width + 1) // 2
            height4, width4 = (height + 3) // 4, (width + 3) // 4
            group_cost = cost.index_select(0, selected)[..., :height, :width]
            group_attention = AttentionPyramid(
                attention.p2.index_select(0, selected)[..., :height, :width],
                attention.p3.index_select(0, selected)[..., :height3, :width3],
                attention.p4.index_select(0, selected)[..., :height4, :width4],
            )
            group_valid = spatial_valid.index_select(0, selected)[..., :height, :width]
            group_refined = self.aggregation(group_cost, group_attention, group_valid)
            padded = F.pad(group_refined, (0, full_width - width, 0, full_height - height))
            refined = refined.index_copy(0, selected, padded)
        return refined

    def _aggregate_cost(
        self, cost: Tensor, attention: AttentionPyramid, metadata: CropMetadata, spatial_valid: Tensor
    ) -> Tensor:
        if metadata.fixed_size is not None:
            return self.aggregation(cost, attention, spatial_valid)
        return self._aggregate_by_native_size(cost, attention, metadata, spatial_valid)

    def forward(self, left: Tensor, right: Tensor, attention: AttentionPyramid, metadata: CropMetadata, spatial_valid: Tensor) -> RoiDisparity:
        with record_function("stereo.idisp"):
            return self._forward(left, right, attention, metadata, spatial_valid)

    def _forward(self, left, right, attention, metadata, spatial_valid):
        with record_function("stereo.cost_volume"):
            volume = build_local_cost_volume(left, right, self.search, spatial_valid, self.temperature)
        amp = torch.autocast("cuda", dtype=torch.float16) if self.amp_aggregation else nullcontext()
        with record_function("stereo.local_aggregation"), amp:
            refined = self._aggregate_cost(volume.cost, attention, metadata, volume.spatial_valid).to(left.dtype)
        with record_function("stereo.disparity_regression"):
            logits = torch.where(
                volume.spatial_valid,
                refined.masked_fill(~volume.valid_bins, float("-inf")),
                torch.zeros_like(refined),
            )
            probability = torch.softmax(logits, dim=1)
            offsets = torch.arange(-self.search.dmax_roi, self.search.dmax_roi + 1,
                                   device=left.device, dtype=left.dtype)
            residual = (probability * offsets[None, :, None, None]).sum(1)
            peak = probability.max(1).values
            safe = probability.clamp_min(torch.finfo(probability.dtype).tiny)
            entropy = -(safe * safe.log()).sum(1)
            absolute = restore_absolute_disparity(residual, metadata)
        return RoiDisparity(residual, absolute, volume.valid_bins.any(1), peak, entropy)


def build_local_cost_volume(
    left_roi: Tensor,
    right_roi: Tensor,
    search: ResidualSearch,
    spatial_valid: Tensor,
    temperature: LearnableTemperature,
) -> LocalCostVolume:
    """Correlation volume `[N, 2D+1, Hroi, Wroi]`; never accepts full-image maps."""
    if left_roi.shape != right_roi.shape or left_roi.ndim != 4:
        raise ValueError("left/right RoI features must have equal NCHW shape")
    if spatial_valid.shape != (left_roi.shape[0], 1, left_roi.shape[-2], left_roi.shape[-1]):
        raise ValueError("spatial_valid must be [N,1,H,W] for the RoI features")
    dmax = search.dmax_roi
    left_roi = F.normalize(left_roi, p=2, dim=1)
    right_roi = F.normalize(right_roi, p=2, dim=1)
    padded = F.pad(right_roi, (dmax, dmax))
    padded_valid = F.pad(spatial_valid.bool(), (dmax, dmax))
    width = left_roi.shape[-1]
    costs, valid = [], []
    tau = temperature()
    for offset in range(-dmax, dmax + 1):
        shifted_valid = padded_valid[..., dmax - offset:dmax - offset + width]
        combined_valid = spatial_valid[:, 0].bool() & shifted_valid[:, 0]
        correlation = (left_roi * padded[..., dmax - offset:dmax - offset + width]).sum(1)
        costs.append((correlation / tau).masked_fill(~combined_valid, 0.0))
        valid.append(combined_valid)
    return LocalCostVolume(torch.stack(costs, 1), torch.stack(valid, 1), spatial_valid.bool())


def regress_residual_disparity(cost_volume: Tensor | LocalCostVolume, search: ResidualSearch) -> Tensor:
    """Differentiable local soft-argmax in P2-RoI pixels."""
    if isinstance(cost_volume, LocalCostVolume):
        logits = cost_volume.cost.masked_fill(~cost_volume.valid_bins, float("-inf"))
    else:
        logits = cost_volume
    offsets = torch.arange(-search.dmax_roi, search.dmax_roi + 1, device=logits.device, dtype=logits.dtype)
    probability = torch.softmax(logits, dim=1)
    return (probability * offsets[None, :, None, None]).sum(dim=1)


def restore_absolute_disparity(instance_residual: Tensor, metadata: CropMetadata) -> Tensor:
    """`Df = Di_roi * C/Wroi + (cL-cR)` using the sole crop-coordinate source."""
    scale = (metadata.c / metadata.w_roi.to(metadata.c.dtype)).to(instance_residual.device)
    offset = (metadata.c_l - metadata.c_r).to(instance_residual.device)
    return instance_residual * scale[:, None, None] + offset[:, None, None]


def overflow_mask(residual_gt: Tensor, dmax_roi: int) -> Tensor:
    """Pixels outside the fixed residual-bin support must not be clamped."""
    return residual_gt.abs() > dmax_roi


def disparity_loss(absolute_prediction: Tensor, absolute_target: Tensor, valid: Tensor, beta: float = 1.0) -> Tensor:
    """Image-domain Smooth-L1, macro-averaged over nonempty instances."""
    if absolute_prediction.shape != absolute_target.shape or valid.shape != absolute_target.shape:
        raise ValueError("prediction, target, and valid mask must have matching shape")
    value = F.smooth_l1_loss(absolute_prediction, absolute_target, beta=beta, reduction="none")
    valid_float = valid.to(value.dtype)
    per_instance_count = valid_float.flatten(1).sum(1)
    per_instance_loss = (value * valid_float).flatten(1).sum(1) / per_instance_count.clamp_min(1)
    nonempty = per_instance_count > 0
    if nonempty.any():
        return per_instance_loss[nonempty].mean()
    return value.sum() * 0
