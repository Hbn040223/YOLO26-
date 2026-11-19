"""Shared continuous crop geometry for mask and local-disparity RoIs."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor
from torch.nn import functional as F
from .local_aggregation import AttentionPyramid


ROI_SAMPLING_MODES = ("native_pad", "fixed_56")


@dataclass(frozen=True)
class CropMetadata:
    c_l: Tensor
    c_r: Tensor
    c: Tensor
    y0: Tensor
    h_crop: Tensor
    w_roi: Tensor
    h_roi: Tensor
    stride: int = 4
    fixed_size: int | None = None


def make_pair_crop_windows(
    boxes6: Tensor, image_height: int, image_width: int, margin: float, *,
    roi_sampling: str = "native_pad",
) -> CropMetadata:
    """Make paired continuous 2D windows with shared vertical geometry."""
    if boxes6.ndim != 2 or boxes6.shape[1] != 6:
        raise ValueError("boxes6 must have shape [N, 6]")
    if min(image_height, image_width) <= 0 or margin < 0:
        raise ValueError("image dimensions must be positive and margin non-negative")
    if roi_sampling not in ROI_SAMPLING_MODES:
        raise ValueError(f"unknown RoI sampling mode: {roi_sampling}")
    left_start, left_end = boxes6[:, 0] - margin, boxes6[:, 2] + margin
    right_start, right_end = boxes6[:, 4] - margin, boxes6[:, 5] + margin
    c = torch.maximum(left_end - left_start, right_end - right_start)
    if torch.any(c > image_width):
        raise ValueError("a paired crop is wider than the image")
    c_l = torch.minimum(left_start.clamp_min(0), image_width - c).clamp_min(0)
    c_r = torch.minimum(right_start.clamp_min(0), image_width - c).clamp_min(0)
    top, bottom = boxes6[:, 1] - margin, boxes6[:, 3] + margin
    h_crop = bottom - top
    if torch.any(h_crop > image_height):
        raise ValueError("a paired crop is taller than the image")
    y0 = torch.minimum(top.clamp_min(0), image_height - h_crop).clamp_min(0)
    fixed_size = 56 if roi_sampling == "fixed_56" else None
    if fixed_size is None:
        w_roi = torch.ceil(c / 4).to(torch.long)
        h_roi = torch.ceil(h_crop / 4).to(torch.long)
    else:
        w_roi = torch.full_like(c, fixed_size, dtype=torch.long)
        h_roi = torch.full_like(h_crop, fixed_size, dtype=torch.long)
    return CropMetadata(
        c_l=c_l, c_r=c_r, c=c, y0=y0, h_crop=h_crop, w_roi=w_roi, h_roi=h_roi,
        fixed_size=fixed_size,
    )


def _roi_grid_shape(metadata: CropMetadata, stride: int) -> tuple[Tensor, Tensor, int, int]:
    # P3 and P4 span the same crop as P2 at half and quarter its grid size.
    reduction = stride // metadata.stride
    heights = (metadata.h_roi + reduction - 1) // reduction
    widths = (metadata.w_roi + reduction - 1) // reduction
    if metadata.fixed_size is not None:
        size = (metadata.fixed_size + reduction - 1) // reduction
        return heights, widths, size, size
    return heights, widths, int(heights.max().item()), int(widths.max().item())


def sample_roi_feature(
    features: Tensor, x0: Tensor, metadata: CropMetadata, stride: int
) -> tuple[Tensor, Tensor]:
    """Sample features directly on the crop grid chosen by CropMetadata."""
    if features.ndim != 4 or stride not in (4, 8, 16):
        raise ValueError("features must be NCHW and stride must be 4, 8, or 16")
    count, _, height, width = features.shape
    if x0.shape != (count,) or metadata.c.shape != (count,) or metadata.h_crop.shape != (count,):
        raise ValueError("crop metadata must align with the feature batch")
    heights, widths, max_height, max_width = _roi_grid_shape(metadata, stride)
    rows = torch.arange(max_height, device=features.device, dtype=features.dtype) + .5
    cols = torch.arange(max_width, device=features.device, dtype=features.dtype) + .5
    y = metadata.y0[:, None, None] + rows[None, :, None] / heights[:, None, None] * metadata.h_crop[:, None, None]
    x = x0[:, None, None] + cols[None, None, :] / widths[:, None, None] * metadata.c[:, None, None]
    grid_x = 2 * ((x / stride) + .5) / width - 1
    grid_y = 2 * ((y / stride) + .5) / height - 1
    grid = torch.stack((grid_x.expand(-1, max_height, -1), grid_y.expand(-1, -1, max_width)), dim=-1)
    sampled = F.grid_sample(features, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    if metadata.fixed_size is not None:
        return sampled, torch.ones((count, 1, max_height, max_width), device=features.device, dtype=torch.bool)
    valid = (rows[None, :, None] < heights[:, None, None]) & (cols[None, None, :] < widths[:, None, None])
    spatial_valid = valid[:, None]
    return sampled.masked_fill(~spatial_valid, 0), spatial_valid


class StereoRoIAlign(torch.nn.Module):
    """Sample the real paired P2 RoIs from shared 2D crop metadata."""

    def __init__(self, stride: int = 4) -> None:
        super().__init__()
        if stride != 4:
            raise ValueError("paired matching RoIs must use stride-4 P2 features")
        self.stride = stride

    def forward(self, p2_left: Tensor, p2_right: Tensor, boxes6: Tensor, metadata: CropMetadata) -> tuple[Tensor, Tensor, Tensor, CropMetadata]:
        if p2_left.shape != p2_right.shape or p2_left.shape[0] != boxes6.shape[0]:
            raise ValueError("paired P2 features and boxes must have matching batch size")
        left, spatial_valid = sample_roi_feature(p2_left, metadata.c_l, metadata, self.stride)
        right, right_valid = sample_roi_feature(p2_right, metadata.c_r, metadata, self.stride)
        if not torch.equal(spatial_valid, right_valid):
            raise RuntimeError("paired RoIs must share one spatial validity mask")
        return left, right, spatial_valid, metadata


class LeftRoIAlign(torch.nn.Module):
    """Sample one left-view semantic feature without fabricating a right crop."""

    def __init__(self, stride: int) -> None:
        super().__init__()
        if stride not in (8, 16):
            raise ValueError("left semantic RoIs must use stride 8 or 16")
        self.stride = stride

    def forward(self, feature: Tensor, metadata: CropMetadata) -> tuple[Tensor, Tensor]:
        return sample_roi_feature(feature, metadata.c_l, metadata, self.stride)


def align_left_attention_pyramid(left_p2_roi: Tensor, p3: Tensor, p4: Tensor, metadata: CropMetadata) -> AttentionPyramid:
    """Reuse P2_L(cL) and sample only the required left P3/P4 RoIs."""
    p3_roi, _ = LeftRoIAlign(stride=8)(p3, metadata)
    p4_roi, _ = LeftRoIAlign(stride=16)(p4, metadata)
    return AttentionPyramid(left_p2_roi, p3_roi, p4_roi)


def align_left_mask_target(mask: Tensor, boxes6: Tensor, metadata: CropMetadata, output_size: int = 28) -> Tensor:
    """Map original left instance masks to the mask head's 28x28 crop geometry."""
    if mask.ndim != 4 or mask.shape[1] != 1 or mask.shape[0] != boxes6.shape[0]:
        raise ValueError("mask must be [N,1,H,W] and align with paired boxes")
    height, width = mask.shape[-2:]
    rows = torch.arange(output_size, device=mask.device, dtype=torch.float32) + .5
    cols = torch.arange(output_size, device=mask.device, dtype=torch.float32) + .5
    y = metadata.y0[:, None, None] + rows[None, :, None] / output_size * metadata.h_crop[:, None, None]
    x = metadata.c_l[:, None, None] + cols[None, None, :] / output_size * metadata.c[:, None, None]
    grid_x = 2 * (x + .5) / width - 1
    grid_y = 2 * (y + .5) / height - 1
    grid = torch.stack((grid_x.expand(-1, output_size, -1), grid_y.expand(-1, -1, output_size)), -1)
    return (F.grid_sample(mask.float(), grid, mode="nearest", padding_mode="zeros", align_corners=False) >= .5).to(mask.dtype)


def align_mask_spatial_valid(spatial_valid: Tensor, output_size: int = 28) -> Tensor:
    """Project padded P2 validity to the fixed mask-logit grid.

    Mask targets use the same continuous crop as P2.  Nearest-neighbor
    resampling preserves the binary contract, so padded P2 columns/rows never
    enter the foreground mask loss.
    """
    if spatial_valid.ndim != 4 or spatial_valid.shape[1] != 1:
        raise ValueError("spatial_valid must have shape [N, 1, H, W]")
    return F.interpolate(spatial_valid.to(torch.float32), size=(output_size, output_size), mode="nearest").to(torch.bool)


def _full_resolution_label_grid(
    metadata: CropMetadata, image_height: int, image_width: int, device: torch.device, dtype: torch.dtype
) -> tuple[Tensor, Tensor]:
    """Grid over the native image for labels, not over stride-4 features."""
    heights, widths, max_height, max_width = _roi_grid_shape(metadata, stride=4)
    rows = torch.arange(max_height, device=device, dtype=dtype) + .5
    cols = torch.arange(max_width, device=device, dtype=dtype) + .5
    y0 = metadata.y0.to(device=device, dtype=dtype)
    x0 = metadata.c_l.to(device=device, dtype=dtype)
    h_crop = metadata.h_crop.to(device=device, dtype=dtype)
    c = metadata.c.to(device=device, dtype=dtype)
    heights_d = heights.to(device=device, dtype=dtype)
    widths_d = widths.to(device=device, dtype=dtype)
    y = y0[:, None, None] + rows[None, :, None] / heights_d[:, None, None] * h_crop[:, None, None]
    x = x0[:, None, None] + cols[None, None, :] / widths_d[:, None, None] * c[:, None, None]
    grid_x = 2 * (x + .5) / image_width - 1
    grid_y = 2 * (y + .5) / image_height - 1
    grid = torch.stack((grid_x.expand(-1, max_height, -1), grid_y.expand(-1, -1, max_width)), dim=-1)
    valid = (rows[None, :, None] < heights[:, None, None]) & (cols[None, None, :] < widths[:, None, None])
    return grid, valid[:, None]


def align_roi_labels(
    label: Tensor,
    valid: Tensor,
    metadata: CropMetadata,
    stride: int = 4,
) -> tuple[Tensor, Tensor]:
    """Crop image-domain labels to exactly the P2 RoI grid.

    ``label`` and ``valid`` are full-image ``[1, 1, H, W]`` tensors.  The returned
    tensors are ``[N, 1, Hroi, Wroi]`` with padding cells marked invalid.
    """
    if label.ndim != 4 or label.shape[1] != 1 or valid.shape != label.shape or label.shape[0] != 1:
        raise ValueError("label and valid must both be [1, 1, H, W]")
    if stride != 4:
        raise ValueError("P2 label alignment requires stride 4")
    count = metadata.c.shape[0]
    if count == 0:
        empty = label.new_empty((0, 1, 0, 0))
        return empty, empty.bool()
    grid, spatial_valid = _full_resolution_label_grid(metadata, label.shape[-2], label.shape[-1], label.device, label.dtype)
    sampled_label = F.grid_sample(label.expand(count, -1, -1, -1), grid, mode="nearest", padding_mode="zeros", align_corners=False)
    sampled_valid = F.grid_sample(
        valid.expand(count, -1, -1, -1).to(label.dtype), grid, mode="nearest", padding_mode="zeros", align_corners=False
    )
    valid_roi = (sampled_valid >= .5) & spatial_valid & torch.isfinite(sampled_label) & (sampled_label > 0)
    return sampled_label.masked_fill(~valid_roi, 0), valid_roi


def align_disparity_roi(
    disparity: Tensor,
    disparity_valid: Tensor,
    metadata: CropMetadata,
    stride: int = 4,
) -> tuple[Tensor, Tensor]:
    """Crop full-image disparity to the P2 RoI grid in image-pixel units."""
    return align_roi_labels(disparity, disparity_valid, metadata, stride)


def align_instance_mask_roi(mask: Tensor, metadata: CropMetadata, stride: int = 4) -> Tensor:
    """Nearest-neighbor instance mask on the P2 RoI grid for evaluation only."""
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError("mask must be [N, 1, H, W]")
    if stride != 4:
        raise ValueError("P2 mask alignment requires stride 4")
    count = metadata.c.shape[0]
    if count == 0:
        return mask.new_empty((0, 1, 0, 0), dtype=torch.bool)
    grid, spatial_valid = _full_resolution_label_grid(metadata, mask.shape[-2], mask.shape[-1], mask.device, torch.float32)
    sampled = F.grid_sample(
        mask.float().expand(count, -1, -1, -1), grid, mode="nearest", padding_mode="zeros", align_corners=False
    )
    return (sampled >= .5) & spatial_valid



def resample_roi_map_to_crop(source: Tensor, source_crop: CropMetadata, target_crop: CropMetadata) -> tuple[Tensor, Tensor]:
    """Resample a local RoI map onto another crop grid via image coordinates."""
    if source.ndim != 3 or source.shape[0] != source_crop.c.shape[0] or source.shape[0] != target_crop.c.shape[0]:
        raise ValueError("source and both crop metadata objects must align by instance")
    count = source.shape[0]
    heights, widths, max_height, max_width = _roi_grid_shape(target_crop, stride=4)
    rows = torch.arange(max_height, device=source.device, dtype=source.dtype) + .5
    cols = torch.arange(max_width, device=source.device, dtype=source.dtype) + .5
    target_h = heights.to(device=source.device, dtype=source.dtype)
    target_w = widths.to(device=source.device, dtype=source.dtype)
    x = target_crop.c_l[:, None, None].to(source.dtype) + cols[None, None, :] / target_w[:, None, None] * target_crop.c[:, None, None].to(source.dtype)
    y = target_crop.y0[:, None, None].to(source.dtype) + rows[None, :, None] / target_h[:, None, None] * target_crop.h_crop[:, None, None].to(source.dtype)
    grid_x = 2 * (x - source_crop.c_l[:, None, None].to(source.dtype)) / source_crop.c[:, None, None].to(source.dtype) - 1
    grid_y = 2 * (y - source_crop.y0[:, None, None].to(source.dtype)) / source_crop.h_crop[:, None, None].to(source.dtype) - 1
    grid = torch.stack((grid_x.expand(-1, max_height, -1), grid_y.expand(-1, -1, max_width)), dim=-1)
    target_valid = (rows[None, :, None] < heights[:, None, None]) & (cols[None, None, :] < widths[:, None, None])
    source_valid = (x >= source_crop.c_l[:, None, None]) & (x < (source_crop.c_l + source_crop.c)[:, None, None]) & (y >= source_crop.y0[:, None, None]) & (y < (source_crop.y0 + source_crop.h_crop)[:, None, None])
    valid = target_valid & source_valid
    sampled = F.grid_sample(source[:, None], grid, mode="bilinear", padding_mode="zeros", align_corners=False)[:, 0]
    return sampled.masked_fill(~valid, 0), valid
