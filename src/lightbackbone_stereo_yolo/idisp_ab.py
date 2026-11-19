"""Image-coordinate comparison for disparity-only experiments."""
from __future__ import annotations

import torch
from torch.nn import functional as F

from .training import supervision_valid


def project_disparity(disparity, crop, image_hw):
    """Project each valid native rectangle to the original left image."""
    image_height, image_width = image_hw
    native = disparity.absolute
    max_height, max_width = native.shape[-2:]
    cols = torch.arange(image_width, device=native.device, dtype=native.dtype)[None, None, :]
    rows = torch.arange(image_height, device=native.device, dtype=native.dtype)[None, :, None]
    left = crop.c_l[:, None, None]
    top = crop.y0[:, None, None]
    width = crop.c[:, None, None]
    height = crop.h_crop[:, None, None]
    out_width = crop.w_roi[:, None, None]
    out_height = crop.h_roi[:, None, None]
    x = (cols - left) * out_width / width - .5
    y = (rows - top) * out_height / height - .5
    # Clamp to each native rectangle before normalizing to the batch tensor.
    x = torch.minimum(x.clamp_min(0), out_width - 1)
    y = torch.minimum(y.clamp_min(0), out_height - 1)
    grid = torch.stack((
        (2 * (x + .5) / max_width - 1).expand(-1, image_height, -1),
        (2 * (y + .5) / max_height - 1).expand(-1, -1, image_width),
    ), -1)
    source_valid = disparity.valid & torch.isfinite(native)
    prediction = F.grid_sample(
        native.masked_fill(~source_valid, 0)[:, None], grid, align_corners=False,
    )[:, 0]
    weights = F.grid_sample(source_valid[:, None].to(native.dtype), grid, align_corners=False)[:, 0]
    inside = (cols >= left) & (cols < left + width) & (rows >= top) & (rows < top + height)
    valid = inside & (weights >= 1 - 1e-5) & torch.isfinite(prediction)
    return prediction, valid


def original_foreground(target, device, indices=None):
    truth = target["disparity"].to(device)
    masks = target["masks"].to(device)[:, 0]
    if indices is not None:
        masks = masks[indices]
    valid = target["disparity_valid"].to(device)
    included = supervision_valid(truth, valid, None, masks)
    return truth, included


def error_totals(prediction, truth, valid):
    """Return additive pixel-domain error counts for a frame or cohort."""
    error = (prediction - truth).abs()
    return {
        "pixels": int(valid.sum().item()),
        "epe_sum": float(error.masked_fill(~valid, 0).sum(dtype=torch.float64).item()),
        "d1_hits": int(((error > 3.) & (error > .05 * truth.abs()) & valid).sum().item()),
    }


def summarize_error(totals):
    pixels = totals["pixels"]
    return {
        **totals,
        "epe": totals["epe_sum"] / pixels if pixels else None,
        "d1": totals["d1_hits"] / pixels if pixels else None,
    }
