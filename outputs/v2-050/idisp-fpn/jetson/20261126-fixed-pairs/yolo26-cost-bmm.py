import torch
from torch.nn import functional as F

def batched_cost_volume(left_roi, right_roi, search, spatial_valid, temperature):
    from lightbackbone_stereo_yolo.local_disparity import LocalCostVolume

    dmax = search.dmax_roi
    width = left_roi.shape[-1]
    left = F.normalize(left_roi, p=2, dim=1).permute(0, 2, 3, 1).contiguous()
    right = F.normalize(right_roi, p=2, dim=1).permute(0, 2, 1, 3).contiguous()
    correlations = left @ right
    positions = torch.arange(width, device=left_roi.device)
    offsets = torch.arange(-dmax, dmax+1, device=left_roi.device)
    right_positions = positions[:, None] - offsets
    valid_position = (right_positions >= 0) & (right_positions < width)
    gather_index = right_positions.clamp(0, width-1)
    cost = correlations.gather(-1, gather_index.expand(*correlations.shape[:2], -1, -1))
    valid_pixels = spatial_valid[:, 0].bool()
    valid = valid_pixels[..., None] & valid_pixels[..., gather_index] & valid_position
    cost = (cost / temperature()).masked_fill(~valid, 0.)
    return LocalCostVolume(cost.permute(0, 3, 1, 2).contiguous(),
                           valid.permute(0, 3, 1, 2).contiguous(), spatial_valid.bool())
