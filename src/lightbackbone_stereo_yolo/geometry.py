"""Continuous 6D paired-box geometry used by Stereo-YOLO.

The right-view terms follow Disp R-CNN: its centre and width are expressed
relative to the *left* anchor width.  A pair therefore stays one object rather
than two independently matched detections.
"""
from __future__ import annotations

import torch
from torch import Tensor


def _validate_pairs(boxes: Tensor) -> None:
    if boxes.ndim != 2 or boxes.shape[1] != 6:
        raise ValueError("paired boxes must have shape [N, 6]")


def _validate_anchors(anchors: Tensor) -> None:
    if anchors.ndim != 2 or anchors.shape[1] != 4:
        raise ValueError("left anchors must have shape [N, 4]")


def pair_boxes_to_lr(pairs: Tensor) -> tuple[Tensor, Tensor]:
    """Expand `[x1L,y1,x2L,y2,x1R,x2R]` into aligned four-coordinate boxes."""

    _validate_pairs(pairs)
    left = pairs[:, :4]
    right = torch.stack((pairs[:, 4], pairs[:, 1], pairs[:, 5], pairs[:, 3]), dim=1)
    return left, right


def encode_boxes6(target_pairs: Tensor, left_anchors: Tensor) -> Tensor:
    """Encode a paired target against a left-only anchor with six continuous terms."""

    _validate_pairs(target_pairs)
    _validate_anchors(left_anchors)
    if target_pairs.shape[0] != left_anchors.shape[0]:
        raise ValueError("target pairs and anchors must have matching length")
    target_left, target_right = pair_boxes_to_lr(target_pairs)
    anchor_width = (left_anchors[:, 2] - left_anchors[:, 0]).clamp_min(torch.finfo(target_pairs.dtype).eps)
    anchor_height = (left_anchors[:, 3] - left_anchors[:, 1]).clamp_min(torch.finfo(target_pairs.dtype).eps)
    anchor_cx = (left_anchors[:, 0] + left_anchors[:, 2]) * 0.5
    anchor_cy = (left_anchors[:, 1] + left_anchors[:, 3]) * 0.5
    left_width = (target_left[:, 2] - target_left[:, 0]).clamp_min(torch.finfo(target_pairs.dtype).eps)
    left_height = (target_left[:, 3] - target_left[:, 1]).clamp_min(torch.finfo(target_pairs.dtype).eps)
    right_width = (target_right[:, 2] - target_right[:, 0]).clamp_min(torch.finfo(target_pairs.dtype).eps)
    left_cx = (target_left[:, 0] + target_left[:, 2]) * 0.5
    left_cy = (target_left[:, 1] + target_left[:, 3]) * 0.5
    right_cx = (target_right[:, 0] + target_right[:, 2]) * 0.5
    return torch.stack((
        (left_cx - anchor_cx) / anchor_width,
        (left_cy - anchor_cy) / anchor_height,
        torch.log(left_width / anchor_width),
        torch.log(left_height / anchor_height),
        (right_cx - anchor_cx) / anchor_width,
        torch.log(right_width / anchor_width),
    ), dim=1)


def decode_boxes6(encoded: Tensor, left_anchors: Tensor, max_log_scale: float = 4.135) -> Tensor:
    """Decode six regression terms to one left/right-bound 6D pair per anchor."""

    if encoded.ndim != 2 or encoded.shape[1] != 6:
        raise ValueError("encoded boxes must have shape [N, 6]")
    _validate_anchors(left_anchors)
    if encoded.shape[0] != left_anchors.shape[0]:
        raise ValueError("encoded boxes and anchors must have matching length")
    anchors = left_anchors.to(dtype=encoded.dtype)
    width = anchors[:, 2] - anchors[:, 0]
    height = anchors[:, 3] - anchors[:, 1]
    cx = (anchors[:, 0] + anchors[:, 2]) * 0.5
    cy = (anchors[:, 1] + anchors[:, 3]) * 0.5
    left_cx = encoded[:, 0] * width + cx
    left_cy = encoded[:, 1] * height + cy
    left_width = torch.exp(encoded[:, 2].clamp(max=max_log_scale)) * width
    left_height = torch.exp(encoded[:, 3].clamp(max=max_log_scale)) * height
    right_cx = encoded[:, 4] * width + cx
    right_width = torch.exp(encoded[:, 5].clamp(max=max_log_scale)) * width
    return torch.stack((
        left_cx - 0.5 * left_width,
        left_cy - 0.5 * left_height,
        left_cx + 0.5 * left_width,
        left_cy + 0.5 * left_height,
        right_cx - 0.5 * right_width,
        right_cx + 0.5 * right_width,
    ), dim=1)


from dataclasses import dataclass


@dataclass(frozen=True)
class AnchorFreeAssignment:
    """Dense point targets for one left-driven, paired stereo frame."""

    labels: Tensor
    matched_indices: Tensor
    classes: Tensor
    box6: Tensor
    box_valid_mask: Tensor


def point_levels(pyramid: object) -> tuple[tuple[int, int, int], ...]:
    """The single P3/P4/C5 point-grid definition shared by train and inference."""
    return (
        (pyramid.p3_left.shape[-2], pyramid.p3_left.shape[-1], 8),
        (pyramid.p4_left.shape[-2], pyramid.p4_left.shape[-1], 16),
        (pyramid.c5_left.shape[-2], pyramid.c5_left.shape[-1], 32),
    )


def generate_point_locations(
    levels: tuple[tuple[int, int, int], ...],
    device: torch.device | None = None,
) -> Tensor:
    """Return flattened [cx, cy, stride] points in P3/P4/C5 head order."""
    resolved_device = torch.device("cpu") if device is None else torch.device(device)
    points: list[Tensor] = []
    for height, width, stride in levels:
        if height < 1 or width < 1 or stride < 1:
            raise ValueError("point levels require positive height, width and stride")
        y, x = torch.meshgrid(
            torch.arange(height, device=resolved_device),
            torch.arange(width, device=resolved_device),
            indexing="ij",
        )
        points.append(torch.stack((
            (x.reshape(-1) + .5) * stride,
            (y.reshape(-1) + .5) * stride,
            torch.full((height * width,), float(stride), device=resolved_device),
        ), dim=1))
    return torch.cat(points, dim=0) if points else torch.empty((0, 3), device=resolved_device)


def _validate_anchorfree_locations(locations: Tensor) -> None:
    if locations.ndim != 2 or locations.shape[1] != 3:
        raise ValueError("anchor-free locations must have shape [N, 3]")


def encode_anchorfree_boxes6(target_pairs: Tensor, matched_locations: Tensor) -> Tensor:
    """Encode paired boxes against their matched point locations."""
    _validate_pairs(target_pairs)
    _validate_anchorfree_locations(matched_locations)
    if target_pairs.shape[0] != matched_locations.shape[0]:
        raise ValueError("target pairs and matched locations must have matching length")
    if not target_pairs.numel():
        return target_pairs.new_empty((0, 6))
    location = matched_locations.to(dtype=target_pairs.dtype)
    stride = location[:, 2].clamp_min(torch.finfo(target_pairs.dtype).eps)
    cx, cy = location[:, 0], location[:, 1]
    left = target_pairs[:, :4]
    right_cx = (target_pairs[:, 4] + target_pairs[:, 5]) * .5
    raw_ltrb = torch.stack((cx - left[:, 0], cy - left[:, 1], left[:, 2] - cx, left[:, 3] - cy), dim=1)
    safe_ltrb = raw_ltrb.clamp_min(-.99 * stride[:, None])
    right_width = (target_pairs[:, 5] - target_pairs[:, 4]).clamp_min(1e-3)
    return torch.cat((
        torch.log1p(safe_ltrb / stride[:, None]),
        torch.asinh(((cx - right_cx) / stride))[:, None],
        torch.log(right_width / stride)[:, None],
    ), dim=1)


RIGHT_OFFSET_MODES = ("legacy_clamp", "signed")


def right_offset_mode_from_config(config, override: str | None = None) -> str:
    mode = config.get("right_offset_mode", "legacy_clamp") if override is None else override
    if mode not in RIGHT_OFFSET_MODES:
        raise ValueError(f"unsupported right offset mode: {mode!r}")
    return mode


def decode_anchorfree_boxes6(
    encoded: Tensor, locations: Tensor, *, right_offset_mode: str = "legacy_clamp",
) -> Tensor:
    """Decode direct 6D point-relative predictions to pair-preserving boxes."""
    if encoded.ndim != 2 or encoded.shape[1] != 6:
        raise ValueError("encoded boxes must have shape [N, 6]")
    _validate_anchorfree_locations(locations)
    if encoded.shape[0] != locations.shape[0]:
        raise ValueError("encoded boxes and locations must have matching length")
    location = locations.to(dtype=encoded.dtype)
    stride = location[:, 2].clamp_min(torch.finfo(encoded.dtype).eps)
    cx, cy = location[:, 0], location[:, 1]
    ltrb = torch.expm1(encoded[:, :4]).clamp_min(1e-3) * stride[:, None]
    point_to_right_center = torch.sinh(encoded[:, 4]) * stride
    if right_offset_mode == "legacy_clamp":
        point_to_right_center = point_to_right_center.clamp_min(0)
    elif right_offset_mode != "signed":
        raise ValueError(f"unsupported right offset mode: {right_offset_mode!r}")
    right_cx = cx - point_to_right_center
    right_width = (torch.exp(encoded[:, 5].clamp(max=8.0)) * stride).clamp_min(1e-3)
    return torch.stack((
        cx - ltrb[:, 0],
        cy - ltrb[:, 1],
        cx + ltrb[:, 2],
        cy + ltrb[:, 3],
        right_cx - right_width * .5,
        right_cx + right_width * .5,
    ), dim=1)


def _point_scale_match(stride: Tensor, max_side: Tensor) -> Tensor:
    """P3/P4/P5 scale assignment."""
    p3 = (stride[:, None] == 8) & (max_side[None, :] > 0) & (max_side[None, :] <= 64)
    p4 = (stride[:, None] == 16) & (max_side[None, :] > 64) & (max_side[None, :] <= 128)
    p5 = (stride[:, None] == 32) & (max_side[None, :] > 128)
    return p3 | p4 | p5


ASSIGNMENT_MODES = ("legacy", "inside_rescue")


def assignment_mode_from_config(config, override: str | None = None) -> str:
    mode = config.get("assignment_mode", "legacy") if override is None else override
    if mode not in ASSIGNMENT_MODES:
        raise ValueError(f"unsupported assignment mode: {mode!r}")
    return mode


def _assign_inside_rescue(
    locations: Tensor, targets: Tensor, inside: Tensor, primary: Tensor, in_image: Tensor,
    valid_box: Tensor, area: Tensor, labels: Tensor, matched: Tensor,
) -> None:
    """Rescue unrepresented targets before assigning any classification-only points."""
    cx, cy, stride = locations.unbind(dim=1)
    center = (targets[:, :2] + targets[:, 2:4]) * .5
    count = targets.shape[0]
    represented = torch.bincount(matched[labels == 1], minlength=count) > 0
    ordered = area.argsort(stable=True)
    missing = ordered[~represented[ordered]].tolist()
    remaining = []
    for gt in missing:
        if not bool(valid_box[gt]):
            remaining.append(gt)
            continue
        choices = in_image & (labels != 1) & inside[:, gt]
        same_level = choices & primary[:, gt]
        if same_level.any():
            choices = same_level
        elif choices.any():
            finest_stride = stride[choices].min()
            choices = choices & (stride == finest_stride)
        else:
            remaining.append(gt)
            continue
        distance = (cx - center[gt, 0]).square() + (cy - center[gt, 1]).square()
        point = distance.masked_fill(~choices, float("inf")).argmin()
        labels[point] = 1
        matched[point] = gt

    classifiable = torch.isfinite(targets).all(dim=1) & (area > 0)
    for gt in remaining:
        if not bool(classifiable[gt]):
            continue
        choices = in_image & (labels != 1) & primary[:, gt]
        if not choices.any():
            continue
        distance = (cx - center[gt, 0]).square() + (cy - center[gt, 1]).square()
        point = distance.masked_fill(~choices, float("inf")).argmin()
        labels[point] = 1
        matched[point] = gt


def assign_anchorfree_left_targets(
    locations: Tensor,
    boxes6: Tensor,
    classes: Tensor,
    center_radius_in_strides: float = 2.5,
    *,
    assignment_mode: str = "legacy",
    image_hw: tuple[int, int] | None = None,
) -> AnchorFreeAssignment:
    """Assign left-box centre points while carrying a bound right-box target."""
    _validate_anchorfree_locations(locations)
    _validate_pairs(boxes6)
    if classes.ndim != 1 or classes.shape[0] != boxes6.shape[0]:
        raise ValueError("classes must be [N] and match paired boxes")
    if center_radius_in_strides <= 0:
        raise ValueError("center_radius_in_strides must be positive")
    if assignment_mode not in ASSIGNMENT_MODES:
        raise ValueError(f"unsupported assignment mode: {assignment_mode!r}")
    if assignment_mode == "inside_rescue" and (image_hw is None or min(image_hw) <= 0):
        raise ValueError("inside_rescue requires positive image height and width")
    count = locations.shape[0]
    labels = torch.zeros(count, dtype=torch.long, device=locations.device)
    matched = torch.full_like(labels, -1)
    class_target = torch.zeros(count, dtype=torch.long, device=locations.device)
    box_target = locations.new_zeros((count, 6))
    box_valid = torch.zeros(count, dtype=torch.bool, device=locations.device)
    if boxes6.numel() == 0 or count == 0:
        return AnchorFreeAssignment(labels, matched, class_target, box_target, box_valid)

    targets = boxes6.to(device=locations.device, dtype=locations.dtype)
    target_classes = classes.to(device=locations.device, dtype=torch.long)
    cx, cy, stride = locations[:, 0], locations[:, 1], locations[:, 2]
    left = targets[:, :4]
    inside = (
        (cx[:, None] >= left[None, :, 0])
        & (cx[:, None] <= left[None, :, 2])
        & (cy[:, None] >= left[None, :, 1])
        & (cy[:, None] <= left[None, :, 3])
    )
    labels[inside.any(dim=1)] = -1
    target_cx = (left[:, 0] + left[:, 2]) * .5
    target_cy = (left[:, 1] + left[:, 3]) * .5
    in_center = (
        (cx[:, None] - target_cx[None, :]).abs() <= center_radius_in_strides * stride[:, None]
    ) & (
        (cy[:, None] - target_cy[None, :]).abs() <= center_radius_in_strides * stride[:, None]
    )
    max_side = torch.maximum(left[:, 2] - left[:, 0], left[:, 3] - left[:, 1])
    primary = _point_scale_match(stride, max_side)
    candidates = inside & in_center & primary
    width, height = left[:, 2] - left[:, 0], left[:, 3] - left[:, 1]
    area = (width * height).clamp_min(0)
    if assignment_mode == "inside_rescue":
        area = width.clamp_min(0) * height.clamp_min(0)
        height_px, width_px = image_hw
        in_image = (cx >= 0) & (cx < width_px) & (cy >= 0) & (cy < height_px)
        valid_box = torch.isfinite(targets).all(dim=1) & (width > 0) & (height > 0)
        valid_box &= (targets[:, 5] - targets[:, 4]) > 1e-3
        # Preserve ordinary classification positives for finite GTs with degenerate right width.
        candidates &= in_image[:, None] & torch.isfinite(targets).all(dim=1)[None, :]
    candidate_area = torch.where(candidates, area[None, :], torch.full_like(candidates, float("inf"), dtype=targets.dtype))
    selected_area, selected_target = candidate_area.min(dim=1)
    positive = torch.isfinite(selected_area)
    labels[positive] = 1
    matched[positive] = selected_target[positive]

    if assignment_mode == "inside_rescue":
        _assign_inside_rescue(locations, targets, inside, primary, in_image, valid_box, area, labels, matched)
    else:
        for target_index in range(targets.shape[0]):
            if (matched == target_index).any():
                continue
            level = primary[:, target_index]
            available = level & (labels != 1)
            choices = available if available.any() else level
            if not choices.any():
                continue
            distance = (cx - target_cx[target_index]).square() + (cy - target_cy[target_index]).square()
            point_index = torch.where(choices, distance, torch.full_like(distance, float("inf"))).argmin()
            labels[point_index] = 1
            matched[point_index] = target_index

    positive = labels == 1
    if positive.any():
        target_index = matched[positive]
        box_target[positive] = encode_anchorfree_boxes6(targets[target_index], locations[positive])
        class_target[positive] = target_classes[target_index]
        left_for_point = left[target_index]
        raw_ltrb = torch.stack((
            locations[positive, 0] - left_for_point[:, 0],
            locations[positive, 1] - left_for_point[:, 1],
            left_for_point[:, 2] - locations[positive, 0],
            left_for_point[:, 3] - locations[positive, 1],
        ), dim=1)
        right_width = targets[target_index, 5] - targets[target_index, 4]
        box_valid[positive] = (raw_ltrb >= 0).all(dim=1) & (right_width > 1e-3)
        if assignment_mode == "inside_rescue":
            box_valid[positive] &= valid_box[target_index] & in_image[positive]
    return AnchorFreeAssignment(labels, matched, class_target, box_target, box_valid)
