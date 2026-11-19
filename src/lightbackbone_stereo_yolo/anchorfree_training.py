"""Anchor-free detector training adapter; leaves the baseline anchor path untouched."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .detection import _elementwise_iou, anchorfree_detection_loss
from .geometry import (
    _point_scale_match, assign_anchorfree_left_targets, decode_anchorfree_boxes6,
    encode_anchorfree_boxes6, generate_point_locations,
    pair_boxes_to_lr, point_levels,
)
from .local_disparity import disparity_loss
from .training_metrics import _pairwise_iou
from .mask import mask_loss
from .training import (
    build_instance_supervision,
    instance_metrics,
    repeat_roi_features,
)


def build_anchorfree_detection_targets(
    locations: Tensor, boxes6: Tensor, classes: Tensor, *,
    assignment_mode: str = "legacy", image_hw: tuple[int, int] | None = None,
    ignore_mask: Tensor | None = None,
) -> dict[str, Tensor]:
    assignment = assign_anchorfree_left_targets(
        locations, boxes6, classes, assignment_mode=assignment_mode, image_hw=image_hw,
    )
    matched_boxes6 = locations.new_zeros((locations.shape[0], 6))
    positive = assignment.labels == 1
    matched_boxes6[positive] = boxes6.to(locations)[assignment.matched_indices[positive]]
    left = matched_boxes6[:, :4]
    point = locations[:, :2]
    left_valid = positive & torch.isfinite(left).all(dim=1)
    left_valid &= (left[:, 2:] > left[:, :2]).all(dim=1)
    left_valid &= ((point >= left[:, :2]) & (point <= left[:, 2:])).all(dim=1)
    if image_hw is not None:
        height, width = image_hw
        left_valid &= (point[:, 0] >= 0) & (point[:, 0] < width)
        left_valid &= (point[:, 1] >= 0) & (point[:, 1] < height)
    labels = assignment.labels
    if ignore_mask is not None:
        if image_hw is None or ignore_mask.shape != image_hw or min(image_hw) <= 0:
            raise ValueError("detector ignore mask must match the image height and width")
        height, width = image_hw
        x, y = point.long().unbind(1)
        inside_image = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        ignore_mask = ignore_mask.to(device=locations.device, dtype=torch.bool)
        ignored = ~inside_image | ignore_mask[y.clamp(0, height - 1), x.clamp(0, width - 1)]
        labels = labels.masked_fill((labels == 0) & ignored, -1)
    return {
        "left_box_valid_mask": left_valid,
        "matched_boxes6": matched_boxes6,
        "labels": labels,
        "classes": assignment.classes,
        "box6": assignment.box6,
        "box_valid_mask": assignment.box_valid_mask,
        "matched_indices": assignment.matched_indices,
    }



def _image_point_ignore(locations: Tensor, image_hw: tuple[int, int], ignore_mask: Tensor | None) -> Tensor:
    height, width = image_hw
    if min(image_hw) <= 0 or (ignore_mask is not None and ignore_mask.shape != image_hw):
        raise ValueError("detector image and ignore mask must share a positive height and width")
    points = locations[:, :2]
    blocked = ~((points >= 0) & (points < points.new_tensor((width, height)))).all(1)
    if ignore_mask is not None:
        x, y = points.long().unbind(1)
        blocked |= ignore_mask.to(device=points.device, dtype=torch.bool)[
            y.clamp(0, height - 1), x.clamp(0, width - 1)
        ]
    return blocked


def _inside_left_boxes(points: Tensor, boxes6: Tensor) -> Tensor:
    return ((points[:, None] >= boxes6[None, :, :2]) & (points[:, None] <= boxes6[None, :, 2:4])).all(2)


def build_car_supervision_targets(
    locations: Tensor, boxes6: Tensor, classes: Tensor, *,
    assignment_mode: str, image_hw: tuple[int, int], ignore_mask: Tensor | None,
    supervise_left_only: bool,
) -> dict[str, Tensor]:
    """Protect the existing paired positives while supervising available left-only points."""
    classes = classes.to(device=locations.device)
    boxes6 = boxes6.to(locations)
    pair = (boxes6[:, 5] - boxes6[:, 4]) > 1e-3
    paired_ids = pair.nonzero().flatten()
    left_ids = (~pair).nonzero().flatten()
    kwargs = {"assignment_mode": assignment_mode, "image_hw": image_hw, "ignore_mask": ignore_mask}
    target = build_anchorfree_detection_targets(locations, boxes6[paired_ids], classes[paired_ids], **kwargs)
    paired_positive = target["labels"] == 1
    target["matched_indices"][paired_positive] = paired_ids[target["matched_indices"][paired_positive]]
    left_boxes = boxes6[left_ids]
    target["labels"].masked_fill_((target["labels"] == 0) & _inside_left_boxes(locations[:, :2], left_boxes).any(1), -1)
    if supervise_left_only and left_ids.numel():
        left_target = build_anchorfree_detection_targets(locations, left_boxes, classes[left_ids], **kwargs)
        take = left_target["left_box_valid_mask"] & ~paired_positive
        take &= ~_image_point_ignore(locations, image_hw, ignore_mask)
        left_target["matched_indices"][take] = left_ids[left_target["matched_indices"][take]]
        for field in target:
            target[field][take] = left_target[field][take]
    return target


@torch.no_grad()
def build_one_to_one_targets(
    predictions: dict[str, Tensor], many: dict[str, Tensor], locations: Tensor,
    *, right_offset_mode: str,
) -> tuple[dict[str, Tensor], Tensor]:
    """Select one regression-valid point per GT from the exclusive dense assignment."""
    eligible = (many["labels"] == 1) & many["box_valid_mask"] & many["left_box_valid_mask"]
    points = eligible.nonzero().flatten()
    winner = torch.zeros_like(eligible)
    quality = locations.new_zeros(locations.shape[0])
    if points.numel():
        gt = many["matched_indices"][points]
        target = many["matched_boxes6"][points]
        pred = decode_anchorfree_boxes6(
            predictions["box6"][points].float(), locations[points], right_offset_mode=right_offset_mode,
        )
        pred = torch.where(torch.isfinite(pred), pred, torch.zeros_like(pred))
        left_pred, right_pred = pair_boxes_to_lr(pred)
        left_gt, right_gt = pair_boxes_to_lr(target)
        left_iou = _elementwise_iou(left_pred, left_gt)
        joint_iou = torch.minimum(left_iou, _elementwise_iou(right_pred, right_gt)).clamp(0, 1)
        class_prob = predictions["class_logits"][points].float().softmax(dim=1)
        class_prob = class_prob.gather(1, many["classes"][points, None]).squeeze(1)
        centers = (target[:, :2] + target[:, 2:4]) * .5
        distance = ((locations[points, :2] - centers) / locations[points, 2:]).square().sum(1)
        alignment = joint_iou * (.5 + .5 * class_prob) + .01 * left_iou + .001 / (1. + distance)
        # First reduce by GT on-device, then choose the lowest dense ID for ties.
        groups, inverse = gt.unique(return_inverse=True)
        best = alignment.new_full((groups.numel(),), -torch.inf)
        best.scatter_reduce_(0, inverse, alignment, reduce="amax")
        winners = points.new_full((groups.numel(),), locations.shape[0])
        choices = alignment == best[inverse]
        winners.scatter_reduce_(0, inverse[choices], points[choices], reduce="amin")
        winner[winners] = True
        quality[points] = torch.where(winner[points], joint_iou, 0)

    # A real left-only object stays ignored by the paired branch. Its siblings
    # with valid paired geometry compete for the one selectable prediction.
    labels = many["labels"].masked_fill((many["labels"] == 1) & ~eligible, -1)
    labels = labels.masked_fill(eligible & ~winner, 0)
    return {
        **many, "labels": labels, "matched_indices": many["matched_indices"].masked_fill(~winner, -1),
        "left_box_valid_mask": winner, "box_valid_mask": winner,
    }, quality



ONE_TO_ONE_ASSIGNMENT_MODES = ("legacy", "global_b1", "global_b2_cross_scale", "global_b2_quality")
_DETECTION_LEVELS = ((8, "selected_p3"), (16, "selected_p4"), (32, "selected_p5"))
ONE_TO_ONE_ASSIGNMENT_COUNT_KEYS = (
    "no_legal", "top12_unmatched", "top12_recovered", "top12_reassigned", "full_pool_conflicts",
    *(name for _, name in _DETECTION_LEVELS),
)


def one_to_one_assignment_mode_from_config(config, override: str | None = None) -> str:
    mode = config.get("one_to_one_assignment_mode", "legacy") if override is None else override
    if mode not in ONE_TO_ONE_ASSIGNMENT_MODES:
        raise ValueError(f"unsupported one-to-one assignment mode: {mode!r}")
    return mode


def _solve_frame_assignment(eligible: Tensor, affinity: Tensor) -> Tensor:
    """Return one local point per GT, preferring cardinality over total affinity."""
    from scipy.optimize import linear_sum_assignment

    gt_count = eligible.shape[1]
    selected = torch.full((gt_count,), -1, dtype=torch.long, device=eligible.device)
    point_count = eligible.shape[0]
    if not gt_count or not point_count:
        return selected
    # Every normalized affinity is at most 1. One more match beats any affinity gain.
    bonus = gt_count + 1
    costs = (bonus + affinity.T.double() / 1.011).masked_fill(~eligible.T, -bonus)
    costs = torch.cat((costs, costs.new_zeros((gt_count, gt_count))), dim=1)
    # GT and dense point order are fixed. SciPy breaks equal-cost ties for that order.
    rows, cols = linear_sum_assignment(costs.cpu().numpy(), maximize=True)
    real = cols < point_count
    selected[torch.as_tensor(rows[real], device=selected.device)] = torch.as_tensor(
        cols[real], device=selected.device,
    )
    return selected


def _rank_quality_edges(affinity: Tensor, eligible: Tensor, distance: Tensor) -> tuple[Tensor, Tensor]:
    """Order within a GT by quality, then distance, then its original dense IDs."""
    center_order = distance.argsort(dim=0, stable=True)
    center_ordered = affinity.masked_fill(~eligible, -torch.inf).gather(0, center_order)
    ranked = center_order.gather(0, center_ordered.argsort(dim=0, descending=True, stable=True))
    scores = affinity.gather(0, ranked)
    valid = eligible.gather(0, ranked)
    neighbors = valid[:-1] & valid[1:]
    equal = neighbors & (scores[:-1] == scores[1:])
    tied = torch.zeros_like(valid)
    tied[:-1] |= equal
    tied[1:] |= equal
    # The Hungarian solver also sees the tie order. Keep the total perturbation
    # smaller than an individual nonzero gap between ranked scores.
    gap = (scores[:-1] - scores[1:]).double()
    limit = scores.new_tensor([1e-8], dtype=torch.float64)
    step = torch.cat((limit, gap[neighbors & (gap > 0)])).amin()
    rank_weight = 1. - torch.arange(ranked.shape[0], device=ranked.device, dtype=torch.float64)[:, None] / max(1, ranked.shape[0])
    bonus_by_rank = tied * rank_weight * (step / (4 * (ranked.shape[1] + 1)))
    bonus = torch.zeros_like(affinity, dtype=torch.float64).scatter_(0, ranked, bonus_by_rank)
    return ranked, affinity.double() + bonus


def _centered_edges(valid_edges: Tensor, distance: Tensor, stride: Tensor, cross_scale: bool) -> Tensor:
    centered = valid_edges & (distance.abs() <= 2.5).all(2)
    if not cross_scale:
        return torch.where(centered.any(0), centered, valid_edges)
    eligible = torch.zeros_like(valid_edges)
    for level_stride, _ in _DETECTION_LEVELS:
        on_level = (stride == level_stride)[:, None]
        centered_level = centered & on_level
        eligible |= torch.where(centered_level.any(0), centered_level, valid_edges & on_level)
    return eligible


def _select_left_only_points(
    predictions: dict[str, Tensor], locations: Tensor, boxes6: Tensor, classes: Tensor,
    legal: Tensor, cross_scale: bool, right_offset_mode: str,
) -> tuple[Tensor, Tensor]:
    active = legal.any(1).nonzero().flatten()
    if not active.numel():
        return active, active
    points = locations[active]
    # Only the left coordinates participate in this pass.
    decoded_left = decode_anchorfree_boxes6(
        predictions["box6"][active].float(), points, right_offset_mode=right_offset_mode,
    )[:, :4]
    probability = predictions["class_logits"][active].float().softmax(dim=1)[:, classes]
    distance = ((points[:, None, :2] - (boxes6[:, :2] + boxes6[:, 2:4]) * .5) / points[:, None, 2:])
    valid_left = torch.isfinite(decoded_left).all(1) & (decoded_left[:, 2:] > decoded_left[:, :2]).all(1)
    eligible = _centered_edges(legal[active] & valid_left[:, None], distance, points[:, 2], cross_scale)
    left_iou = _pairwise_iou(decoded_left, boxes6[:, :4])
    affinity = left_iou * (.5 + .5 * probability) + .001 / (1. + distance.square().sum(2))
    eligible &= torch.isfinite(affinity)
    ranked = affinity.masked_fill(~eligible, -torch.inf).argsort(dim=0, descending=True, stable=True)[:12]
    top12 = torch.zeros_like(eligible).scatter_(0, ranked, True) & eligible
    selected = _solve_frame_assignment(top12, affinity)
    if (eligible.any(0) & (selected < 0)).any():
        selected = _solve_frame_assignment(eligible, affinity)
    assigned = (selected >= 0).nonzero().flatten()
    return active[selected[assigned]], assigned


@torch.no_grad()
def build_global_one_to_one_targets(
    predictions: dict[str, Tensor], boxes6: Tensor, classes: Tensor, locations: Tensor, *,
    image_hw: tuple[int, int], ignore_mask: Tensor | None = None, right_offset_mode: str,
    assignment_mode: str = "global_b1",
    supervise_left_only: bool = False,
) -> tuple[dict[str, Tensor], Tensor, dict[str, int]]:
    """Build independent GT-to-point edges and select one point per target."""
    if assignment_mode not in ONE_TO_ONE_ASSIGNMENT_MODES[1:]:
        raise ValueError(f"unsupported global one-to-one assignment: {assignment_mode!r}")
    cross_scale = assignment_mode == "global_b2_cross_scale"
    boxes6 = boxes6.to(locations)
    classes = classes.to(device=locations.device, dtype=torch.long)
    point = locations[:, :2]
    stride = locations[:, 2]
    blocked = _image_point_ignore(locations, image_hw, ignore_mask)
    left = boxes6[:, :4]
    size = left[:, 2:] - left[:, :2]
    inside = _inside_left_boxes(point, boxes6)
    left_valid = torch.isfinite(left).all(1) & (size > 0).all(1)
    pair_valid = left_valid & torch.isfinite(boxes6[:, 4:]).all(1) & ((boxes6[:, 5] - boxes6[:, 4]) > 1e-3)
    legal_geometry = inside & ~blocked[:, None]
    if cross_scale:
        legal_geometry &= ((stride == 8) | (stride == 16) | (stride == 32))[:, None]
    else:
        legal_geometry &= _point_scale_match(stride, size.amax(1))
    legal = legal_geometry & pair_valid[None, :]
    labels = torch.zeros(locations.shape[0], dtype=torch.long, device=locations.device)
    labels.masked_fill_(blocked | inside.any(1), -1)
    owners = torch.full_like(labels, -1)
    box_target = locations.new_zeros((locations.shape[0], 6))
    raw_target = torch.zeros_like(box_target)
    class_target = torch.zeros_like(labels)
    quality = locations.new_zeros(locations.shape[0])
    counts = dict.fromkeys(ONE_TO_ONE_ASSIGNMENT_COUNT_KEYS, 0)
    counts["no_legal"] = len(boxes6)

    active_points = legal.any(1).nonzero().flatten()
    if active_points.numel():
        decoded = decode_anchorfree_boxes6(
            predictions["box6"][active_points].float(), locations[active_points], right_offset_mode=right_offset_mode,
        )
        probability = predictions["class_logits"][active_points].float().softmax(dim=1)
        valid_prediction = torch.isfinite(decoded).all(1) & torch.isfinite(probability).all(1)
        valid_prediction &= (decoded[:, 2:4] > decoded[:, :2]).all(1) & (decoded[:, 5] > decoded[:, 4])
        active_points = active_points[valid_prediction]
        decoded = decoded[valid_prediction]
        probability = probability[valid_prediction]
        distance = (
            (locations[active_points, None, :2] - (left[:, :2] + left[:, 2:]) * .5)
            / stride[active_points, None, None]
        )
        valid_edges = legal[active_points]
        eligible = _centered_edges(valid_edges, distance, stride[active_points], cross_scale)
        pred_left, pred_right = pair_boxes_to_lr(decoded)
        gt_left, gt_right = pair_boxes_to_lr(boxes6)
        left_iou = _pairwise_iou(pred_left, gt_left)
        joint_iou = torch.minimum(left_iou, _pairwise_iou(pred_right, gt_right)).clamp(0, 1)
        center_distance = distance.square().sum(2)
        class_probability = probability[:, classes]
        if assignment_mode == "global_b2_quality":
            mean_iou = .5 * (left_iou + joint_iou)
            affinity = class_probability.sqrt() * mean_iou.square()
            eligible &= torch.isfinite(affinity)
            ranked, affinity = _rank_quality_edges(affinity, eligible, center_distance)
        else:
            affinity = joint_iou * (.5 + .5 * class_probability) + .01 * left_iou
            affinity = affinity + .001 / (1. + center_distance)
            eligible &= torch.isfinite(affinity)
            ranked = affinity.masked_fill(~eligible, -torch.inf).argsort(dim=0, descending=True, stable=True)
        labels[active_points[eligible.any(1)]] = 0
        top12 = torch.zeros_like(eligible).scatter_(0, ranked[:12], True) & eligible
        first = _solve_frame_assignment(top12, affinity)
        gt_has_edge = eligible.any(0)
        missing = gt_has_edge & (first < 0)
        selected = _solve_frame_assignment(eligible, affinity) if missing.any() else first
        counts["no_legal"] = int((~gt_has_edge).sum())
        counts["top12_unmatched"] = int(missing.sum())
        counts["full_pool_conflicts"] = int((gt_has_edge & (selected < 0)).sum())
        counts["top12_recovered"] = counts["top12_unmatched"] - counts["full_pool_conflicts"]
        counts["top12_reassigned"] = int(((first >= 0) & (selected != first)).sum())
        assigned_gt = (selected >= 0).nonzero().flatten()
        local_points = selected[assigned_gt]
        matched_points = active_points[local_points]
        owners[matched_points] = assigned_gt
        labels[matched_points] = 1
        raw_target[matched_points] = boxes6[assigned_gt]
        class_target[matched_points] = classes[assigned_gt]
        box_target[matched_points] = encode_anchorfree_boxes6(raw_target[matched_points], locations[matched_points])
        quality[matched_points] = joint_iou[local_points, assigned_gt]
        selected_strides = stride[matched_points]
        level_counts = torch.stack([(selected_strides == level).sum() for level, _ in _DETECTION_LEVELS]).tolist()
        counts.update({name: count for (_, name), count in zip(_DETECTION_LEVELS, level_counts)})

    paired_positive = labels == 1
    if supervise_left_only:
        left_ids = (left_valid & (boxes6[:, 4] == boxes6[:, 5])).nonzero().flatten()
        points, local_gt = _select_left_only_points(
            predictions, locations, boxes6[left_ids], classes[left_ids],
            legal_geometry[:, left_ids] & ~paired_positive[:, None], cross_scale, right_offset_mode,
        )
        gt_ids = left_ids[local_gt]
        owners[points] = gt_ids
        labels[points] = 1
        raw_target[points] = boxes6[gt_ids]
        class_target[points] = classes[gt_ids]
        box_target[points] = encode_anchorfree_boxes6(raw_target[points], locations[points])
        counts["left_only_selected"] = points.numel()
    return {
        "labels": labels, "matched_indices": owners, "classes": class_target,
        "box6": box_target, "matched_boxes6": raw_target,
        "left_box_valid_mask": labels == 1, "box_valid_mask": paired_positive,
    }, quality, counts


def one_to_one_duplicate_loss(
    objectness: Tensor, many: dict[str, Tensor], one: dict[str, Tensor], margin: float = 1.,
) -> Tensor:
    """Separate each selected point from its highest-scoring sibling, per GT."""
    selected = one["labels"] == 1
    winner = selected & one.get("box_valid_mask", selected)
    siblings = (
        (many["labels"] == 1) & many["box_valid_mask"] & many["left_box_valid_mask"] & ~selected
    )
    if not siblings.any():
        return objectness.sum() * 0
    owners = many["matched_indices"]
    winner_owners = one.get("matched_indices", owners)[winner]
    same_target = winner_owners[:, None] == owners[siblings][None, :]
    has_sibling = same_target.any(dim=1)
    hardest = objectness[siblings][None, :].expand_as(same_target)
    hardest = hardest.masked_fill(~same_target, -torch.inf).amax(dim=1)
    penalty = (margin + hardest[has_sibling] - objectness[winner][has_sibling]).relu()
    return penalty.sum() / has_sibling.sum().clamp_min(1)


def _frame_predictions(dense) -> dict[str, Tensor]:
    return {"objectness": dense.objectness[0], "class_logits": dense.class_logits[0], "box6": dense.box6[0]}


def training_forward_anchorfree(
    encoder: nn.Module,
    detector: nn.Module,
    instance_head: nn.Module,
    left: Tensor,
    right: Tensor,
    target: dict[str, Tensor],
    dmax_roi: int,
    roi_margin: float = 16.,
    detection_weight: float = 1.,
    mask_weight: float = 1.,
    disparity_weight: float = 1.,
    supervision_mode: str = "foreground",
    collect_metrics: bool = False,
    *,
    right_offset_mode: str = "legacy_clamp",
    assignment_mode: str = "legacy",
    box_supervision_mode: str = "joint",
    left_iou_weight: float = 0.,
    one_to_one_rank_weight: float = 0.,
    one_to_one_score_target: str = "floor",
    one_to_one_assignment_mode: str = "legacy",
    car_left_only_supervision: str = "legacy",
) -> dict[str, object]:
    if left.shape[0] != 1:
        raise ValueError("GT-RoI training currently accepts one stereo pair per step")
    pyramid = encoder(left, right)
    sparse = None
    if getattr(detector, "one_to_one_enabled", False):
        dense, sparse = detector.forward_training(pyramid)
    else:
        dense = detector(pyramid)
    locations = generate_point_locations(point_levels(pyramid), left.device)
    boxes6 = target["boxes6"].to(left.device)
    detection_boxes = target.get("detection_boxes6", target["boxes6"]).to(left.device)
    detection_classes = target.get("detection_classes", target["classes"]).to(left.device)
    car_control = "detection_boxes6" in target and car_left_only_supervision in ("off", "on")
    supervise_left_only = car_control and car_left_only_supervision == "on"
    dense_kwargs = {
        "assignment_mode": assignment_mode, "image_hw": left.shape[-2:],
        "ignore_mask": target.get("detection_ignore_mask"),
    }
    if car_control:
        detection_targets = build_car_supervision_targets(
            locations, detection_boxes, detection_classes, supervise_left_only=supervise_left_only, **dense_kwargs,
        )
    else:
        detection_targets = build_anchorfree_detection_targets(
            locations, detection_boxes, detection_classes, **dense_kwargs,
        )
    detection = anchorfree_detection_loss(
        _frame_predictions(dense),
        detection_targets,
        locations,
        right_offset_mode=right_offset_mode,
        box_supervision_mode=box_supervision_mode,
        left_iou_weight=left_iou_weight,
    )
    sparse_metrics = {}
    if sparse is not None:
        predicted = _frame_predictions(sparse)
        mode = one_to_one_assignment_mode_from_config({}, one_to_one_assignment_mode)
        assignment_counts = {}
        if mode != "legacy":
            sparse_targets, quality, assignment_counts = build_global_one_to_one_targets(
                predicted, detection_boxes, detection_classes, locations, image_hw=left.shape[-2:],
                ignore_mask=target.get("detection_ignore_mask"), right_offset_mode=right_offset_mode,
                assignment_mode=mode, supervise_left_only=supervise_left_only,
            )
        else:
            sparse_targets, quality = build_one_to_one_targets(
                predicted, detection_targets, locations, right_offset_mode=right_offset_mode,
            )
        loss_one = anchorfree_detection_loss(
            predicted, sparse_targets, locations, right_offset_mode=right_offset_mode,
            box_supervision_mode=box_supervision_mode, one_to_one_quality=quality,
            one_to_one_score_target=one_to_one_score_target,
            left_iou_weight=left_iou_weight,
        )
        rank_loss = None
        if one_to_one_rank_weight:
            rank_loss = one_to_one_duplicate_loss(predicted["objectness"], detection_targets, sparse_targets)
            loss_one["loss_obj"] = loss_one["loss_obj"] + one_to_one_rank_weight * rank_loss
        sparse_metrics = {
            "one_to_one_positives": (sparse_targets["labels"] == 1).sum(),
            "loss_one_to_one": sum(loss_one[key] for key in ("loss_obj", "loss_cls", "loss_box6")).detach(),
        }
        if car_control:
            provided_pairs = int(((detection_boxes[:, 5] - detection_boxes[:, 4]) > 1e-3).sum())
            dense_left_only = detection_targets["left_box_valid_mask"] & ~detection_targets["box_valid_mask"]
            sparse_left_only = sparse_targets["left_box_valid_mask"] & ~sparse_targets["box_valid_mask"]
            sparse_metrics["car_supervision_counts"] = {
                "provided_pairs": provided_pairs, "provided_left_only": len(detection_boxes) - provided_pairs,
                "dense_left_only_positions": int(dense_left_only.sum()),
                "sparse_left_only_positions": int(sparse_left_only.sum()),
                "sparse_paired_positions": int(sparse_targets["box_valid_mask"].sum()),
            }
        if assignment_counts:
            sparse_metrics["one_to_one_assignment_counts"] = assignment_counts
        if rank_loss is not None:
            sparse_metrics["loss_one_to_one_rank"] = rank_loss.detach()
        detection = {key: (value + loss_one[key]) * .5 for key, value in detection.items()}
    zero = dense.objectness.sum() * 0
    metrics = {"iou_sum": 0., "iou_count": 0, "epe_sum": 0., "epe_count": 0, "d1_hit": 0, "d1_count": 0}
    if mask_weight == 0. and disparity_weight == 0.:
        loss_detector = detection["loss_obj"] + detection["loss_cls"] + detection["loss_box6"]
        return {
            "loss_obj": detection["loss_obj"], "loss_cls": detection["loss_cls"], "loss_box6": detection["loss_box6"],
            "loss_left": detection["loss_left"], "loss_disparity": detection["loss_disparity"],
            "loss_right_width": detection["loss_right_width"], "loss_pair_iou": detection["loss_pair_iou"],
            "loss_left_iou": detection["loss_left_iou"],
            "loss_mask": zero, "loss_disp": zero, "loss_total": detection_weight * loss_detector, "metrics": metrics, **sparse_metrics,
        }

    if boxes6.numel():
        roi_pyramid = repeat_roi_features(pyramid, boxes6.shape[0])
        instance_output = instance_head(roi_pyramid, boxes6, left.shape[-2], left.shape[-1], margin=roi_margin)
        supervision = build_instance_supervision(
            instance_output, boxes6, target["masks"].to(left.device),
            target["disparity"].to(left.device).unsqueeze(1),
            target["disparity_valid"].to(left.device).unsqueeze(1), dmax_roi, supervision_mode,
        )
        loss_mask = mask_loss(instance_output.mask_logits, supervision.mask_target, supervision.mask_validity)
        loss_disp = disparity_loss(instance_output.disparity, supervision.disparity_target[:, 0], supervision.disparity_valid[:, 0])
        if collect_metrics:
            metrics = instance_metrics(instance_output, supervision, target["masks"].to(left.device), dmax_roi)
    else:
        loss_mask, loss_disp = zero, zero
    loss_detector = detection["loss_obj"] + detection["loss_cls"] + detection["loss_box6"]
    total = detection_weight * loss_detector + mask_weight * loss_mask + disparity_weight * loss_disp
    return {
        "loss_obj": detection["loss_obj"], "loss_cls": detection["loss_cls"], "loss_box6": detection["loss_box6"],
        "loss_left": detection["loss_left"], "loss_disparity": detection["loss_disparity"],
        "loss_right_width": detection["loss_right_width"], "loss_pair_iou": detection["loss_pair_iou"],
        "loss_left_iou": detection["loss_left_iou"],
        "loss_mask": loss_mask, "loss_disp": loss_disp, "loss_total": total, "metrics": metrics, **sparse_metrics,
    }

