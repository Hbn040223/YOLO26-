"""Left-anchor-assigned, pair-preserving stereo detection primitives."""
from __future__ import annotations

from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from .geometry import decode_boxes6


_ANCHOR_CACHE_LIMIT = 32
_ANCHOR_CACHE: "OrderedDict[tuple[tuple[tuple[int, int, int, float], ...], str], Tensor]" = OrderedDict()


@dataclass(frozen=True)
class Assignment:
    labels: Tensor  # 1 positive, 0 negative, -1 ignore
    matched_indices: Tensor


@dataclass(frozen=True)
class PairDetections:
    boxes6: Tensor
    scores: Tensor
    indices: Tensor
    classes: Tensor | None = None


@dataclass(frozen=True)
class ContainmentEvent:
    suppressor_id: int
    suppressed_id: int
    left_coverage: float
    right_coverage: float


def suppress_contained_pairs(
    pairs: PairDetections, threshold: float = .90,
) -> tuple[PairDetections, tuple[ContainmentEvent, ...]]:
    """Greedily remove lower-scoring stereo pairs covered by one larger pair."""
    if not 0 < threshold <= 1:
        raise ValueError("containment threshold must be in (0, 1]")
    if pairs.scores.numel() < 2:
        return pairs, ()
    if pairs.classes is None:
        raise ValueError("containment suppression requires candidate classes")

    order = pairs.scores.argsort(descending=True, stable=True)
    # The greedy decision is sequential. Transfer the compact candidate table
    # in one pass and compute coverage in FP32 even with FP16 inference.
    rows = torch.cat((pairs.boxes6[order].float(), pairs.scores[order, None].float(),
                      pairs.classes[order, None]), dim=1)
    rows = rows.detach().cpu().numpy()
    boxes = np.stack((rows[:, :4], rows[:, [4, 1, 5, 3]]), axis=1)
    sizes = boxes[:, :, 2:] - boxes[:, :, :2]
    areas = np.maximum(sizes, 0).prod(axis=-1)
    valid = np.isfinite(boxes).all(axis=(1, 2)) & (sizes > 0).all(axis=(1, 2))
    suppressed = np.zeros(len(rows), dtype=bool)
    keep = []
    events = []
    for index, row in enumerate(rows):
        if suppressed[index]:
            continue
        keep.append(index)
        if not valid[index]:
            continue
        candidates = np.flatnonzero(valid & ~suppressed & (rows[:, 6] < row[6])
                                    & (rows[:, 7] == row[7]) & (areas < areas[index]).all(axis=1))
        small = boxes[candidates]
        large = boxes[index]
        overlap = np.minimum(small[:, :, 2:], large[:, 2:]) - np.maximum(small[:, :, :2], large[:, :2])
        coverage = np.maximum(overlap, 0).prod(axis=-1) / areas[candidates]
        contained = (coverage >= threshold).all(axis=1)
        suppressed[candidates[contained]] = True
        for removed, (left, right) in zip(candidates[contained], coverage[contained]):
            events.append((index, removed, float(left), float(right)))
    selected = order[torch.as_tensor(keep, dtype=torch.long, device=order.device)]
    candidate_ids = pairs.indices[order].tolist() if events else []
    records = tuple(ContainmentEvent(candidate_ids[large], candidate_ids[small], left, right)
                    for large, small, left, right in events)
    return PairDetections(pairs.boxes6[selected], pairs.scores[selected], pairs.indices[selected],
                          pairs.classes[selected]), records


@dataclass(frozen=True)
class DenseStereoPredictions:
    """One joint objectness/class/6D prediction at every stereo feature cell."""

    objectness: Tensor
    class_logits: Tensor
    box6: Tensor


class StereoYoloHead(nn.Module):
    """Concat-baseline detector that reuses P3/P4/C5 from LightStereo directly."""

    def __init__(self, in_channels: int, num_classes: int, hidden_channels: int | None = None) -> None:
        super().__init__()
        hidden_channels = hidden_channels or in_channels
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
        )
        self.objectness_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.class_head = nn.Conv2d(hidden_channels, num_classes, kernel_size=1)
        self.box6_head = nn.Conv2d(hidden_channels, 6, kernel_size=1)

    def forward(self, pyramid: object) -> DenseStereoPredictions:
        objectness, classes, boxes = [], [], []
        for left_name, right_name in (("p3_left", "p3_right"), ("p4_left", "p4_right"), ("c5_left", "c5_right")):
            fused = self.fusion(torch.cat((getattr(pyramid, left_name), getattr(pyramid, right_name)), dim=1))
            objectness.append(self.objectness_head(fused).flatten(1))
            classes.append(self.class_head(fused).flatten(2).transpose(1, 2))
            boxes.append(self.box6_head(fused).flatten(2).transpose(1, 2))
        return DenseStereoPredictions(torch.cat(objectness, 1), torch.cat(classes, 1), torch.cat(boxes, 1))


def _iou(left: Tensor, target: Tensor) -> Tensor:
    tl = torch.maximum(left[:, None, :2], target[None, :, :2])
    br = torch.minimum(left[:, None, 2:], target[None, :, 2:])
    inter = (br - tl).clamp_min(0).prod(-1)
    area_l = (left[:, 2] - left[:, 0]).clamp_min(0) * (left[:, 3] - left[:, 1]).clamp_min(0)
    area_t = (target[:, 2] - target[:, 0]).clamp_min(0) * (target[:, 3] - target[:, 1]).clamp_min(0)
    return inter / (area_l[:, None] + area_t[None, :] - inter).clamp_min(1e-8)


def assign_left_iou_targets(anchors: Tensor, targets6: Tensor, positive_iou: float = .5, negative_iou: float = .4) -> Assignment:
    if not 0 <= negative_iou < positive_iou <= 1:
        raise ValueError("require 0 <= negative_iou < positive_iou <= 1")
    labels = torch.full((anchors.shape[0],), -1, dtype=torch.long, device=anchors.device)
    matched = torch.full_like(labels, -1)
    if targets6.numel() == 0:
        return Assignment(torch.zeros_like(labels), matched)
    overlaps = _iou(anchors, targets6[:, :4])
    quality, best = overlaps.max(dim=1)
    labels[quality < negative_iou] = 0
    labels[quality >= positive_iou] = 1
    matched[labels == 1] = best[labels == 1]
    _, forced = overlaps.max(dim=0)
    labels[forced] = 1
    matched[forced] = torch.arange(targets6.shape[0], device=anchors.device)
    return Assignment(labels, matched)


def detection_loss(predictions: dict[str, Tensor], targets: dict[str, Tensor]) -> dict[str, Tensor]:
    labels = targets["labels"]
    positive = labels == 1
    active = labels >= 0
    obj_target = positive.to(predictions["objectness"].dtype)
    loss_obj = F.binary_cross_entropy_with_logits(predictions["objectness"][active], obj_target[active]) if active.any() else predictions["objectness"].sum() * 0
    loss_cls = F.cross_entropy(predictions["class_logits"][positive], targets["classes"][positive]) if positive.any() else predictions["class_logits"].sum() * 0
    loss_box6 = F.smooth_l1_loss(predictions["box6"][positive], targets["box6"][positive]) if positive.any() else predictions["box6"].sum() * 0
    return {"loss_obj": loss_obj, "loss_cls": loss_cls, "loss_box6": loss_box6}


def paired_nms(
    boxes6: Tensor, scores: Tensor, iou_threshold: float = .5, *, classes: Tensor | None = None, profile=None,
) -> PairDetections:
    """Suppress one whole pair only when both views overlap above the threshold."""
    if not 0 <= iou_threshold <= 1:
        raise ValueError("paired NMS IoU must be between 0 and 1")
    order = scores.argsort(descending=True, stable=True)
    indices = order
    count = order.numel()
    if count > 1:
        span = profile or nullcontext
        with torch.no_grad():
            sorted_boxes = boxes6[order]
            left = sorted_boxes[:, :4]
            right = sorted_boxes[:, [4, 1, 5, 3]]
            sorted_classes = classes[order] if classes is not None else None
            suppressed = np.zeros(count, dtype=bool)
            kept = []
            # The normal capped pool fits in one block. Bound temporary memory
            # for callers that turn the pre-NMS candidate cap off.
            rows_per_block = max(1, 1024**2 // count)
            for start in range(0, count, rows_per_block):
                stop = min(start + rows_per_block, count)
                with span("paired_iou"):
                    paired_iou = torch.minimum(_iou(left[start:stop], left[start:]),
                                               _iou(right[start:stop], right[start:]))
                    removes = ~(paired_iou <= iou_threshold)
                    if sorted_classes is not None:
                        removes &= sorted_classes[start:stop, None] == sorted_classes[None, start:]
                # Transfer one boolean mask per block. Greedy suppression uses
                # score order on the CPU without a per-candidate CUDA sync.
                with span("decision_mask_d2h"):
                    removes_cpu = removes.cpu().numpy()
                with span("greedy_cpu"):
                    for index in range(start, stop):
                        if not suppressed[index]:
                            kept.append(index)
                            suppressed[index + 1:] |= removes_cpu[index - start, index + 1 - start:]
            indices = order[torch.as_tensor(kept, dtype=torch.long, device=order.device)]
    return PairDetections(boxes6[indices], scores[indices], indices,
                          classes[indices] if classes is not None else None)


def decode_dense_predictions(
    predictions: dict[str, Tensor],
    anchors: Tensor,
    score_threshold: float = .05,
    nms_threshold: float = .5,
    pre_nms_topk: int = 500,
) -> PairDetections:
    """Decode one image's dense stereo predictions; empty inputs stay empty pairs."""
    if pre_nms_topk <= 0:
        raise ValueError("pre_nms_topk must be positive")
    objectness = predictions["objectness"].reshape(-1)
    classes = predictions["class_logits"].reshape(objectness.numel(), -1)
    box6 = predictions["box6"].reshape(objectness.numel(), 6)
    class_probability, class_index = classes.softmax(1).max(1)
    scores = objectness.sigmoid() * class_probability
    chosen = scores >= score_threshold
    if not chosen.any():
        empty = anchors.new_empty((0, 6))
        return PairDetections(empty, scores.new_empty((0,)), torch.empty((0,), device=anchors.device, dtype=torch.long), class_index.new_empty((0,)))
    chosen_indices = chosen.nonzero().flatten()
    if chosen_indices.numel() > pre_nms_topk:
        _, top_order = scores[chosen_indices].topk(pre_nms_topk)
        chosen_indices = chosen_indices[top_order]
    decoded = decode_boxes6(box6[chosen_indices], anchors[chosen_indices])
    kept = paired_nms(decoded, scores[chosen_indices], nms_threshold, classes=class_index[chosen_indices])
    return PairDetections(kept.boxes6, kept.scores, chosen_indices[kept.indices], class_index[chosen_indices][kept.indices])


def generate_grid_anchors(levels: tuple[tuple[int, int, int, float], ...], device: torch.device | None = None) -> Tensor:
    """One static square left-view anchor per P3/P4/C5 feature cell.

    Each level is `(height, width, stride, anchor_size)` in image pixels.
    """
    resolved_device = torch.device("cpu") if device is None else torch.device(device)
    normalized_levels = tuple((int(height), int(width), int(stride), float(size)) for height, width, stride, size in levels)
    key = normalized_levels, str(resolved_device)
    cached = _ANCHOR_CACHE.get(key)
    if cached is not None:
        _ANCHOR_CACHE.move_to_end(key)
        return cached
    anchors: list[Tensor] = []
    for height, width, stride, size in normalized_levels:
        y, x = torch.meshgrid(torch.arange(height, device=resolved_device), torch.arange(width, device=resolved_device), indexing="ij")
        cx, cy = (x.reshape(-1) + .5) * stride, (y.reshape(-1) + .5) * stride
        half = size * .5
        anchors.append(torch.stack((cx - half, cy - half, cx + half, cy + half), 1).float())
    result = (torch.cat(anchors, 0) if anchors else torch.empty((0, 4), device=resolved_device)).detach()
    _ANCHOR_CACHE[key] = result
    _ANCHOR_CACHE.move_to_end(key)
    if len(_ANCHOR_CACHE) > _ANCHOR_CACHE_LIMIT:
        _ANCHOR_CACHE.popitem(last=False)
    return result



def _elementwise_iou(boxes: Tensor, targets: Tensor) -> Tensor:
    top_left = torch.maximum(boxes[:, :2], targets[:, :2])
    bottom_right = torch.minimum(boxes[:, 2:], targets[:, 2:])
    inter = (bottom_right - top_left).clamp_min(0).prod(dim=1)
    area_boxes = (boxes[:, 2] - boxes[:, 0]).clamp_min(0) * (boxes[:, 3] - boxes[:, 1]).clamp_min(0)
    area_targets = (targets[:, 2] - targets[:, 0]).clamp_min(0) * (targets[:, 3] - targets[:, 1]).clamp_min(0)
    return inter / (area_boxes + area_targets - inter).clamp_min(1e-8)


BOX_SUPERVISION_MODES = ("joint", "split_left")
ONE_TO_ONE_SCORE_TARGETS = ("floor", "joint_iou")


def box_supervision_mode_from_config(config, override: str | None = None) -> str:
    mode = config.get("box_supervision_mode", "joint") if override is None else override
    if mode not in BOX_SUPERVISION_MODES:
        raise ValueError(f"unsupported box supervision mode: {mode!r}")
    return mode


def anchorfree_detection_loss(
    predictions: dict[str, Tensor], targets: dict[str, Tensor], locations: Tensor,
    *, right_offset_mode: str = "legacy_clamp", box_supervision_mode: str = "joint",
    one_to_one_quality: Tensor | None = None,
    one_to_one_score_target: str = "floor",
    left_iou_weight: float = 0.,
) -> dict[str, Tensor]:
    """Regress left and paired coordinates on their own valid points in split mode."""
    from .geometry import decode_anchorfree_boxes6, pair_boxes_to_lr

    if not 0 <= left_iou_weight < float("inf"):
        raise ValueError("left IoU loss weight must be finite and nonnegative")
    labels = targets["labels"]
    positive = labels == 1
    active = labels >= 0
    paired_valid = positive & targets["box_valid_mask"].bool()
    if box_supervision_mode == "split_left":
        left_valid = positive & targets["left_box_valid_mask"].bool()
        paired_valid = paired_valid & left_valid
    elif box_supervision_mode == "joint":
        left_valid = paired_valid
    else:
        raise ValueError(f"unsupported box supervision mode: {box_supervision_mode!r}")
    zero = predictions["box6"].sum() * 0
    loss_obj = F.binary_cross_entropy_with_logits(
        predictions["objectness"][active], positive[active].to(predictions["objectness"].dtype)
    ) if active.any() else predictions["objectness"].sum() * 0
    if one_to_one_quality is not None:
        logits = predictions["objectness"]
        quality = one_to_one_quality.detach()[paired_valid].to(logits.dtype)
        if one_to_one_score_target == "floor":
            positive_target = .5 + .5 * quality
        elif one_to_one_score_target == "joint_iou":
            positive_target = quality
        else:
            raise ValueError(f"unsupported one-to-one score target: {one_to_one_score_target!r}")
        positive_bce = F.binary_cross_entropy_with_logits(
            logits[paired_valid], positive_target, reduction="sum",
        ) / paired_valid.sum().clamp_min(1)
        negatives = F.softplus(logits[labels == 0])
        keep = min(negatives.numel(), max(128, 16 * positive_target.numel()))
        negative_bce = negatives.topk(keep).values.sum() / max(keep, 1)
        loss_obj = positive_bce + negative_bce
    loss_cls = F.cross_entropy(predictions["class_logits"][positive], targets["classes"][positive]) if positive.any() else predictions["class_logits"].sum() * 0
    if not paired_valid.any():
        loss_left = loss_disp = loss_width = loss_pair_iou = loss_left_iou = zero
    else:
        predicted = predictions["box6"][paired_valid]
        expected = targets["box6"][paired_valid].to(dtype=predicted.dtype)
        loss_left = F.smooth_l1_loss(predicted[:, :4], expected[:, :4])
        loss_disp = F.smooth_l1_loss(predicted[:, 4], expected[:, 4])
        loss_width = F.smooth_l1_loss(predicted[:, 5], expected[:, 5])
        decoded_prediction = decode_anchorfree_boxes6(
            predicted, locations[paired_valid], right_offset_mode=right_offset_mode,
        )
        if right_offset_mode == "signed":
            decoded_target = targets["matched_boxes6"][paired_valid].to(decoded_prediction)
            if not torch.isfinite(decoded_prediction).all():
                raise FloatingPointError("non-finite decoded anchor-free training box")
        else:
            decoded_target = decode_anchorfree_boxes6(expected, locations[paired_valid])
        pred_left, pred_right = pair_boxes_to_lr(decoded_prediction)
        target_left, target_right = pair_boxes_to_lr(decoded_target)
        left_iou = _elementwise_iou(pred_left, target_left)
        loss_left_iou = 1. - left_iou.mean()
        loss_pair_iou = 1. - torch.minimum(left_iou, _elementwise_iou(pred_right, target_right)).mean()
    if box_supervision_mode == "split_left" and (left_valid & ~paired_valid).any():
        # Preserve the original paired loss calculation when the two valid sets match.
        predicted_left = predictions["box6"][left_valid, :4]
        expected_left = targets["box6"][left_valid, :4].to(predicted_left)
        loss_left = F.smooth_l1_loss(predicted_left, expected_left)
    loss_box6 = (loss_left + loss_disp + loss_width) / 3. + loss_pair_iou
    if left_iou_weight:
        loss_box6 = loss_box6 + left_iou_weight * loss_left_iou
    return {
        "loss_obj": loss_obj,
        "loss_cls": loss_cls,
        "loss_box6": loss_box6,
        "loss_left": loss_left.detach(),
        "loss_disparity": loss_disp.detach(),
        "loss_right_width": loss_width.detach(),
        "loss_pair_iou": loss_pair_iou.detach(),
        "loss_left_iou": loss_left_iou.detach(),
    }


def anchorfree_prediction_scores(predictions: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
    objectness = predictions["objectness"].reshape(-1)
    if not objectness.numel():
        return objectness, objectness.new_empty((0,), dtype=torch.long)
    logits = predictions["class_logits"].reshape(objectness.numel(), -1)
    probability, classes = logits.softmax(dim=1).max(dim=1)
    return objectness.sigmoid() * probability, classes


def anchorfree_candidate_indices(scores: Tensor, score_threshold: float, pre_nms_topk: int) -> tuple[Tensor, Tensor]:
    score_indices = (scores >= score_threshold).nonzero().flatten()
    topk_indices = score_indices
    if pre_nms_topk > 0 and score_indices.numel() > pre_nms_topk:
        # nonzero supplies ascending dense IDs, including ties at the cutoff.
        order = scores[score_indices].argsort(descending=True, stable=True)
        topk_indices = score_indices[order[:pre_nms_topk]]
    return score_indices, topk_indices


def suppress_anchorfree_candidates(
    decoded: Tensor, indices: Tensor, scores: Tensor, classes: Tensor, nms_threshold: float,
    *, apply_nms: bool = True, class_aware_nms: bool = True,
) -> tuple[PairDetections, PairDetections]:
    # Finite encoded values can still overflow in expm1 or sinh.
    valid = torch.isfinite(decoded).all(dim=1)
    indices = indices[valid]
    pre = PairDetections(decoded[valid], scores[indices], indices, classes[indices])
    return select_anchorfree_pairs(pre, nms_threshold, apply_nms=apply_nms,
                                   class_aware_nms=class_aware_nms), pre


def select_anchorfree_pairs(
    pre: PairDetections, nms_threshold: float, *, apply_nms: bool, class_aware_nms: bool = True,
) -> PairDetections:
    """Run both ablation arms from the same valid pre-NMS pool."""
    if not apply_nms:
        order = pre.scores.argsort(descending=True, stable=True)
    else:
        order = paired_nms(pre.boxes6, pre.scores, nms_threshold,
                           classes=pre.classes if class_aware_nms else None).indices
    classes = pre.classes[order] if pre.classes is not None else None
    return PairDetections(pre.boxes6[order], pre.scores[order], pre.indices[order], classes)


def decode_anchorfree_predictions(
    predictions: dict[str, Tensor],
    locations: Tensor,
    score_threshold: float = .05,
    nms_threshold: float = .5,
    pre_nms_topk: int = 500,
    *,
    right_offset_mode: str = "legacy_clamp",
    apply_nms: bool = True,
    class_aware_nms: bool = True,
) -> PairDetections:
    """Decode aligned stereo pairs, optionally applying pair-preserving NMS."""
    from .geometry import decode_anchorfree_boxes6

    scores, classes = anchorfree_prediction_scores(predictions)
    if locations.shape[0] != scores.numel():
        raise ValueError("locations must match flattened predictions")
    _, chosen = anchorfree_candidate_indices(scores, score_threshold, pre_nms_topk)
    decoded = decode_anchorfree_boxes6(
        predictions["box6"].reshape(-1, 6)[chosen], locations[chosen], right_offset_mode=right_offset_mode,
    )
    final, _ = suppress_anchorfree_candidates(
        decoded, chosen, scores, classes, nms_threshold, apply_nms=apply_nms, class_aware_nms=class_aware_nms,
    )
    return final
