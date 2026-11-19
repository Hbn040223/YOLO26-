"""Read-only detection score diagnostics on fixed decoded candidates."""
from __future__ import annotations

import torch
from torch import Tensor

from .geometry import pair_boxes_to_lr
from .training_metrics import _pairwise_iou


def audit_score_candidates(
    boxes: Tensor, objectness: Tensor, class_probabilities: Tensor,
    targets: Tensor, target_labels: Tensor, threshold: float = .25,
) -> list[dict]:
    """Attribute low scores and check score ordering among left-localized candidates."""
    probable, predicted_classes = class_probabilities.max(dim=1)
    scores = objectness * probable
    valid = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores)
    left, right = pair_boxes_to_lr(boxes)
    gt_left, gt_right = pair_boxes_to_lr(targets)
    left_iou = _pairwise_iou(left, gt_left)
    right_iou = _pairwise_iou(right, gt_right)
    quality = torch.minimum(left_iou, right_iou)
    rows = []

    def describe(pred: int, gt: int) -> dict:
        gt_class = int(target_labels[gt])
        return {"dense_index": pred, "objectness": float(objectness[pred]),
                "best_class_probability": float(probable[pred]),
                "gt_class_probability": float(class_probabilities[pred, gt_class]),
                "pred_class": int(predicted_classes[pred]), "final_score": float(scores[pred]),
                "iou_left": float(left_iou[pred, gt]), "iou_right": float(right_iou[pred, gt]),
                "joint_iou": float(quality[pred, gt])}

    def best_by_score(mask: Tensor, gt: int) -> dict | None:
        indices = mask.nonzero().flatten()
        if not indices.numel():
            return None
        return describe(int(indices[scores[indices].argmax()]), gt)

    for gt in range(len(targets)):
        good_left = valid & (left_iou[:, gt] >= .5)
        good_joint = good_left & (right_iou[:, gt] >= .5)
        best_left = best_by_score(good_left, gt)
        best_joint = best_by_score(good_joint, gt)
        best_bad_right = best_by_score(good_left & ~good_joint, gt)
        best_geometry = None
        if good_left.any():
            max_quality = quality[good_left, gt].amax()
            best_geometry = best_by_score(good_left & (quality[:, gt] == max_quality), gt)
        max_joint_obj = None
        if best_joint is None:
            blocker = "no_joint_geometry"
        else:
            max_joint_obj = float(objectness[good_joint].max())
            if best_joint["final_score"] >= threshold:
                blocker = "score_pass"
            elif max_joint_obj < threshold:
                blocker = "objectness_below_threshold"
            else:
                blocker = "product_below_threshold"
        comparable = best_joint is not None and best_bad_right is not None
        rows.append({"gt_index": gt, "gt_class": int(target_labels[gt]),
                     "left_filtered_by_score": best_left is not None and best_left["final_score"] < threshold,
                     "joint_score_blocker": blocker,
                     "best_left_score": best_left, "best_joint_score": best_joint,
                     "best_joint_geometry": best_geometry, "best_left_only_score": best_bad_right,
                     "max_joint_objectness": max_joint_obj,
                     "comparable_rank": comparable,
                     "bad_right_outranks_joint": comparable and best_bad_right["final_score"] > best_joint["final_score"]})
    return rows


def budget_keep_indices(frames: list[Tensor], budget: int) -> list[Tensor]:
    """Select globally strongest post-NMS predictions, breaking ties by frame and index."""
    if budget < 0:
        raise ValueError("candidate budget must be non-negative")
    if not frames:
        return []
    all_scores = torch.cat(frames)
    selected = torch.zeros_like(all_scores, dtype=torch.bool)
    selected[all_scores.argsort(descending=True, stable=True)[:budget]] = True
    return [mask.nonzero().flatten() for mask in selected.split([values.numel() for values in frames])]
