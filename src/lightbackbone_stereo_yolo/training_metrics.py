"""Shared Pred-RoI matching metrics for paired stereo detections."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor
from .geometry import pair_boxes_to_lr


@dataclass(frozen=True)
class PairedMatch:
    prediction_indices: Tensor
    target_indices: Tensor
    qualities: Tensor
    matched_count: int
    target_count: int


def _pairwise_iou(first: Tensor, second: Tensor) -> Tensor:
    tl=torch.maximum(first[:,None,:2],second[None,:,:2])
    br=torch.minimum(first[:,None,2:],second[None,:,2:])
    inter=(br-tl).clamp_min(0).prod(-1)
    area_first=(first[:,2]-first[:,0]).clamp_min(0)*(first[:,3]-first[:,1]).clamp_min(0)
    area_second=(second[:,2]-second[:,0]).clamp_min(0)*(second[:,3]-second[:,1]).clamp_min(0)
    return inter/(area_first[:,None]+area_second[None,:]-inter).clamp_min(1e-8)


def paired_match_eligibility(
    predictions: Tensor, targets: Tensor, minimum_quality: float, *,
    prediction_classes: Tensor | None = None, target_classes: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Return min bilateral IoU and the same-GT eligibility mask."""
    pl, pr = pair_boxes_to_lr(predictions)
    tl, tr = pair_boxes_to_lr(targets)
    quality = torch.minimum(_pairwise_iou(pl, tl), _pairwise_iou(pr, tr))
    eligible = quality >= minimum_quality
    if prediction_classes is not None or target_classes is not None:
        if (prediction_classes is None or target_classes is None
                or prediction_classes.shape != (len(predictions),) or target_classes.shape != (len(targets),)):
            raise ValueError("both class tensors must align with their paired boxes")
        eligible &= prediction_classes[:, None] == target_classes[None, :]
    return quality, eligible


def greedy_match_paired_detections(
    predictions: Tensor, targets: Tensor, scores: Tensor, minimum_quality: float = .5, *,
    prediction_classes: Tensor | None = None, target_classes: Tensor | None = None,
) -> PairedMatch:
    """Greedy one-to-one match sorted by min(left IoU, right IoU), then score."""
    if predictions.ndim!=2 or predictions.shape[1]!=6 or targets.ndim!=2 or targets.shape[1]!=6:
        raise ValueError("paired boxes must have shape [N, 6]")
    if scores.ndim!=1 or scores.shape[0]!=predictions.shape[0]:
        raise ValueError("scores must match predictions")
    if not predictions.numel() or not targets.numel():
        empty=torch.empty(0,dtype=torch.long,device=predictions.device)
        return PairedMatch(empty,empty, predictions.new_empty(0),0,targets.shape[0])
    quality, eligible = paired_match_eligibility(
        predictions, targets, minimum_quality,
        prediction_classes=prediction_classes, target_classes=target_classes,
    )
    candidates=[]
    for prediction_index,target_index in eligible.nonzero().tolist():
        candidates.append((float(quality[prediction_index,target_index]),float(scores[prediction_index]),prediction_index,target_index))
    candidates.sort(key=lambda item:(-item[0],-item[1],item[2],item[3]))
    used_predictions=set(); used_targets=set(); kept=[]
    for q,_,p,t in candidates:
        if p not in used_predictions and t not in used_targets:
            used_predictions.add(p);used_targets.add(t);kept.append((p,t,q))
    device=predictions.device
    if not kept:
        empty=torch.empty(0,dtype=torch.long,device=device)
        return PairedMatch(empty,empty,predictions.new_empty(0),0,targets.shape[0])
    return PairedMatch(torch.tensor([v[0] for v in kept],device=device),torch.tensor([v[1] for v in kept],device=device),torch.tensor([v[2] for v in kept],device=device),len(kept),targets.shape[0])






def _car_deployment_counts(
    raw, raw_classes, deployed, targets, target_classes, car_class, near_indices, *, near_details=None,
):
    """Score .40 car-only outputs inside the fixed stereo validation cohort."""
    car_targets = target_classes == car_class
    target_ids = car_targets.nonzero().flatten()
    car_gt = targets[car_targets]
    near = torch.isin(target_ids, target_ids.new_tensor(near_indices))
    keep = deployed.classes == car_class
    predictions = deployed.boxes6[keep]
    scores = deployed.scores[keep]
    match = greedy_match_paired_detections(predictions, car_gt, scores, .5)
    pairs_left, pairs_right = pair_boxes_to_lr(predictions)
    gt_left, gt_right = pair_boxes_to_lr(car_gt)
    left_iou = _pairwise_iou(pairs_left, gt_left)
    joint = torch.minimum(left_iou, _pairwise_iou(pairs_right, gt_right))
    raw_left, raw_right = pair_boxes_to_lr(raw)
    raw_left_iou = _pairwise_iou(raw_left, gt_left)
    raw_joint = torch.minimum(raw_left_iou, _pairwise_iou(raw_right, gt_right))
    unmatched = torch.ones(len(predictions), dtype=torch.bool, device=predictions.device)
    unmatched[match.prediction_indices] = False
    duplicate = ((joint[unmatched][:, match.target_indices] >= .50).any(1)).sum()
    matched_near = near[match.target_indices]
    class_raw_joint = raw_joint[raw_classes == car_class]
    if near_details is not None and near_indices:
        near_ids = near.nonzero().flatten()
        candidates = (joint[:, near_ids] >= .50).sum(0).tolist()
        raw_for_near = class_raw_joint[:, near_ids]
        best_raw = raw_for_near.amax(0).tolist() if raw_for_near.shape[0] else [0.] * len(near_ids)
        best_left_roi = left_iou[:, near_ids]
        best_left = best_left_roi.amax(0).tolist() if best_left_roi.shape[0] else [0.] * len(near_ids)
        matched = {
            gt: (dense_id, score, q)
            for gt, dense_id, score, q in zip(
                match.target_indices.tolist(), deployed.indices[keep][match.prediction_indices].tolist(),
                scores[match.prediction_indices].tolist(), match.qualities.tolist(),
            )
        }
        for local_id, gt_id, num, q_raw, q_left in zip(
            near_ids.tolist(), target_ids[near_ids].tolist(), candidates, best_raw, best_left,
        ):
            dense_id, score, q = matched.get(local_id, (None, None, None))
            near_details.append({
                "gt_index": gt_id, "raw_best_car_joint_iou": q_raw, "deployed_best_car_left_iou": q_left,
                "qualifying_car_pairs": num, "matched_dense_id": dense_id,
                "matched_score": score, "matched_joint_iou": q,
            })
    return {
        "car_gt": len(car_gt),
        "near_gt": int(near.sum()),
        "outputs": len(predictions),
        "paired_hits": match.matched_count,
        "qualified_duplicates": int(duplicate),
        "near_raw_left_hits": int(((raw_left_iou >= .50).any(0) & near).sum()),
        "near_raw_paired_hits": int(((class_raw_joint >= .50).any(0) & near).sum()),
        "near_left_hits_050": int(((left_iou >= .50).any(0) & near).sum()),
        "near_left_hits_075": int(((left_iou >= .75).any(0) & near).sum()),
        "near_paired_hits": int(matched_near.sum()),
        "near_paired_hits_iou_075": int((matched_near & (match.qualities >= .75)).sum()),
    }


@torch.no_grad()
def evaluate_anchorfree_detector_frame(encoder, detector, left: Tensor, right: Tensor, target: dict,
                                       score_threshold: float = .25, pre_score_threshold: float = .01,
                                       pre_nms_topk: int = 500, *,
                                       right_offset_mode: str = "legacy_clamp",
                                       car_cohort: tuple[int, list[int]] | None = None) -> dict:
    """Detector-only validation: raw, score-filtered pre-NMS, and deployed paired metrics."""
    from .detection import anchorfree_prediction_scores, decode_anchorfree_predictions
    from .geometry import decode_anchorfree_boxes6, generate_point_locations, point_levels
    pyramid = encoder(left, right)
    dense = detector(pyramid)
    locations = generate_point_locations(point_levels(pyramid), left.device)
    targets = target["boxes6"].to(left.device)
    dense_dict = {"objectness": dense.objectness[0], "class_logits": dense.class_logits[0], "box6": dense.box6[0]}
    encoded = dense_dict["box6"].reshape(-1, 6)
    scores, classes = anchorfree_prediction_scores(dense_dict)
    def hits(boxes: Tensor) -> Tensor:
        if not boxes.numel() or not targets.numel():
            return torch.zeros(targets.shape[0], dtype=torch.bool, device=targets.device)
        return _pairwise_iou(boxes[:, :4], targets[:, :4]).amax(dim=0) >= .5
    finite = torch.isfinite(encoded).all(1)
    raw = decode_anchorfree_boxes6(encoded[finite], locations[finite], right_offset_mode=right_offset_mode)
    finite_raw = torch.isfinite(raw).all(1)
    raw = raw[finite_raw]
    chosen = ((scores >= pre_score_threshold) & finite).nonzero().flatten()
    pre = decode_anchorfree_boxes6(encoded[chosen], locations[chosen], right_offset_mode=right_offset_mode) if chosen.numel() else targets.new_empty((0, 6))
    pre = pre[torch.isfinite(pre).all(1)]
    deployed = decode_anchorfree_predictions(
        dense_dict, locations, score_threshold=score_threshold,
        pre_nms_topk=pre_nms_topk, right_offset_mode=right_offset_mode,
        apply_nms=not getattr(detector, "one_to_one_enabled", False),
    )
    match = greedy_match_paired_detections(deployed.boxes6, targets, deployed.scores, .5)
    pred_left, pred_right = pair_boxes_to_lr(deployed.boxes6)
    target_left, target_right = pair_boxes_to_lr(targets)
    qualifies = torch.minimum(_pairwise_iou(pred_left, target_left), _pairwise_iou(pred_right, target_right)) >= .5
    duplicate_pairs = qualifies.any(dim=1).sum() - match.matched_count
    raw_hits = hits(raw)
    deployed_hits = hits(deployed.boxes6)
    paired_hits = torch.zeros_like(raw_hits)
    paired_hits[match.target_indices] = True
    longest = (targets[:, 2:4] - targets[:, :2]).amax(dim=1)
    groups = {"le16": longest <= 16, "16_to_32": (longest > 16) & (longest <= 32),
              "32_to_64": (longest > 32) & (longest <= 64), "gt64": longest > 64}
    size_counts = torch.stack([
        torch.stack((group.sum(), (raw_hits & group).sum(), (deployed_hits & group).sum(),
                     (paired_hits & group).sum()))
        for group in groups.values()
    ]).tolist()
    size_groups = {name: dict(zip(("gt", "raw_left_hits", "deployed_left_hits", "paired_hits"), counts))
                   for name, counts in zip(groups, size_counts)}
    result = {"target_count": int(targets.shape[0]), "raw_dense_left_hits": int(raw_hits.sum()),
            "pre_nms_left_hits_score_001": int(hits(pre).sum()), "deployment_left_hits_score_025": int(deployed_hits.sum()),
            "paired_hits_score_025": match.matched_count, "post_nms_pairs_score_025": int(deployed.boxes6.shape[0]),
            "paired_duplicate_predictions": int(duplicate_pairs), "detection_size_groups": size_groups}
    if car_cohort is not None:
        if not getattr(detector, "one_to_one_enabled", False):
            raise ValueError("the car cohort audit requires one-to-one deployment")
        car_pairs = decode_anchorfree_predictions(
            dense_dict, locations, score_threshold=.40, pre_nms_topk=pre_nms_topk,
            right_offset_mode=right_offset_mode, apply_nms=False,
        )
        car_class, near_indices = car_cohort
        valid_score = torch.isfinite(scores[finite][finite_raw])
        near_details = []
        result["car_deployment_score_040"] = _car_deployment_counts(
            raw[valid_score], classes[finite][finite_raw][valid_score], car_pairs, targets,
            target["classes"].to(left.device), car_class, near_indices, near_details=near_details,
        )
        result["car_deployment_near_states"] = near_details
    return result

def evaluate_pred_roi_frame(
    encoder, detector, instance_head, left: Tensor, right: Tensor, target: dict,
    dmax_roi: int, detector_variant: str, score_threshold: float=.05,
    roi_margin: float=16., minimum_quality: float=.5,
    *, right_offset_mode: str = "legacy_clamp",
) -> dict:
    """Evaluate prediction-driven RoIs using the same mask/iDisp metric formula as GT-RoI."""
    from .detection import decode_anchorfree_predictions, decode_dense_predictions, generate_grid_anchors
    from .geometry import generate_point_locations, point_levels
    from .training import build_instance_supervision, instance_metrics, repeat_roi_features

    pyramid=encoder(left,right)
    dense=detector(pyramid)
    dense_dict={"objectness":dense.objectness[0],"class_logits":dense.class_logits[0],"box6":dense.box6[0]}
    if detector_variant=="stereo_yolo26_pan":
        pairs=decode_anchorfree_predictions(
            dense_dict, generate_point_locations(point_levels(pyramid), left.device), score_threshold,
            right_offset_mode=right_offset_mode, apply_nms=not getattr(detector, "one_to_one_enabled", False),
        )
    elif detector_variant=="baseline":
        levels=((pyramid.p3_left.shape[-2],pyramid.p3_left.shape[-1],8,32.),(pyramid.p4_left.shape[-2],pyramid.p4_left.shape[-1],16,64.),(pyramid.c5_left.shape[-2],pyramid.c5_left.shape[-1],32,128.))
        pairs=decode_dense_predictions(dense_dict,generate_grid_anchors(levels,left.device),score_threshold)
    else:
        raise ValueError("unsupported detector_variant")
    boxes=target["boxes6"].to(left.device)
    match=greedy_match_paired_detections(pairs.boxes6,boxes,pairs.scores,minimum_quality)
    result={"matched_count":match.matched_count,"target_count":match.target_count,"coverage":match.matched_count/max(match.target_count,1),"iou_sum":0.,"iou_count":0,"epe_sum":0.,"epe_count":0,"d1_hit":0,"d1_count":0}
    if not match.matched_count:
        return result
    predicted=pairs.boxes6[match.prediction_indices]
    target_indices=match.target_indices
    roi_pyramid=repeat_roi_features(pyramid,predicted.shape[0])
    instance=instance_head(roi_pyramid,predicted,left.shape[-2],left.shape[-1],margin=roi_margin)
    masks=target["masks"].to(left.device)[target_indices]
    disparity=target["disparity"].to(left.device).unsqueeze(1)
    disparity_valid=target["disparity_valid"].to(left.device).unsqueeze(1)
    supervision=build_instance_supervision(instance,predicted,masks,disparity,disparity_valid,dmax_roi)
    result.update(instance_metrics(instance,supervision,masks,dmax_roi))
    return result


def greedy_match_left_detections(predictions: Tensor, targets: Tensor, scores: Tensor, minimum_quality: float = .5) -> PairedMatch:
    """One-to-one matching using only left-box IoU, for diagnostic use."""
    if not predictions.numel() or not targets.numel():
        empty = torch.empty(0, dtype=torch.long, device=predictions.device)
        return PairedMatch(empty, empty, predictions.new_empty(0), 0, targets.shape[0])
    quality = _pairwise_iou(predictions[:, :4], targets[:, :4])
    candidates = [(float(quality[p, t]), float(scores[p]), p, t) for p, t in (quality >= minimum_quality).nonzero().tolist()]
    candidates.sort(key=lambda value: (-value[0], -value[1], value[2], value[3]))
    used_predictions, used_targets, kept = set(), set(), []
    for quality_value, _, prediction_index, target_index in candidates:
        if prediction_index not in used_predictions and target_index not in used_targets:
            used_predictions.add(prediction_index); used_targets.add(target_index); kept.append((prediction_index, target_index, quality_value))
    if not kept:
        empty = torch.empty(0, dtype=torch.long, device=predictions.device)
        return PairedMatch(empty, empty, predictions.new_empty(0), 0, targets.shape[0])
    return PairedMatch(torch.tensor([item[0] for item in kept], device=predictions.device), torch.tensor([item[1] for item in kept], device=predictions.device), torch.tensor([item[2] for item in kept], device=predictions.device), len(kept), targets.shape[0])


def analyze_paired_detections(predictions: Tensor, targets: Tensor, scores: Tensor, minimum_quality: float = .5) -> dict:
    """Separate formal paired coverage from left-first right-geometry diagnostics."""
    left_match = greedy_match_left_detections(predictions, targets, scores, minimum_quality)
    paired_match = greedy_match_paired_detections(predictions, targets, scores, minimum_quality)
    result = {"target_count": int(targets.shape[0]), "left_matched_count": left_match.matched_count, "right_pass_given_left_count": 0, "paired_matched_count": paired_match.matched_count, "dR_error_signed": [], "wR_error_signed": []}
    if not left_match.matched_count:
        return result
    predicted = predictions[left_match.prediction_indices]
    target = targets[left_match.target_indices]
    _, predicted_right = pair_boxes_to_lr(predicted)
    _, target_right = pair_boxes_to_lr(target)
    right_iou = _pairwise_iou(predicted_right, target_right).diag()
    result["right_pass_given_left_count"] = int((right_iou >= minimum_quality).sum().item())
    predicted_dr = (predicted[:, :2].sum(1) * .5) - ((predicted[:, 4] + predicted[:, 5]) * .5)
    target_dr = (target[:, :2].sum(1) * .5) - ((target[:, 4] + target[:, 5]) * .5)
    result["dR_error_signed"] = (predicted_dr - target_dr).detach().cpu().tolist()
    result["wR_error_signed"] = ((predicted[:, 5] - predicted[:, 4]) - (target[:, 5] - target[:, 4])).detach().cpu().tolist()
    return result

def audit_left_misses(targets: Tensor, pre_nms: Tensor, pre_scores: Tensor, post_nms: Tensor, post_scores: Tensor, minimum_iou: float = .5, reference_score: float = .25) -> dict:
    """Classify every GT by the earliest stage that prevents a left-box hit."""
    result = {"nms_suppressed": 0, "below_reference_score": 0, "no_pre_nms_left_match": 0, "kept_at_reference": 0}
    pre_quality = _pairwise_iou(pre_nms[:, :4], targets[:, :4]) if pre_nms.numel() else targets.new_empty((0, targets.shape[0]))
    post_quality = _pairwise_iou(post_nms[:, :4], targets[:, :4]) if post_nms.numel() else targets.new_empty((0, targets.shape[0]))
    for target_index in range(targets.shape[0]):
        if not pre_quality.numel() or not bool((pre_quality[:, target_index] >= minimum_iou).any()):
            result["no_pre_nms_left_match"] += 1
            continue
        post_hits = post_quality[:, target_index] >= minimum_iou if post_quality.numel() else torch.zeros(0, dtype=torch.bool, device=targets.device)
        if not bool(post_hits.any()):
            result["nms_suppressed"] += 1
        elif float(post_scores[post_hits].max()) < reference_score:
            result["below_reference_score"] += 1
        else:
            result["kept_at_reference"] += 1
    return result

def fixed_roi_disparity_metrics(gt_prediction: Tensor, pred_prediction: Tensor, target: Tensor, instance_mask: Tensor, gt_crop_valid: Tensor, pred_crop_valid: Tensor) -> dict:
    """Compare fixed GT/Pred RoIs only where both crops cover valid instance pixels."""
    foreground = instance_mask.bool()
    common = foreground & gt_crop_valid.bool() & pred_crop_valid.bool()
    denominator = foreground & gt_crop_valid.bool()
    common_per_instance = common.flatten(1).sum(1)
    denominator_per_instance = denominator.flatten(1).sum(1)
    common_count = int(common_per_instance.sum().item())
    denominator_count = int(denominator_per_instance.sum().item())
    coverage_per_instance = (common_per_instance.to(torch.float64) / denominator_per_instance.clamp_min(1).to(torch.float64)).tolist()
    empty_instances = int((common_per_instance == 0).sum().item())
    if not common_count:
        return {"common_pixel_count": 0, "empty_instances": empty_instances, "gt_epe": 0., "pred_epe": 0., "delta_epe": 0., "relative_epe_increase": 0., "gt_d1": 0., "pred_d1": 0., "delta_d1": 0., "mask_coverage": 0., "coverage_per_instance": coverage_per_instance}
    gt_error = (gt_prediction - target).abs(); pred_error = (pred_prediction - target).abs()
    gt_epe = float(gt_error[common].mean().item()); pred_epe = float(pred_error[common].mean().item())
    gt_d1 = float(((gt_error > 3.) & (gt_error > .05 * target.abs()) & common).sum().item() / common_count)
    pred_d1 = float(((pred_error > 3.) & (pred_error > .05 * target.abs()) & common).sum().item() / common_count)
    return {"common_pixel_count": common_count, "empty_instances": empty_instances, "gt_epe": gt_epe, "pred_epe": pred_epe, "delta_epe": pred_epe - gt_epe, "relative_epe_increase": (pred_epe - gt_epe) / max(gt_epe, 1e-8), "gt_d1": gt_d1, "pred_d1": pred_d1, "delta_d1": pred_d1 - gt_d1, "mask_coverage": common_count / max(denominator_count, 1), "coverage_per_instance": coverage_per_instance}

def evaluate_fixed_roi_pairs(pyramid: object, instance_head: object, left: Tensor, target: dict, gt_boxes6: Tensor, pred_boxes6: Tensor, target_indices: Tensor, dmax_roi: int, roi_margin: float) -> tuple[dict, object, object]:
    """A3: compare cached matched GT/predicted pairs without detector decode or matching."""
    from .training import build_instance_supervision, repeat_roi_features
    from .roi import align_instance_mask_roi, resample_roi_map_to_crop
    if gt_boxes6.shape != pred_boxes6.shape or gt_boxes6.ndim != 2 or gt_boxes6.shape[1] != 6:
        raise ValueError("fixed GT and predicted paired boxes must have the same [N,6] shape")
    if target_indices.shape != (gt_boxes6.shape[0],):
        raise ValueError("target_indices must select one GT per fixed pair")
    count = gt_boxes6.shape[0]
    roi_pyramid = repeat_roi_features(pyramid, count)
    gt_output = instance_head(roi_pyramid, gt_boxes6, left.shape[-2], left.shape[-1], margin=roi_margin)
    pred_output = instance_head(roi_pyramid, pred_boxes6, left.shape[-2], left.shape[-1], margin=roi_margin)
    masks = target["masks"].to(left.device)[target_indices]
    disparity = target["disparity"].to(left.device).unsqueeze(1)
    disparity_valid = target["disparity_valid"].to(left.device).unsqueeze(1)
    gt_supervision = build_instance_supervision(gt_output, gt_boxes6, masks, disparity, disparity_valid, dmax_roi, supervision_mode="foreground")
    pred_disparity, overlap = resample_roi_map_to_crop(pred_output.disparity, pred_output.crop, gt_output.crop)
    pred_valid, _ = resample_roi_map_to_crop(pred_output.validity.float(), pred_output.crop, gt_output.crop)
    instance_mask = align_instance_mask_roi(masks, gt_output.crop)[:, 0]
    metrics = fixed_roi_disparity_metrics(
        gt_output.disparity, pred_disparity, gt_supervision.disparity_target[:, 0], instance_mask,
        gt_supervision.disparity_valid[:, 0], overlap & (pred_valid >= .5),
    )
    return metrics, gt_output, pred_output
