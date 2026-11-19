"""Read-only paired-detection diagnosis for an existing checkpoint."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import List
from zipfile import ZipFile
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]


def parse_thresholds(value: str) -> list[float]:
    thresholds = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    if not thresholds or any(item < 0 or item > 1 for item in thresholds):
        raise ValueError("thresholds must be non-empty probabilities")
    return thresholds



def parse_topks(value: str) -> list[int]:
    topks = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not topks or any(item < 0 for item in topks):
        raise ValueError("pre-NMS top-k values must be non-negative integers")
    return list(dict.fromkeys(topks))


def sweep_settings(thresholds: list[float], topks: list[int]) -> list[tuple[float, int]]:
    return [(threshold, topk) for threshold in thresholds for topk in topks]


def _setting_key(threshold: float, topk: int) -> str:
    return "score_{:.6g}_topk_{}".format(threshold, "all" if topk == 0 else topk)
def _iou(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    top_left = torch.maximum(first[:, None, :2], second[None, :, :2])
    bottom_right = torch.minimum(first[:, None, 2:], second[None, :, 2:])
    inter = (bottom_right - top_left).clamp_min(0).prod(-1)
    area_first = (first[:, 2] - first[:, 0]).clamp_min(0) * (first[:, 3] - first[:, 1]).clamp_min(0)
    area_second = (second[:, 2] - second[:, 0]).clamp_min(0) * (second[:, 3] - second[:, 1]).clamp_min(0)
    return inter / (area_first[:, None] + area_second[None, :] - inter).clamp_min(1e-8)


def _right_boxes(pairs: torch.Tensor) -> torch.Tensor:
    return torch.stack((pairs[:, 4], pairs[:, 1], pairs[:, 5], pairs[:, 3]), 1)


def _summary(values: List[float]) -> dict:
    if not values:
        return {"count": 0, "median": None, "p50_abs": None, "p90_abs": None, "p95_abs": None}
    tensor = torch.tensor(values, dtype=torch.float64)
    absolute = tensor.abs()
    return {"count": len(values), "median": float(tensor.median()), "p50_abs": float(torch.quantile(absolute, .5)), "p90_abs": float(torch.quantile(absolute, .9)), "p95_abs": float(torch.quantile(absolute, .95))}


def _dense_inputs(dense) -> dict[str, torch.Tensor]:
    return {"objectness": dense.objectness[0], "class_logits": dense.class_logits[0], "box6": dense.box6[0]}


def _candidate_trace(scores, classes, boxes, threshold, topk, *, apply_nms=True):
    from lightbackbone_stereo_yolo.detection import anchorfree_candidate_indices, suppress_anchorfree_candidates

    score_ids, topk_ids = anchorfree_candidate_indices(scores, threshold, topk)
    pairs, pre = suppress_anchorfree_candidates(boxes[topk_ids], topk_ids, scores, classes, .5, apply_nms=apply_nms)
    counts = {"score_filtered": int(score_ids.numel()), "pre_nms_topk": int(topk_ids.numel()),
              "invalid_after_decode": int(topk_ids.numel() - pre.indices.numel()),
              "nms": int(pairs.indices.numel()), "post_nms_max_det": int(pairs.indices.numel()),
              "topk_saturated": topk > 0 and score_ids.numel() > topk}
    return pairs, pre, counts, score_ids, topk_ids


def _decode_anchorfree(dense, locations, threshold, topk, *, right_offset_mode="legacy_clamp", apply_nms=True):
    scores, classes, boxes = _prepare_audit(dense, locations, right_offset_mode)
    return _candidate_trace(scores, classes, boxes, threshold, topk, apply_nms=apply_nms)[:3]


def _prepare_audit(dense, locations, mode):
    from lightbackbone_stereo_yolo.detection import anchorfree_prediction_scores
    from lightbackbone_stereo_yolo.geometry import decode_anchorfree_boxes6

    inputs = _dense_inputs(dense)
    scores, classes = anchorfree_prediction_scores(inputs)
    boxes = decode_anchorfree_boxes6(inputs["box6"].reshape(-1, 6), locations, right_offset_mode=mode)
    return scores, classes, boxes


def audit_frame(dense, locations, targets, threshold, topk, *, right_offset_mode="legacy_clamp", prepared=None, apply_nms=True):
    if prepared is None:
        prepared = _prepare_audit(dense, locations, right_offset_mode)
    scores, classes, boxes = prepared
    pairs, pre, counts, score_ids, topk_ids = _candidate_trace(scores, classes, boxes, threshold, topk, apply_nms=apply_nms)
    finite = torch.isfinite(boxes).all(dim=1)
    left_iou = _iou(boxes[:, :4], targets[:, :4])
    right_iou = _iou(_right_boxes(boxes), _right_boxes(targets))
    left_good = (left_iou >= .5) & finite[:, None]
    joint_good = left_good & (right_iou >= .5)
    raw_ids = torch.arange(boxes.shape[0], device=boxes.device)
    stage_ids = {"raw": raw_ids, "score": score_ids, "topk": topk_ids, "nms": pairs.indices}
    flags = {name: (left_good[idx].any(dim=0), joint_good[idx].any(dim=0)) for name, idx in stage_ids.items()}
    result = _frame_diagnosis(pairs.boxes6, pairs.scores, targets, .5, pairs.indices)
    for i, status in enumerate(result["statuses"]):
        status["stage_left_hit"] = {name: bool(left[i]) for name, (left, _) in flags.items()}
        status["stage_joint_hit"] = {name: bool(joint[i]) for name, (_, joint) in flags.items()}
    result["stage_hits"] = {name: {"left": int(left.sum()), "joint": int(joint.sum())}
                            for name, (left, joint) in flags.items()}
    return {**result, "counts": counts, "pairs": pairs, "pre_nms": pre,
            "pre_nms_topk_indices": topk_ids}


def _frame_diagnosis(predictions: torch.Tensor, scores: torch.Tensor, targets: torch.Tensor, minimum_iou: float, source_indices: torch.Tensor | None = None) -> dict:
    from lightbackbone_stereo_yolo.training_metrics import greedy_match_left_detections, greedy_match_paired_detections
    left = greedy_match_left_detections(predictions, targets, scores, minimum_iou)
    paired = greedy_match_paired_detections(predictions, targets, scores, minimum_iou)
    statuses = [{"status": "no_left", "left_hit": False, "right_pass_given_left": False,
                 "formal_paired": False} for _ in range(targets.shape[0])]
    dr_error, wr_error, right_iou = [], [], []
    if left.matched_count:
        pred = predictions[left.prediction_indices]
        gt = targets[left.target_indices]
        left_iou = _iou(pred[:, :4], gt[:, :4]).diag()
        right = _iou(_right_boxes(pred), _right_boxes(gt)).diag()
        pred_dr = (pred[:, 0] + pred[:, 2] - pred[:, 4] - pred[:, 5]) * .5
        gt_dr = (gt[:, 0] + gt[:, 2] - gt[:, 4] - gt[:, 5]) * .5
        for row, target_index in enumerate(left.target_indices.tolist()):
            pair_index = int(left.prediction_indices[row])
            dense_index = int(source_indices[pair_index]) if source_indices is not None else pair_index
            passed = bool(right[row] >= minimum_iou)
            statuses[target_index] = {
                "status": "left_only" if not passed else "left_right",
                "left_hit": True,
                "right_pass_given_left": passed,
                "formal_paired": False,
                "prediction_index": dense_index,
                "score": float(scores[pair_index]),
                "left_iou": float(left_iou[row]),
                "right_iou": float(right[row]),
            }
            dr_error.append(float(pred_dr[row] - gt_dr[row]))
            wr_error.append(float((pred[row, 5] - pred[row, 4]) - (gt[row, 5] - gt[row, 4])))
            right_iou.append(float(right[row]))
    for pair_index, target_index in zip(paired.prediction_indices.tolist(), paired.target_indices.tolist()):
        statuses[target_index]["formal_paired"] = True
        statuses[target_index]["status"] = "formal_paired"
        statuses[target_index]["formal_prediction_index"] = int(source_indices[pair_index]) if source_indices is not None else int(pair_index)
    return {"left_matched": left.matched_count, "right_pass_given_left": sum(item["right_pass_given_left"] for item in statuses), "formal_paired": paired.matched_count, "statuses": statuses, "dR_errors": dr_error, "wR_errors": wr_error, "right_iou": right_iou}


def _empty_aggregate() -> dict:
    return {
        "eligible_gt": 0,
        "left_matched": 0,
        "right_pass_given_left": 0,
        "formal_paired": 0,
        "candidate_stages": {"score_filtered": 0, "pre_nms_topk": 0, "invalid_after_decode": 0, "nms": 0, "post_nms_max_det": 0, "topk_saturated_frames": 0},
        "dR_errors": [], "wR_errors": [], "right_iou": [], "per_frame": [],
        "stage_hits": {stage: {"left": 0, "joint": 0} for stage in ("raw", "score", "topk", "nms")},
        "groups": {group: {"matched": 0, "eligible_gt": 0} for group in ("lt10m", "person", "unknown_depth")},
    }


def _finalize_aggregate(aggregate: dict) -> None:
    eligible = max(aggregate["eligible_gt"], 1)
    aggregate["left_recall"] = aggregate["left_matched"] / eligible
    aggregate["right_pass_rate_given_left"] = aggregate["right_pass_given_left"] / max(aggregate["left_matched"], 1)
    aggregate["paired_coverage"] = aggregate["formal_paired"] / eligible
    aggregate["paired_precision"] = aggregate["formal_paired"] / max(aggregate["candidate_stages"]["post_nms_max_det"], 1)
    aggregate["dR_error"] = _summary(aggregate.pop("dR_errors"))
    aggregate["wR_error"] = _summary(aggregate.pop("wR_errors"))
    aggregate["right_iou"] = _summary(aggregate["right_iou"])


def _state_transitions(reference: dict, current: dict) -> dict:
    transitions = {"gained_formal_pair": 0, "lost_formal_pair": 0, "replaced_formal_prediction": 0}
    def by_frame(aggregate):
        records = aggregate["per_frame"]
        result = {record["frame_id"]: record["statuses"] for record in records}
        if len(result) != len(records):
            raise ValueError("duplicate frame ID in diagnosis")
        return result
    ref, cur = by_frame(reference), by_frame(current)
    if ref.keys() != cur.keys():
        raise ValueError("diagnoses do not cover the same frames")
    for frame_id, reference_statuses in ref.items():
        current_statuses = cur[frame_id]
        if len(reference_statuses) != len(current_statuses):
            raise ValueError(f"GT count differs for frame {frame_id}")
        for reference_status, current_status in zip(reference_statuses, current_statuses):
            reference_pair = bool(reference_status.get("formal_paired"))
            current_pair = bool(current_status.get("formal_paired"))
            transitions["gained_formal_pair"] += int(current_pair and not reference_pair)
            transitions["lost_formal_pair"] += int(reference_pair and not current_pair)
            transitions["replaced_formal_prediction"] += int(current_pair and reference_pair and current_status.get("formal_prediction_index") != reference_status.get("formal_prediction_index"))
    return transitions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--thresholds", default="0.01,0.05,0.10,0.25")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pre-nms-topks", "--pre-nms-topk", dest="pre_nms_topks", default="500", help="comma-separated limits; 0 keeps all score-filtered candidates")
    parser.add_argument("--reference-threshold", type=float, default=.25)
    parser.add_argument("--reference-topk", type=int, default=500)
    parser.add_argument("--right-offset-mode", choices=("legacy_clamp", "signed"))
    diagnostic_mode = parser.add_mutually_exclusive_group()
    diagnostic_mode.add_argument("--compare-right-offset", action="store_true")
    diagnostic_mode.add_argument("--audit-score-breakdown", action="store_true",
                                 help="attribute score-gated misses and measure objectness-only ranking at the same budget")
    diagnostic_mode.add_argument("--export-roi-candidates", type=Path)
    return parser.parse_args()


def _rectified_calibration(calibration_zip, frame_id):
    stem = frame_id.split("_", 1)[0]
    with calibration_zip.open(f"training/calib_cam_to_cam/{stem}.txt") as handle:
        projection = {}
        for line in handle.read().decode("ascii").splitlines():
            key, _, values = line.partition(":")
            if key in ("P_rect_02", "P_rect_03"):
                projection[key] = [float(value) for value in values.split()]
    p2, p3 = projection["P_rect_02"], projection["P_rect_03"]
    bf, cx_delta = p2[3] - p3[3], p2[2] - p3[2]
    if not (math.isfinite(bf) and math.isfinite(cx_delta) and abs(p2[0] - p3[0]) <= 1e-4 and bf > 0):
        raise ValueError(f"unexpected rectified stereo calibration for {frame_id}")
    return {"bf_px_m": bf, "cx_delta_px": cx_delta}


def _target_groups(sample, mapping, calibration_zip, *, calibration=None):
    if calibration is None:
        calibration = _rectified_calibration(calibration_zip, sample["frame_id"])
    disparity = sample["disparity"][0]
    valid = sample["disparity_valid"][0]
    groups = []
    for index, label in enumerate(sample["classes"].tolist()):
        pixels = disparity[valid & sample["masks"][index, 0]]
        depth = None
        if pixels.numel():
            corrected = float(torch.quantile(pixels, .5)) - calibration["cx_delta_px"]
            if corrected > 0:
                depth = calibration["bf_px_m"] / corrected
        groups.append({"depth_m": depth, "person": label == mapping.get(24)})
    return groups


def _fixed_candidate_right_geometry(locations, dense, targets, topk_ids, legacy, signed):
    scores, _, old_boxes = legacy
    _, _, new_boxes = signed
    if not torch.equal(old_boxes[:, :4], new_boxes[:, :4]):
        # NaNs are handled by the finite masks and compare equal only by position.
        torch.testing.assert_close(old_boxes[:, :4], new_boxes[:, :4], rtol=0, atol=0, equal_nan=True)
    ids = topk_ids.sort().values
    ids = ids[scores[ids].argsort(descending=True, stable=True)]
    ids = ids[torch.isfinite(old_boxes[ids]).all(1) & torch.isfinite(new_boxes[ids]).all(1)]
    if not ids.numel() or not targets.numel():
        return []
    left_max, winner = _iou(old_boxes[ids, :4], targets[:, :4]).max(dim=0)
    gt_idx = (left_max >= .5).nonzero().flatten()
    pred_idx = ids[winner[gt_idx]]
    old = _right_boxes(old_boxes[pred_idx])
    new = _right_boxes(new_boxes[pred_idx])
    gt = _right_boxes(targets[gt_idx])
    old_iou = _iou(old, gt).diag().tolist()
    new_iou = _iou(new, gt).diag().tolist()
    gt_offset = (locations[pred_idx, 0] - (gt[:, 0] + gt[:, 2]) * .5).tolist()
    predicted_offset = (dense.box6[0].reshape(-1, 6)[pred_idx, 4].sinh() * locations[pred_idx, 2]).tolist()
    return [{"gt_index": t, "dense_index": p, "gt_point_offset_px": offset,
             "pred_point_offset_px": predicted, "right_iou_legacy": old_q, "right_iou_signed": new_q}
            for t, p, offset, predicted, old_q, new_q in zip(gt_idx.tolist(), pred_idx.tolist(), gt_offset,
                                                            predicted_offset, old_iou, new_iou)]


def _accumulate(aggregate, diagnosis, frame_id, groups, audit_left=False, targets=None):
    for key in ("left_matched", "right_pass_given_left", "formal_paired"):
        aggregate[key] += diagnosis[key]
    statuses = diagnosis["statuses"]
    if len(statuses) != len(groups):
        raise ValueError(f"group and GT counts disagree for {frame_id}")
    aggregate["eligible_gt"] += len(statuses)
    for key, counter in aggregate["candidate_stages"].items():
        source = "topk_saturated" if key == "topk_saturated_frames" else key
        aggregate["candidate_stages"][key] = counter + int(diagnosis["counts"][source])
    for stage, hits in diagnosis["stage_hits"].items():
        for metric, count in hits.items():
            aggregate["stage_hits"][stage][metric] += count
    for key in ("dR_errors", "wR_errors", "right_iou"):
        aggregate[key].extend(diagnosis[key])
    for status, group in zip(statuses, groups):
        status.update(group)
        depth = group["depth_m"]
        selected = {"lt10m": depth is not None and depth < 10,
                    "unknown_depth": depth is None, "person": group["person"]}
        for name, include in selected.items():
            if include:
                aggregate["groups"][name]["eligible_gt"] += 1
                aggregate["groups"][name]["matched"] += int(status["formal_paired"])
    aggregate["per_frame"].append({"frame_id": frame_id, "statuses": statuses,
                                   "candidate_stages": diagnosis["counts"],
                                   "post_nms_dense_indices": diagnosis["pairs"].indices.tolist()})
    if audit_left:
        from lightbackbone_stereo_yolo.training_metrics import audit_left_misses
        pre, pairs = diagnosis["pre_nms"], diagnosis["pairs"]
        audit = audit_left_misses(targets, pre.boxes6, pre.scores, pairs.boxes6, pairs.scores)
        current = aggregate.setdefault("left_miss_audit", {key: 0 for key in audit})
        for key, value in audit.items():
            current[key] += value


def _finalize_score_breakdown(score_frames, reference, sweep):
    from lightbackbone_stereo_yolo.score_diagnostics import budget_keep_indices
    from lightbackbone_stereo_yolo.training_metrics import greedy_match_paired_detections

    rows = [row for frame in score_frames for row in frame["targets"]]
    low_left = [row for row in rows if row["left_filtered_by_score"]]
    low_keys = {(frame["frame_id"], row["gt_index"])
                for frame in score_frames for row in frame["targets"] if row["left_filtered_by_score"]}
    matched_by_threshold = {}
    for setting, aggregate in sweep.items():
        good = {(frame["frame_id"], i)
                for frame in aggregate["per_frame"] for i, row in enumerate(frame["statuses"])
                if row["formal_paired"]}
        matched_by_threshold[setting] = len(good & low_keys)

    budget = reference["candidate_stages"]["nms"]
    selection = budget_keep_indices([frame["objectness_pool"].scores for frame in score_frames], budget)
    ref_matched = {(frame["frame_id"], i)
                   for frame in reference["per_frame"] for i, row in enumerate(frame["statuses"])
                   if row["formal_paired"]}
    obj_matched = set()
    for frame, keep in zip(score_frames, selection):
        pool = frame["objectness_pool"]
        match = greedy_match_paired_detections(pool.boxes6[keep], frame["boxes6_gt"], pool.scores[keep], .5)
        obj_matched.update((frame["frame_id"], i) for i in match.target_indices.tolist())
    total = sum(pick.numel() for pick in selection)
    groups = {}
    for name, include in (("lt10m", lambda row: row["depth_m"] is not None and row["depth_m"] < 10),
                          ("person", lambda row: row["person"])):
        ids = {(frame["frame_id"], row["gt_index"])
               for frame in score_frames for row in frame["targets"] if include(row)}
        groups[name] = {"gt": len(ids), "matched": len(obj_matched & ids)}
    return {
        "low_score_left_targets": len(low_left),
        "joint_blocker_all": dict(Counter(row["joint_score_blocker"] for row in rows)),
        "joint_blocker_among_low_score_left": dict(Counter(row["joint_score_blocker"] for row in low_left)),
        "bad_right_ranks_higher": sum(row["bad_right_outranks_joint"] for row in rows),
        "comparable_left_localized_rank": sum(row["comparable_rank"] for row in rows),
        "low_score_left_formally_recovered_in_sweep": matched_by_threshold,
        "objectness_only_at_reference_budget": {
            "budget_requested": budget, "post_nms_pairs": total,
            "matched": len(obj_matched), "eligible_gt": len(rows),
            "paired_precision": len(obj_matched) / max(total, 1),
            "gained": len(obj_matched - ref_matched), "lost": len(ref_matched - obj_matched),
            "groups": groups,
        },
        "per_frame": [{"frame_id": frame["frame_id"], "targets": frame["targets"]} for frame in score_frames],
    }


def main() -> None:
    args = parse_args()
    thresholds = parse_thresholds(args.thresholds)
    topks = parse_topks(args.pre_nms_topks)
    settings = sweep_settings(thresholds, topks)
    reference_key = _setting_key(args.reference_threshold, args.reference_topk)
    if (args.reference_threshold, args.reference_topk) not in settings:
        raise ValueError("reference threshold/top-k must be included in the sweep")
    device = torch.device(args.device)
    from lightbackbone_stereo_yolo.data.frame_dataset import KittiFrameDataset, build_frame_records
    from lightbackbone_stereo_yolo.lightstereo_encoder import LightStereoFeatureEncoder
    from lightbackbone_stereo_yolo.stereo_yolo26 import StereoYOLO26PanHead
    from lightbackbone_stereo_yolo.training import StereoYoloTrainingModel
    from lightbackbone_stereo_yolo.model import StereoLocalInstanceHead
    from lightbackbone_stereo_yolo.geometry import generate_point_locations, point_levels
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = payload["config"]
    frames = build_frame_records(config["dataset_root"], config["val_json"])
    mapping = payload["class_mapping"]
    dataset = KittiFrameDataset(frames, mapping, unknown_class="skip")
    encoder = LightStereoFeatureEncoder.from_checkpoint(config["lightstereo_source"], config["checkpoint"], adapter_channels=config["adapter_channels"])
    detector = StereoYOLO26PanHead(config["adapter_channels"], len(mapping), one_to_one=config.get("one_to_one", False))
    apply_nms = not detector.one_to_one_enabled
    model = StereoYoloTrainingModel(encoder, detector, StereoLocalInstanceHead(config["adapter_channels"], payload["dmax_roi"],
                                    roi_sampling=config.get("roi_sampling", "native_pad"))).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    from lightbackbone_stereo_yolo.detection import decode_anchorfree_predictions
    from lightbackbone_stereo_yolo.geometry import right_offset_mode_from_config

    configured_mode = right_offset_mode_from_config(config, args.right_offset_mode)
    if args.export_roi_candidates and len(settings) != 1:
        raise ValueError("candidate export requires exactly one score threshold and top-k")
    exported_frames = []
    modes = ("legacy_clamp", "signed") if args.compare_right_offset else (configured_mode,)
    reference_mode = "legacy_clamp" if args.compare_right_offset else configured_mode
    all_aggregates = {mode: {_setting_key(th, k): _empty_aggregate() for th, k in settings} for mode in modes}
    fixed_geometry = []
    official_differences = []
    score_frames = []
    calibration_path = Path(config["dataset_root"]) / "data_scene_flow_calib.zip"
    with torch.no_grad(), ZipFile(calibration_path) as calibration_zip:
        for frame_number, sample in enumerate(dataset, 1):
            left = sample["left"].unsqueeze(0).to(device)
            right = sample["right"].unsqueeze(0).to(device)
            pyramid = model.encoder(left, right)
            dense = model.detector(pyramid)
            locations = generate_point_locations(point_levels(pyramid), device)
            targets = sample["boxes6"].to(device)
            frame_id = sample["frame_id"]
            calibration = _rectified_calibration(calibration_zip, frame_id)
            groups = _target_groups(sample, mapping, calibration_zip, calibration=calibration)
            if args.export_roi_candidates:
                pairs = decode_anchorfree_predictions(
                    _dense_inputs(dense), locations, args.reference_threshold, pre_nms_topk=args.reference_topk,
                    right_offset_mode=configured_mode, apply_nms=apply_nms,
                )
                exported_frames.append({
                    "frame_id": frame_id, "image_hw": list(left.shape[-2:]), "calibration": calibration,
                    "boxes6": pairs.boxes6.tolist(), "scores": pairs.scores.tolist(),
                    "classes": pairs.classes.tolist(), "dense_ids": pairs.indices.tolist(),
                    "nms_order": list(range(1, pairs.indices.numel() + 1)),
                    "gt_boxes6": targets.tolist(), "gt_classes": sample["classes"].tolist(),
                    "gt_groups": [{**g, "near": g["depth_m"] is not None and 0 < g["depth_m"] < 10}
                                  for g in groups],
                })
                if frame_number % 10 == 0:
                    print(f"exported {frame_number}/{len(dataset)} frames", flush=True)
                continue
            prepared = {mode: _prepare_audit(dense, locations, mode) for mode in modes}
            reference_results = {}
            for mode in modes:
                for threshold, topk in settings:
                    key = _setting_key(threshold, topk)
                    diagnosis = audit_frame(dense, locations, targets, threshold, topk,
                                            right_offset_mode=mode, prepared=prepared[mode], apply_nms=apply_nms)
                    _accumulate(all_aggregates[mode][key], diagnosis, frame_id, groups,
                                audit_left=(threshold == .01 and topk == 0), targets=targets)
                    if key == reference_key:
                        reference_results[mode] = diagnosis
                        official = decode_anchorfree_predictions(
                            _dense_inputs(dense), locations, threshold,
                            pre_nms_topk=topk, right_offset_mode=mode, apply_nms=apply_nms,
                        )
                        trace_ids = diagnosis["pairs"].indices
                        if not torch.equal(trace_ids, official.indices):
                            a, b = set(trace_ids.tolist()), set(official.indices.tolist())
                            official_differences.append({"frame_id": frame_id, "mode": mode,
                                                         "trace_only": sorted(a - b), "official_only": sorted(b - a),
                                                         "same_set_different_order": a == b})
            if args.compare_right_offset:
                old, new = reference_results["legacy_clamp"], reference_results["signed"]
                if not torch.equal(old["pre_nms_topk_indices"], new["pre_nms_topk_indices"]):
                    raise AssertionError(f"pre-NMS candidates changed before geometry for {frame_id}")
                fixed_geometry.extend({"frame_id": frame_id, **row} for row in _fixed_candidate_right_geometry(
                    locations, dense, targets, old["pre_nms_topk_indices"], prepared["legacy_clamp"], prepared["signed"],
                ))
            if args.audit_score_breakdown:
                from lightbackbone_stereo_yolo.detection import PairDetections
                from lightbackbone_stereo_yolo.score_diagnostics import audit_score_candidates

                inputs = _dense_inputs(dense)
                objectness = inputs["objectness"].reshape(-1).sigmoid()
                probabilities = inputs["class_logits"].reshape(objectness.numel(), -1).softmax(dim=1)
                _, classes, boxes = prepared[configured_mode]
                target_boxes_cpu = targets.cpu()
                rows = audit_score_candidates(boxes.cpu(), objectness.cpu(), probabilities.cpu(),
                                              target_boxes_cpu, sample["classes"], args.reference_threshold)
                for row, group in zip(rows, groups):
                    row.update(group)
                pool, _, _, _, _ = _candidate_trace(objectness, classes, boxes, 0., args.reference_topk, apply_nms=apply_nms)
                pool_cpu = PairDetections(pool.boxes6.cpu(), pool.scores.cpu(), pool.indices.cpu())
                score_frames.append({"frame_id": frame_id, "targets": rows, "boxes6_gt": target_boxes_cpu,
                                     "objectness_pool": pool_cpu})
            if frame_number % 10 == 0:
                print(f"audited {frame_number}/{len(dataset)} frames", flush=True)
    if args.export_roi_candidates:
        digest = hashlib.sha256()
        with Path(args.checkpoint).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        cache = {
            "checkpoint": args.checkpoint, "checkpoint_sha256": digest.hexdigest(),
            "checkpoint_epoch": payload.get("epoch"), "validation_manifest": config["val_json"],
            "roi_geometry": {"dmax_roi": int(payload["dmax_roi"]),
                             "margin_image_px": 16.,
                             "roi_sampling": model.instance_head.roi_sampling},
            "protocol": {"score": args.reference_threshold, "pre_nms_topk": args.reference_topk,
                         "paired_nms_iou": .5 if apply_nms else None,
                         "minimum_joint_iou": .5, "right_offset_mode": configured_mode},
            "frames": exported_frames,
        }
        if not apply_nms:
            cache["protocol"]["detector_head"] = "one_to_one"
        args.export_roi_candidates.write_text(json.dumps(cache, allow_nan=False), encoding="utf-8")
        print(f"wrote {len(exported_frames)} frames to {args.export_roi_candidates}", flush=True)
        return
    reference = all_aggregates[reference_mode][reference_key]
    for aggregates in all_aggregates.values():
        for aggregate in aggregates.values():
            if len(aggregate["per_frame"]) != len(dataset):
                raise AssertionError("expected one diagnostic record per frame")
            _finalize_aggregate(aggregate)
            aggregate["state_transition_from_reference"] = _state_transitions(reference, aggregate)
    result = {"checkpoint": args.checkpoint, "checkpoint_epoch": payload.get("epoch"),
              "frames": len(dataset), "minimum_joint_iou": .5, "reference": reference_key,
              "reference_mode": reference_mode, "detector_head": "one_to_many" if apply_nms else "one_to_one",
              "settings": all_aggregates[reference_mode],
              "official_trace_differences": official_differences,
              "depth_method": "median valid GT foreground disparity, frame-specific rectified P02/P03"}
    if args.compare_right_offset:
        signed = all_aggregates["signed"]
        legacy_budget = reference["candidate_stages"]["nms"]
        budget_key = min(signed, key=lambda key: (abs(signed[key]["candidate_stages"]["nms"] - legacy_budget),
                                                 key != reference_key, key))
        result.update({"signed_settings": signed, "matched_candidate_budget_setting": budget_key,
                       "fixed_candidate_right_geometry": fixed_geometry})
    if args.audit_score_breakdown:
        result["score_breakdown"] = _finalize_score_breakdown(score_frames, reference, all_aggregates[configured_mode])
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({mode: {key: {"matched": value["formal_paired"], "eligible_gt": value["eligible_gt"],
                                  "pairs": value["candidate_stages"]["nms"]}
                             for key, value in aggregates.items()}
                      for mode, aggregates in all_aggregates.items()}), flush=True)


if __name__ == "__main__":
    main()
