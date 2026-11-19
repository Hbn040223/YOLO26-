"""Re-match retained stereo pairs using a single post-NMS candidate cache."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from lightbackbone_stereo_yolo.detection import PairDetections
from lightbackbone_stereo_yolo.geometry import pair_boxes_to_lr
from lightbackbone_stereo_yolo.roi import make_pair_crop_windows
from lightbackbone_stereo_yolo.roi_budget import RectifiedStereoCalibration, coarse_depth_m, select_roi_pairs
from lightbackbone_stereo_yolo.training_metrics import _pairwise_iou, greedy_match_paired_detections

NEAR_M = 10.
MIN_NEAR_HITS = 19
CROWDED_FRAME = "000005_10"
REFERENCE = (40, 287, 23, 21, 1461, 168, 20, 1)


def read_frames(cache):
    frames = {}
    for row in cache["frames"]:
        frame_id = row["frame_id"]
        boxes = torch.tensor(row["boxes6"], dtype=torch.float32).reshape(-1, 6)
        scores = torch.tensor(row["scores"], dtype=torch.float32)
        ids = torch.tensor(row["dense_ids"], dtype=torch.long)
        classes = torch.tensor(row["classes"], dtype=torch.long)
        targets = torch.tensor(row["gt_boxes6"], dtype=torch.float32).reshape(-1, 6)
        count = ids.numel()
        if (frame_id in frames or not len(boxes) == len(scores) == len(classes) == count
                or ids.unique().numel() != count or (scores[1:] > scores[:-1]).any()
                or row["nms_order"] != list(range(1, count + 1))
                or not len(targets) == len(row["gt_groups"]) == len(row["gt_classes"])):
            raise ValueError(f"invalid candidate cache fields in {frame_id}")
        calib = row["calibration"]
        frames[frame_id] = (
            PairDetections(boxes, scores, ids, classes), targets, row["gt_groups"],
            RectifiedStereoCalibration(**calib) if calib else None,
        )
    return frames


def quantile(values, q):
    return float(torch.quantile(torch.tensor(values, dtype=torch.float64), q)) if values else None


def summarize(frames, limit, quota):
    selected_ids = {}
    matches = {}
    for frame_id, (pairs, targets, groups, calib) in frames.items():
        selected = select_roi_pairs(pairs, max_pairs=limit, near_quota=quota, calibration=calib)
        match = greedy_match_paired_detections(selected.boxes6, targets, selected.scores)
        selected_ids[frame_id] = selected.indices.tolist()
        original_rows = {dense_id: i for i, dense_id in enumerate(pairs.indices.tolist())}
        for pred_index, gt_index, iou in zip(match.prediction_indices.tolist(), match.target_indices.tolist(),
                                            match.qualities.tolist()):
            dense_id = int(selected.indices[pred_index])
            matches[(frame_id, gt_index)] = {
                "dense_id": dense_id, "nms_rank": original_rows[dense_id] + 1,
                "score": float(selected.scores[pred_index]), "joint_iou": iou,
            }
    gt = {(frame_id, i): g for frame_id, (_, _, groups, _) in frames.items() for i, g in enumerate(groups)}
    near = {key for key, g in gt.items() if g["near"]}
    people = {key for key, g in gt.items() if g["person"]}
    counts = [len(ids) for ids in selected_ids.values()]
    total_pairs = sum(counts)
    return {
        "frames": len(frames), "gt": len(gt), "near_gt": len(near), "person_gt": len(people),
        "pairs": total_pairs, "matched": len(matches), "near_matched": len(near & matches.keys()),
        "person_matched": len(people & matches.keys()),
        "precision_proxy": len(matches) / total_pairs if total_pairs else 0.,
        "pairs_per_frame_p50": quantile(counts, .5), "pairs_per_frame_p90": quantile(counts, .9),
        "k": limit, "r": quota, "selected_dense_ids": selected_ids,
    }, matches, near


def transitions(reference, current, near):
    lost = reference.keys() - current.keys()
    gained = current.keys() - reference.keys()
    replaced = {key for key in current.keys() & reference.keys()
                if current[key]["dense_id"] != reference[key]["dense_id"]}
    return {
        name: [{"frame_id": k[0], "gt_index": k[1], "near": k in near,
                "before": reference.get(k), "after": current.get(k)} for k in sorted(keys)]
        for name, keys in (("lost", lost), ("gained", gained), ("replaced", replaced))
    }


def compare_common_near(reference, current, near):
    common = sorted(near & reference.keys() & current.keys())
    result = {"count": len(common)}
    for name, arm in (("reference", reference), ("candidate", current)):
        values = [arm[key]["joint_iou"] for key in common]
        result[name] = {"p10": quantile(values, .1), "median": quantile(values, .5)}
    return result


def crowded_review(frames, baseline, arms):
    pairs, targets, groups, calibration = frames[CROWDED_FRAME]
    depths = coarse_depth_m(pairs.boxes6, calibration)
    pl, pr = pair_boxes_to_lr(pairs.boxes6)
    tl, tr = pair_boxes_to_lr(targets)
    li, ri = _pairwise_iou(pl, tl), _pairwise_iou(pr, tr)
    rank_by_id = {dense_id: i for i, dense_id in enumerate(pairs.indices.tolist())}

    def describe(dense_id, gt_index):
        row = rank_by_id[dense_id]
        return {"dense_id": dense_id, "nms_rank": row + 1, "score": float(pairs.scores[row]),
                "proxy_depth_m": float(depths[row]) if torch.isfinite(depths[row]) else None,
                "left_iou": float(li[row, gt_index]), "right_iou": float(ri[row, gt_index])}

    review = []
    for gt_index, group in enumerate(groups):
        key = (CROWDED_FRAME, gt_index)
        if key not in baseline or group["depth_m"] is None or not 0 < group["depth_m"] < NEAR_M:
            continue
        review.append({
            "gt_index": gt_index, "depth_m": group["depth_m"],
            "original": describe(baseline[key]["dense_id"], gt_index),
            "settings": {
                name: {
                    "selected": [describe(idx, gt_index) for idx in summary["selected_dense_ids"][CROWDED_FRAME]],
                    "qualified_selected_dense_ids": [
                        idx for idx in summary["selected_dense_ids"][CROWDED_FRAME]
                        if min(li[rank_by_id[idx], gt_index], ri[rank_by_id[idx], gt_index]) >= .5
                    ],
                    "formal_match": matches.get(key),
                } for name, (summary, matches) in arms.items()
            },
        })
    return review


def cost_proxy(frames, image_sizes, arms, geometry):
    """Estimate the current P2 cost-volume tensor size without running the instance head."""
    dlocal = 2 * geometry["dmax_roi"] + 1
    for summary, _ in arms.values():
        per_frame = {}
        crop_errors = {}
        for frame_id, dense_ids in summary["selected_dense_ids"].items():
            pairs = frames[frame_id][0]
            n = len(dense_ids)
            if not n:
                per_frame[frame_id] = {"shape": [0, dlocal, 0, 0], "cells": 0}
                continue
            id_to_row = {dense_id: i for i, dense_id in enumerate(pairs.indices.tolist())}
            boxes = pairs.boxes6[[id_to_row[dense_id] for dense_id in dense_ids]]
            try:
                crop = make_pair_crop_windows(
                    boxes, *image_sizes[frame_id], geometry["margin_image_px"],
                    roi_sampling=geometry.get("roi_sampling", "native_pad"),
                )
                h, w = int(crop.h_roi.max()), int(crop.w_roi.max())
                per_frame[frame_id] = {"shape": [n, dlocal, h, w], "cells": n * dlocal * h * w}
            except ValueError as error:
                crop_errors[frame_id] = str(error)
        cells = [row["cells"] for row in per_frame.values()]
        summary["cost_volume_proxy"] = {
            "frames": per_frame, "geometry_errors": crop_errors, "cells_sum": sum(cells),
            "cells_p50": quantile(cells, .5), "cells_p95": quantile(cells, .95),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    cache = json.loads(args.cache.read_text(encoding="utf-8"))
    frames = read_frames(cache)
    unlimited, baseline, near = summarize(frames, None, 0)
    protocol = {"score": .25, "pre_nms_topk": 500, "paired_nms_iou": .5,
                "minimum_joint_iou": .5, "right_offset_mode": "legacy_clamp"}
    reference_keys = ("frames", "gt", "near_gt", "person_gt", "pairs", "matched", "near_matched",
                      "person_matched")
    observed = tuple(unlimited[key] for key in reference_keys)
    # The training log is zero-based and the checkpoint stores the next epoch to resume.
    logged_epoch = int(cache["checkpoint_epoch"]) - 1
    if observed != REFERENCE or logged_epoch != 33 or cache["protocol"] != protocol:
        args.output.write_text(json.dumps({"baseline_mismatch": unlimited, "protocol": cache["protocol"],
                                           "checkpoint_epoch": cache["checkpoint_epoch"]}, indent=2))
        raise ValueError(f"Tail reference mismatch: {dict(zip(reference_keys, observed))}")

    arms = {"unlimited": (unlimited, baseline)}

    def evaluate(k, r):
        name = f"k{k}_r{r}"
        if name in arms:
            return arms[name][0]["passes"]
        summary, matches, _ = summarize(frames, k, r)
        summary["baseline_near_preserved"] = len(near & matches.keys() & baseline.keys())
        summary["vs_unlimited"] = transitions(baseline, matches, near)
        summary["common_near_iou_vs_unlimited"] = compare_common_near(baseline, matches, near)
        if r:
            score_summary, score_matches = arms[f"k{k}_r0"]
            if any(len(summary["selected_dense_ids"][frame_id])
                   != len(score_summary["selected_dense_ids"][frame_id]) for frame_id in frames):
                raise AssertionError("score and near strategies must have identical per-frame budgets")
            summary["vs_same_k_score"] = transitions(score_matches, matches, near)
            summary["common_near_iou_vs_same_k_score"] = compare_common_near(score_matches, matches, near)
            summary["overall_ge_same_k_score"] = summary["matched"] >= score_summary["matched"]
        summary["passes"] = summary["near_matched"] >= MIN_NEAR_HITS and summary.get("overall_ge_same_k_score", True)
        arms[name] = (summary, matches)
        return summary["passes"]

    def evaluate_quotas(k):
        evaluate(k, 0)
        results = [evaluate(k, r) for r in (1, 2, 3, 4)]
        return any([arms[f"k{k}_r0"][0]["passes"], *results])

    for k in (5, 10, 15, 20, 30):
        evaluate(k, 0)
    if evaluate_quotas(10):
        for k in range(5, 10):
            evaluate_quotas(k)
    else:
        for k in (15, 20, 30):
            if evaluate_quotas(k):
                break

    eligible = [name for name, (row, _) in arms.items() if row.get("passes")]
    selected = min(eligible, key=lambda name: (
        arms[name][0]["k"], -arms[name][0]["baseline_near_preserved"],
        -arms[name][0]["near_matched"], -arms[name][0]["matched"], arms[name][0]["r"],
    ), default=None)
    image_sizes = {row["frame_id"]: row["image_hw"] for row in cache["frames"]}
    cost_proxy(frames, image_sizes, arms, cache["roi_geometry"])
    result = {
        "source": {key: val for key, val in cache.items() if key != "frames"},
        "training_log_epoch": logged_epoch, "old_training_log_pairs": 1462,
        "reference_recomputed_pairs": 1461,
        "minimum_near_matches": MIN_NEAR_HITS, "selected": selected,
        "settings": {name: {**row, "matches": [
            {"frame_id": f, "gt_index": i, **v} for (f, i), v in sorted(matches.items())
        ]} for name, (row, matches) in arms.items()},
        "crowded_frame": crowded_review(frames, baseline, arms),
    }
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    for name, (row, _) in arms.items():
        keys = ("pairs", "matched", "near_matched", "baseline_near_preserved", "person_matched",
                "precision_proxy", "pairs_per_frame_p50", "pairs_per_frame_p90", "passes")
        print(name, json.dumps({key: row[key] for key in keys if key in row}))
    print(f"selected={selected}", flush=True)


if __name__ == "__main__":
    main()
