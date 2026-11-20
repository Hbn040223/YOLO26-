"""Compare paired NMS on/off on one frozen detector and validation split."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from infer import ROOT, load_published_model
from lightbackbone_stereo_yolo.data.frame_dataset import KittiFrameDataset, build_frame_records
from lightbackbone_stereo_yolo.detection import (
    PairDetections, anchorfree_candidate_indices, anchorfree_prediction_scores, select_anchorfree_pairs,
)
from lightbackbone_stereo_yolo.geometry import (
    decode_anchorfree_boxes6, generate_point_locations, pair_boxes_to_lr, point_levels,
)
from lightbackbone_stereo_yolo.training_metrics import (
    _pairwise_iou, greedy_match_paired_detections, paired_match_eligibility,
)


def probabilities(value: str) -> list[float]:
    values = sorted({float(item) for item in value.split(",")})
    if not values or any(not 0 <= item <= 1 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated probabilities in [0, 1]")
    return values


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validation_dataset(dataset_root, split_json, bundle):
    normalization = bundle["normalization"]
    return KittiFrameDataset(
        build_frame_records(dataset_root, split_json, require_files=True), bundle["class_mapping"],
        mean=normalization["mean"], std=normalization["std"], unknown_class="skip",
    )


@torch.inference_mode()
def capture_candidates(args) -> dict:
    model, bundle = load_published_model(args.weights, args.openstereo_source, args.device)
    if model.detector_variant != "stereo_yolo26_pan":
        raise ValueError("paired NMS evaluation requires stereo_yolo26_pan")
    dataset = validation_dataset(args.dataset_root, args.split_json, bundle)
    frames = []
    for sample in dataset:
        left, right = (sample[side].unsqueeze(0).to(args.device) for side in ("left", "right"))
        pyramid = model.encoder(left, right)
        dense = model.detector(pyramid)
        scores, classes = anchorfree_prediction_scores(
            {"objectness": dense.objectness[0], "class_logits": dense.class_logits[0]},
        )
        locations = generate_point_locations(point_levels(pyramid), left.device)
        boxes = decode_anchorfree_boxes6(dense.box6[0], locations, right_offset_mode=model.right_offset_mode)
        # Keep invalid decoded rows in the archive so they still count toward top-K.
        valid = boxes.isfinite().all(1).tolist()
        finite_scores = scores.isfinite().tolist()
        frames.append({
            "frame_id": sample["frame_id"],
            "scores": [score if finite else None for score, finite in zip(scores.tolist(), finite_scores)],
            "classes": classes.tolist(),
            "boxes6": [box if finite else None for box, finite in zip(boxes.tolist(), valid)],
            "gt_boxes6": sample["boxes6"].tolist(), "gt_classes": sample["classes"].tolist(),
        })
    return {
        "schema": "anchorfree-pre-score-v1",
        "weights": str(args.weights), "weights_sha256": sha256(args.weights),
        "checkpoint_epoch": bundle.get("epoch"), "checkpoint_config": bundle["config"],
        "head": "one_to_one" if model.detector.one_to_one_enabled else "one_to_many",
        "split": str(args.split_json), "split_sha256": sha256(args.split_json),
        "right_offset_mode": model.right_offset_mode, "frames": frames,
    }


def counts(pairs, targets, target_classes, gt_iou, class_aware):
    labels = ({"prediction_classes": pairs.classes, "target_classes": target_classes} if class_aware else {})
    paired = greedy_match_paired_detections(pairs.boxes6, targets, pairs.scores, gt_iou, **labels)
    quality, eligible = paired_match_eligibility(pairs.boxes6, targets, gt_iou, **labels)
    qualifying = eligible.any(1)
    owned = torch.zeros(len(targets), dtype=torch.long, device=targets.device)
    if qualifying.any():
        # Ownership is independent of the one-to-one TP assignment. The first
        # maximum is the fixed GT-row tie-break.
        owners = quality[qualifying].masked_fill(~eligible[qualifying], float("-inf")).argmax(1)
        owned = torch.bincount(owners, minlength=len(targets))
    duplicates = (owned - 1).clamp_min(0)
    overlapping = (_pairwise_iou(targets[:, :4], targets[:, :4]) > 0)
    overlapping.fill_diagonal_(False)
    overlapping &= target_classes[:, None] == target_classes[None, :]
    return {
        "pairs": len(pairs.indices), "gt": len(targets), "hits": paired.matched_count,
        "duplicates": int(duplicates.sum()), "overlap_gt_duplicates": int(duplicates[overlapping.any(1)].sum()),
        "overlap_gt": int(overlapping.any(1).sum()),
        "matched_gt_ids": paired.target_indices.tolist(),
        "matched_dense_ids": pairs.indices[paired.prediction_indices].tolist(),
        "kept_dense_ids": pairs.indices.tolist(),
    }


def single_view_hits(boxes, targets, gt_iou):
    # This preserves the existing class-agnostic left recall definition.
    pred_left, pred_right = pair_boxes_to_lr(boxes)
    gt_left, gt_right = pair_boxes_to_lr(targets)
    return (
        int((_pairwise_iou(pred_left, gt_left) >= gt_iou).any(0).sum()),
        int((_pairwise_iou(pred_right, gt_right) >= gt_iou).any(0).sum()),
    )


def evaluate(cache, scores, nms_ious, topk, gt_iou, *, class_aware=True, bootstrap=1000):
    if cache.get("schema") != "anchorfree-pre-score-v1" or not cache.get("frames"):
        raise ValueError("expected a non-empty pre-score candidate archive")
    arms = (None, *nms_ious)
    rows = {(score, iou): [] for score in scores for iou in arms}
    hist = torch.zeros((2, 10), dtype=torch.long)
    frame_ids = set()
    for frame in cache["frames"]:
        frame_id = frame["frame_id"]
        if frame_id in frame_ids:
            raise ValueError(f"duplicate frame ID: {frame_id}")
        frame_ids.add(frame_id)
        boxes = torch.tensor([box if box is not None else [float("nan")] * 6 for box in frame["boxes6"]]).reshape(-1, 6)
        values = torch.tensor([value if value is not None else -1. for value in frame["scores"]])
        classes = torch.tensor(frame["classes"], dtype=torch.long)
        targets = torch.tensor(frame["gt_boxes6"]).reshape(-1, 6)
        target_classes = torch.tensor(frame["gt_classes"], dtype=torch.long)
        if not len(boxes) == len(values) == len(classes) or len(targets) != len(target_classes):
            raise ValueError(f"misaligned candidate archive fields in {frame_id}")
        valid = boxes.isfinite().all(1) & (values >= 0)
        labels = ({"prediction_classes": classes[valid], "target_classes": target_classes} if class_aware else {})
        _, eligible = paired_match_eligibility(boxes[valid], targets, gt_iou, **labels)
        score_bins = (values[valid] * 10).long().clamp(0, 9)
        for group, mask in enumerate((eligible.any(1), ~eligible.any(1))):
            hist[group] += torch.bincount(score_bins[mask], minlength=10)
        for score in scores:
            selected, ids = anchorfree_candidate_indices(values, score, topk)
            valid_ids = ids[valid[ids]]
            pre = PairDetections(boxes[valid_ids], values[valid_ids], valid_ids, classes[valid_ids])
            left_hits, right_hits = single_view_hits(pre.boxes6, targets, gt_iou)
            for iou in arms:
                picked = select_anchorfree_pairs(pre, .5 if iou is None else iou,
                                                 apply_nms=iou is not None, class_aware_nms=True)
                rows[score, iou].append({
                    "frame_id": frame_id, "after_score": len(selected), "after_topk": len(ids),
                    "pre_nms_dense_ids": valid_ids.tolist(), "pre_nms_left_hits": left_hits,
                    "pre_nms_right_hits": right_hits, **counts(picked, targets, target_classes, gt_iou, class_aware),
                })
    draws = torch.randint(len(frame_ids), (bootstrap, len(frame_ids)), generator=torch.Generator().manual_seed(17))

    def totals(frame_rows):
        return torch.tensor([[row[key] for key in ("gt", "pairs", "hits", "duplicates")]
                             for row in frame_rows], dtype=torch.float64)

    def rates(values):
        return values[..., 2:3] / values[..., :2].clamp_min(1)

    settings = []
    for (score, iou), frames in rows.items():
        values = totals(frames)
        gt, pairs, hits, duplicates = (int(value) for value in values.sum(0))
        result = {
            "score_threshold": score, "nms_iou": iou, "gt": gt, "pairs": pairs, "hits": hits,
            "duplicates": duplicates, "coverage": hits / max(1, gt), "precision": hits / max(1, pairs),
            "overlap_gt": sum(row["overlap_gt"] for row in frames),
            "overlap_gt_duplicates": sum(row["overlap_gt_duplicates"] for row in frames),
            "pre_nms_left_hits": sum(row["pre_nms_left_hits"] for row in frames),
            "pre_nms_right_hits": sum(row["pre_nms_right_hits"] for row in frames),
            "frames": frames,
        }
        if iou is not None:
            reference = rows[score, None]
            result["lost_gt_count"] = sum(len(set(before["matched_gt_ids"]) - set(after["matched_gt_ids"]))
                                          for before, after in zip(reference, frames))
            if bootstrap:
                delta = rates(values[draws].sum(1)) - rates(totals(reference)[draws].sum(1))
                interval = torch.quantile(delta, delta.new_tensor([.025, .975]), dim=0).T.tolist()
                result["paired_bootstrap_delta_95ci"] = dict(zip(("coverage", "precision"), interval))
        settings.append(result)
    return {
        "source": {key: value for key, value in cache.items() if key != "frames"},
        "protocol": {"matcher": "quality_first", "class_aware_matching": class_aware,
                     "class_aware_nms": True, "gt_joint_iou": gt_iou, "pre_nms_topk": topk,
                     "bootstrap_frames": bootstrap, "bootstrap_seed": 17,
                     "single_view_recall_matching": "class_agnostic",
                     "duplicate_ownership": "max_eligible_joint_iou_then_lowest_gt_row",
                     "overlap_gt": "same_class_with_positive_left_iou",
                     "prediction_denominator": "all_kept_pairs",
                     "ground_truth_denominator": "valid_paired_annotations_from_manifest",
                     "final_pair_cap": None},
        "score_histogram": {"edges": [i / 10 for i in range(11)],
                            "eligible_for_same_gt": hist[0].tolist(), "other_valid": hist[1].tolist()},
        "settings": settings,
    }


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=ROOT / "models/stereo-yolo26-fixed56-seed17-e20.pt")
    parser.add_argument("--openstereo-source", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--split-json", type=Path)
    parser.add_argument("--candidates", type=Path, help="rerun from a pre-score archive without inference")
    parser.add_argument("--save-candidates", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--scores", type=probabilities, default=[.05, .10, .15, .25, .35, .40])
    parser.add_argument("--nms-ious", type=probabilities, default=[.5, .6, .7])
    parser.add_argument("--gt-iou", type=float, default=.5)
    parser.add_argument("--topk", type=int, default=500, help="0 means unlimited")
    parser.add_argument("--bootstrap", type=int, default=1000, help="paired frame resamples, 0 disables")
    parser.add_argument("--class-agnostic-matching", action="store_true", help="reuse legacy matcher semantics")
    args = parser.parse_args()
    if not 0 < args.gt_iou <= 1 or args.topk < 0 or args.bootstrap < 0:
        parser.error("require gt-iou in (0, 1] and nonnegative topk/bootstrap")
    if args.candidates and (args.save_candidates or args.openstereo_source or args.dataset_root or args.split_json):
        parser.error("--candidates is exclusive with model capture options")
    if not args.candidates and not (args.openstereo_source and args.dataset_root and args.split_json):
        parser.error("provide --candidates or all of --openstereo-source, --dataset-root and --split-json")
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    cache = json.loads(args.candidates.read_text(encoding="utf-8")) if args.candidates else capture_candidates(args)
    result = evaluate(cache, args.scores, args.nms_ious, args.topk, args.gt_iou,
                      class_aware=not args.class_agnostic_matching, bootstrap=args.bootstrap)
    if args.save_candidates:
        write_json(args.save_candidates, cache)
    write_json(args.output, result)


if __name__ == "__main__":
    main()
