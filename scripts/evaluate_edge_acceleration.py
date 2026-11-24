"""Evaluate crop or containment on a frozen split in original image coordinates."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import torch

from benchmark_edge_acceleration import CONFIGURATIONS, configure, revision
from diagnose_pred_roi import _target_groups
from evaluate_paired_nms import sha256, validation_dataset, write_json
from infer import ROOT, load_published_model
from lightbackbone_stereo_yolo.detection import PairDetections, suppress_contained_pairs
from lightbackbone_stereo_yolo.idisp_ab import error_totals, original_foreground, project_disparity, summarize_error
from lightbackbone_stereo_yolo.roi_selection import subset_crop
from lightbackbone_stereo_yolo.training_metrics import greedy_match_paired_detections, paired_match_eligibility


CLASSES = {24: "Pedestrian", 26: "Car", 27: "Truck", 28: "Bus"}
REVIEW_FRAMES = {"000005", "000115", "000126", "000083"}


def crop_geometry(box, crop, nearby):
    top, bottom = float(box[1]), float(box[3])
    return {
        "fully_cropped": bottom < crop,
        "truncated": top < crop <= bottom,
        "near_crop_line": top <= crop + nearby and bottom >= crop - nearby,
    }


def make_pairs(output):
    boxes = torch.cat((output.boxes_left, output.boxes_right[:, [0, 2]]), dim=1)
    return PairDetections(boxes.cpu(), output.scores.cpu(), output.pair_ids.cpu(), output.classes.cpu())


def match(pairs, sample, iou):
    gt = sample["boxes6"]
    labels = {"prediction_classes": pairs.classes, "target_classes": sample["classes"]}
    selected = greedy_match_paired_detections(pairs.boxes6, gt, pairs.scores, iou, **labels)
    return dict(zip(selected.target_indices.tolist(), selected.prediction_indices.tolist()))


def containment_audit(baseline, trial, c0, c1, events, sample, iou, gt_ids):
    quality, eligible = paired_match_eligibility(
        baseline.boxes6, sample["boxes6"], iou,
        prediction_classes=baseline.classes, target_classes=sample["classes"],
    )
    baseline_owner = {int(baseline.indices[index]): gt for gt, index in c0.items()}
    candidate_index = {value: index for index, value in enumerate(baseline.indices.tolist())}

    def owner(pair_id):
        if pair_id in baseline_owner:
            return baseline_owner[pair_id]
        index = candidate_index[pair_id]
        choices = eligible[index]
        if choices.any():
            return int(quality[index].masked_fill(~choices, float("-inf")).argmax())
        return None

    def audit_row(event):
        gt, suppressor_gt = owner(event.suppressed_id), owner(event.suppressor_id)
        decision = "review_unmatched"
        if gt is not None and suppressor_gt is not None:
            decision = "same_gt_candidate" if gt == suppressor_gt else "different_gt_candidate"
        return {
            **asdict(event), "gt_id": gt_ids[gt] if gt is not None else None,
            "suppressor_gt_id": gt_ids[suppressor_gt] if suppressor_gt is not None else None,
            "c0_gt_hit": gt in c0, "decision": decision,
            "c1_replacement_pair_id": int(trial.indices[c1[gt]]) if gt in c1 else None,
        }

    return [audit_row(event) for event in events]


def projected(output, matched, sample, *, foreground=False):
    if not matched:
        return {}
    targets, predictions = zip(*matched.items())
    selected = torch.as_tensor(predictions, device=output.disparity.device)
    crop = subset_crop(output.crop, selected)
    valid = output.foreground_validity if foreground else output.validity
    disparity = SimpleNamespace(absolute=output.disparity[selected], valid=valid[selected])
    maps, masks = project_disparity(disparity, crop, output.image_hw)
    return dict(zip(targets, zip(maps.cpu(), masks.cpu())))


def rectangular_support(sample):
    truth = sample["disparity"]
    _, height, width = truth.shape
    x = torch.arange(width)[None, None, :]
    y = torch.arange(height)[None, :, None]
    x1, y1, x2, y2 = sample["boxes6"][:, :4].unbind(1)
    return ((x >= x1[:, None, None]) & (x <= x2[:, None, None])
            & (y >= y1[:, None, None]) & (y <= y2[:, None, None])
            & sample["disparity_valid"] & torch.isfinite(truth) & (truth > 0))


def disparity_scores(predictions, truth, support):
    totals = {"pixels": 0, "epe_sum": 0., "d1_hits": 0}
    for gt, (prediction, validity) in predictions.items():
        error = error_totals(prediction, truth[0], support[gt] & validity)
        for field, value in error.items():
            totals[field] += value
    return {**totals, "gt_valid_pixels": int(support.sum()), "matched_gt": len(predictions)}


def reduce_pixel_metrics(rows):
    if rows[0] is None:
        return None
    values = {key: sum(row[key] for row in rows) for key in rows[0]}
    denominator = values.get("gt_valid_pixels")
    if denominator is not None:
        values["pixel_coverage"] = values["pixels"] / denominator if denominator else None
    return summarize_error(values)


@torch.inference_mode()
def evaluate(model, dataset, settings, mapping, device, iou, nearby, calibration):
    geometry = {(setting[1], name): {"gt": 0, "fully_cropped": [], "truncated": [],
                                    "near_crop_line": [], "review_gt_ids": []}
                for setting in settings.values() if setting[1] for name in CLASSES.values()}
    classes = {index: CLASSES[raw] for raw, index in mapping.items() if raw in CLASSES}
    fg_class = mapping.get(26) if mapping.get(26) in model.foreground_class_indices else None
    frames = []
    for sample in dataset:
        frame_id = sample["frame_id"]
        gt_ids = [f"{frame_id}:{i}" for i in range(len(sample["boxes6"]))]
        gt_classes = [classes[label] for label in sample["classes"].tolist()]
        near = set()
        if calibration is not None:
            groups = _target_groups(sample, mapping, calibration)
            near = {i for i, g in enumerate(groups) if gt_classes[i] == "Car"
                    and g["depth_m"] is not None and 0 < g["depth_m"] < 10}
        for crop, name in geometry:
            row = geometry[crop, name]
            rows = [i for i, label in enumerate(gt_classes) if label == name]
            row["gt"] += len(rows)
            for i in rows:
                if frame_id.split("_")[0] in REVIEW_FRAMES or i in near:
                    row["review_gt_ids"].append(gt_ids[i])
                for field, affected in crop_geometry(sample["boxes6"][i], crop, nearby).items():
                    if affected:
                        row[field].append(gt_ids[i])

        left, right = (sample[side][None].to(device) for side in ("left", "right"))
        table = {"frame_id": frame_id, "gt_ids": gt_ids, "gt_classes": gt_classes,
                 "near_car_ids": [gt_ids[i] for i in sorted(near)], "arms": {}}
        pairs, matches, rect_maps, fg_maps = {}, {}, {}, {}
        truth, fg_support = original_foreground(sample, "cpu")
        rect_support = rectangular_support(sample)
        if fg_class is not None:
            fg_support &= (sample["classes"] == fg_class)[:, None, None]
            eligible_fg = fg_support.flatten(1).any(1)
        for arm, setting in settings.items():
            configure(model, setting)
            result = model(left, right)
            pairs[arm] = make_pairs(result)
            matches[arm] = match(pairs[arm], sample, iou)
            rect_maps[arm] = projected(result, matches[arm], sample)
            if fg_class is not None:
                fg_matches = {gt: p for gt, p in matches[arm].items() if eligible_fg[gt]}
                fg_maps[arm] = projected(result, fg_matches, sample, foreground=True)
            table["arms"][arm] = {
                "match_pair_ids_by_gt": {gt_ids[g]: int(pairs[arm].indices[p]) for g, p in matches[arm].items()},
                "rectangular": disparity_scores(rect_maps[arm], truth, rect_support),
                "car_foreground": None if fg_class is None else {
                    **disparity_scores(fg_maps[arm], truth, fg_support),
                    "eligible_gt": int(eligible_fg.sum()),
                },
            }
        if "C1" in pairs:
            replay, events = suppress_contained_pairs(pairs["C0"], .90)
            if not torch.equal(replay.indices, pairs["C1"].indices):
                raise RuntimeError(f"{frame_id}: C1 online candidate IDs differ from C0 replay")
            table["c0_candidates"] = {"boxes6": pairs["C0"].boxes6.tolist(),
                                      "classes": pairs["C0"].classes.tolist(), "scores": pairs["C0"].scores.tolist(),
                                      "pair_ids": pairs["C0"].indices.tolist()}
            table["suppression_events"] = containment_audit(
                pairs["C0"], replay, matches["C0"], matches["C1"], events, sample, iou, gt_ids)

        first = next(iter(settings))
        for arm in settings:
            seen, baseline = set(matches[arm]), set(matches[first])
            row = table["arms"][arm]
            row["newly_hit_gt_ids"] = [gt_ids[i] for i in sorted(seen - baseline)]
            row["lost_gt_ids"] = [gt_ids[i] for i in sorted(baseline - seen)]
            cohorts = {name: {i for i, label in enumerate(gt_classes) if label == name}
                       for name in CLASSES.values()}
            if calibration is not None:
                cohorts["near_car"] = near
            row["coverage"] = {name: {"gt": len(targets), "hits": len(targets & seen),
                                      "lost": len(targets & (baseline - seen)),
                                      "gained": len(targets & (seen - baseline))}
                               for name, targets in cohorts.items()}
            for family, maps, support in (("rectangular", rect_maps, rect_support),
                                          ("car_foreground", fg_maps, fg_support)):
                if not maps:
                    continue
                common = {name: {} for name in (first, arm)}
                for gt in maps[first].keys() & maps[arm].keys():
                    base_pred, base_valid = maps[first][gt]
                    trial_pred, trial_valid = maps[arm][gt]
                    valid = base_valid & trial_valid
                    common[first][gt] = (base_pred, valid)
                    common[arm][gt] = (trial_pred, valid)
                row[family + "_common"] = {
                    "baseline": disparity_scores(common[first], truth, support),
                    "trial": disparity_scores(common[arm], truth, support),
                }
        frames.append(table)
    if not frames:
        raise ValueError("validation split contains no frames")
    summary = {}
    for arm in settings:
        rows = [frame["arms"][arm] for frame in frames]
        summary[arm] = {
            "coverage": {name: {key: sum(row["coverage"][name][key] for row in rows)
                                for key in ("gt", "hits", "lost", "gained")}
                         for name in rows[0]["coverage"]},
            **{key: reduce_pixel_metrics([row[key] for row in rows])
               for key in ("rectangular", "car_foreground")},
        }
        for family in ("rectangular", "car_foreground"):
            key = family + "_common"
            if key in rows[0]:
                summary[arm][key] = {role: reduce_pixel_metrics([row[key][role] for row in rows])
                                     for role in ("baseline", "trial")}
    return {
        "summary": summary, "frames": frames,
        "crop_gt_geometry": [{"crop_top_px": crop, "class": name, **row}
                             for (crop, name), row in geometry.items()],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("crop", "containment"), required=True)
    parser.add_argument("--weights", type=Path, default=ROOT / "models/stereo-yolo26-fixed56-seed17-e20.pt")
    parser.add_argument("--openstereo-source", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--calibration-zip", type=Path, help="enable the frozen under-10 m Car cohort")
    parser.add_argument("--near-crop-line-px", type=int, required=True)
    parser.add_argument("--match-iou", type=float, default=.50)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.near_crop_line_px < 0 or not 0 < args.match_iou <= 1:
        parser.error("crop band must be nonnegative and match IoU in (0, 1]")
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    model, bundle = load_published_model(args.weights, args.openstereo_source, device, apply_paired_nms=True)
    dataset = validation_dataset(args.dataset_root, args.split_json, bundle)
    settings = CONFIGURATIONS[args.stage]
    with ZipFile(args.calibration_zip) if args.calibration_zip else nullcontext() as calibration:
        result = evaluate(model, dataset, settings, bundle["class_mapping"], device,
                          args.match_iou, args.near_crop_line_px, calibration)
    result["protocol"] = {
        "commit": revision(ROOT), "openstereo_commit": revision(args.openstereo_source),
        "weights_sha256": sha256(args.weights), "split_sha256": sha256(args.split_json),
        "class_mapping": bundle["class_mapping"], "mask_threshold": model.mask_threshold,
        "foreground_mask_raw_classes": bundle.get("foreground_mask_raw_classes", ()),
        "foreground_class_indices": model.foreground_class_indices,
        "paired_match_iou": args.match_iou, "near_crop_line_px": args.near_crop_line_px,
        "near_car_distance_m_exclusive": 10 if args.calibration_zip else None,
        "device": str(device), "settings": settings,
    }
    write_json(args.output, result)


if __name__ == "__main__":
    main()
