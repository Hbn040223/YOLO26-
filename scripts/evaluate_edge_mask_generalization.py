"""Compare Mask ownership and left-edge recentering in separate server experiments."""
import argparse
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import torch
from torch.nn import functional as F

from diagnose_pred_roi import _rectified_calibration
from edge_run_evidence import digest
from evaluate_fixed_pair_features import image_array
from evaluate_paired_nms import write_json
from infer import load_published_model, load_rectified_pair
from lightbackbone_stereo_yolo.idisp_ab import error_totals, project_disparity, summarize_error
from lightbackbone_stereo_yolo.mask import foreground_disparity_validity, mask_logit_cutoff
from lightbackbone_stereo_yolo.roi_selection import near10_measure
from lightbackbone_stereo_yolo.training_metrics import _pairwise_iou, greedy_match_paired_detections


def foreground(model, pairs, out):
    classes = pairs.classes.new_tensor(model.foreground_class_indices)
    supported = (pairs.classes[:, None] == classes).any(1)
    mask = foreground_disparity_validity(out.mask_logits, out.mask_validity, out.disparity,
                                        out.validity, supported, mask_threshold=model.mask_threshold)
    return mask, supported


def pixels_in_box(rows, cols, box):
    return (cols >= box[0]) & (cols < box[2]) & (rows >= box[1]) & (rows < box[3])


def pixel_competition(model, pairs, out, hw, mask, supported, targets, labels, protection=()):
    """Replay margin-zero competition and optional overlap-limited variants."""
    crop = out.crop
    probability = F.interpolate((out.mask_logits - mask_logit_cutoff(model.mask_threshold)).sigmoid(),
                                size=out.disparity.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    projected, valid = project_disparity(SimpleNamespace(absolute=probability, valid=out.validity), crop, hw)
    if protection:
        spatial = F.interpolate(out.mask_validity.float(), size=probability.shape[-2:], mode="nearest")[:, 0].bool()
        overlap_prob, overlap_valid = project_disparity(
            SimpleNamespace(absolute=probability, valid=out.validity & spatial & torch.isfinite(out.disparity)),
            crop, hw)
    left, right = pairs.boxes6[:, :4], pairs.boxes6[:, [4, 1, 5, 3]]
    duplicate = (_pairwise_iou(left, left) >= .5) & (_pairwise_iou(right, right) >= .5)
    rivals = (pairs.classes[:, None] == pairs.classes[None, :]) & ~duplicate
    rivals.fill_diagonal_(False)
    boxes = left.new_tensor([target["left_box"] for target in targets]).reshape(-1, 4)
    quality = _pairwise_iou(left, boxes)
    results = {"trial": mask.clone()}
    results.update({threshold: mask.clone() for threshold in protection})
    pixels = []
    for k in torch.where(supported)[0].tolist():
        h, w = probability.shape[-2:]
        rows = (crop.y0[k] + (torch.arange(h, device=left.device) + .5) * crop.h_crop[k] / h).long()
        cols = (crop.c_l[k] + (torch.arange(w, device=left.device) + .5) * crop.c[k] / w).long()
        rows, cols = rows.clamp(0, hw[0] - 1)[:, None], cols.clamp(0, hw[1] - 1)
        if rivals[k].any():
            rival_prob = projected[rivals[k]][:, rows, cols]
            rival_support = valid[rivals[k]][:, rows, cols]
            strongest = rival_prob.masked_fill(~rival_support, 0).max(0).values
            results["trial"][k] &= ~((strongest > .5) & (strongest > probability[k]))
            if protection:
                # Compete only where both decoded left boxes and valid foreground overlap.
                rival_boxes = left[rivals[k]][:, :, None, None]
                valid_overlap = (overlap_valid[rivals[k]][:, rows, cols] &
                                 pixels_in_box(rows, cols, rival_boxes.transpose(0, 1)))
                rival_max = overlap_prob[rivals[k]][:, rows, cols].masked_fill(~valid_overlap, 0).max(0).values
                loses_overlap = (pixels_in_box(rows, cols, left[k]) &
                                 (rival_max > .5) & (rival_max > probability[k]))
                for threshold in protection:
                    results[threshold][k] &= ~(loses_overlap & (probability[k] < threshold))
        row = {"pair_id": int(pairs.indices[k]), "total_before": int(mask[k].sum()),
               "total_after": int(results["trial"][k].sum())}
        variants = {str(t): {"total_after": int(results[t][k].sum())} for t in protection}
        if variants:
            row["overlap_protected"] = variants
        if targets and float(quality[k].max()) >= .5:
            target = targets[int(quality[k].argmax())]
            ids = labels[rows, cols]
            own = ids == target["id"]
            other = (ids > 0) & ~own
            row.update(gt_id=target["id"], gt_depth_m=target["depth_m"])
            if target.get("mask_available", True):
                row.update(own_before=int((mask[k] & own).sum()), own_after=int((results["trial"][k] & own).sum()),
                           other_before=int((mask[k] & other).sum()),
                           other_after=int((results["trial"][k] & other).sum()))
                for threshold in protection:
                    variants[str(threshold)].update(own_after=int((results[threshold][k] & own).sum()),
                                                    other_after=int((results[threshold][k] & other).sum()))
        pixels.append(row)
    return results, pixels


def score(pairs, distances, car, targets, *, bilateral=False, ignore_boxes=()):
    active = torch.where((pairs.classes == car) & torch.isfinite(distances) & (distances <= 10))[0]
    unknown = torch.where((pairs.classes == car) & ~torch.isfinite(distances))[0]
    boxes = pairs.boxes6[active]
    left = boxes[:, :4]
    gt_left = boxes.new_tensor([x["left_box"] for x in targets]).reshape(-1, 4)
    gt_right = boxes.new_tensor([x["right_box"] for x in targets]).reshape(-1, 4) if bilateral else gt_left
    gt = torch.cat((gt_left, gt_right[:, [0, 2]]), 1)
    match_boxes = boxes if bilateral else torch.cat((left, left[:, [0, 2]]), 1)
    match = greedy_match_paired_detections(match_boxes, gt, pairs.scores[active])
    kept = {}
    near_hits = {}
    unknown_depth = []
    fp = {}
    for p, t in zip(match.prediction_indices.tolist(), match.target_indices.tolist()):
        target = targets[t]
        pair_id = int(pairs.indices[active[p]])
        kept[p] = target["id"]
        depth = target["depth_m"]
        if depth is None:
            unknown_depth.append(pair_id)
        elif 0 < depth <= 10:
            near_hits[str(target["id"])] = pair_id
        else:
            fp[str(pair_id)] = "matched_far_gt"
    quality = _pairwise_iou(left, gt_left)
    if bilateral:
        quality = torch.minimum(quality, _pairwise_iou(boxes[:, [4, 1, 5, 3]], gt_right))
    ignored = []
    zones = left.new_tensor(ignore_boxes).reshape(-1, 4)
    intersection = (torch.minimum(left[:, None, 2:], zones[:, 2:]) -
                    torch.maximum(left[:, None, :2], zones[:, :2])).clamp_min(0).prod(-1)
    area = (left[:, 2:] - left[:, :2]).clamp_min(0).prod(-1)
    overlap = (intersection > .5 * area[:, None]).any(1)
    for p, k in enumerate(active.tolist()):
        if p not in kept:
            pair_id = int(pairs.indices[k])
            if bool((quality[p] >= .5).any()):
                fp[str(pair_id)] = "duplicate"
            elif overlap[p]:
                ignored.append(pair_id)
            else:
                fp[str(pair_id)] = "unmatched"
    return {"near_gt_ids": [x["id"] for x in targets if x["depth_m"] is not None and 0 < x["depth_m"] <= 10],
            "near_hits": near_hits, "fp": fp, "unknown_gt_depth_pairs": unknown_depth,
            "unknown_measured_pairs": pairs.indices[unknown].tolist(), "kept_car_pairs": pairs.indices[active].tolist(),
            "ignored_pairs": ignored}


def run_arm(model, left, pyramid, dense, geometry, calib, recenter):
    head = model.instance_head
    head.recenter_left_edge = recenter
    captured = []
    hook = head.register_forward_hook(lambda _, __, out: captured.append(out))
    route = {}
    try:
        final = model.decode_and_refine(left, pyramid, dense, input_geometry=geometry, route=route, calibration=calib)
    finally:
        hook.remove()
    pairs = route["idisp"]
    if not captured:
        assert len(pairs.indices) == 0
        empty = pairs.boxes6.new_empty(0)
        return pairs, None, None, None, empty
    instance, = captured
    mask, supported = foreground(model, pairs, instance)
    valid = torch.where(supported[:, None, None], mask, instance.validity)
    distance = near10_measure(instance.disparity, valid, calib)
    keep = torch.isfinite(distance) & (distance <= 10)
    torch.testing.assert_close(pairs.indices[keep], final.pair_ids)
    torch.testing.assert_close(distance[keep], final.near10_distance_m)
    return pairs, instance, mask, supported, distance


def finite_list(values):
    return [float(x) if np.isfinite(x) else None for x in values.tolist()]


def edge_results(base, trial, hw, labels, truth, targets, calib):
    pairs, first, _, _, old_dist = base
    changed_pairs, second, _, _, new_dist = trial
    torch.testing.assert_close(pairs.indices, changed_pairs.indices)
    torch.testing.assert_close(pairs.boxes6, changed_pairs.boxes6)
    if first is None:
        return []
    before = first.crop.c_l - first.crop.c_r
    after = second.crop.c_l - second.crop.c_r
    changed = torch.where(before != after)[0]
    same = before == after
    torch.testing.assert_close(first.disparity[same], second.disparity[same], rtol=0, atol=0, equal_nan=True)
    if not len(changed):
        return []
    projected = [project_disparity(SimpleNamespace(absolute=o.disparity, valid=o.validity), o.crop, hw)
                 for o in (first, second)]
    result = []
    for k in changed.tolist():
        left_box = pairs.boxes6[k:k+1, :4]
        gt_boxes = left_box.new_tensor([x["left_box"] for x in targets]).reshape(-1, 4)
        ious = _pairwise_iou(left_box, gt_boxes)[0]
        row = {"pair_id": int(pairs.indices[k]), "box6": pairs.boxes6[k].tolist(),
               "d0_before": float(before[k]), "d0_after": float(after[k]),
               "distance_before": finite_list(old_dist[k:k+1])[0], "distance_after": finite_list(new_dist[k:k+1])[0],
               "valid_before": int(first.validity[k].sum()), "valid_after": int(second.validity[k].sum())}
        if len(ious) and float(ious.max()) >= .5:
            target = targets[int(ious.argmax())]
            ry = pairs.boxes6[k:k+1, [4, 1, 5, 3]]
            right_box = target.get("right_box")
            row.update(gt_id=target["id"], gt_depth_m=target["depth_m"], left_iou=float(ious.max()),
                       right_iou=float(_pairwise_iou(ry, ry.new_tensor([right_box]))[0, 0]) if right_box else None)
            if truth is not None:
                gt = (labels == target["id"]) & (truth > 0)
                shared = gt & projected[0][1][k] & projected[1][1][k]
                row["common_pixel_error"] = [summarize_error(error_totals(d[k], truth, shared)) for d, _ in projected]
                corrected = truth[gt] - calib[1]
                row["nominal_overflow"] = [int(((corrected + calib[1] - d0[k]).abs() >
                    16 * o.crop.c[k] / o.crop.w_roi[k]).sum()) for o, d0 in ((first, before), (second, after))]
        result.append(row)
    return result


def split_overlap(train, val):
    sets = []
    for path in (train, val):
        sets.append({Path(item["file_name"]).stem for item in json.loads(path.read_text())["images"]})
    return sets


def development_manifest(root, train, val, calibration, prior_manifest, case_frames):
    training, validation = split_overlap(train, val)
    if training & validation:
        raise ValueError("overlapping training and validation frame IDs")
    known_frames = {row["frame_id"]: {m["id"]: m for m in row["instances"] if m["right_box_verified"]}
                    for row in json.loads(prior_manifest.read_text())}
    with ZipFile(calibration) as archive:
        for frame in sorted(case_frames or validation):
            base = root / "training"
            label_path = base / "instance" / (frame + ".png")
            disp_path = base / "disp_occ_0" / (frame + ".png")
            labels = image_array(str(label_path), np.int64)
            disp = image_array(str(disp_path), np.float32) / 256
            calib = _rectified_calibration(archive, frame)
            targets = []
            for id in labels.unique().tolist():
                if id // 256 != 26:
                    continue
                y, x = torch.where(labels == id)
                valid = disp[y, x] > 0
                bf, cx = calib["bf_px_m"], calib["cx_delta_px"]
                corrected = disp[y[valid], x[valid]] - cx
                corrected = corrected[corrected > 0]
                depth = float(bf / corrected.median()) if len(corrected) else None
                left = [int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1]
                right_x = x[valid] - disp[y[valid], x[valid]]
                right_x = right_x[right_x >= 0]
                right = [float(right_x.min()), left[1], float(right_x.max()) + 1, left[3]] if len(right_x) else None
                checked = known_frames.get(frame, {}).get(id)
                if checked:
                    right = checked["right_box"]
                targets.append({"id": id, "depth_m": depth, "left_box": left, "right_box": right})
            yield {"frame_id": frame, "left": str(base / "image_2" / (frame + ".png")),
                   "right": str(base / "image_3" / (frame + ".png")), "labels": str(label_path),
                   "disparity": str(disp_path), "calibration": [calib["bf_px_m"], calib["cx_delta_px"]],
                   "targets": targets}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("mask", "edge"))
    for name in ("weights", "openstereo", "output"):
        parser.add_argument(name, type=Path)
    for name in ("manifest", "development", "train", "val", "calibration", "prior-manifest"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--case-frame", action="append", default=[], help="run only these development diagnostic frames")
    parser.add_argument("--overlap-protect", type=float, nargs="+", default=[],
                        help="additional Mask arms; protect calibrated own probability at or above each threshold")
    args = parser.parse_args()
    if args.overlap_protect and (args.mode != "mask" or
                                 any(not .5 < t <= 1 for t in args.overlap_protect)):
        parser.error("--overlap-protect requires Mask mode and thresholds in (.5, 1]; 1 is overlap-only")
    protection = sorted(set(args.overlap_protect))
    device = torch.device("cuda:0")
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model, bundle = load_published_model(args.weights, args.openstereo, device, score=.48, apply_paired_nms=True,
                                         near10_mode="range", idisp_cost_backend="bmm")
    model.instance_head.idisp.force_fp32_aggregation = True
    idisp = model.instance_head.idisp
    assert idisp.search.dmax_roi == 16 and idisp.aggregation.stem[0].out_channels == 32
    car = bundle["class_mapping"][26]
    frames = json.loads(args.manifest.read_text()) if args.manifest else development_manifest(
        args.development, args.train, args.val, args.calibration, args.prior_manifest, args.case_frame)
    rows = []
    for frame in frames:
        left, right, geometry = load_rectified_pair(Path(frame["left"]), Path(frame["right"]), bundle["normalization"])
        left, right = left.to(device), right.to(device)
        hw = geometry.original_hw
        labels = image_array(frame["labels"], np.int64).to(device)
        if args.development:
            labels = torch.where(labels // 256 == 26, labels, 0)
        assert tuple(labels.shape) == tuple(hw)
        targets = frame["targets"]
        calib = frame["calibration"]
        pyramid = model.encoder(left, right)
        dense = model.detector(pyramid)
        base = run_arm(model, left, pyramid, dense, geometry, calib, False)
        pairs, out, mask, supported, distance = base
        ignore_boxes = frame.get("ignore_boxes", ())
        row = {"frame_id": frame["frame_id"], "idisp_pairs": len(pairs.indices), "targets": targets,
               "baseline": score(pairs, distance, car, targets, ignore_boxes=ignore_boxes)}
        if args.mode == "mask":
            after = distance
            row["pixels"] = []
            variants = {t: distance for t in protection}
            if out is not None:
                masks, row["pixels"] = pixel_competition(model, pairs, out, hw, mask, supported, targets, labels,
                                                        protection)
                for arm, trial in masks.items():
                    valid = torch.where(supported[:, None, None], trial, out.validity)
                    measured = near10_measure(out.disparity, valid, calib)
                    if arm == "trial":
                        after = measured
                    else:
                        variants[arm] = measured
            row["trial"] = score(pairs, after, car, targets, ignore_boxes=ignore_boxes)
            row["distance_before"] = dict(zip(map(str, pairs.indices.tolist()), finite_list(distance)))
            row["distance_after"] = dict(zip(map(str, pairs.indices.tolist()), finite_list(after)))
            if variants:
                row["overlap_protected"] = {
                    str(t): {"score": score(pairs, d, car, targets, ignore_boxes=ignore_boxes),
                             "distances": dict(zip(map(str, pairs.indices.tolist()), finite_list(d)))}
                    for t, d in variants.items()
                }
        else:
            trial = run_arm(model, left, pyramid, dense, geometry, calib, True)
            truth = image_array(frame["disparity"], np.float32).to(device) / 256 if frame.get("disparity") else None
            row["trial"] = score(trial[0], trial[-1], car, targets, ignore_boxes=ignore_boxes)
            row["changed"] = edge_results(base, trial, hw, labels, truth, targets, calib)
            verified = [t for t in targets if t["right_box"] is not None]
            row["baseline_paired_proxy"] = score(pairs, distance, car, verified, bilateral=True)
            row["trial_paired_proxy"] = score(trial[0], trial[-1], car, verified, bilateral=True)
        rows.append(row)
        print(frame["frame_id"], len(pairs.indices), flush=True)
    totals = {}
    for arm in ("baseline", "trial", "baseline_paired_proxy", "trial_paired_proxy"):
        if rows and arm in rows[0]:
            items = [row[arm] for row in rows]
            totals[arm] = {key: sum(len(x[key]) for x in items) for key in
                           ("near_gt_ids", "near_hits", "fp", "unknown_measured_pairs", "kept_car_pairs",
                            "ignored_pairs", "unknown_gt_depth_pairs")}
            totals[arm]["fp_types"] = dict(Counter(reason for row in items for reason in row["fp"].values()))
    for key, a, b in (("new_misses", "baseline", "trial"), ("new_hits", "trial", "baseline")):
        totals[key] = [[row["frame_id"], gt] for row in rows
                       for gt in sorted(row[a]["near_hits"].keys() - row[b]["near_hits"].keys())]
    write_json(args.output, {"mode": args.mode, "weights_sha256": digest(args.weights), "script_sha256": digest(Path(__file__)),
                             "manifest_sha256": digest(args.manifest) if args.manifest else digest(args.val),
                             "settings": {"dmax_roi": 16, "bins": 33, "aggregation_width": 32,
                                          "cost": "bmm", "aggregation": "fp32", "mask_margin": 0, "score": .48,
                                          "overlap_protect": protection},
                             "totals": totals, "frames": rows})


if __name__ == "__main__":
    main()
