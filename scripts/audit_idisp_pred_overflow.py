"""Audit both sampling modes on the same training detections and choose smoke frames."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lightbackbone_stereo_yolo.data.frame_dataset import KittiFrameDataset, build_class_mapping, build_frame_records
from lightbackbone_stereo_yolo.roi import ROI_SAMPLING_MODES, make_pair_crop_windows
from lightbackbone_stereo_yolo.training import build_roi_disparity_targets


def frame_residual(target, boxes, masks, mode, margin):
    truth = target["disparity"]
    crop = make_pair_crop_windows(boxes, *truth.shape[-2:], margin, roi_sampling=mode)
    return crop, build_roi_disparity_targets(truth.unsqueeze(1), target["disparity_valid"].unsqueeze(1), masks, crop)


def audit_instances(target, entry, mode, margin):
    matched = entry["target_indices"]
    if not matched:
        return []
    boxes = torch.tensor(entry["boxes6"], dtype=torch.float32)
    crop, (truth, _, valid, residual) = frame_residual(target, boxes, target["masks"][matched], mode, margin)
    width = crop.c[:, None, None]
    cols = torch.arange(residual.shape[-1], dtype=truth.dtype)[None, None, :] + .5
    x = crop.c_l[:, None, None] + cols * width / crop.w_roi[:, None, None]
    right_x = x - truth[:, 0]
    right_start = crop.c_r[:, None, None]
    outside = (right_x < right_start) | (right_x >= right_start + width)
    rows = []
    for local_idx, gt_idx in enumerate(matched):
        base = valid[local_idx, 0]
        group = entry["gt_groups"][gt_idx]
        rows.append({
            "frame_id": target["frame_id"], "target_index": gt_idx,
            "near_car": group["near_car"], "values": residual[local_idx, 0][base].abs(),
            "outside": int(outside[local_idx][base].sum().item()),
        })
    return rows


def summarize(rows, dmax):
    detail = []
    for row in rows:
        n = row["values"].numel()
        detail.append({
            "frame_id": row["frame_id"], "target_index": row["target_index"], "pixels": n,
            "overflow": int((row["values"] > dmax).sum().item()), "outside": row["outside"],
        })
    pixels = sum(row["pixels"] for row in detail)
    output = {"instances": len(detail), "empty_foreground_instances": sum(not row["pixels"] for row in detail),
              "foreground_pixels": pixels}
    for key in ("overflow", "outside"):
        count = sum(row[key] for row in detail)
        fractions = torch.tensor([row[key] / row["pixels"] for row in detail if row["pixels"]])
        output[key] = {
            "count": count, "ratio": count / pixels if pixels else None,
            "per_instance_p95": float(torch.quantile(fractions, .95)) if fractions.numel() else None,
            "per_instance_max": float(fractions.max()) if fractions.numel() else None,
            "over_half_instances": [row for row in detail if row[key] > .5 * row["pixels"]],
        }
    return output


def pick_overfit_frames(frames, eligible, pressure):
    def choose(order, count):
        for frame in order:
            if len(picked) >= count:
                break
            if frame in eligible and frame not in picked:
                picked.append(frame)
    picked = []
    near = [frame for frame, row in frames.items() if any(g["near_car"] for g in row["gt_groups"])]
    far = [frame for frame, row in frames.items()
           if any(g["car"] and g["depth_m"] is not None and g["depth_m"] > 30 for g in row["gt_groups"])]
    choose(near, 3)
    choose(far, 5)
    choose(sorted(eligible, key=lambda f: (-pressure[f], f)), 8)
    if len(picked) != 8 or not set(picked) & set(near) or not set(picked) & set(far):
        raise RuntimeError("fixed 8 requires shared supervision, near cars and far cars")
    return picked


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = json.loads(args.cache.read_text())
    frames = {row["frame_id"]: row for row in cache["all_frames"]}
    records = build_frame_records(args.dataset_root, cache["manifest"])
    dataset = KittiFrameDataset(records, build_class_mapping(records), load_images=False)
    calibration = {mode: json.loads((args.calibration_dir / f"{mode}_gt.json").read_text())
                   for mode in ROI_SAMPLING_MODES}
    margins = {row["roi_margin"] for row in calibration.values()}
    if len(margins) != 1 or any(Path(row["split_json"]).resolve() != Path(cache["manifest"]).resolve()
                               for row in calibration.values()):
        raise ValueError("both GT calibrations must share the cache manifest and margin")
    margin = margins.pop()
    predicted = defaultdict(list)
    gt = defaultdict(dict)
    for index in range(len(dataset)):
        target = dataset[index]
        frame = frames[target["frame_id"]]
        boxes = target["boxes6"]
        if not torch.equal(boxes, torch.tensor(frame["gt_boxes6"], dtype=boxes.dtype).reshape(-1, 6)):
            raise RuntimeError(f"cached GT boxes changed for {target['frame_id']}")
        for mode in ROI_SAMPLING_MODES:
            predicted[mode].extend(audit_instances(target, frame, mode, margin))
            if boxes.numel():
                _, (_, _, valid, residual) = frame_residual(target, boxes, target["masks"], mode, margin)
                gt[mode][target["frame_id"]] = [r[v].abs() for r, v in zip(residual, valid)]
            else:
                gt[mode][target["frame_id"]] = []
        if (index + 1) % 20 == 0:
            print(f"audited {index + 1}/{len(dataset)} frames", flush=True)
    selection = {}
    counts = {}
    for mode, report in calibration.items():
        dmax = int(report["suggested_dmax_roi"])
        rows = predicted[mode]
        near_rows = [r for r in rows if r["near_car"]]
        start = summarize(rows, dmax)
        start_near = summarize(near_rows, dmax)
        near_values = [r["values"] for r in near_rows if r["values"].numel()]
        wider = None
        # Audit a larger interval when it could recover near-car pixels.
        if near_values and start_near["overflow"]["ratio"] > .01:
            wider = math.ceil(float(torch.quantile(torch.cat(near_values), .995)))
            if mode == "fixed_56":
                wider = min(wider, 55)
        selected = max(dmax, wider or dmax)
        counts[mode] = {frame: sum(bool((values <= selected).any()) for values in instances)
                        for frame, instances in gt[mode].items()}
        pressure = max(counts[mode], key=counts[mode].get)
        audit = {
            "mode": mode, "margin_image_px": margin, "source_cache": str(args.cache.resolve()),
            "suggested_dmax_roi": dmax, "selected_dmax_roi": selected, "disparity_bins": 2 * selected + 1,
            "initial": start, "initial_near_cars": start_near, "selected": summarize(rows, selected),
            "selected_near_cars": summarize(near_rows, selected),
            "rule": "GT P99.5; widen to training matched near-car P99.5 if near overflow exceeds 1%, with fixed grid offset at most 55",
        }
        write_json(args.output_dir / f"{mode}_pred.json", audit)
        selection[mode] = {"dmax_roi": selected, "pressure_frame": pressure,
                           "pressure_supervised_instances": counts[mode][pressure]}
        print(mode, json.dumps(selection[mode]), flush=True)
    eligible = {frame for frame in frames if all(counts[mode][frame] > 0 for mode in ROI_SAMPLING_MODES)}
    pressure = {frame: max(counts[mode][frame] for mode in ROI_SAMPLING_MODES) for frame in eligible}
    overfit = pick_overfit_frames(frames, eligible, pressure)
    write_json(args.output_dir / "fixed_frames.json", {"frame_ids": overfit})
    write_json(args.output_dir / "selection.json", selection)


if __name__ == "__main__":
    main()
