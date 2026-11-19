"""Compare legacy and inside-rescue box supervision on the training split."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lightbackbone_stereo_yolo.data.frame_dataset import KittiFrameDataset, build_class_mapping, build_frame_records
from lightbackbone_stereo_yolo.geometry import assign_anchorfree_left_targets, generate_point_locations


def size_group(side: float) -> str:
    if side <= 16:
        return "le16"
    if side <= 32:
        return "16_to_32"
    if side <= 64:
        return "32_to_64"
    return "gt64"


def box_counts(assignment, num_targets: int) -> torch.Tensor:
    return torch.bincount(assignment.matched_indices[assignment.box_valid_mask], minlength=num_targets)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--train-json", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    records = build_frame_records(args.dataset_root, args.train_json)
    mapping = build_class_mapping(records)
    dataset = KittiFrameDataset(records, mapping, load_images=False)
    locations_by_image_hw = {}
    grouped = {}
    detailed = []
    mode_counts = {"legacy": Counter(), "inside_rescue": Counter()}
    for index in range(len(dataset)):
        sample = dataset[index]
        frame_id = sample["frame_id"]
        height, width = sample["disparity"].shape[-2:]
        image_hw = (height, width)
        if image_hw not in locations_by_image_hw:
            levels = tuple(((height + s - 1) // s, (width + s - 1) // s, s) for s in (8, 16, 32))
            locations_by_image_hw[image_hw] = generate_point_locations(levels)
        locations = locations_by_image_hw[image_hw]
        boxes, classes = sample["boxes6"], sample["classes"]
        assignment = {
            mode: assign_anchorfree_left_targets(locations, boxes, classes, assignment_mode=mode, image_hw=image_hw)
            for mode in mode_counts
        }
        counts = {mode: box_counts(target, len(boxes)) for mode, target in assignment.items()}
        for mode, target in assignment.items():
            mode_counts[mode]["obj_cls_only_points"] += int(((target.labels == 1) & ~target.box_valid_mask).sum())
            mode_counts[mode]["box_valid_points"] += int(target.box_valid_mask.sum())
        new = assignment["inside_rescue"]
        x, y, stride = locations.unbind(1)
        in_image = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        for gt_index, (box, gt_class) in enumerate(zip(boxes, classes)):
            extent = box[2:4] - box[:2]
            size = float(extent.max())
            primary_stride = 8 if size <= 64 else (16 if size <= 128 else 32)
            old_count = int(counts["legacy"][gt_index])
            new_count = int(counts["inside_rescue"][gt_index])
            recovered = old_count == 0 and new_count > 0
            lost = old_count > 0 and new_count == 0
            note = None
            invalid_reasons = []
            if not torch.isfinite(box).all():
                invalid_reasons.append("nonfinite_box")
            if extent[0] <= 0:
                invalid_reasons.append("nonpositive_left_width")
            if extent[1] <= 0:
                invalid_reasons.append("nonpositive_left_height")
            if box[5] - box[4] <= 1e-3:
                invalid_reasons.append("degenerate_right_width")
            if recovered:
                assigned = (new.matched_indices == gt_index) & new.box_valid_mask
                note = "same_scale_rescue" if (stride[assigned] == primary_stride).any() else "cross_scale_rescue"
            elif new_count == 0:
                if invalid_reasons:
                    note = "invalid_gt_geometry"
                else:
                    inside = in_image & (x >= box[0]) & (x <= box[2]) & (y >= box[1]) & (y <= box[3])
                    note = "inside_point_already_assigned" if inside.any() else "no_inside_point_on_any_scale"
            fields = ("all", size_group(size)) + (("person",) if int(gt_class) == mapping.get(24) else ())
            for name in fields:
                row = grouped.setdefault(name, Counter())
                row.update({"gt": 1, "legacy_no_box": int(old_count == 0), "rescue_no_box": int(new_count == 0),
                            "gained": int(recovered), "lost": int(lost)})
                if note:
                    row[note] += 1
                if note == "invalid_gt_geometry":
                    row.update(invalid_reasons)
            if note:
                instance = {"frame_id": frame_id, "gt_index": gt_index, "class": int(gt_class),
                            "max_side_px": size, "legacy_count": old_count, "rescue_count": new_count,
                            "reason": note}
                if invalid_reasons:
                    foreground = sample["masks"][gt_index, 0] & sample["disparity_valid"][0]
                    instance.update({"invalid_reasons": invalid_reasons, "box6": box.tolist(),
                                     "valid_foreground_disparity_pixels": int(foreground.sum())})
                detailed.append(instance)
    output = {"frames": len(dataset), "groups": grouped, "points": mode_counts, "instances": detailed}
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in output.items() if key != "instances"}, indent=2))


if __name__ == "__main__":
    main()
