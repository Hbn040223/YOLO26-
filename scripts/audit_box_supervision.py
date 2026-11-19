"""Audit how many left boxes can be supervised independently of right labels."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from audit_anchorfree_assignment import size_group
from lightbackbone_stereo_yolo.anchorfree_training import build_anchorfree_detection_targets
from lightbackbone_stereo_yolo.data.frame_dataset import KittiFrameDataset, build_class_mapping, build_frame_records
from lightbackbone_stereo_yolo.geometry import generate_point_locations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split-json", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    records = build_frame_records(args.dataset_root, args.split_json)
    mapping = build_class_mapping(records)
    dataset = KittiFrameDataset(records, mapping, load_images=False)
    locations_by_shape = {}
    grouped = {}
    points = Counter()
    left_only_instances = []
    for index in range(len(dataset)):
        sample = dataset[index]
        height, width = shape = sample["disparity"].shape[-2:]
        if shape not in locations_by_shape:
            levels = tuple(((height + s - 1) // s, (width + s - 1) // s, s) for s in (8, 16, 32))
            locations_by_shape[shape] = generate_point_locations(levels)
        locations = locations_by_shape[shape]
        boxes, cls = sample["boxes6"], sample["classes"]
        targets = build_anchorfree_detection_targets(locations, boxes, cls, image_hw=shape)
        joint = targets["box_valid_mask"]
        left = targets["left_box_valid_mask"]
        paired = joint & left
        matched = targets["matched_indices"]
        joint_per_gt = torch.bincount(matched[joint], minlength=len(boxes))
        left_per_gt = torch.bincount(matched[left], minlength=len(boxes))
        paired_per_gt = torch.bincount(matched[paired], minlength=len(boxes))
        points.update({"joint": int(joint.sum()), "split_left": int(left.sum()), "split_paired": int(paired.sum()),
                       "split_left_only": int((left & ~paired).sum())})
        for i, (box, label) in enumerate(zip(boxes, cls)):
            side = float((box[2:4] - box[:2]).max())
            joint_count, left_count, paired_count = int(joint_per_gt[i]), int(left_per_gt[i]), int(paired_per_gt[i])
            recovered = not joint_count and left_count > 0
            groups = ("all", size_group(side)) + (("person",) if int(label) == mapping.get(24) else ())
            for group in groups:
                grouped.setdefault(group, Counter()).update(
                    {"gt": 1, "joint_supervised_gt": int(joint_count > 0),
                     "split_left_supervised_gt": int(left_count > 0),
                     "split_paired_supervised_gt": int(paired_count > 0),
                     "new_left_supervised_gt": int(recovered),
                     "lost_paired_gt": int(joint_count > 0 and paired_count == 0)}
                )
            if recovered:
                left_only_instances.append({"frame_id": sample["frame_id"], "gt_index": i,
                                            "max_side_px": side, "class_id": int(label),
                                            "left_points": left_count, "right_width": float(box[5] - box[4])})
    result = {"frames": len(dataset), "groups": grouped, "points": points, "left_only_instances": left_only_instances}
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "left_only_instances"}), flush=True)


if __name__ == "__main__":
    main()
