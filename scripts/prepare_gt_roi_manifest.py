"""Stage KITTI stereo-box review candidates without marking projections verified."""
import argparse
import json
import math
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw

def rectified_calibration(archive, frame):
    stem = frame.split("_", 1)[0]
    with archive.open(f"training/calib_cam_to_cam/{stem}.txt") as handle:
        projections = {}
        for line in handle.read().decode("ascii").splitlines():
            key, _, values = line.partition(":")
            if key in ("P_rect_02", "P_rect_03"):
                projections[key] = [float(value) for value in values.split()]
    left, right = projections["P_rect_02"], projections["P_rect_03"]
    bf, cx_delta = left[3] - right[3], left[2] - right[2]
    if not (math.isfinite(bf) and math.isfinite(cx_delta) and abs(left[0] - right[0]) <= 1e-4 and bf > 0):
        raise ValueError(f"invalid stereo calibration for {frame}")
    return {"bf_px_m": bf, "cx_delta_px": cx_delta}


def bounds(x, y):
    return [int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1]


def propose_instances(labels, disparity, calibration):
    h, w = labels.shape
    rows = []
    for instance_id in np.unique(labels):
        if instance_id // 256 != 26:
            continue
        mask = labels == instance_id
        y, x = np.where(mask)
        ly, lx = np.where(mask & (disparity > 0))
        if len(lx) == 0:
            continue
        values = disparity[ly, lx]
        rx = np.rint(lx - values).astype(int)
        supported = (rx >= 0) & (rx < w)
        if not supported.any():
            continue
        d = float(np.median(values)) - calibration["cx_delta_px"]
        rows.append(dict(id=int(instance_id), category_id=26, left_box=bounds(x, y),
                         right_box=bounds(rx[supported], ly[supported]), right_box_verified=False,
                         right_box_source="projected_from_left_disp_occ_0_pending_review",
                         gt_depth_m=calibration["bf_px_m"] / d if d > 0 else None,
                         valid_fraction=float(len(lx) / len(x)),
                         touches_edge=bool(x.min() == 0 or y.min() == 0 or x.max() == w - 1 or y.max() == h - 1)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--calibration-zip", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    split = json.loads(args.split_json.read_text())
    frame_ids = {Path(image["file_name"]).stem for image in split["images"]}
    training = args.dataset_root.resolve() / "training"
    focus = {"000004_10": [9486], "000005_10": [9542, 9584], "000058_10": [9524, 9642]}
    rows = []
    with ZipFile(args.calibration_zip) as archive:
        for frame in (*focus, *sorted(frame_ids - set(focus))):
            paths = {name: training / folder / (frame + ".png") for name, folder in
                     (("left", "image_2"), ("right", "image_3"), ("instance_map", "instance"), ("disparity", "disp_occ_0"))}
            if not all(path.is_file() for path in paths.values()):
                continue
            calibration = rectified_calibration(archive, frame)
            with Image.open(paths["instance_map"]) as labels, Image.open(paths["disparity"]) as disparity:
                instances = propose_instances(np.asarray(labels), np.asarray(disparity, dtype=np.float32) / 256., calibration)
            if instances:
                rows.append(dict(frame_id=frame, split="training", right_box_source="projection_pending_review",
                                 **{k: str(v) for k, v in paths.items()}, **calibration,
                                 instances=instances, diagnostic_pair_ids=focus.get(frame, [])))
    fixed = [row for row in rows if row["diagnostic_pair_ids"]]
    other = [row for row in rows if not row["diagnostic_pair_ids"]]
    selectors = (lambda x: x["touches_edge"],
                 lambda x: x["valid_fraction"] < .7,
                 lambda x: x["gt_depth_m"] is not None and 9 <= x["gt_depth_m"] <= 11,
                 lambda x: x["gt_depth_m"] is not None and 10 < x["gt_depth_m"] <= 15,
                 lambda x: x["left_box"][2] - x["left_box"][0] >= 200)
    for matches in selectors:
        for row in other:
            if row not in fixed and any(matches(item) for item in row["instances"]):
                fixed.append(row)
                break
    fixed.extend([row for row in other if row not in fixed][:10 - len(fixed)])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixed, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    for row in fixed:
        views = []
        for side in ("left", "right"):
            with Image.open(row[side]) as source:
                view = source.convert("RGB")
            draw = ImageDraw.Draw(view)
            for item in row["instances"]:
                box = item[f"{side}_box"]
                draw.rectangle(box, outline="orange", width=2)
                draw.text((box[0], max(0, box[1] - 12)), str(item["id"]), fill="yellow")
            views.append(view)
        left, right = views
        preview = Image.new("RGB", (left.width, left.height + right.height))
        preview.paste(left, (0, 0))
        preview.paste(right, (0, left.height))
        preview.save(args.output.parent / (row["frame_id"] + "-review.jpg"), quality=90)


if __name__ == "__main__":
    main()
