"""Calibrate the RoI residual-disparity search range from the training split.

Reports the P99.5 of ``|(Df - (cL - cR)) * Wroi / C|`` directly in RoI bins.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lightbackbone_stereo_yolo.data.frame_dataset import (  # noqa: E402
    KittiFrameDataset,
    build_class_mapping,
    build_frame_records,
)
from lightbackbone_stereo_yolo.calibration import derive_global_dmax  # noqa: E402
from lightbackbone_stereo_yolo.roi import ROI_SAMPLING_MODES, make_pair_crop_windows  # noqa: E402
from lightbackbone_stereo_yolo.training import build_roi_disparity_targets  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split-json", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--quantile", type=float, default=.995)
    parser.add_argument("--roi-margin", type=float, default=16.)
    parser.add_argument("--roi-sampling", choices=ROI_SAMPLING_MODES, default="native_pad")
    parser.add_argument("--box-margin", type=float, default=0., help="image-pixel detection-box error margin")
    parser.add_argument("--mask-margin", type=float, default=0., help="image-pixel mask-edge error margin")
    parser.add_argument("--stride", type=int, default=4, help="fixed P2 label-alignment stride")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.quantile < 1:
        raise SystemExit("quantile must be strictly between zero and one")
    if min(args.roi_margin, args.box_margin, args.mask_margin) < 0:
        raise SystemExit("margins must be non-negative")
    if args.stride != 4:
        raise SystemExit("local iDisp calibration requires stride=4")
    frames = build_frame_records(args.dataset_root, args.split_json)
    dataset = KittiFrameDataset(frames, build_class_mapping(frames), load_images=False)
    residual_chunks = []
    scale_chunks = []
    crop_width_chunks = []
    per_instance = []
    pixels = 0
    instances = 0
    for index in range(len(dataset)):
        sample = dataset[index]
        boxes6 = sample["boxes6"]
        if boxes6.numel() == 0:
            continue
        disparity = sample["disparity"]
        crop = make_pair_crop_windows(
            boxes6, disparity.shape[-2], disparity.shape[-1], args.roi_margin, roi_sampling=args.roi_sampling,
        )
        _, _, calibration_valid, residual = build_roi_disparity_targets(
            disparity[None], sample["disparity_valid"][None], sample["masks"], crop,
        )
        values = residual[calibration_valid].abs().flatten()
        if values.numel():
            residual_chunks.append(values)
        scale_chunks.append(crop.w_roi.to(torch.float32) / crop.c)
        crop_width_chunks.append(crop.c)
        pixels += int(values.numel())
        instances += boxes6.shape[0]
        for instance_index in range(boxes6.shape[0]):
            instance_values = residual[instance_index, 0][calibration_valid[instance_index, 0]].abs().flatten()
            per_instance.append((sample["frame_id"], instance_index, instance_values))
    if not residual_chunks:
        raise SystemExit("no valid instance pixels were found for calibration")
    margin_img_px = args.box_margin + args.mask_margin
    calibration = derive_global_dmax(
        torch.cat(residual_chunks), torch.cat(scale_chunks), quantile=args.quantile, margin_img_px=margin_img_px,
    )
    dmax_without_margin = int(torch.ceil(torch.tensor(calibration.roi_residual_q_bins)).item())
    overflow_ratios = []
    overflow_gt_50 = []
    empty_instances = 0
    for frame_id, instance_index, values in per_instance:
        if not values.numel():
            empty_instances += 1
            continue
        ratio = float((values > calibration.dmax_roi).to(torch.float32).mean().item())
        overflow_ratios.append(ratio)
        if ratio > .5:
            overflow_gt_50.append({"frame_id": frame_id, "instance_index": instance_index, "overflow_ratio": ratio})
    scales = torch.cat(scale_chunks)
    crop_widths = torch.cat(crop_width_chunks)
    overflow_all = torch.cat(residual_chunks) > calibration.dmax_roi
    report = {
        "schema_version": 2,
        "roi_sampling": args.roi_sampling,
        "frames": len(dataset),
        "instances": instances,
        "foreground_valid_pixels": pixels,
        "quantile": args.quantile,
        "supervision_mode": "foreground",
        "roi_residual_quantile_bins": calibration.roi_residual_q_bins,
        "dmax_roi_without_margin": dmax_without_margin,
        "margin_img_px": margin_img_px,
        "margin_roi_global_bins": calibration.margin_roi_global_bins,
        "suggested_dmax_roi": calibration.dmax_roi,
        "global_foreground_overflow_ratio": float(overflow_all.to(torch.float32).mean().item()),
        "per_instance_overflow_ratio_p95": float(torch.quantile(torch.tensor(overflow_ratios), .95).item()) if overflow_ratios else 0.,
        "per_instance_overflow_ratio_max": max(overflow_ratios, default=0.),
        "overflow_gt_50_instance_count": len(overflow_gt_50),
        "overflow_gt_50_instances": overflow_gt_50,
        "empty_foreground_instance_count": empty_instances,
        "roi_bin_per_image_px": {
            "min": float(scales.min().item()), "p50": float(torch.quantile(scales, .5).item()),
            "p95": float(torch.quantile(scales, .95).item()), "max": float(scales.max().item()),
            "crop_width_lt_32_instances": int((crop_widths < 32).sum().item()),
        },
        "roi_margin": args.roi_margin,
        "box_margin": args.box_margin,
        "mask_margin": args.mask_margin,
        "stride": 4,
        "split_json": args.split_json,
    }
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
