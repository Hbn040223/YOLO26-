"""Run the published fixed_56 model on a rectified stereo pair."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import sys
import warnings

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lightbackbone_stereo_yolo.data.frame_dataset import normalized_rgb
from lightbackbone_stereo_yolo.lightstereo_encoder import LightStereoFeatureEncoder, build_lightstereo_backbone
from lightbackbone_stereo_yolo.mask import mask_logit_cutoff
from lightbackbone_stereo_yolo.roi_selection import RoiPairSelection
from lightbackbone_stereo_yolo.model import StereoLocalInstanceHead, StereoYoloLocalModel
from lightbackbone_stereo_yolo.stereo_yolo26 import StereoYOLO26PanHead
from lightbackbone_stereo_yolo.stereo_input import StereoInputGeometry, load_rectified_pair
from edge_run_evidence import capture_route, serialize_route, source_info


def resolve_mask_threshold(bundle, override=None):
    # The published E25 stores bias-folded logits. New exports store raw logits.
    if override is None:
        threshold = float(bundle.get("mask_threshold", .50))
        mask_logit_cutoff(threshold)
        return threshold
    cutoff = mask_logit_cutoff(override)
    folded_threshold = bundle.get("mask_evaluation", {}).get("threshold_before_logit_calibration")
    if folded_threshold is None:
        return override
    shift = cutoff - mask_logit_cutoff(folded_threshold)
    if shift >= 0:
        return 1 / (1 + math.exp(-shift))
    value = math.exp(shift)
    return value / (1 + value)


def load_published_model(weights, openstereo_source, device, *, score=.40, topk=500,
                         apply_paired_nms=None, paired_nms_iou=.5, mask_threshold=None, roi_max_distance=None,
                         bf_px_m=None, cx_delta_px=None, crop_top_px=0, containment_threshold=None):
    bundle = torch.load(weights, map_location="cpu", weights_only=True)
    config = bundle["config"]
    channels = config["adapter_channels"]
    encoder = LightStereoFeatureEncoder(build_lightstereo_backbone(openstereo_source), channels)
    detector = StereoYOLO26PanHead(
        channels, len(bundle["class_mapping"]), one_to_one=config["one_to_one"],
        one_to_one_only=config.get("one_to_one_only", False),
    )
    instance = StereoLocalInstanceHead(
        channels, bundle["dmax_roi"], roi_sampling=config["roi_sampling"],
        mask_head_variant=config.get("mask_head_variant", "light"),
    )
    selection = None
    if roi_max_distance is not None:
        if bf_px_m is None or cx_delta_px is None:
            raise ValueError("near selection needs explicit per-frame bf_px_m and cx_delta_px")
        selection = RoiPairSelection(bundle["class_mapping"][26], bf_px_m, cx_delta_px,
                                     max_distance_m=roi_max_distance)
    model = StereoYoloLocalModel(
        encoder, detector, instance, score_threshold=score, pre_nms_topk=topk,
        detector_variant=config["detector_variant"], right_offset_mode=config["right_offset_mode"],
        apply_paired_nms=apply_paired_nms, paired_nms_iou=paired_nms_iou,
        foreground_class_indices=tuple(bundle["class_mapping"][key]
                                       for key in bundle.get("foreground_mask_raw_classes", ())),
        mask_threshold=resolve_mask_threshold(bundle, mask_threshold), roi_selection=selection,
        crop_top_px=crop_top_px, containment_threshold=containment_threshold,
    )
    model.load_state_dict(bundle["model"], strict=True)
    return model.to(device).eval(), bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openstereo-source", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=ROOT / "models/stereo-yolo26-fixed56-seed17-e20.pt")
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--score", type=float, default=.40)
    parser.add_argument("--paired-nms", action="store_true", help="opt in to class-aware stereo pair suppression")
    parser.add_argument("--paired-nms-iou", type=float, default=.5)
    parser.add_argument("--crop-top-px", type=int, default=0, help="crop this many rows from both rectified images")
    parser.add_argument("--containment-suppression", action="store_true", help="suppress contained pairs after paired NMS")
    parser.add_argument("--containment-threshold", type=float, default=.90)
    parser.add_argument("--mask-threshold", type=float, help="override the probability cutoff in the unfurled Mask logit space")
    parser.add_argument("--roi-max-distance", type=float, help="opt in to experimental near-Car foreground selection")
    parser.add_argument("--bf-px-m", type=float, help="fB from this stereo frame calibration")
    parser.add_argument("--cx-delta-px", type=float, help="left cx minus right cx in pixels")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-path", choices=("prepared", "model"), default="prepared")
    parser.add_argument("--trace-json", type=Path, help="record loaded source, Backbone shape and NMS/iDisp pair IDs")
    parser.add_argument("--expect-commit", help="fail before inference if this commit is not checked out")
    args = parser.parse_args()
    if args.containment_suppression and not args.paired_nms:
        parser.error("--containment-suppression requires --paired-nms")
    evidence = source_info(ROOT, __file__, args.openstereo_source, args.weights) if (
        args.trace_json or args.expect_commit) else None
    if args.expect_commit and evidence["commit"] != args.expect_commit:
        parser.error(f"expected commit {args.expect_commit}, actual {evidence['commit']}")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model, bundle = load_published_model(
        args.weights, args.openstereo_source, args.device, score=args.score,
        apply_paired_nms=True if args.paired_nms else None, paired_nms_iou=args.paired_nms_iou,
        mask_threshold=args.mask_threshold, roi_max_distance=args.roi_max_distance,
        bf_px_m=args.bf_px_m, cx_delta_px=args.cx_delta_px,
        crop_top_px=args.crop_top_px,
        containment_threshold=args.containment_threshold if args.containment_suppression else None,
    )
    if not model.foreground_class_indices:
        warnings.warn("This checkpoint has no supported foreground classes. Use foreground_evaluation.pt for Car foreground.",
                      stacklevel=2)
    normalization = bundle["normalization"]
    timings = {}
    cpu_crop = args.crop_top_px if args.input_path == "prepared" else 0
    left, right, decoded_geometry = load_rectified_pair(args.left, args.right, normalization, cpu_crop, timings=timings)
    geometry = StereoInputGeometry(decoded_geometry.original_hw, args.crop_top_px)
    left, right = left.to(args.device), right.to(args.device)
    with torch.inference_mode():
        with capture_route(model) if args.trace_json else nullcontext() as route:
            result = (model.forward_prepared(left, right, geometry, route=route) if args.input_path == "prepared"
                      else model(left, right, route=route))
        arrays = {name: getattr(result, name).cpu().numpy() for name in (
            "boxes_left", "boxes_right", "scores", "classes", "disparity", "validity", "mask_validity",
            "pair_ids", "foreground_validity",
        )}
        if result.left_only is not None:
            arrays.update({
                "left_only_boxes": result.left_only.boxes.cpu().numpy(),
                "left_only_scores": result.left_only.scores.cpu().numpy(),
                "left_only_ids": result.left_only.ids.cpu().numpy(),
                "left_only_classes": result.left_only.classes.cpu().numpy(),
                "left_only_distance_m": result.left_only.distance_m.cpu().numpy(),
            })
        arrays["roi_selection_enabled"] = np.asarray(model.roi_selection is not None)
        arrays["mask_probability"] = result.mask_logits.sigmoid().cpu().numpy()
        arrays["foreground_disparity"] = result.disparity.masked_fill(
            ~result.foreground_validity, torch.nan).cpu().numpy()
        arrays["foreground_supported"] = np.asarray(bool(model.foreground_class_indices))
        arrays["mask_threshold_on_saved_logits"] = np.asarray(model.mask_threshold)
        arrays["foreground_supported_raw_classes"] = np.asarray(
            bundle.get("foreground_mask_raw_classes", ()), dtype=np.int64)
    for name in ("c_l", "c_r", "c", "y0", "h_crop", "w_roi", "h_roi"):
        arrays[name] = getattr(result.crop, name).cpu().numpy() if result.crop is not None else np.empty(0)
    arrays["image_hw"] = np.asarray(result.image_hw)
    arrays["crop_top_px"] = np.asarray(result.crop_top_px)
    arrays["encoder_padded_hw"] = np.asarray(result.encoder_padded_hw, dtype=np.int64)
    arrays["cropped_hw"] = np.asarray(geometry.cropped_hw)
    arrays["input_path"] = np.asarray(args.input_path)
    arrays["containment_pair_ids"] = np.asarray(
        [(event.suppressor_id, event.suppressed_id) for event in result.containment_events], dtype=np.int64,
    ).reshape(-1, 2)
    arrays["containment_coverages"] = np.asarray(
        [(event.left_coverage, event.right_coverage) for event in result.containment_events], dtype=np.float32,
    ).reshape(-1, 2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    if args.trace_json:
        args.trace_json.parent.mkdir(parents=True, exist_ok=True)
        args.trace_json.write_text(json.dumps({
            **evidence, **geometry.as_dict(), "input_path": args.input_path,
            "input_nchw": list(left.shape), "device": args.device,
            "detector_variant": model.detector_variant, "score": model.score_threshold, "topk": model.pre_nms_topk,
            "paired_nms_iou": model.paired_nms_iou, "containment_threshold": model.containment_threshold,
            "roi_sampling": model.instance_head.roi_sampling, **serialize_route(route), **timings,
        }, indent=2) + "\n", encoding="utf-8")
    print(f"{len(result.scores)} paired detections, disparity shape {tuple(result.disparity.shape)}")
    if result.left_only is not None:
        print(f"{len(result.left_only.ids)} left-only detections with unknown distance")
    print(args.output)


if __name__ == "__main__":
    main()
