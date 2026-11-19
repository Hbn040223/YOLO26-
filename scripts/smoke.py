"""GPU smoke and component profiler for the isolated Stereo-YOLO experiment.

The script profiles the deployed *local* path only. It deliberately never
constructs LightStereo's global cost-volume, aggregation, Context Network, or
full-image disparity head. Local RoI work uses one valid KITTI target so an
untrained detector's arbitrary decoded boxes cannot invalidate the profile.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Callable, TypeVar

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lightbackbone_stereo_yolo.data.kitti import KittiInstanceDataset, build_kitti_manifest
from lightbackbone_stereo_yolo.detection import StereoYoloHead, decode_dense_predictions, generate_grid_anchors
from lightbackbone_stereo_yolo.lightstereo_encoder import LightStereoFeatureEncoder
from lightbackbone_stereo_yolo.local_aggregation import AttentionPyramid
from lightbackbone_stereo_yolo.local_disparity import build_local_cost_volume
from lightbackbone_stereo_yolo.model import StereoLocalInstanceHead
from lightbackbone_stereo_yolo.roi import ROI_SAMPLING_MODES, make_pair_crop_windows


T = TypeVar("T")


def profile_cuda(name: str, device: torch.device, operation: Callable[[], T]) -> tuple[T, dict[str, float | str]]:
    """Synchronize and reset peak memory around exactly one measured operation."""
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    output = operation()
    end.record()
    torch.cuda.synchronize(device)
    return output, {
        "stage": name,
        "latency_ms": round(float(start.elapsed_time(end)), 3),
        "peak_memory_mb": round(torch.cuda.max_memory_allocated(device) / 2**20, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split-json", required=True)
    parser.add_argument("--lightstereo-source", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dmax-roi", type=int, default=16)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--score-threshold", type=float, default=.05)
    parser.add_argument("--pre-nms-topk", type=int, default=500)
    parser.add_argument("--roi-margin", type=float, default=16.0)
    parser.add_argument("--roi-sampling", choices=ROI_SAMPLING_MODES, default="native_pad")
    args = parser.parse_args()
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("This profiler requires a CUDA device; do not use CPU timings as the baseline")
    if args.dmax_roi < 1 or args.pre_nms_topk < 1 or args.roi_margin < 0:
        raise ValueError("dmax-roi/pre-nms-topk must be positive and roi-margin non-negative")

    device = torch.device(args.device)
    records = build_kitti_manifest(args.dataset_root, args.split_json)
    if not records:
        raise RuntimeError("manifest has no valid KITTI instance records")
    sample = KittiInstanceDataset(records)[0]
    left = sample["left"].unsqueeze(0).to(device)
    right = sample["right"].unsqueeze(0).to(device)

    encoder = LightStereoFeatureEncoder.from_checkpoint(args.lightstereo_source, args.checkpoint).to(device).eval()
    with torch.inference_mode():
        pyramid, encoder_stats = profile_cuda("encoder", device, lambda: encoder(left, right))
        detector = StereoYoloHead(pyramid.p3_left.shape[1], args.num_classes).to(device).eval()
        dense, detector_stats = profile_cuda("detector", device, lambda: detector(pyramid))
        levels = (
            (pyramid.p3_left.shape[-2], pyramid.p3_left.shape[-1], 8, 32.0),
            (pyramid.p4_left.shape[-2], pyramid.p4_left.shape[-1], 16, 64.0),
            (pyramid.c5_left.shape[-2], pyramid.c5_left.shape[-1], 32, 128.0),
        )
        anchors = generate_grid_anchors(levels, device)
        pairs, nms_stats = profile_cuda(
            "decode_and_paired_nms",
            device,
            lambda: decode_dense_predictions(
                {"objectness": dense.objectness[0], "class_logits": dense.class_logits[0], "box6": dense.box6[0]},
                anchors,
                score_threshold=args.score_threshold,
                pre_nms_topk=args.pre_nms_topk,
            ),
        )

        # A valid GT pair makes local profiling deterministic even before detector training.
        boxes6 = sample["boxes6"].reshape(1, 6).to(device)
        instance = StereoLocalInstanceHead(
            pyramid.p2_left.shape[1], args.dmax_roi, roi_sampling=args.roi_sampling,
        ).to(device).eval()

        def sample_rois() -> tuple[object, torch.Tensor, torch.Tensor, torch.Tensor, AttentionPyramid]:
            crop = make_pair_crop_windows(
                boxes6, left.shape[-2], left.shape[-1], args.roi_margin, roi_sampling=args.roi_sampling,
            )
            left_roi, right_roi, spatial_valid, _ = instance.roi(pyramid.p2_left, pyramid.p2_right, boxes6, crop)
            p3_roi, _ = instance.p3_roi(pyramid.p3_left, crop)
            p4_roi, _ = instance.p4_roi(pyramid.p4_left, crop)
            return crop, left_roi, right_roi, spatial_valid, AttentionPyramid(left_roi, p3_roi, p4_roi)

        roi_data, roi_stats = profile_cuda("roi_sampling", device, sample_rois)
        crop, left_roi, right_roi, spatial_valid, attention = roi_data
        volume, cost_stats = profile_cuda(
            "local_cost_volume",
            device,
            lambda: build_local_cost_volume(left_roi, right_roi, instance.idisp.search, spatial_valid, instance.idisp.temperature),
        )
        _, aggregation_stats = profile_cuda(
            "local_aggregation", device,
            lambda: instance.idisp._aggregate_cost(volume.cost, attention, crop, spatial_valid),
        )
        disparity, output_stats = profile_cuda(
            "local_head_output", device,
            lambda: instance.idisp(left_roi, right_roi, attention, crop, spatial_valid),
        )

    report = {
        "records": len(records),
        "input_hw": tuple(left.shape[-2:]),
        "roi_sampling": args.roi_sampling,
        "local_cost_shape": tuple(volume.cost.shape),
        "dlocal": 2 * args.dmax_roi + 1,
        "detected_candidates_after_nms": int(pairs.boxes6.shape[0]),
        "profiled_local_rois": int(boxes6.shape[0]),
        "p2": tuple(pyramid.p2_left.shape),
        "p3": tuple(pyramid.p3_left.shape),
        "p4": tuple(pyramid.p4_left.shape),
        "finite_local_disparity": bool(torch.isfinite(disparity.absolute[disparity.valid]).all()),
        "stages": [encoder_stats, detector_stats, nms_stats, roi_stats, cost_stats, aggregation_stats, output_stats],
    }
    print(report)


if __name__ == "__main__":
    main()
