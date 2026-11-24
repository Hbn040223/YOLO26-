"""Alternate frozen baseline, crop, or containment settings on a KITTI split."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from pathlib import Path
import subprocess
from time import perf_counter
from unittest.mock import patch

import torch

from evaluate_paired_nms import sha256, validation_dataset, write_json
from infer import ROOT, load_published_model
from edge_run_evidence import source_info
from lightbackbone_stereo_yolo import model as inference_model
from lightbackbone_stereo_yolo.stereo_input import load_rectified_pair


CONFIGURATIONS = {
    "baseline": {"nms_off": (False, 0, None), "nms_on": (True, 0, None)},
    "crop": {f"A{i}": (True, crop, None) for i, crop in enumerate((0, 64, 96, 128))},
    "containment": {"C0": (True, 0, None), "C1": (True, 0, .90)},
}


def revision(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def profile_forward(model, left, right, device, geometry=None):
    """Measure synchronized stage boundaries independently of the latency runs."""
    marks = {}

    def stamp(name):
        synchronize(device)
        marks[name] = perf_counter()

    def boundary(stack, module, name, *, before=False):
        hook = module.register_forward_pre_hook if before else module.register_forward_hook
        handle = hook(lambda *_: stamp(name))
        stack.callback(handle.remove)

    def pair_suppression(*args, **kwargs):
        stamp("containment_start")
        result = suppress(*args, **kwargs)
        stamp("containment_end")
        return result

    encoder = model.encoder
    backbone = encoder.backbone
    suppress = inference_model.suppress_contained_pairs
    with ExitStack() as stack:
        for module, name, before in (
            (encoder, "encoder_start", True), (encoder, "encoder_end", False),
            (backbone, "backbone_start", True), (backbone.block4, "mobilenet_end", False),
            (backbone, "fpn_end", False), (encoder.adapters[0], "adapter_start", True),
            (encoder.adapters[-1], "adapter_end", False),
            (model.detector, "detector_start", True), (model.detector, "detector_end", False),
            (model.instance_head, "instance_start", True), (model.instance_head, "instance_end", False),
        ):
            boundary(stack, module, name, before=before)
        stack.enter_context(patch.object(inference_model, "suppress_contained_pairs", pair_suppression))
        if geometry is None:
            model(left, right)
        else:
            model.forward_prepared(left, right, geometry)
        stamp("output_end")

    def elapsed(start, end):
        return (marks[end] - marks[start]) * 1000 if start in marks and end in marks else 0.

    return {
        "encoder_ms": elapsed("encoder_start", "encoder_end"),
        "padding_and_pack_ms": elapsed("encoder_start", "backbone_start"),
        "mobilenet_ms": elapsed("backbone_start", "mobilenet_end"),
        "fpn_ms": elapsed("mobilenet_end", "fpn_end"),
        "adapters_and_intermediate_slicing_ms": elapsed("adapter_start", "adapter_end"),
        "detector_ms": elapsed("detector_start", "detector_end"),
        "postprocess_and_dispatch_ms": elapsed(
            "detector_end", "instance_start" if "instance_start" in marks else "output_end"),
        "containment_ms": elapsed("containment_start", "containment_end"),
        "roi_mask_idisp_ms": elapsed("instance_start", "instance_end"),
    }


def measure(model, record, normalization, device, input_path="prepared"):
    """Time file decode through host outputs, and separately model forward."""
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = perf_counter()
    input_timings = {}
    crop = model.crop_top_px if input_path == "prepared" else 0
    left, right, geometry = load_rectified_pair(record.left_image, record.right_image, normalization, crop,
                                               timings=input_timings)
    decoded = perf_counter()
    left, right = left.to(device), right.to(device)
    synchronize(device)
    on_device = perf_counter()
    output = (model.forward_prepared(left, right, geometry) if input_path == "prepared" else model(left, right))
    synchronize(device)
    forward_done = perf_counter()
    for tensor in (output.boxes_left, output.boxes_right, output.pair_ids, output.scores, output.classes,
                   output.mask_logits, output.mask_validity, output.disparity, output.validity,
                   output.foreground_validity):
        tensor.cpu()
    if output.crop is not None:
        for name in ("c_l", "c_r", "c", "y0", "h_crop", "w_roi", "h_roi"):
            getattr(output.crop, name).cpu()
    synchronize(device)
    end = perf_counter()
    return {
        "image_hw": output.image_hw, "encoder_padded_hw": output.encoder_padded_hw,
        "cropped_hw": (output.image_hw[0] - output.crop_top_px, output.image_hw[1]),
        **input_timings,
        "pairs": len(output.pair_ids), "preprocess_ms": (decoded - start) * 1000,
        "h2d_ms": (on_device - decoded) * 1000, "forward_ms": (forward_done - on_device) * 1000,
        "output_d2h_ms": (end - forward_done) * 1000, "end_to_end_ms": (end - start) * 1000,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
    }


def configure(model, setting):
    model.apply_paired_nms, model.crop_top_px, model.containment_threshold = setting


def summarize(rows):
    result = {}
    for key in rows[0]:
        values = [row[key] for row in rows]
        if values[0] is None or isinstance(values[0], (tuple, list)):
            continue
        data = torch.tensor(values, dtype=torch.float64)
        p50, p95 = torch.quantile(data, data.new_tensor([.50, .95])).tolist()
        result[key] = {"p50": p50, "p95": p95, "sample_variance": float(data.var()) if len(values) > 1 else None,
                       "samples": len(values)}
    return result


def bootstrap_p95_differences(timings, arms):
    """Resample paired frames, carrying every repetition into each draw."""
    by_frame = {}
    for row in timings:
        by_frame.setdefault(row["frame_id"], []).append([row[arm]["end_to_end_ms"] for arm in arms])
    data = torch.tensor(list(by_frame.values()), dtype=torch.float64)
    count = len(data)
    if count < 2:
        return {}
    generator = torch.Generator().manual_seed(17)
    draws = torch.randint(count, (2000, count), generator=generator)
    resampled = data[draws].flatten(1, 2)
    tail = torch.quantile(resampled, .95, dim=1)
    observed = torch.quantile(data.flatten(0, 1), .95, dim=0)
    return {
        arm: {"p95_difference_ms": float(observed[i] - observed[0]),
              "ci95_ms": torch.quantile(tail[:, i] - tail[:, 0], tail.new_tensor([.025, .975])).tolist(),
              "frames": count, "bootstrap_draws": len(draws)}
        for i, arm in enumerate(arms) if i
    }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=CONFIGURATIONS, required=True)
    parser.add_argument("--weights", type=Path, default=ROOT / "models/stereo-yolo26-fixed56-seed17-e20.pt")
    parser.add_argument("--openstereo-source", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--power-mode", required=True, help="record the actual board mode and clock setting")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--profile", action="store_true", help="run synchronized profiling after timing")
    parser.add_argument("--input-path", choices=("prepared", "model"), default="prepared")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 0:
        parser.error("repeats must be positive and warmup nonnegative")
    device = torch.device(args.device)
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model, bundle = load_published_model(args.weights, args.openstereo_source, device, apply_paired_nms=True)
    records = validation_dataset(args.dataset_root, args.split_json, bundle).records
    if not records:
        parser.error("the split has no frames")
    settings = CONFIGURATIONS[args.stage]
    arms = list(settings)
    for _ in range(args.warmup):
        for arm, setting in settings.items():
            configure(model, setting)
            measure(model, records[0], bundle["normalization"], device, args.input_path)
    timings = []
    for frame, record in enumerate(records):
        for repeat in range(args.repeats):
            offset = (frame + repeat) % len(arms)
            row = {"frame_id": record.frame_id, "repeat": repeat}
            for arm in arms[offset:] + arms[:offset]:
                configure(model, settings[arm])
                row[arm] = measure(model, record, bundle["normalization"], device, args.input_path)
            timings.append(row)
    profiles = []
    if args.profile:
        for record in records:
            normalization = bundle["normalization"]
            row = {"frame_id": record.frame_id}
            for arm, setting in settings.items():
                configure(model, setting)
                crop = model.crop_top_px if args.input_path == "prepared" else 0
                left, right, geometry = load_rectified_pair(record.left_image, record.right_image, normalization, crop)
                row[arm] = profile_forward(model, left.to(device), right.to(device), device,
                                           geometry if args.input_path == "prepared" else None)
            profiles.append(row)
    write_json(args.output, {
        "commit": revision(ROOT), "openstereo_commit": revision(args.openstereo_source),
        "weight_sha256": sha256(args.weights), "split_sha256": sha256(args.split_json),
        "settings": settings, "input_path": args.input_path, "torch_version": str(torch.__version__), "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "power_mode": args.power_mode, "precision": str(next(model.parameters()).dtype),
        "score": model.score_threshold, "topk": model.pre_nms_topk, "paired_nms_iou": model.paired_nms_iou,
        "roi_margin": 16, "roi_sampling": model.instance_head.roi_sampling, "dmax_roi": bundle["dmax_roi"],
        "class_mapping": bundle["class_mapping"],
        "foreground_mask_raw_classes": bundle.get("foreground_mask_raw_classes", ()),
        "foreground_class_indices": model.foreground_class_indices,
        "warmup": args.warmup, "timings": timings, "profiles": profiles,
        "summary": {arm: summarize([row[arm] for row in timings]) for arm in arms},
        "end_to_end_p95_paired_frame_bootstrap": bootstrap_p95_differences(timings, arms),
        "profile_summary": {arm: summarize([row[arm] for row in profiles]) for arm in arms} if profiles else {},
        "runtime_source": source_info(ROOT, __file__, args.openstereo_source, args.weights),
    })


if __name__ == "__main__":
    main()
