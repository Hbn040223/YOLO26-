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
from infer import ROOT, load_published_model, normalized_rgb
from lightbackbone_stereo_yolo import model as inference_model


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


def profile_forward(model, left, right, device):
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
        model(left, right)
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


def measure(model, record, normalization, device):
    """Time file decode through host outputs, and separately model forward."""
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = perf_counter()
    images = [normalized_rgb(path, normalization["mean"], normalization["std"])[None]
              for path in (record.left_image, record.right_image)]
    decoded = perf_counter()
    left, right = (image.to(device) for image in images)
    synchronize(device)
    on_device = perf_counter()
    output = model(left, right)
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
            measure(model, records[0], bundle["normalization"], device)
    timings = []
    for frame, record in enumerate(records):
        for repeat in range(args.repeats):
            offset = (frame + repeat) % len(arms)
            row = {"frame_id": record.frame_id, "repeat": repeat}
            for arm in arms[offset:] + arms[:offset]:
                configure(model, settings[arm])
                row[arm] = measure(model, record, bundle["normalization"], device)
            timings.append(row)
    profiles = []
    if args.profile:
        for record in records:
            normalization = bundle["normalization"]
            left, right = (normalized_rgb(path, normalization["mean"], normalization["std"])[None].to(device)
                           for path in (record.left_image, record.right_image))
            row = {"frame_id": record.frame_id}
            for arm, setting in settings.items():
                configure(model, setting)
                row[arm] = profile_forward(model, left, right, device)
            profiles.append(row)
    write_json(args.output, {
        "commit": revision(ROOT), "openstereo_commit": revision(args.openstereo_source),
        "weight_sha256": sha256(args.weights), "split_sha256": sha256(args.split_json),
        "settings": settings, "torch_version": str(torch.__version__), "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "power_mode": args.power_mode, "precision": str(next(model.parameters()).dtype),
        "score": model.score_threshold, "topk": model.pre_nms_topk, "paired_nms_iou": model.paired_nms_iou,
        "roi_margin": 16, "roi_sampling": model.instance_head.roi_sampling, "dmax_roi": bundle["dmax_roi"],
        "class_mapping": bundle["class_mapping"],
        "foreground_mask_raw_classes": bundle.get("foreground_mask_raw_classes", ()),
        "foreground_class_indices": model.foreground_class_indices,
        "warmup": args.warmup, "timings": timings, "profiles": profiles,
        "summary": {arm: summarize([row[arm] for row in timings]) for arm in arms},
        "profile_summary": {arm: summarize([row[arm] for row in profiles]) for arm in arms} if profiles else {},
    })


if __name__ == "__main__":
    main()
