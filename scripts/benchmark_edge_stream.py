"""Benchmark the online stereo pipeline on a repeating sequence of decoded pairs."""
from bisect import bisect_right
from math import ceil, isfinite
from pathlib import Path
from time import perf_counter, sleep
from types import SimpleNamespace
from zipfile import ZipFile
import argparse
from contextlib import nullcontext

import numpy as np
from PIL import Image
import torch

from diagnose_pred_roi import _rectified_calibration
from edge_run_evidence import digest, source_info
from evaluate_paired_nms import validation_dataset, write_json
from infer import ROOT, load_published_model
from lightbackbone_stereo_yolo.stereo_input import prepare_rectified_images


def run_stream(frames, infer_pair, *, rate_fps, duration_s):
    """Replay virtual arrivals independently of the serial consumer's availability."""
    if not frames or not isfinite(duration_s) or duration_s <= 0:
        raise ValueError("a nonempty frame set and positive finite duration are required")
    if rate_fps is not None and (not isfinite(rate_fps) or rate_fps <= 0):
        raise ValueError("replay rate must be positive and finite")
    rows = []
    arrivals = (None if rate_fps is None else
                [i / rate_fps for i in range(ceil(duration_s * rate_fps)) if i / rate_fps < duration_s])
    start = perf_counter()
    queue_peak = 0
    while True:
        sequence = len(rows)
        now = perf_counter() - start
        if arrivals is None:
            if now >= duration_s:
                break
            arrival = now
        else:
            if sequence == len(arrivals):
                break
            arrival = arrivals[sequence]
            if arrival > now:
                sleep(arrival - now)
        frame_id, frame = frames[sequence % len(frames)]
        begun = perf_counter() - start
        details = infer_pair(frame)
        finished = perf_counter() - start
        if arrivals is not None:
            queue_peak = max(queue_peak, bisect_right(arrivals, finished) - sequence - 1)
        rows.append({
            "frame_id": frame_id, "sequence_id": sequence, "arrival_s": arrival,
            "started_s": begun, "finished_s": finished, "queue_ms": (begun - arrival) * 1000,
            "service_ms": (finished - begun) * 1000, "latency_ms": (finished - arrival) * 1000,
            "details": details,
        })
    completed = sum(row["finished_s"] <= duration_s for row in rows)
    latencies = [row["latency_ms"] for row in rows]
    return {
        "rate_hz": rate_fps, "duration_s": duration_s, "submitted": len(rows), "completed": completed,
        "in_flight_at_end": len(rows) - completed, "fps": completed / duration_s, "dropped": 0,
        "queue_peak": queue_peak,
        "timeouts": None if rate_fps is None else sum(value > 1000 / rate_fps for value in latencies),
        "latency_p50_ms": float(np.percentile(latencies, 50)) if rows else None,
        "latency_p95_ms": float(np.percentile(latencies, 95)) if rows else None,
        "rows": rows,
    }


def decoded_frames(dataset, calibration_zip=None):
    frames = []
    for record in dataset.records:
        with Image.open(record.left_image) as left, Image.open(record.right_image) as right:
            pixels = (left.convert("RGB"), right.convert("RGB"))
        base = record.left_image.stem
        frame_id = getattr(record, "frame_id", base)
        values = _rectified_calibration(calibration_zip, frame_id) if calibration_zip is not None else None
        calibration = (values["bf_px_m"], values["cx_delta_px"]) if values is not None else None
        frames.append((frame_id, (*pixels, calibration)))
    return frames


def load_frontends(manifests, precision, device, evidence):
    from benchmark_edge_tensorrt import TensorRTFrontend
    from lightbackbone_stereo_yolo.edge_deployment import manifest_geometry
    import json
    import tensorrt

    frontends = {}
    for path in manifests:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        geometry = manifest_geometry(manifest)
        if (manifest["backbone"] != "v2" or geometry.crop_top_px
                or any(manifest[field] != evidence[field] for field in ("openstereo_commit", "weights_sha256"))):
            raise ValueError(f"engine manifest does not match the full-frame V2 model: {path}")
        if geometry.original_hw in frontends:
            raise ValueError(f"duplicate engine for {geometry.original_hw}")
        entry = manifest["engines"][precision]
        engine = path.parent / entry["engine"]
        if digest(engine) != entry["sha256"]:
            raise ValueError(f"engine digest mismatch: {engine}")
        frontends[geometry.original_hw] = TensorRTFrontend(tensorrt, engine, device, geometry.cropped_nchw)
    return frontends


@torch.inference_mode()
def infer_frame(model, frame, normalization, device, precision, frontends=None, *, trace=False):
    from torch.profiler import record_function
    from lightbackbone_stereo_yolo.edge_deployment import FEATURE_OUTPUTS, unpack_frontend_outputs

    left, right, calibration = frame
    timings = {}
    l, r, geometry = prepare_rectified_images(left, right, normalization, timings=timings)
    route = {}
    start = perf_counter()
    with record_function("stereo.h2d"):
        l, r = l.to(device), r.to(device)
    timings["h2d_ms"] = (perf_counter() - start) * 1000
    if frontends is not None:
        with record_function("stereo.frontend"):
            frontend = frontends[geometry.original_hw]
            pyramid, dense = unpack_frontend_outputs(frontend(l, r), geometry.cropped_hw, geometry.padded_hw)
    else:
        amp = torch.autocast("cuda", dtype=torch.float16) if precision == "amp-fp16" else nullcontext()
        with record_function("stereo.frontend"), amp:
            pyramid = model.encoder(l, r)
            dense = model.detector(pyramid)
    # Keep geometric decoding and RoI coordinates in FP32 for all frontends.
    if precision != "fp32" or frontends is not None:
        pyramid = SimpleNamespace(**{name: getattr(pyramid, name).float()
            for name in FEATURE_OUTPUTS},
            metadata=getattr(pyramid, "metadata", None))
        dense = SimpleNamespace(**{name: getattr(dense, name).float() for name in ("objectness", "class_logits", "box6")})
    start = perf_counter()
    with record_function("stereo.decode_and_refine"):
        output = model.decode_and_refine(l, pyramid, dense, input_geometry=geometry, route=route,
                                         calibration=calibration)
    with record_function("stereo.output"):
        for name in ("pair_ids", "boxes_left", "boxes_right", "scores", "classes", "disparity",
                     "validity", "foreground_validity", "mask_logits", "mask_validity", "near10_distance_m"):
            value = getattr(output, name)
            if value is not None:
                value.cpu()
        if output.left_only is not None:
            for name in ("boxes", "ids", "scores", "classes", "distance_m"):
                getattr(output.left_only, name).cpu()
        if output.crop is not None:
            for name in ("c_l", "c_r", "c", "y0", "h_crop", "w_roi", "h_roi"):
                getattr(output.crop, name).cpu()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    timings["postprocess_and_output_ms"] = (perf_counter() - start) * 1000
    timings["idisp_pairs"] = len(route["idisp"].indices)
    timings["output_pairs"] = len(output.pair_ids)
    if trace:
        return output, route, timings
    return timings


def add_input_arguments(parser):
    parser.add_argument("--openstereo-source", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=ROOT / "models/stereo-yolo26-fixed56-seed17-e20.pt")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--calibration-zip", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--near10-prefilter-m", type=float, default=15.)
    parser.add_argument("--near10-mask-iou", type=float, default=.60)
    parser.add_argument("--near10-review-box-iou", type=float,
                        help="allow Mask review at this bilateral IoU even if box disparities disagree")
    parser.add_argument("--output", type=Path, required=True)


def add_runtime_arguments(parser):
    parser.add_argument("--precision", choices=("fp32", "amp-fp16", "trt-fp32", "trt-fp16"), default="fp32")
    parser.add_argument("--engine-manifest", type=Path, action="append", default=[])


def configure_runtime(parser, args, model, frames, device, evidence):
    if args.precision != "fp32" and device.type != "cuda":
        parser.error("accelerated precision modes require CUDA")
    if args.precision.startswith("trt-") != bool(args.engine_manifest):
        parser.error("TensorRT modes require --engine-manifest for each input size")
    if not frames:
        parser.error("the evaluation split contains no frames")
    model.instance_head.idisp.amp_aggregation = args.precision == "amp-fp16"
    if not args.engine_manifest:
        return None
    frontends = load_frontends(args.engine_manifest, args.precision.removeprefix("trt-"), device, evidence)
    sizes = {(frame[0].height, frame[0].width) for _, frame in frames}
    missing = sizes.difference(frontends)
    if missing:
        parser.error(f"missing TensorRT engines for input sizes {sorted(missing)}")
    return frontends


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_input_arguments(parser)
    add_runtime_arguments(parser)
    parser.add_argument("--near10-policy", choices=("baseline", "range", "range-dedup"), default="baseline")
    parser.add_argument("--duration-s", type=float, default=120.)
    parser.add_argument("--rate-fps", type=float)
    parser.add_argument("--profile", action="store_true", help="run a separate instrumented pass over the input set")
    args = parser.parse_args()
    if args.near10_policy != "baseline" and args.calibration_zip is None:
        parser.error("range policies require --calibration-zip")
    device = torch.device(args.device)
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    evidence = source_info(ROOT, __file__, args.openstereo_source, args.weights)
    model, bundle = load_published_model(args.weights, args.openstereo_source, device, apply_paired_nms=True,
        near10_mode=args.near10_policy if args.calibration_zip else None,
        near10_prefilter_m=args.near10_prefilter_m, near10_mask_iou=args.near10_mask_iou,
        near10_review_box_iou=args.near10_review_box_iou)
    dataset = validation_dataset(args.dataset_root, args.split_json, bundle, load_images=False)
    if args.calibration_zip:
        with ZipFile(args.calibration_zip) as archive:
            frames = decoded_frames(dataset, archive)
    else:
        frames = decoded_frames(dataset)
    frontends = configure_runtime(parser, args, model, frames, device, evidence)
    def infer(frame):
        return infer_frame(model, frame, bundle["normalization"], device, args.precision, frontends)
    for index in range(10):
        infer(frames[index % len(frames)][1])
    report = run_stream(frames, infer, rate_fps=args.rate_fps, duration_s=args.duration_s)
    if args.profile:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        profile_rows = []
        for frame_id, frame in frames:
            with torch.profiler.profile(activities=activities) as profiler:
                infer(frame)
            spans = {}
            for event in profiler.key_averages():
                if event.key.startswith("stereo."):
                    device_time = (event.device_time_total if hasattr(event, "device_time_total")
                                   else event.cuda_time_total)
                    spans[event.key] = {"cpu_ms": event.cpu_time_total / 1000,
                                        "cuda_ms": device_time / 1000, "calls": event.count}
            profile_rows.append({"frame_id": frame_id, "spans": spans})
        report["profile_rows"] = profile_rows
        report["profile_p95_ms"] = {
            name: {key: float(np.percentile([row["spans"].get(name, {}).get(key, 0.) for row in profile_rows], 95))
                   for key in ("cpu_ms", "cuda_ms")}
            for name in sorted({name for row in profile_rows for name in row["spans"]})
        }
    report["protocol"] = {
        **evidence, "device": str(device), "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "split_sha256": digest(args.split_json), "precision": args.precision, "policy": args.near10_policy,
        "engine_manifests": {str(path): digest(path) for path in args.engine_manifest},
        "calibration_sha256": digest(args.calibration_zip) if args.calibration_zip else None,
        "prefilter_m": args.near10_prefilter_m, "mask_iou": args.near10_mask_iou,
        "review_box_iou": args.near10_review_box_iou,
    }
    write_json(args.output, report)


if __name__ == "__main__":
    main()
