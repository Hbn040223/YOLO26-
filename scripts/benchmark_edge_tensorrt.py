"""Compare static TensorRT frontends with PyTorch on one registered stereo pair."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from functools import partial
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import torch

from benchmark_edge_acceleration import revision, summarize, synchronize
from benchmark_edge_backbone import device_pairs, elapsed
from evaluate_paired_nms import sha256, write_json
from infer import ROOT, load_published_model, normalized_rgb
from lightbackbone_stereo_yolo import detection
from lightbackbone_stereo_yolo.edge_deployment import (
    FRONTEND_OUTPUTS, StaticStereoFrontend, unpack_frontend_outputs,
)


class TensorRTFrontend:
    """Run a static, linear-I/O engine on the current PyTorch CUDA stream."""

    def __init__(self, trt, path, device, input_shape):
        self.trt = trt
        logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(logger)
        self.engine = self.runtime.deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"TensorRT could not deserialize {path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"TensorRT could not create a context for {path}")
        types = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16}
        names = {self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)}
        self.input_names = ("left", "right")
        if names != set((*self.input_names, *FRONTEND_OUTPUTS)):
            raise RuntimeError(f"unexpected TensorRT I/O tensors in {path}")
        self.buffers = {}
        for name in (*self.input_names, *FRONTEND_OUTPUTS):
            shape = tuple(self.engine.get_tensor_shape(name))
            input_tensor = name in self.input_names
            expected_mode = trt.TensorIOMode.INPUT if input_tensor else trt.TensorIOMode.OUTPUT
            if (self.engine.get_tensor_mode(name) != expected_mode or self.engine.get_tensor_format(name) != trt.TensorFormat.LINEAR
                    or min(shape) < 1 or (input_tensor and shape != input_shape)):
                raise RuntimeError(f"unexpected format, mode or static shape for {name} in {path}")
            self.buffers[name] = torch.empty(shape, device=device, dtype=types[self.engine.get_tensor_dtype(name)])
            if not self.context.set_tensor_address(name, self.buffers[name].data_ptr()):
                raise RuntimeError(f"TensorRT could not bind {name} in {path}")
        self.device = device

    def __call__(self, left, right):
        for name, tensor in zip(self.input_names, (left, right)):
            self.buffers[name].copy_(tensor)
        if not self.context.execute_async_v3(torch.cuda.current_stream(self.device).cuda_stream):
            raise RuntimeError("TensorRT frontend execution failed")
        return tuple(self.buffers[name].float() for name in FRONTEND_OUTPUTS)


def tensor_errors(reference, candidate):
    result = {}
    for name, expected, actual in zip(FRONTEND_OUTPUTS, reference, candidate):
        if actual.shape != expected.shape:
            raise RuntimeError(f"frontend output shape differs for {name}")
        if not bool(torch.isfinite(actual).all()):
            raise RuntimeError(f"non-finite frontend output in {name}")
        error = (actual - expected).abs()
        result[name] = {"mean_absolute": float(error.mean()), "max_absolute": float(error.max())}
    return result


def profile_native_dispatch(model, device, dispatch):
    """Time isolated stages in a separate, synchronized dispatch."""
    timings = dict.fromkeys(("paired_iou", "decision_mask_d2h", "greedy_cpu", "roi_mask_idisp"), 0.)
    instance_start = None

    @contextmanager
    def span(name):
        synchronize(device)
        start = perf_counter()
        try:
            yield
        finally:
            synchronize(device)
            timings[name] += (perf_counter() - start) * 1000

    def start_instance(*_):
        nonlocal instance_start
        synchronize(device)
        instance_start = perf_counter()

    def end_instance(*_):
        synchronize(device)
        timings["roi_mask_idisp"] = (perf_counter() - instance_start) * 1000

    before = model.instance_head.register_forward_pre_hook(start_instance)
    after = model.instance_head.register_forward_hook(end_instance)
    try:
        with patch.object(detection, "paired_nms", partial(detection.paired_nms, profile=span)):
            dispatch()
    finally:
        before.remove()
        after.remove()
    return {name + "_ms": duration for name, duration in timings.items()}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="export_edge_frontend.py JSON manifest")
    parser.add_argument("--weights", type=Path, default=ROOT / "models/stereo-yolo26-fixed56-seed17-e20.pt")
    parser.add_argument("--openstereo-source", type=Path, required=True)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--power-mode", required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        import tensorrt as trt
    except ImportError:
        parser.error("TensorRT 8.6–10.x Python bindings are required")
    device = torch.device(args.device)
    if device.type != "cuda" or args.warmup < 0 or args.repeats < 1:
        parser.error("CUDA, positive repeats and nonnegative warmup are required")
    torch.set_num_threads(2)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if (manifest["weights_sha256"] != sha256(args.weights)
            or manifest["openstereo_commit"] != revision(args.openstereo_source)):
        parser.error("checkpoint or OpenStereo revision differs from the export manifest")
    crop = manifest["crop_top_px"]
    model, bundle = load_published_model(args.weights, args.openstereo_source, device, apply_paired_nms=True,
                                         crop_top_px=crop)
    normalization = bundle["normalization"]
    left, right = (normalized_rgb(path, normalization["mean"], normalization["std"])[None].to(device)
                   for path in (args.left, args.right))
    if list(left.shape[-2:]) != manifest["original_hw"] or right.shape != left.shape:
        parser.error("stereo dimensions must match the static manifest")
    cropped = (left[..., crop:, :].contiguous(), right[..., crop:, :].contiguous())
    frontends = {"pytorch_fp32": StaticStereoFrontend(model.encoder, model.detector).eval()}
    for precision in ("fp32", "fp16"):
        built = manifest["engines"][precision]
        engine = args.manifest.parent / built["engine"]
        if sha256(engine) != built["sha256"]:
            raise RuntimeError(f"TensorRT engine failed checksum validation: {engine}")
        frontends[f"tensorrt_{precision}"] = TensorRTFrontend(trt, engine, device, tuple(cropped[0].shape))

    def dispatch(tensors, replay=None):
        pyramid, dense = unpack_frontend_outputs(tensors, tuple(manifest["input_hw"]), tuple(manifest["encoder_padded_hw"]))
        return model.decode_and_refine(left, pyramid, dense, replay_pairs=replay)

    reference = frontends["pytorch_fp32"](*cropped)
    reference_result = model(left, right)
    torch.testing.assert_close(dispatch(reference).pair_ids, reference_result.pair_ids, rtol=0, atol=0)
    replay = device_pairs(reference_result)
    numerical = {}
    for arm, frontend in frontends.items():
        tensors = frontend(*cropped)
        selected = dispatch(tensors)
        numerical[arm] = {
            "frontend_errors_against_pytorch": tensor_errors(reference, tensors),
            "native_pair_ids": selected.pair_ids.tolist(),
            "native_ordered_ids_match_pytorch": torch.equal(selected.pair_ids, reference_result.pair_ids),
        }

    rows = []
    names = tuple(frontends)
    for iteration in range(-args.warmup, args.repeats):
        start = iteration % len(names)
        row = {"repeat": iteration}
        for arm in names[start:] + names[:start]:
            frontend = frontends[arm]
            full_ms, output = elapsed(device, lambda: dispatch(frontend(*cropped), replay))
            frontend_ms, tensors = elapsed(device, lambda: frontend(*cropped))
            native_ms, _ = elapsed(device, lambda: dispatch(tensors))
            if not torch.equal(output.pair_ids, replay.indices):
                raise RuntimeError(f"{arm} changed frozen RoI IDs")
            row[arm] = {"frozen_forward_ms": full_ms, "frontend_ms": frontend_ms,
                        "native_decode_nms_roi_ms": native_ms, "frozen_pairs": len(replay.indices)}
        if iteration >= 0:
            rows.append(row)
    profiles = {arm: [] for arm in names}
    for iteration in range(args.repeats):
        start = iteration % len(names)
        for arm in names[start:] + names[:start]:
            tensors = frontends[arm](*cropped)
            profiles[arm].append(profile_native_dispatch(model, device, lambda: dispatch(tensors)))
    write_json(args.output, {
        "commit": revision(ROOT), "export_manifest_sha256": sha256(args.manifest),
        "tensorrt_version": trt.__version__, "device_name": torch.cuda.get_device_name(device),
        "power_mode": args.power_mode, "encoder_padded_hw": manifest["encoder_padded_hw"], "crop_top_px": crop,
        "matched_load_pair_ids": replay.indices.tolist(), "numerical": numerical,
        "summary": {arm: summarize([row[arm] for row in rows]) for arm in names}, "timings": rows,
        "native_stage_summary": {arm: summarize(values) for arm, values in profiles.items()},
        "native_stage_timings": profiles,
    })


if __name__ == "__main__":
    main()
