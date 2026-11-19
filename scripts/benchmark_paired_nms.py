"""Benchmark paired NMS off/on at one chosen operating point on the same split."""
from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import torch

from evaluate_paired_nms import sha256, validation_dataset, write_json
from infer import ROOT, load_published_model


LATENCY_FIELDS = ("end_to_end_ms", "decode_and_roi_dispatch_ms", "instance_head_ms")


def measure_forward(model, left, right):
    """Measure batch-one forward after input transfer and before output export."""
    is_cuda = left.is_cuda
    stages = ("detector_done", "instance_start", "instance_end", "forward_end")
    if is_cuda:
        stream = torch.cuda.current_stream(left.device)
        events = {stage: torch.cuda.Event(enable_timing=True) for stage in stages}
        # Initialize events outside the timed region. Elapsed times use the
        # measurements from the subsequent records on this same stream.
        for event in events.values():
            event.record(stream)
    marks = {}

    def stamp(stage):
        if is_cuda:
            marks[stage] = events[stage]
            events[stage].record(stream)
        else:
            marks[stage] = perf_counter()

    def duration(first, last):
        if is_cuda:
            return marks[first].elapsed_time(marks[last])
        return 1000 * (marks[last] - marks[first])

    hooks = (
        model.detector.register_forward_hook(lambda *_: stamp("detector_done")),
        model.instance_head.register_forward_pre_hook(lambda *_: stamp("instance_start")),
        model.instance_head.register_forward_hook(lambda *_: stamp("instance_end")),
    )
    try:
        if is_cuda:
            torch.cuda.synchronize(left.device)
        start = perf_counter()
        output = model(left, right)
        stamp("forward_end")
        if is_cuda:
            torch.cuda.synchronize(left.device)
        latency = 1000 * (perf_counter() - start)
        ran_instance = "instance_start" in marks
        return {
            "pairs": output.scores.numel(),
            "end_to_end_ms": latency,
            "decode_and_roi_dispatch_ms": duration("detector_done", "instance_start" if ran_instance else "forward_end"),
            "instance_head_ms": duration("instance_start", "instance_end") if ran_instance else 0.,
        }
    finally:
        for hook in hooks:
            hook.remove()


def percentiles(values):
    data = torch.tensor(values, dtype=torch.float64)
    p50, p95 = torch.quantile(data, data.new_tensor([.50, .95])).tolist()
    return {"p50": p50, "p95": p95}


@torch.inference_mode()
def benchmark(model, dataset, device, *, warmup, repeats):
    if not len(dataset):
        raise ValueError("benchmark split contains no frames")
    if repeats < 1 or warmup < 0:
        raise ValueError("repeats must be positive and warmup nonnegative")
    rows = []
    previous_override = model.apply_paired_nms
    try:
        for index in range(len(dataset)):
            sample = dataset[index]
            left, right = (sample[side].unsqueeze(0).to(device) for side in ("left", "right"))
            if index == 0:
                for _ in range(warmup):
                    for enable in (False, True):
                        model.apply_paired_nms = enable
                        model(left, right)

            for repeat in range(repeats):
                row = {"frame_id": sample["frame_id"], "repeat": repeat}
                # Alternate arm order to reduce cache and scheduling bias.
                order = (False, True) if (index + repeat) % 2 == 0 else (True, False)
                for enable in order:
                    model.apply_paired_nms = enable
                    row["on" if enable else "off"] = measure_forward(model, left, right)
                if row["on"]["pairs"] > row["off"]["pairs"]:
                    raise RuntimeError(f"NMS increased the number of pairs on {sample['frame_id']}")
                rows.append(row)
    finally:
        model.apply_paired_nms = previous_override

    summary = {arm: {key: percentiles([row[arm][key] for row in rows]) for key in LATENCY_FIELDS}
               for arm in ("off", "on")}
    for arm in ("off", "on"):
        summary[arm]["pairs_per_forward"] = percentiles([row[arm]["pairs"] for row in rows])
    summary["on_minus_off"] = {
        key: percentiles([row["on"][key] - row["off"][key] for row in rows])
        for key in (*LATENCY_FIELDS, "pairs")
    }
    return {"frames": len(dataset), "paired_timing_samples": len(rows), "summary": summary, "timings": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=ROOT / "models/stereo-yolo26-fixed56-seed17-e20.pt")
    parser.add_argument("--openstereo-source", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--score", type=float, default=.40)
    parser.add_argument("--topk", type=int, default=500, help="0 means unlimited")
    parser.add_argument("--nms-iou", type=float, default=.5)
    parser.add_argument("--warmup", type=int, default=2, help="unmeasured runs per arm on the first frame")
    parser.add_argument("--repeats", type=int, default=3, help="measured runs per arm on each frame")
    args = parser.parse_args()
    if not 0 <= args.score <= 1 or not 0 <= args.nms_iou <= 1 or args.topk < 0 or args.repeats < 1 or args.warmup < 0:
        parser.error("invalid score, NMS IoU, top-K, warmup, or repeats")
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        parser.error("timing is supported on CPU or CUDA")
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu for a CPU measurement")
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model, bundle = load_published_model(args.weights, args.openstereo_source, device,
                                         score=args.score, topk=args.topk, paired_nms_iou=args.nms_iou)
    if model.detector_variant != "stereo_yolo26_pan":
        parser.error("paired NMS evaluation requires stereo_yolo26_pan")
    dataset = validation_dataset(args.dataset_root, args.split_json, bundle)
    result = benchmark(model, dataset, device, warmup=args.warmup, repeats=args.repeats)
    result["protocol"] = {
        "weights_sha256": sha256(args.weights), "checkpoint_epoch": bundle.get("epoch"),
        "checkpoint_config": bundle["config"], "split_sha256": sha256(args.split_json),
        "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "torch_version": str(torch.__version__), "torch_threads": torch.get_num_threads(),
        "batch_size": 1, "score_threshold": args.score, "pre_nms_topk": args.topk,
        "class_aware_nms_iou": args.nms_iou, "warmup_per_arm": args.warmup, "repeats_per_frame": args.repeats,
    }
    write_json(args.output, result)


if __name__ == "__main__":
    main()
