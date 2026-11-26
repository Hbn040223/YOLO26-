"""Benchmark iDisp on nested sets of actual post-range stereo pairs."""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from edge_run_evidence import digest, source_info
from evaluate_paired_nms import write_json
from infer import ROOT, add_idisp_cost_argument, load_published_model, load_rectified_pair, parse_score_threshold
from lightbackbone_stereo_yolo import local_disparity, model as model_module
from lightbackbone_stereo_yolo.local_aggregation import AttentionPyramid
from lightbackbone_stereo_yolo.roi_selection import subset_crop


def ranked_indices(pair_ids, scores):
    if len(set(pair_ids)) != len(pair_ids):
        raise ValueError("iDisp candidate IDs must be unique")
    return sorted(range(len(pair_ids)), key=lambda i: (-scores[i], pair_ids[i]))


def load_order(loads, round_id):
    ordinary = [n for n in loads if n not in (6, 9)]
    paired = [n for n in loads if n in (6, 9)]
    return ordinary + sorted(paired, reverse=bool(round_id % 2))


def distribution(samples):
    values = np.asarray(samples)
    rng = np.random.default_rng(17)
    indices = rng.integers(len(values), size=(10000, len(values)))
    quantiles = (50, 95)
    intervals = np.percentile(np.percentile(values[indices], quantiles, axis=1), (2.5, 97.5), axis=1).T
    return {f"p{q}": {"ms": float(point), "ci95_ms": interval.tolist()}
            for q, point, interval in zip(quantiles, np.percentile(values, quantiles), intervals)}


def time_call(call, device):
    cuda = device.type == "cuda"
    if cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    begin = perf_counter()
    if cuda:
        start.record()
    call()
    if cuda:
        end.record()
        end.synchronize()
    return {"wall_ms": (perf_counter() - begin) * 1000,
            "cuda_event_ms": start.elapsed_time(end) if cuda else None,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if cuda else None}


@contextmanager
def capture_stages(device, spans):
    tracked = {"stereo.roi_sample", "stereo.cost_volume", "stereo.local_aggregation", "stereo.disparity_regression"}

    @contextmanager
    def record(name):
        if name in tracked:
            start = perf_counter()
            events = None
            if device.type == "cuda":
                events = tuple(torch.cuda.Event(enable_timing=True) for _ in range(2))
                events[0].record()
            yield
            if events is not None:
                events[1].record()
            spans[name.removeprefix("stereo.")] = ((perf_counter() - start) * 1000, events)
        else:
            yield

    with ExitStack() as stack:
        for module in (model_module, local_disparity):
            stack.enter_context(patch.object(module, "record_function", record))
        yield


def benchmark_loads(loads, calls, device, warmup, repeats, reference):
    summaries, rows = {}, []
    keys = ["wall_ms"] + (["cuda_event_ms"] if device.type == "cuda" else [])
    for mode, call in calls.items():
        for n in loads:
            if n:
                for _ in range(warmup):
                    call(n)
        for round_id in range(repeats):
            order = load_order(loads, round_id)
            zero = {"wall_ms": 0., "cuda_event_ms": 0. if device.type == "cuda" else None,
                    "peak_allocated_bytes": None}
            measured = {n: time_call(lambda: call(n), device) if n else zero for n in order}
            delta = ({key: measured[9][key] - measured[6][key] for key in keys}
                     if 6 in measured and 9 in measured else None)
            rows.extend(dict(reference, mode=mode, round_id=round_id, N=n, order=order,
                             paired_delta_ms=delta if n in (6, 9) else None, **measured[n]) for n in order)
        selected = [row for row in rows if row["mode"] == mode]
        summary = {str(n): {key: distribution([row[key] for row in selected if row["N"] == n])
                            for key in keys} for n in loads}
        if 6 in loads and 9 in loads:
            summary["delta_9_minus_6"] = {key: distribution(
                [row["paired_delta_ms"][key] for row in selected if row["N"] == 9]) for key in keys}
        summaries[mode] = summary
    return rows, summaries


def parse_loads(raw):
    try:
        loads = sorted(set(map(int, raw.split(","))))
    except ValueError as error:
        raise argparse.ArgumentTypeError("loads must be comma-separated nonnegative integers") from error
    if min(loads) < 0:
        raise argparse.ArgumentTypeError("loads must be nonnegative")
    return loads


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("openstereo-source", "weights", "left", "right", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--bf-px-m", type=float, required=True)
    parser.add_argument("--cx-delta-px", type=float, required=True)
    parser.add_argument("--split", choices=("training", "testing"), required=True)
    parser.add_argument("--frame-id")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--score", type=parse_score_threshold, default=.48)
    parser.add_argument("--loads", type=parse_loads, default="0,2,6,9")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=60)
    parser.add_argument("--precision", choices=("fp32", "amp-fp16"), default="fp32")
    parser.add_argument("--profile", action="store_true", help="measure four RoI stages in separate rounds")
    add_idisp_cost_argument(parser)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1:
        parser.error("--warmup must be nonnegative and --repeats must be positive")
    device = torch.device(args.device)
    if args.precision == "amp-fp16" and device.type != "cuda":
        parser.error("AMP requires CUDA")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    with torch.cuda.device(device) if device.type == "cuda" else nullcontext():
        benchmark(args, device)


def benchmark(args, device):
    model, bundle = load_published_model(
        args.weights, args.openstereo_source, device, score=args.score, apply_paired_nms=True,
        near10_mode="range", idisp_cost_backend=args.idisp_cost_backend)
    l, r, geometry = load_rectified_pair(args.left, args.right, bundle["normalization"], 0)
    l, r = l.to(device), r.to(device)
    head = model.instance_head
    captured = []
    route = {}
    calls = []
    with (model.encoder.register_forward_hook(lambda _, _args, result: captured.append(result)),
          head.idisp.register_forward_pre_hook(lambda *_: calls.append(1))):
        model.forward_prepared(l, r, geometry, route=route, calibration=(args.bf_px_m, args.cx_delta_px))
    pairs = route["idisp"]
    pair_ids = pairs.indices.tolist()
    order = ranked_indices(pair_ids, pairs.scores.tolist())
    loads = [n for n in args.loads if n <= len(order)]
    selected = order[:max(loads, default=0)]
    boxes = pairs.boxes6[selected]
    pyramid = captured[0]
    features = ("p2_left", "p2_right", "p3_left", "p4_left")
    if args.precision == "amp-fp16" and selected:
        with torch.autocast("cuda", dtype=torch.float16):
            pyramid = model.encoder(l, r)
        pyramid = SimpleNamespace(**{name: getattr(pyramid, name).float() for name in features})
        head.idisp.amp_aggregation = True

    def sample(n):
        expanded = SimpleNamespace(**{name: getattr(pyramid, name).expand(n, -1, -1, -1) for name in features})
        return head.forward_disparity(expanded, boxes[:n], *geometry.original_hw)

    cache = {}
    if selected:
        inputs = []
        with head.idisp.register_forward_pre_hook(lambda _, values: inputs.append(values)):
            sample(len(selected))
        left, right, attention, crop, spatial = inputs[0]
        for n in loads:
            if n:
                index = torch.arange(n, device=device)
                cache[n] = (left[:n], right[:n], AttentionPyramid(attention.p2[:n], attention.p3[:n], attention.p4[:n]),
                            subset_crop(crop, index), spatial[:n])

    frame_id = args.frame_id or args.left.stem
    reference = {"frame_id": frame_id, "pair_ids_9": [pair_ids[i] for i in order[:9]] if len(order) >= 9 else [],
                 "pair_ids_6": [pair_ids[i] for i in order[:6]] if len(order) >= 6 else []}
    rows, summary = benchmark_loads(loads, {"roi_to_idisp": sample, "cached_roi": lambda n: head.idisp(*cache[n])},
                                    device, args.warmup, args.repeats, reference)
    stage_rows = []
    if args.profile:
        for n in cache:
            for round_id in range(args.repeats):
                spans = {}
                with capture_stages(device, spans):
                    sample(n)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                stage_rows.append(dict(frame_id=frame_id, N=n, round_id=round_id,
                                       stages={name: {"host_wall_ms": wall,
                                                      "cuda_event_ms": events[0].elapsed_time(events[1]) if events else None}
                                               for name, (wall, events) in spans.items()}))
    write_json(args.output, {
        **reference, "split": args.split, "source": source_info(ROOT, __file__, args.openstereo_source, args.weights),
        "left_sha256": digest(args.left), "right_sha256": digest(args.right),
        "calibration": [args.bf_px_m, args.cx_delta_px], "device": str(device), "score": args.score,
        "torch": torch.__version__, "cuda": torch.version.cuda, "precision": args.precision,
        "idisp_cost_backend": args.idisp_cost_backend, "tf32": False, "warmup": args.warmup, "repeats": args.repeats,
        "actual_pair_ids": pair_ids, "actual_idisp_calls": len(calls),
        "reference_boxes6": boxes.tolist(), "missing_loads": sorted(set(args.loads) - set(loads)),
        "roi_shapes": {n: list(values[0].shape) for n, values in cache.items()}, "rows": rows,
        "summary": summary, "stage_rows": stage_rows,
    })


if __name__ == "__main__":
    main()
