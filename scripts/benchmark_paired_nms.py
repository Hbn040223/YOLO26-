"""Benchmark paired NMS off/on at one chosen operating point on the same split."""
from __future__ import annotations

import argparse
from dataclasses import fields, replace
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import patch

import torch

from evaluate_paired_nms import sha256, validation_dataset, write_json
from infer import ROOT, load_published_model
from lightbackbone_stereo_yolo import detection
from lightbackbone_stereo_yolo.geometry import pair_boxes_to_lr
from lightbackbone_stereo_yolo.idisp_ab import error_totals, original_foreground, project_disparity, summarize_error
from lightbackbone_stereo_yolo.roi import align_left_mask_target


LATENCY_FIELDS = ("end_to_end_ms", "decode_and_roi_dispatch_ms", "paired_nms_ms", "instance_head_ms")


def measure_forward(model, left, right):
    """Measure batch-one forward after input transfer and before output export."""
    is_cuda = left.is_cuda
    stages = ("detector_done", "nms_start", "nms_end", "instance_start", "instance_end", "forward_end")
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

    suppress = detection.paired_nms

    def time_nms(*args, **kwargs):
        stamp("nms_start")
        result = suppress(*args, **kwargs)
        stamp("nms_end")
        return result

    hooks = (
        model.detector.register_forward_hook(lambda *_: stamp("detector_done")),
        model.instance_head.register_forward_pre_hook(lambda *_: stamp("instance_start")),
        model.instance_head.register_forward_hook(lambda *_: stamp("instance_end")),
    )
    try:
        if is_cuda:
            torch.cuda.synchronize(left.device)
        start = perf_counter()
        with patch.object(detection, "paired_nms", time_nms):
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
            "paired_nms_ms": duration("nms_start", "nms_end") if "nms_start" in marks else 0.,
            "instance_head_ms": duration("instance_start", "instance_end") if ran_instance else 0.,
        }
    finally:
        for hook in hooks:
            hook.remove()


def percentiles(values):
    data = torch.tensor(values, dtype=torch.float64)
    p50, p95 = torch.quantile(data, data.new_tensor([.50, .95])).tolist()
    return {"p50": p50, "p95": p95}


def trace_route(model, left, right, enable):
    """Trace the live decode selection and the boxes reaching the instance head."""
    select, suppress = detection.select_anchorfree_pairs, detection.paired_nms
    trace = {"nms_calls": 0, "roi": None, "pre": None, "kept": None}

    def select_pairs(pre, *args, **kwargs):
        trace["pre"] = pre
        trace["kept"] = select(pre, *args, **kwargs)
        return trace["kept"]

    def count_nms(*args, **kwargs):
        trace["nms_calls"] += 1
        return suppress(*args, **kwargs)

    def inspect_roi(_module, args, output):
        torch.testing.assert_close(args[1], trace["kept"].boxes6, rtol=0, atol=0)
        trace["roi"] = output

    hook = model.instance_head.register_forward_hook(inspect_roi)
    model.apply_paired_nms = enable
    try:
        with patch.object(detection, "select_anchorfree_pairs", select_pairs), patch.object(detection, "paired_nms", count_nms):
            trace["output"] = model(left, right)
    finally:
        hook.remove()
    kept = trace["kept"]
    assert (trace["roi"] is not None) == (kept is not None and bool(kept.indices.numel()))
    if kept is not None:
        output = trace["output"]
        boxes_l, boxes_r = pair_boxes_to_lr(kept.boxes6)
        for actual, expected in ((output.boxes_left, boxes_l), (output.boxes_right, boxes_r),
                                 (output.scores, kept.scores), (output.classes, kept.classes)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    return trace


def verify_live_replay(model, left, right, expected, sample):
    traces = {arm: trace_route(model, left, right, arm == "on") for arm in ("off", "on")}
    result = {}
    for arm, trace in traces.items():
        ids = lambda value: value.indices.tolist() if value is not None else []
        pre, kept = ids(trace["pre"]), ids(trace["kept"])
        replay = expected[arm]
        if pre != replay["pre_nms_dense_ids"] or kept != replay["kept_dense_ids"]:
            raise RuntimeError(f"{sample['frame_id']}: live {arm} dense IDs differ from replay")
        calls = int(arm == "on" and trace["pre"] is not None)
        if trace["nms_calls"] != calls:
            raise RuntimeError(f"{sample['frame_id']}: unexpected NMS call count for {arm}")
        result[arm] = {"pre_nms_dense_ids": pre, "roi_dense_ids": kept, "nms_calls": calls}
    before, after = (traces[arm]["pre"] for arm in ("off", "on"))
    if before is not None:
        for name in ("boxes6", "scores", "classes", "indices"):
            torch.testing.assert_close(getattr(before, name), getattr(after, name), rtol=0, atol=0)
    matched = {arm: dict(zip(row["matched_gt_ids"], row["matched_dense_ids"])) for arm, row in expected.items()}
    shared = sorted(matched["off"].keys() & matched["on"].keys())
    result["common_matched_gt"] = len(shared)
    result["reassigned_gt"] = sum(matched["off"][gt] != matched["on"][gt] for gt in shared)
    for arm in traces:
        result[arm]["disparity"] = {"pixels": 0, "epe_sum": 0., "d1_hits": 0}
        result[arm]["mask_iou_sum"], result[arm]["mask_instances"] = 0., 0
    if shared:
        gt_ids = torch.tensor(shared, device=left.device)
        truth, base = original_foreground(sample, left.device, gt_ids)
        prediction, validity = {}, {}
        for arm, trace in traces.items():
            index_by_id = {dense_id: i for i, dense_id in enumerate(result[arm]["roi_dense_ids"])}
            i = torch.tensor([index_by_id[matched[arm][gt]] for gt in shared], device=left.device)
            roi = trace["roi"]
            crop = replace(roi.crop, **{field.name: getattr(roi.crop, field.name)[i]
                                        for field in fields(roi.crop)
                                        if isinstance(getattr(roi.crop, field.name), torch.Tensor)})
            disparity = SimpleNamespace(absolute=roi.disparity[i], valid=roi.validity[i])
            prediction[arm], validity[arm] = project_disparity(disparity, crop, left.shape[-2:])
            gt_mask = align_left_mask_target(sample["masks"].to(left.device)[gt_ids], trace["kept"].boxes6[i], crop).bool()
            pred_mask = roi.mask_logits[i] > 0
            mask_valid = roi.mask_validity[i]
            union = ((gt_mask | pred_mask) & mask_valid).flatten(1).sum(1)
            intersection = ((gt_mask & pred_mask) & mask_valid).flatten(1).sum(1)
            measured = union > 0
            result[arm]["mask_iou_sum"] = float((intersection[measured] / union[measured]).sum())
            result[arm]["mask_instances"] = int(measured.sum())
        common = base & validity["off"] & validity["on"]
        for arm in prediction:
            result[arm]["disparity"] = error_totals(prediction[arm], truth, common)
    return result


@torch.inference_mode()
def benchmark(model, dataset, device, *, warmup, repeats, replay=None):
    if not len(dataset):
        raise ValueError("benchmark split contains no frames")
    if repeats < 1 or warmup < 0:
        raise ValueError("repeats must be positive and warmup nonnegative")
    rows, routes = [], []
    previous_override = model.apply_paired_nms
    try:
        for index in range(len(dataset)):
            sample = dataset[index]
            left, right = (sample[side].unsqueeze(0).to(device) for side in ("left", "right"))
            if replay is not None:
                routes.append({"frame_id": sample["frame_id"], **verify_live_replay(
                    model, left, right, replay[sample["frame_id"]], sample,
                )})
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
    output = {"frames": len(dataset), "paired_timing_samples": len(rows), "summary": summary, "timings": rows}
    if replay is not None:
        for arm in ("off", "on"):
            summary[arm]["disparity_on_shared_gt"] = summarize_error({
                key: sum(row[arm]["disparity"][key] for row in routes) for key in ("pixels", "epe_sum", "d1_hits")
            })
            mask_count = sum(row[arm]["mask_instances"] for row in routes)
            summary[arm]["mask_iou_on_shared_gt"] = (
                sum(row[arm]["mask_iou_sum"] for row in routes) / mask_count if mask_count else None
            )
        output["live_replay"] = {
            "all_pre_nms_ids_boxes_scores_classes_identical": True,
            "all_roi_ids_match_replay": True,
            "common_matched_gt": sum(row["common_matched_gt"] for row in routes),
            "reassigned_gt": sum(row["reassigned_gt"] for row in routes),
            "routes": routes,
        }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=ROOT / "models/stereo-yolo26-fixed56-seed17-e20.pt")
    parser.add_argument("--openstereo-source", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay", type=Path, help="repeat the evaluator's candidate IDs through the live RoI path")
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
    replay = None
    if args.replay is not None:
        report = json.loads(args.replay.read_text(encoding="utf-8"))
        if (report["source"]["weights_sha256"] != sha256(args.weights)
                or report["source"]["split_sha256"] != sha256(args.split_json)
                or report["protocol"]["pre_nms_topk"] != args.topk):
            parser.error("replay checkpoint, split or top-K differs from the benchmark")
        chosen = {}
        for setting in report["settings"]:
            if setting["score_threshold"] == args.score and setting["nms_iou"] in (None, args.nms_iou):
                arm = "off" if setting["nms_iou"] is None else "on"
                chosen[arm] = {row["frame_id"]: row for row in setting["frames"]}
        frame_ids = {record.frame_id for record in dataset.records}
        if set(chosen) != {"off", "on"} or any(set(rows) != frame_ids for rows in chosen.values()):
            parser.error("replay must cover both benchmark arms and every validation frame")
        replay = {frame_id: {arm: rows[frame_id] for arm, rows in chosen.items()} for frame_id in frame_ids}
    result = benchmark(model, dataset, device, warmup=args.warmup, repeats=args.repeats, replay=replay)
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
