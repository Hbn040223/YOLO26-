"""Evaluate R0–R2 on a fixed rectified split with per-class 10 m denominators."""
import argparse
from collections import Counter
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import torch

from benchmark_edge_stream import add_input_arguments, add_runtime_arguments, configure_runtime, decoded_frames, infer_frame
from diagnose_pred_roi import _target_groups
from edge_run_evidence import digest, serialize_route, source_info
from evaluate_edge_acceleration import CLASSES, disparity_scores, make_pairs, match, projected, reduce_pixel_metrics
from evaluate_paired_nms import validation_dataset, write_json
from infer import ROOT, load_published_model, parse_score_threshold
from lightbackbone_stereo_yolo.roi_selection import subset_pairs
from lightbackbone_stereo_yolo.training_metrics import paired_match_eligibility


def eligible(pairs, sample, iou):
    return paired_match_eligibility(pairs.boxes6.cpu(), sample["boxes6"], iou,
        prediction_classes=pairs.classes.cpu(), target_classes=sample["classes"])[1]


def frame_metrics(output, route, sample, depths, mapping, iou):
    pairs = make_pairs(output)
    near = torch.tensor([depth is not None and 0 < depth <= 10. for depth in depths], dtype=torch.bool)
    unknown = torch.tensor([depth is None for depth in depths], dtype=torch.bool)
    final_matrix = eligible(pairs, sample, iou)
    idisp_matrix = eligible(route["idisp"], sample, iou)
    ignored = final_matrix[:, unknown].any(1) & ~final_matrix[:, ~unknown].any(1)
    kept = subset_pairs(pairs, ~ignored)
    near_indices = torch.where(near)[0]
    near_sample = {"boxes6": sample["boxes6"][near], "classes": sample["classes"][near]}
    matching = match(kept, near_sample, iou)
    kept_indices = torch.where(~ignored)[0]
    matches = {int(near_indices[t]): int(kept_indices[p]) for t, p in matching.items()}
    maps = projected(output, matches, sample)
    car_label = mapping.get(26, -1)
    foreground = projected(output, {t: p for t, p in matches.items()
                                     if int(sample["classes"][t]) == car_label}, sample, foreground=True)
    truth = sample["disparity"]
    support = sample["masks"][:, 0] & sample["disparity_valid"] & torch.isfinite(truth) & (truth > 0)
    classes = {}
    for raw, name in CLASSES.items():
        label = mapping.get(raw, -1)
        same_class = sample["classes"] == label
        targets = torch.where(same_class & near)[0].tolist()
        fitted = {t: p for t, p in maps.items() if t in targets}
        counts = []
        for gt in targets:
            entering = route["idisp"].indices.cpu()[idisp_matrix[:, gt]].tolist()
            final = pairs.indices[final_matrix[:, gt]].tolist()
            ambiguous = bool((final_matrix[final_matrix[:, gt]][:, near].sum(1) > 1).any()
                             or (idisp_matrix[idisp_matrix[:, gt]][:, near].sum(1) > 1).any())
            counts.append({"gt": gt, "depth_m": depths[gt], "idisp_ids": entering, "output_ids": final,
                           "ambiguous": ambiguous, "matched": gt in matches})
        predictions = int((kept.classes == label).sum())
        classes[name] = {
            "gt": len(targets), "unknown_gt": int((same_class & unknown).sum()),
            "predicted": predictions, "ignored_predictions": int((ignored & (pairs.classes == label)).sum()),
            "tp": len(fitted), "fp": predictions - len(fitted), "fn": len(targets) - len(fitted),
            "idisp_pairs": int((route["idisp"].classes == label).sum()), "targets": counts,
            "disparity": disparity_scores(fitted, truth, support & (same_class & near)[:, None, None]),
        }
        if raw == 26:
            car_support = support & (same_class & near)[:, None, None]
            classes[name]["foreground"] = disparity_scores(foreground, truth, car_support)
    return classes, maps, support


def summarize(frames, arm, *, baseline=None):
    result = {}
    for name in CLASSES.values():
        rows = [frame["arms"][arm]["classes"][name] for frame in frames]
        keys = ("gt", "unknown_gt", "predicted", "ignored_predictions", "tp", "fp", "fn", "idisp_pairs")
        totals = {key: sum(row[key] for row in rows) for key in keys}
        gt = totals["gt"]
        targets = [target for row in rows for target in row["targets"]]
        totals.update(
            recall=totals["tp"] / gt if gt else None,
            precision=totals["tp"] / totals["predicted"] if totals["predicted"] else None,
            disparity=reduce_pixel_metrics([row["disparity"] for row in rows]),
            ambiguous_targets=sum(target["ambiguous"] for target in targets),
            coverage_status="N/A" if gt == 0 else "small_sample" if gt < 10 else "reported",
        )
        for stage in ("idisp", "output"):
            counts = [len(target[stage + "_ids"]) for target in targets]
            totals[stage + "_distribution"] = {key: counts.count(n) for key, n in (("zero", 0), ("one", 1))}
            totals[stage + "_distribution"]["multiple"] = sum(count > 1 for count in counts)
            totals[stage + "_extra_pairs"] = sum(max(count - 1, 0) for count in counts)
        if name == "Car":
            complete = sum(not target["ambiguous"] and len(target["idisp_ids"]) == 1
                           and target["idisp_ids"] == target["output_ids"] for target in targets)
            totals["exactly_one_pair_targets"] = complete
            totals["all_targets_exactly_one_pair"] = complete == gt if gt else None
            totals["foreground"] = reduce_pixel_metrics([row["foreground"] for row in rows])
        totals["new_missed_gt"] = sum(
            (frame["frame_id"], target["gt"]) in baseline[name] and not target["matched"]
            for frame, row in zip(frames, rows) for target in row["targets"])
        if arm != "R0" and "shared_pixel_errors" in frames[0]["arms"][arm]:
            totals["common_valid_pixels"] = {
                key: reduce_pixel_metrics([frame["arms"][arm]["shared_pixel_errors"][name][key]
                                           for frame in frames])
                for key in ("R0", arm)
            }
        result[name] = totals
    return result


def matched_ids(frames, arm):
    return {name: {(frame["frame_id"], target["gt"]) for frame in frames
                   for target in frame["arms"][arm]["classes"][name]["targets"] if target["matched"]}
            for name in CLASSES.values()}


def score_trace(route):
    return {stage: dict(zip(pairs.indices.tolist(), pairs.scores.tolist()))
            for stage in ("pre_nms", "idisp", "near10_final") if (pairs := route.get(stage)) is not None}


@torch.inference_mode()
def evaluate_r2_training(model, dataset, frames, normalization, mapping, device, baseline):
    rows = []
    for (frame_id, frame), sample in zip(frames, dataset):
        if sample["frame_id"] != frame_id:
            raise ValueError("training evaluation frames do not align")
        bf, cx = frame[2]
        groups = _target_groups(sample, mapping, None, calibration={"bf_px_m": bf, "cx_delta_px": cx})
        output, route, _ = infer_frame(model, frame, normalization, device, "fp32", trace=True)
        metrics, _, _ = frame_metrics(output, route, sample, [g["depth_m"] for g in groups], mapping, .5)
        rows.append({"frame_id": frame_id, "arms": {"R2": {"classes": metrics}}})
    return summarize(rows, "R2", baseline=baseline), rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_input_arguments(parser)
    add_runtime_arguments(parser)
    parser.add_argument("--match-iou", type=float, default=.50)
    parser.add_argument("--comparison-weights", type=Path,
                        help="run a second checkpoint in R2 and compare common prediction pixels")
    parser.add_argument("--comparison-score", type=parse_score_threshold,
                        help="threshold for --comparison-weights; defaults to --score")
    parser.add_argument("--range-only", action="store_true", help="run R0 and R1 without Car foreground weights")
    args = parser.parse_args()
    if args.calibration_zip is None or not 0 < args.match_iou <= 1:
        parser.error("a calibration zip and match IoU in (0, 1] are required")
    if args.comparison_score is not None and args.comparison_weights is None:
        parser.error("--comparison-score requires --comparison-weights")
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    model, bundle = load_published_model(args.weights, args.openstereo_source, device, apply_paired_nms=True,
        score=args.score,
        idisp_cost_backend=args.idisp_cost_backend,
        near10_mode="range" if args.range_only else "range-dedup",
        near10_prefilter_m=args.near10_prefilter_m, near10_mask_iou=args.near10_mask_iou,
        near10_review_box_iou=args.near10_review_box_iou)
    mapping = bundle["class_mapping"]
    comparison = None
    if args.comparison_weights:
        if args.range_only or args.engine_manifest:
            parser.error("cross-checkpoint R2 comparison requires the native or AMP backend")
        comparison, second = load_published_model(args.comparison_weights, args.openstereo_source, device,
            score=args.score if args.comparison_score is None else args.comparison_score,
            idisp_cost_backend=args.idisp_cost_backend,
            apply_paired_nms=True, near10_mode="range-dedup", near10_prefilter_m=args.near10_prefilter_m,
            near10_mask_iou=args.near10_mask_iou, near10_review_box_iou=args.near10_review_box_iou)
        if second["class_mapping"] != mapping or second["normalization"] != bundle["normalization"]:
            parser.error("comparison requires identical classes and input normalization")
        comparison.instance_head.idisp.amp_aggregation = args.precision == "amp-fp16"
        comparison.instance_head.idisp.force_fp32_aggregation = args.force_fp32_aggregation
        comparison.instance_head.idisp.skip_fixed_spatial_mask = args.skip_fixed_spatial_mask
    dataset = validation_dataset(args.dataset_root, args.split_json, bundle, load_images=False)
    modes = {"R0": "baseline", "R1": "range"}
    if not args.range_only:
        modes["R2"] = "range-dedup"
    policy = model.near10_policy
    rows = []
    evidence = source_info(ROOT, __file__, args.openstereo_source, args.weights)
    with ZipFile(args.calibration_zip) as archive:
        frames = decoded_frames(dataset, archive)
        frontends = configure_runtime(parser, args, model, frames, device, evidence)
        for record, (frame_id, frame), sample in zip(dataset.records, frames, dataset):
            if sample["frame_id"] != frame_id:
                raise ValueError(f"pixels and ground truth do not align: {frame_id} vs {sample['frame_id']}")
            bf, cx = frame[2]
            groups = _target_groups(sample, mapping, archive, calibration={"bf_px_m": bf, "cx_delta_px": cx})
            annotated = Counter(annotation.category_id for annotation in record.annotations)
            paired_targets = Counter(sample["classes"].tolist())
            row = {"frame_id": frame_id, "arms": {}}
            for arm, mode in modes.items():
                model.near10_policy = replace(policy, mode=mode)
                start = perf_counter()
                output, route, timings = infer_frame(model, frame, bundle["normalization"], device,
                                                     args.precision, frontends, trace=True)
                elapsed = (perf_counter() - start) * 1000
                metrics, maps, support = frame_metrics(output, route, sample, [g["depth_m"] for g in groups],
                                                       mapping, args.match_iou)
                for raw, name in CLASSES.items():
                    # The paired dataset omits annotations without valid pixels to project.
                    metrics[name]["unknown_gt"] += annotated[raw] - paired_targets[mapping.get(raw, -1)]
                row["arms"][arm] = {"classes": metrics, "route": serialize_route(route),
                                    "timings": {**timings, "frame_ms": elapsed}}
                if arm == "R2":
                    row["arms"][arm]["candidate_scores"] = score_trace(route)
                if arm == "R0":
                    reference = maps
                else:
                    shared = reference.keys() & maps.keys()
                    errors = {}
                    for raw, name in CLASSES.items():
                        mask = torch.zeros_like(support)
                        targets = [gt for gt in shared if int(sample["classes"][gt]) == mapping.get(raw)]
                        for gt in targets:
                            mask[gt] = support[gt] & reference[gt][1] & maps[gt][1]
                        errors[name] = {
                            "R0": disparity_scores({gt: reference[gt] for gt in targets}, sample["disparity"], mask),
                            arm: disparity_scores({gt: maps[gt] for gt in targets}, sample["disparity"], mask),
                        }
                    row["arms"][arm]["shared_pixel_errors"] = errors
            if comparison is not None:
                other, other_route, _ = infer_frame(comparison, frame, bundle["normalization"], device,
                                                     args.precision, trace=True)
                second_metrics, second_maps, _ = frame_metrics(
                    other, other_route, sample, [g["depth_m"] for g in groups], mapping, args.match_iou)
                common = maps.keys() & second_maps.keys()
                pixels = torch.zeros_like(support)
                for gt in common:
                    if int(sample["classes"][gt]) == mapping.get(26):
                        pixels[gt] = support[gt] & maps[gt][1] & second_maps[gt][1]
                row["comparison"] = {"car": second_metrics["Car"], "classes": second_metrics,
                                     "route": serialize_route(other_route), "candidate_scores": score_trace(other_route),
                                     "common_car_pixels": {
                    "candidate": disparity_scores({gt: maps[gt] for gt in common}, sample["disparity"], pixels),
                    "reference": disparity_scores({gt: second_maps[gt] for gt in common}, sample["disparity"], pixels)}}
            rows.append(row)
    if not rows:
        parser.error("the evaluation split contains no frames")
    comparison_report = None
    if comparison is not None:
        baseline = {name: {(row["frame_id"], target["gt"]) for row in rows
                           for target in row["comparison"]["classes"][name]["targets"] if target["matched"]}
                    for name in CLASSES.values()}
        actual = matched_ids(rows, "R2")
        missed = {name: sorted(baseline[name] - actual[name]) for name in CLASSES.values()}
        comparison_report = {"weights_sha256": digest(args.comparison_weights), "score": comparison.score_threshold,
                             "new_missed_car_gt": missed["Car"], "new_missed_gt_by_class": missed,
                             "common_car_pixels": {key: reduce_pixel_metrics(
                                 [row["comparison"]["common_car_pixels"][key] for row in rows])
                                 for key in ("candidate", "reference")}}
    write_json(args.output, {
        "cross_checkpoint_r2": comparison_report,
        "frames": rows, "arms": {arm: summarize(rows, arm, baseline=matched_ids(rows, "R0")) for arm in modes},
        "protocol": {**evidence, "precision": args.precision, "score": model.score_threshold,
                     "idisp_cost_backend": args.idisp_cost_backend,
                     "force_fp32_aggregation": args.force_fp32_aggregation,
                     "skip_fixed_spatial_mask": args.skip_fixed_spatial_mask,
                     "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                     "paired_nms_iou": model.paired_nms_iou, "pre_nms_topk": model.pre_nms_topk,
                     "engine_manifests": {str(path): digest(path) for path in args.engine_manifest},
                     "split_sha256": digest(args.split_json), "calibration_sha256": digest(args.calibration_zip),
                     "match_iou": args.match_iou, "near_m": 10., "prefilter_m": args.near10_prefilter_m,
                     "mask_iou": args.near10_mask_iou, "review_box_iou": args.near10_review_box_iou,
                     "small_sample_reporting_below": 10},
    })


if __name__ == "__main__":
    main()
