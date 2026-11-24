"""Evaluate R0–R2 on a fixed rectified split with per-class 10 m denominators."""
import argparse
from collections import Counter
from dataclasses import replace
from time import perf_counter
from zipfile import ZipFile

import torch

from benchmark_edge_stream import add_input_arguments, add_runtime_arguments, configure_runtime, decoded_frames, infer_frame
from diagnose_pred_roi import _target_groups
from edge_run_evidence import digest, serialize_route, source_info
from evaluate_edge_acceleration import CLASSES, disparity_scores, make_pairs, match, projected, reduce_pixel_metrics
from evaluate_paired_nms import validation_dataset, write_json
from infer import ROOT, load_published_model
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
    return classes, maps, support


def summarize(frames, arm):
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
        if name == "Car":
            complete = sum(not target["ambiguous"] and len(target["idisp_ids"]) == 1
                           and target["idisp_ids"] == target["output_ids"] for target in targets)
            totals["exactly_one_pair_targets"] = complete
            totals["all_targets_exactly_one_pair"] = complete == gt if gt else None
        totals["new_missed_gt"] = sum(
            target["gt"] in {old["gt"] for old in frame["arms"]["R0"]["classes"][name]["targets"]
                             if old["matched"]} and not target["matched"]
            for frame, row in zip(frames, rows) for target in row["targets"])
        if arm != "R0":
            totals["common_valid_pixels"] = {
                key: reduce_pixel_metrics([frame["arms"][arm]["shared_pixel_errors"][name][key]
                                           for frame in frames])
                for key in ("R0", arm)
            }
        result[name] = totals
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_input_arguments(parser)
    add_runtime_arguments(parser)
    parser.add_argument("--match-iou", type=float, default=.50)
    parser.add_argument("--range-only", action="store_true", help="run R0 and R1 without Car foreground weights")
    args = parser.parse_args()
    if args.calibration_zip is None or not 0 < args.match_iou <= 1:
        parser.error("a calibration zip and match IoU in (0, 1] are required")
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    model, bundle = load_published_model(args.weights, args.openstereo_source, device, apply_paired_nms=True,
        near10_mode="range" if args.range_only else "range-dedup",
        near10_prefilter_m=args.near10_prefilter_m, near10_mask_iou=args.near10_mask_iou,
        near10_review_box_iou=args.near10_review_box_iou)
    mapping = bundle["class_mapping"]
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
            rows.append(row)
    if not rows:
        parser.error("the evaluation split contains no frames")
    write_json(args.output, {
        "frames": rows, "arms": {arm: summarize(rows, arm) for arm in modes},
        "protocol": {**evidence, "precision": args.precision,
                     "engine_manifests": {str(path): digest(path) for path in args.engine_manifest},
                     "split_sha256": digest(args.split_json), "calibration_sha256": digest(args.calibration_zip),
                     "match_iou": args.match_iou, "near_m": 10., "prefilter_m": args.near10_prefilter_m,
                     "mask_iou": args.near10_mask_iou, "review_box_iou": args.near10_review_box_iou,
                     "small_sample_reporting_below": 10},
    })


if __name__ == "__main__":
    main()
