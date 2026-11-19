"""Compare trained iDisp sampling arms on shared frozen B2a candidates."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cache_a3_fixed_pairs import decode_from_pyramid
from lightbackbone_stereo_yolo.data.frame_dataset import KittiFrameDataset, build_frame_records
from lightbackbone_stereo_yolo.idisp_ab import error_totals, original_foreground, project_disparity, summarize_error
from lightbackbone_stereo_yolo.lightstereo_encoder import LightStereoFeatureEncoder
from lightbackbone_stereo_yolo.model import StereoLocalInstanceHead
from lightbackbone_stereo_yolo.stereo_yolo26 import StereoYOLO26PanHead
from lightbackbone_stereo_yolo.training import repeat_roi_features, roi_residual_target

MODES = ("native_pad", "fixed_56")


def create_models(pair, cache, device):
    payloads = [torch.load(path, map_location="cpu", weights_only=False) for path in pair]
    configs = [payload["config"] for payload in payloads]
    for key in ("idisp_only", "seed", "roi_margin", "initial_weights_sha256", "epoch_order_sha256", "val_json"):
        if configs[0][key] != configs[1][key]:
            raise ValueError(f"paired checkpoints differ in {key}")
    if (not configs[0]["idisp_only"] or [cfg["roi_sampling"] for cfg in configs] != list(MODES)
            or payloads[0]["epoch"] != payloads[1]["epoch"]):
        raise ValueError("expected same-epoch checkpoints in native_pad, fixed_56 order")
    config = configs[0]
    if config["initial_weights_sha256"] != cache["checkpoint_sha256"]:
        raise ValueError("cache was made from a different frozen B2a state")
    if hashlib.sha256(Path(config["val_json"]).read_bytes()).hexdigest() != cache["manifest_sha256"]:
        raise ValueError("validation manifest differs from the cache")
    mapping = payloads[0]["class_mapping"]
    if mapping != payloads[1]["class_mapping"]:
        raise ValueError("paired checkpoints differ in class mapping")
    state = payloads[0]["model"]
    for key in state:
        if not key.startswith("instance_head.idisp.") and not torch.equal(state[key], payloads[1]["model"][key]):
            raise ValueError(f"frozen weights differ across arms for {key}")
    encoder = LightStereoFeatureEncoder.from_checkpoint(
        config["lightstereo_source"], config["checkpoint"], adapter_channels=config["adapter_channels"],
    )
    detector = StereoYOLO26PanHead(config["adapter_channels"], len(mapping), one_to_one=True)
    encoder.load_state_dict({key[len("encoder."):]: value for key, value in state.items() if key.startswith("encoder.")})
    detector.load_state_dict({key[len("detector."):]: value for key, value in state.items() if key.startswith("detector.")})
    heads = {}
    for mode, payload in zip(MODES, payloads):
        head = StereoLocalInstanceHead(config["adapter_channels"], payload["dmax_roi"], roi_sampling=mode)
        head.load_state_dict({key[len("instance_head."):]: value for key, value in payload["model"].items()
                              if key.startswith("instance_head.")})
        heads[mode] = head.to(device).eval()
    return encoder.to(device).eval(), detector.to(device).eval(), heads, config, mapping


def run_head(head, pyramid, boxes, image_hw, margin, include_mask=False):
    if not len(boxes):
        return None
    features = repeat_roi_features(pyramid, len(boxes))
    if include_mask:
        return head(features, boxes, *image_hw, margin=margin)
    return head.forward_disparity(features, boxes, *image_hw, margin=margin)


def evaluate_boxes(heads, pyramid, boxes, sample, image_hw, margin, device, indices=None):
    truth, base = original_foreground(sample, device, indices)
    predictions = {}
    valid = {}
    volumes = {}
    for mode, head in heads.items():
        raw = run_head(head, pyramid, boxes, image_hw, margin)
        if raw is None:
            predictions[mode] = truth.new_empty((0,) + tuple(image_hw))
            valid[mode] = base
            volumes[mode] = None
        else:
            disparity, crop = raw
            prediction, spatial = project_disparity(disparity, crop, image_hw)
            predictions[mode] = prediction
            valid[mode] = base & spatial
            volumes[mode] = crop
    return truth, base, predictions, valid, volumes


def report_setting(data, heads, near_flags):
    truth, base, predictions, valid, volumes = data
    common = valid["native_pad"] & valid["fixed_56"]
    metrics = {}
    for mode, head in heads.items():
        overflow = torch.zeros_like(base)
        if volumes[mode] is not None:
            residual = roi_residual_target(truth.unsqueeze(1), volumes[mode])[:, 0]
            overflow = (residual.abs() > head.idisp.search.dmax_roi) & valid[mode]
        metrics[mode] = {
            "common": error_totals(predictions[mode], truth, common),
            "near_common": error_totals(predictions[mode][near_flags], truth, common[near_flags]),
            "foreground": int(base.sum()),
            "covered": int(valid[mode].sum()),
            "overflow": int(overflow.sum()),
            "empty_coverage_instances": int((~valid[mode].flatten(1).any(1)).sum()),
            "empty_common_instances": int((~common.flatten(1).any(1)).sum()),
        }
    return metrics


def example_figure(path, sample, entry, pred_data, near_matches):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle

    if not near_matches:
        return
    row = min(near_matches, key=lambda i: entry["gt_groups"][entry["target_indices"][i]]["depth_m"])
    box = entry["boxes6"][row]
    truth, base, predictions, validity, crops = pred_data
    left = sample["left"] * torch.tensor([.229, .224, .225])[:, None, None] + torch.tensor([.485, .456, .406])[:, None, None]
    right = sample["right"] * torch.tensor([.229, .224, .225])[:, None, None] + torch.tensor([.485, .456, .406])[:, None, None]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6), layout="constrained")
    for ax, rgb, x1, x2, title in zip(axes[0, :2], (left, right), (box[0], box[4]), (box[2], box[5]), ("Left", "Right")):
        ax.imshow(rgb.clamp(0, 1).permute(1, 2, 0))
        ax.add_patch(Rectangle((x1, box[1]), x2-x1, box[3]-box[1], fill=False, color="orange", linewidth=2))
        ax.set_title(title)
    axes[0, 2].axis("off")
    shown = (truth[0], predictions["native_pad"][row], predictions["fixed_56"][row])
    masks = (base[row], validity["native_pad"][row], validity["fixed_56"][row])
    crop = crops["fixed_56"]
    x0, y0, c, h = (float(value[row]) for value in (crop.c_l, crop.y0, crop.c, crop.h_crop))
    for ax, image, valid, label in zip(axes[1], shown, masks, ("GT foreground", *MODES)):
        heat = ax.imshow(np.ma.masked_where(~valid.cpu().numpy(), image.cpu().numpy()), cmap="turbo", vmin=0, vmax=96)
        ax.set_xlim(x0, x0 + c)
        ax.set_ylim(y0 + h, y0)
        ax.set_title(label)
    for ax in axes.flat:
        ax.set_axis_off()
    fig.colorbar(heat, ax=list(axes[1]), label="Absolute disparity (original px)", shrink=.75)
    fig.suptitle(f"{sample['frame_id']}  |  GT {entry['target_indices'][row]}")
    fig.savefig(path, dpi=120)
    plt.close(fig)


def timed_cuda(call, device, local):
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    if local:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
    else:
        wall = time.perf_counter()
    value = call()
    if local:
        end.record()
    torch.cuda.synchronize(device)
    elapsed = start.elapsed_time(end) if local else (time.perf_counter() - wall) * 1000
    return value, elapsed, torch.cuda.max_memory_allocated(device) / 2**20


def profile_frame(encoder, detector, heads, pyramid, tensors, entry, cache, config, repeats, warmup, device, reverse):
    image_hw = tuple(entry["image_hw"])
    boxes = torch.tensor(entry["all_boxes6"], device=device).reshape(-1, 6)
    margin = config["roi_margin"]
    reports = {mode: {"local_ms": [], "e2e_ms": [], "peak_mib": [], "pairs": len(boxes)} for mode in MODES}

    def end_to_end(head):
        online = encoder(*tensors)
        pairs = decode_from_pyramid(detector, online, device, score=cache["score"], topk=cache["topk"],
                                    right_offset_mode=cache["right_offset_mode"])
        return pairs, run_head(head, online, pairs.boxes6, image_hw, margin, include_mask=True)

    for mode, head in heads.items():
        for _ in range(warmup):
            end_to_end(head)
            run_head(head, pyramid, boxes, image_hw, margin, include_mask=True)
    for rep in range(repeats):
        order = MODES[::-1] if (rep + reverse) % 2 else MODES
        for mode in order:
            head = heads[mode]
            roi, local_ms, local_peak = timed_cuda(
                lambda: run_head(head, pyramid, boxes, image_hw, margin, include_mask=True), device, True,
            )
            if roi is not None:
                count, height, width = roi.disparity.shape
                bins = 2 * head.idisp.search.dmax_roi + 1
                shapes = zip(roi.crop.h_roi.tolist(), roi.crop.w_roi.tolist())
                reports[mode]["cost_shape"] = [count, bins, height, width]
                reports[mode]["cost_elements"] = count * bins * height * width
                reports[mode]["aggregation_calls"] = len(set(shapes))
            del roi
            value, e2e_ms, peak = timed_cuda(lambda: end_to_end(head), device, False)
            pairs, roi_output = value
            if (pairs.indices.tolist() != entry["all_dense_ids"]
                    or not torch.allclose(pairs.boxes6, boxes, atol=.05, rtol=0.)):
                raise RuntimeError(f"online candidates differ from cache for {entry['frame_id']}")
            reports[mode]["local_ms"].append(local_ms)
            reports[mode]["e2e_ms"].append(e2e_ms)
            reports[mode]["peak_mib"].append(max(peak, local_peak))
            del value, pairs, roi_output
    return reports


def compute_pair(pair, cache, device, out, timing_repeats):
    encoder, detector, heads, config, mapping = create_models(pair, cache, device)
    records = build_frame_records(config["dataset_root"], config["val_json"])
    dataset = KittiFrameDataset(records, mapping, unknown_class="skip")
    by_id = {row["frame_id"]: row for row in cache["all_frames"]}
    if set(by_id) != {record.frame_id for record in records}:
        raise ValueError("cache and validation frames differ")
    frames = []
    for index in range(len(dataset)):
        sample = dataset[index]
        entry = by_id[sample["frame_id"]]
        tensors = tuple(sample[side].unsqueeze(0).to(device) for side in ("left", "right"))
        image_hw = tuple(tensors[0].shape[-2:])
        boxes = sample["boxes6"].to(device)
        if not torch.equal(boxes, torch.tensor(entry["gt_boxes6"], device=device).reshape(-1, 6)):
            raise ValueError(f"GT changed in {sample['frame_id']}")
        indices = torch.tensor(entry["target_indices"], dtype=torch.long, device=device)
        near_gt = torch.tensor([g["near_car"] for g in entry["gt_groups"]], dtype=torch.bool, device=device)
        unknown_cars = sum(g["car"] and g["depth_m"] is None for g in entry["gt_groups"])
        pyramid = encoder(*tensors)
        gt = evaluate_boxes(heads, pyramid, boxes, sample, image_hw, config["roi_margin"], device)
        pred = evaluate_boxes(
            heads, pyramid, torch.tensor(entry["boxes6"], device=device).reshape(-1, 6),
            sample, image_hw, config["roi_margin"], device, indices,
        )
        common_four = gt[3]["native_pad"][indices] & gt[3]["fixed_56"][indices] & pred[3]["native_pad"] & pred[3]["fixed_56"]
        row = {
            "frame_id": sample["frame_id"], "gt": report_setting(gt, heads, near_gt),
            "pred": report_setting(pred, heads, near_gt[indices]), "targets": len(boxes), "matches": len(indices),
            "near_targets": int(near_gt.sum()), "near_matches": int(near_gt[indices].sum()),
            "unknown_distance_cars": unknown_cars,
            "four_way": {mode: {"gt": error_totals(gt[2][mode][indices], pred[0], common_four),
                                "pred": error_totals(pred[2][mode], pred[0], common_four)}
                         for mode in MODES},
        }
        if timing_repeats and sample["frame_id"] == "000018_10":
            example_figure(out / "000018_10.png", sample, entry, pred, near_gt[indices].nonzero().flatten().tolist())
        del gt, pred
        if timing_repeats:
            row["timing"] = profile_frame(encoder, detector, heads, pyramid, tensors, entry, cache, config,
                                          timing_repeats, 2, device, index)
        frames.append(row)
        if (index + 1) % 10 == 0:
            print(f"seed={config['seed']} evaluated {index + 1}/{len(dataset)}", flush=True)
    return {"seed": config["seed"], "bins": {mode: 2 * head.idisp.search.dmax_roi + 1 for mode, head in heads.items()},
            "frames": frames}


def aggregate_counts(rows):
    return summarize_error({key: sum(row[key] for row in rows) for key in ("pixels", "epe_sum", "d1_hits")})


def summarize_run(run):
    frames = run["frames"]
    result = {"seed": run["seed"], "bins": run["bins"],
              **{key: sum(row[key] for row in frames)
                 for key in ("targets", "matches", "near_targets", "near_matches", "unknown_distance_cars")}}
    for setting in ("gt", "pred"):
        result[setting] = {}
        for mode in MODES:
            rows = [row[setting][mode] for row in frames]
            counts = {key: sum(row[key] for row in rows) for key in
                      ("foreground", "covered", "overflow", "empty_coverage_instances", "empty_common_instances")}
            result[setting][mode] = {
                **counts,
                "coverage": counts["covered"] / counts["foreground"] if counts["foreground"] else None,
                "overflow_ratio": counts["overflow"] / counts["covered"] if counts["covered"] else None,
                "common": aggregate_counts([row["common"] for row in rows]),
                "near_common": aggregate_counts([row["near_common"] for row in rows]),
            }
    gap = {}
    for mode in MODES:
        a, b = (aggregate_counts([row["four_way"][mode][setting] for row in frames]) for setting in ("gt", "pred"))
        gap[mode] = {"gt": a, "pred": b, "pred_minus_gt_epe": b["epe"] - a["epe"] if a["pixels"] else None}
    a, b = (gap[mode]["pred_minus_gt_epe"] for mode in MODES)
    gap["interaction_epe"] = b - a if a is not None and b is not None else None
    result["four_way"] = gap
    return result


def paired_bootstrap(runs, repetitions=10000):
    reference_ids = [row["frame_id"] for row in runs[0]["frames"]]
    if any([row["frame_id"] for row in run["frames"]] != reference_ids for run in runs):
        raise ValueError("bootstrap requires the same frame order for every seed")
    rng = np.random.default_rng(170205)
    indices = rng.integers(len(reference_ids), size=(repetitions, len(reference_ids)))
    results = {}
    for setting in ("gt", "pred"):
        data = np.asarray([
            [[[row[setting][mode]["common"][key] for key in ("pixels", "epe_sum", "d1_hits")]
              for mode in MODES] for row in run["frames"]] for run in runs
        ], dtype=np.float64)
        total = data.sum(axis=1)
        if not (total[..., 0] > 0).all():
            raise ValueError(f"no common {setting} evaluation pixels")
        by_seed = total[..., 1:] / total[..., :1]
        estimated = by_seed.mean(axis=0)
        if estimated[0, 0] <= 0:
            raise ValueError("relative EPE increase requires a positive native EPE")
        sampled_total = data[:, indices].sum(axis=2)
        keep = (sampled_total[..., 0] > 0).all(axis=(0, 2))
        used = sampled_total[:, keep]
        rate = (used[..., 1:] / used[..., :1]).mean(axis=0)
        rate = rate[rate[:, 0, 0] > 0.]
        draws = {
            "relative_epe_increase": rate[:, 1, 0] / rate[:, 0, 0] - 1.,
            "absolute_d1_increase": rate[:, 1, 1] - rate[:, 0, 1],
        }
        points = (estimated[1, 0] / estimated[0, 0] - 1., estimated[1, 1] - estimated[0, 1])
        results[setting] = {
            name: {"point": float(point), "one_sided_95_upper": float(np.quantile(draw, .95)),
                   "valid_draws": len(draw)}
            for (name, draw), point in zip(draws.items(), points)
        }
        results[setting]["paired_by_seed"] = [
            {"seed": run["seed"], "native_epe": float(seed[0, 0]), "fixed_epe": float(seed[1, 0]),
             "delta_epe": float(seed[1, 0] - seed[0, 0]), "delta_d1": float(seed[1, 1] - seed[0, 1])}
            for run, seed in zip(runs, by_seed)
        ]
    return results


def summarize_timing(run):
    rows = [frame["timing"] for frame in run["frames"] if "timing" in frame]
    return {
        mode: {
            **{key: {"p50": float(np.quantile(values, .5)), "p95": float(np.quantile(values, .95))}
               for key in ("local_ms", "e2e_ms")
               for values in [[sample for row in rows for sample in row[mode][key]]]},
            "peak_mib": max(sample for row in rows for sample in row[mode]["peak_mib"]),
        }
        for mode in MODES
    } if rows else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", nargs=2, action="append", type=Path, required=True, metavar=("NATIVE", "FIXED"))
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--timing-repeats", type=int, default=5)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache = json.loads(args.cache.read_text())
    if cache["score"] != .40 or cache["topk"] != 500 or cache["apply_paired_nms"]:
        raise ValueError("expected the agreed B2a .40/500 one-to-one candidate cache")
    device = torch.device(args.device)
    if args.timing_repeats and device.type != "cuda":
        raise ValueError("timing requires CUDA")
    runs = []
    with torch.inference_mode():
        for index, pair in enumerate(args.pair):
            run = compute_pair(pair, cache, device, args.output_dir, args.timing_repeats if index == 0 else 0)
            path = args.output_dir / f"seed_{run['seed']}_frames.json"
            path.write_text(json.dumps(run, indent=2, allow_nan=False) + "\n")
            runs.append(run)
    if len({run["seed"] for run in runs}) != len(runs):
        raise ValueError("expected distinct seeds")
    bootstrap = paired_bootstrap(runs)
    timing = summarize_timing(runs[0])
    precision_pass = (bootstrap["pred"]["relative_epe_increase"]["one_sided_95_upper"] <= .10
                      and bootstrap["pred"]["absolute_d1_increase"]["one_sided_95_upper"] <= .03)
    timing_pass = timing is not None and timing["fixed_56"]["e2e_ms"]["p95"] < timing["native_pad"]["e2e_ms"]["p95"]
    result = {"seeds": [summarize_run(run) for run in runs], "paired_bootstrap": bootstrap, "timing": timing,
              "precision_pass": precision_pass, "timing_pass": timing_pass,
              "deploy_fixed": precision_pass and timing_pass if len(runs) >= 3 else None,
              "metadata": {"evaluation_cache_sha256": hashlib.sha256(args.cache.read_bytes()).hexdigest(),
                           "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
                           "dtype": "float32", "mask_included_in_timing": True,
                           "timing_repeats": args.timing_repeats, "timing_seed": runs[0]["seed"],
                           "warmup_per_frame_per_arm": 2, "torch_version": torch.__version__}}
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"precision_pass": precision_pass, "timing_pass": timing_pass, "timing": timing}, indent=2))


if __name__ == "__main__":
    main()
