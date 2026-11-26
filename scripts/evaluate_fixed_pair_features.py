"""Compare frozen checkpoints on identical, manually verified rectified stereo boxes.

The manifest is a list of frames. Each has frame_id, left, right, instance_map,
disparity, bf_px_m, cx_delta_px, and instances. Each instance has id (the
integer pixel value in instance_map), category_id, left_box and right_box
(x1,y1,x2,y2). Formal splits require right_box_verified=true per counted
instance. Paths are relative to the manifest. Disparity PNG values are
in 1/256 px. Use a separate manifest for each evaluation split.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from edge_run_evidence import digest, source_info
from evaluate_paired_nms import write_json
from infer import ROOT, load_published_model, load_rectified_pair, parse_score_threshold
from lightbackbone_stereo_yolo.idisp_ab import error_totals, project_disparity, summarize_error
from lightbackbone_stereo_yolo.local_disparity import build_local_cost_volume
from lightbackbone_stereo_yolo.roi_selection import near10_measure


def edge_band(mask):
    """Inner four-pixel band, counting the exterior of the image as background."""
    background = F.pad((~mask).float()[None, None], (4, 4, 4, 4), value=1.)
    return mask & F.max_pool2d(background, 9, stride=1)[0, 0].bool()


def valid_box(box, image_hw):
    if box is None or len(box) != 4 or not all(math.isfinite(x) for x in box):
        return False
    h, w = image_hw
    x1, y1, x2, y2 = box
    return 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h


def correlation_maps(cost, valid, temperature):
    support = valid.any(1)
    logits = (cost / temperature).masked_fill(~valid, -torch.inf)
    probs = torch.where(support[:, None], logits, 0.).softmax(1)
    peak = cost.masked_fill(~valid, -torch.inf).amax(1).masked_fill(~support, 0.)
    top = cost.masked_fill(~valid, -torch.inf).topk(2, dim=1).values
    gap_valid = valid.sum(1) >= 2
    gap = (top[:, 0] - top[:, 1]).masked_fill(~gap_valid, 0.)
    entropy = -(probs * probs.clamp_min(1e-30).log()).sum(1)
    return {"cosine_peak": (peak, support), "entropy": (entropy, support), "peak_margin": (gap, gap_valid)}


def left_right_consistency(left_volume, right_volume, dmax):
    left_cost = left_volume.cost.masked_fill(~left_volume.valid_bins, -torch.inf)
    right_cost = right_volume.cost.masked_fill(~right_volume.valid_bins, -torch.inf)
    left_offset = left_cost.argmax(1) - dmax
    right_offset = right_cost.argmax(1) - dmax
    width = left_offset.shape[-1]
    right_x = torch.arange(width, device=left_offset.device) - left_offset
    index = right_x.clamp(0, width - 1)
    valid = (left_volume.valid_bins.any(1) & (right_x >= 0) & (right_x < width)
             & right_volume.valid_bins.any(1).gather(-1, index))
    consistent = (left_offset + right_offset.gather(-1, index)).abs() <= 1
    return consistent.to(left_cost.dtype), valid


def project_mask(output, hw, threshold):
    probability = output.mask_logits.sigmoid()[:, 0]
    h, w = probability.shape[-2:]
    crop = replace(output.crop, h_roi=torch.full_like(output.crop.h_roi, h),
                   w_roi=torch.full_like(output.crop.w_roi, w))
    values, valid = project_disparity(SimpleNamespace(absolute=probability, valid=output.mask_validity[:, 0]), crop, hw)
    return valid & (values > threshold)


def roi_features(model, pyramid, boxes, hw, temperature):
    inputs = []
    head = model.instance_head
    with head.idisp.register_forward_pre_hook(lambda _, args: inputs.append(args)):
        expanded = SimpleNamespace(**{key: getattr(pyramid, key) for key in
                                    ("p2_left", "p2_right", "p3_left", "p4_left")})
        output = head(expanded, boxes, *hw)
    left, right, _, crop, spatial = inputs[0]
    search = head.idisp.search
    volume = build_local_cost_volume(left, right, search, spatial, lambda: 1.)
    reverse = build_local_cost_volume(right, left, search, spatial, lambda: 1.)
    maps = correlation_maps(volume.cost, volume.valid_bins, temperature)
    maps["left_right_consistency_within_one_bin"] = left_right_consistency(volume, reverse, search.dmax_roi)
    projected = {name: project_disparity(SimpleNamespace(absolute=values, valid=valid), crop, hw)
                 for name, (values, valid) in maps.items()}
    disparity, valid = project_disparity(SimpleNamespace(absolute=output.disparity, valid=output.validity), crop, hw)
    return disparity[0], valid[0], project_mask(output, hw, model.mask_threshold)[0], projected, crop


def region_result(truth, support, first, second):
    common = support & first[1] & second[1]
    mask_common = common & first[2] & second[2]
    denominator = int(support.sum())
    return {label: dict(
        gt_pixels=denominator, valid_pixels=int((support & valid).sum()),
        predicted_mask_valid_pixels=int((support & valid & mask).sum()),
        disparity=summarize_error(error_totals(disp, truth, common)),
        predicted_mask_disparity=summarize_error(error_totals(disp, truth, mask_common)),
        correlation={name: {"pixels": int((support & valid_map[0]).sum()),
                            "mean": float(value[0][support & valid_map[0]].mean()) if (support & valid_map[0]).any() else None}
                     for name, (value, valid_map) in maps.items()})
            for label, (disp, valid, mask, maps, _) in zip(("1.0x", "0.5x"), (first, second))}


def image_array(path, dtype):
    with Image.open(path) as image:
        return torch.from_numpy(np.array(image, dtype=dtype))


def box_iou(prediction, target):
    target = prediction.new_tensor(target)
    intersection = (torch.minimum(prediction[:, 2:], target[2:]) -
                    torch.maximum(prediction[:, :2], target[:2])).clamp_min(0).prod(1)
    area = (prediction[:, 2:] - prediction[:, :2]).clamp_min(0).prod(1)
    return intersection / (area + (target[2:] - target[:2]).prod() - intersection)


def automatic_pair(ids, classes, boxes_left, boxes_right, distances, category, left, right):
    eligible = torch.where(classes == category)[0]
    if not eligible.numel():
        return None
    left_iou = box_iou(boxes_left[eligible], left)
    right_iou = box_iou(boxes_right[eligible], right)
    quality = torch.minimum(left_iou, right_iou)
    i = quality.argmax()
    if quality[i] <= 0:
        return None
    index = eligible[i]
    distance = distances[index] if distances is not None else None
    return dict(pair_id=int(ids[index]), left_iou=float(left_iou[i]), right_iou=float(right_iou[i]),
                distance_m=float(distance) if distance is not None and distance.isfinite() else None)


def predict_pairs(model, left, pyramid, geometry, calibration):
    dense = model.detector(pyramid)
    old_policy = model.near10_policy
    try:
        for policy in ("range", "range-dedup"):
            model.near10_policy = replace(old_policy, mode=policy)
            route = {}
            detected = model.decode_and_refine(left, pyramid, dense, input_geometry=geometry,
                                               route=route, calibration=calibration)
            yield policy, (route["idisp"], detected)
    finally:
        model.near10_policy = old_policy


def compare_automatic(automatic, category, left, right, calibration):
    result = {}
    for policy, (pairs, detected) in automatic.items():
        matches = {
            "entry": automatic_pair(pairs.indices, pairs.classes, pairs.boxes6[:, :4], pairs.boxes6[:, [4, 1, 5, 3]],
                                    None, category, left, right),
            "output": automatic_pair(detected.pair_ids, detected.classes, detected.boxes_left, detected.boxes_right,
                                     detected.near10_distance_m, category, left, right),
        }
        found = matches["output"]
        if found is not None:
            index = torch.where(detected.pair_ids == found["pair_id"])[0][0]
            foreground = detected.foreground_validity[index]
            samples = detected.disparity[index][foreground]
            delta = detected.boxes_left[index, [0, 2]].mean() - detected.boxes_right[index, [0, 2]].mean()
            corrected = float(delta) - calibration[1]
            found["box_center_distance_m"] = calibration[0] / corrected if corrected > 0 else None
            found["foreground_pixels"] = int(foreground.sum())
            found["valid_disparity_pixels"] = int(detected.validity[index].sum())
            found["foreground_disparity_p10_p50_p90_px"] = (
                torch.quantile(samples, samples.new_tensor([.1, .5, .9])).tolist() if samples.numel() else None)
        result[policy] = matches
    return result


def aggregate_regions(frames):
    instances = [row["regions"] for frame in frames for row in frame["instances"] if "regions" in row]
    summary = {}
    for region in ("all", "edge", "interior"):
        summary[region] = {}
        for label in ("1.0x", "0.5x"):
            rows = [r[region][label] for r in instances]
            totals = {key: sum(row[key] for row in rows)
                      for key in ("gt_pixels", "valid_pixels", "predicted_mask_valid_pixels")}
            denominator = totals["gt_pixels"]
            for name in ("disparity", "predicted_mask_disparity"):
                totals[name] = summarize_error({key: sum(row[name][key] for row in rows)
                                                for key in ("pixels", "epe_sum", "d1_hits")})
            totals["coverage"] = totals["valid_pixels"] / denominator if denominator else None
            totals["predicted_mask_coverage"] = totals["predicted_mask_valid_pixels"] / denominator if denominator else None
            summary[region][label] = totals
    return summary


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("openstereo-source", "weights-full", "weights-half", "manifest", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--score-full", type=parse_score_threshold, default=.40)
    parser.add_argument("--score-half", type=parse_score_threshold, default=.48)
    parser.add_argument("--correlation-temperature", type=float, default=.1)
    parser.add_argument("--sample-role", choices=("development", "independent", "smoke"), required=True)
    args = parser.parse_args()
    if not math.isfinite(args.correlation_temperature) or args.correlation_temperature <= 0:
        parser.error("correlation temperature must be finite and positive")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    arms = [load_published_model(weights, args.openstereo_source, args.device, score=score, apply_paired_nms=True,
                                 near10_mode="range", near10_mask_iou=.50, near10_review_box_iou=.45)
            for weights, score in ((args.weights_full, args.score_full), (args.weights_half, args.score_half))]
    if [bundle["config"].get("encoder_backbone", "v2") for _, bundle in arms] != ["v2", "v2-050"]:
        parser.error("weights must identify the 1.0x and 0.5x encoders respectively")
    if len({bundle["dmax_roi"] for _, bundle in arms}) != 1 or any(
            model.instance_head.roi_sampling != "fixed_56" for model, _ in arms):
        parser.error("checkpoints must use fixed_56 with the same disparity search")
    assets = args.manifest.parent
    rows = []
    seen = set()
    for frame in json.loads(args.manifest.read_text()):
        frame_id = frame["frame_id"]
        if frame_id in seen:
            raise ValueError(f"duplicate frame_id: {frame_id}")
        seen.add(frame_id)
        calibration = (frame["bf_px_m"], frame["cx_delta_px"])
        if not all(map(math.isfinite, calibration)) or calibration[0] <= 0:
            raise ValueError(f"invalid calibration for {frame_id}")
        paths = {key: assets / frame[key] for key in ("left", "right", "instance_map", "disparity")}
        instances = frame["instances"]
        if len({x["id"] for x in instances}) != len(instances):
            raise ValueError(f"duplicate instance ID in {frame_id}")
        index_image = image_array(paths["instance_map"], np.int64).to(args.device)
        truth = image_array(paths["disparity"], np.float32).to(args.device) / 256.
        frame_rows = []
        frame_data = dict(frame_id=frame_id,instances=frame_rows, calibration=calibration,
                          right_box_source=frame.get("right_box_source"),
                          input_sha256={key:digest(path) for key,path in paths.items()},automatic={})
        pyramids = []
        automatic = []
        arrays = {}
        for label,(model,bundle) in zip(("1.0x","0.5x"),arms):
            l,r,geometry=load_rectified_pair(paths["left"],paths["right"],bundle["normalization"],0)
            hw = geometry.original_hw
            if tuple(truth.shape) != hw or tuple(index_image.shape) != hw:
                raise ValueError(f"label and image shapes differ for {frame_id}")
            l,r=l.to(args.device),r.to(args.device)
            pyramid = model.encoder(l,r)
            pyramids.append(pyramid)
            attempts = dict(predict_pairs(model, l, pyramid, geometry, calibration))
            automatic.append(attempts)
            frame_data["automatic"][label] = {policy: dict(idisp_ids=pairs.indices.tolist(),
                                                          output_ids=detected.pair_ids.tolist())
                                              for policy, (pairs, detected) in attempts.items()}

        for item in instances:
            left,right=item.get("left_box"),item.get("right_box")
            row=dict(id=item["id"],category_id=item["category_id"],left_box=left,right_box=right)
            frame_rows.append(row)
            if not valid_box(left, hw) or not valid_box(right, hw):
                row["excluded"]="missing or invalid stereo box"
                continue
            if args.sample_role != "smoke" and item.get("right_box_verified") is not True:
                row["excluded"] = "right box not verified"
                continue
            mask = index_image == item["id"]
            if not mask.any():
                row["excluded"]="no foreground instance pixels"
                continue
            boxes=torch.tensor([left + [right[0],right[2]]],dtype=torch.float32,device=args.device)
            row["automatic"] = {label: compare_automatic(attempts, bundle["class_mapping"].get(item["category_id"], -1),
                                                          left, right, calibration)
                                for label, (_, bundle), attempts in zip(("1.0x", "0.5x"), arms, automatic)}
            coarse = ((left[0]+left[2]) - (right[0]+right[2])) / 2 - calibration[1]
            row["box_center_distance_m"] = calibration[0] / coarse if coarse > 0 else None
            support = mask & torch.isfinite(truth) & (truth > 0)
            edge = edge_band(mask)
            fixed=[roi_features(model,pyramid,boxes,hw,args.correlation_temperature)
                   for (model,_),pyramid in zip(arms,pyramids)]
            row["regions"]={name:region_result(truth,px,*fixed) for name,px in
                            (("all",support),("edge",support & edge),("interior",support & ~edge))}
            crop = fixed[0][4]
            range_center = crop.c_l[0] - crop.c_r[0]
            extent = arms[0][1]["dmax_roi"] * crop.c[0] / crop.w_roi[0]
            row["search_range_gt_hits"] = int((support & ((truth - range_center).abs() <= extent)).sum())
            row["search_range_px"] = [float(range_center-extent),float(range_center+extent)]
            row["gt_depth_m"] = float(calibration[0]/(truth[support].median()-calibration[1])) if support.any() and truth[support].median()>calibration[1] else None
            row["prediction"]={}
            for label,(disparity,valid,pred_mask,_,_) in zip(("1.0x","0.5x"),fixed):
                union=int((mask|pred_mask).sum())
                distance=near10_measure(disparity[None],(valid&pred_mask)[None],calibration)[0]
                row["prediction"][label]=dict(mask_iou=int((mask&pred_mask).sum())/union,
                                              distance_m=float(distance) if distance.isfinite() else None)
                key=f'{len(rows)}_{len(frame_rows)-1}_{label}'
                arrays[key+'_mask']=pred_mask.cpu().numpy()
                arrays[key+'_disparity']=disparity.masked_fill(~(valid&pred_mask),0.).cpu().numpy()
                row["prediction"][label]["array_key"]=key
        args.output.parent.mkdir(parents=True,exist_ok=True)
        maps_file = args.output.with_name(f"{args.output.stem}-frame-{len(rows):04d}.npz")
        np.savez_compressed(maps_file, **arrays)
        frame_data["projected_maps"] = maps_file.name
        rows.append(frame_data)
    write_json(args.output,dict(frames=rows,means_use_common_disparity_pixels=True,edge_width_px=4,
                                sample_role=args.sample_role, aggregate=aggregate_regions(rows),
                                manifest_sha256=digest(args.manifest),
                                correlation_temperature=args.correlation_temperature,
                                source={label:source_info(ROOT,__file__,args.openstereo_source,weights)
                                        for label,weights in (("1.0x",args.weights_full),("0.5x",args.weights_half))}))


if __name__ == "__main__":
    main()
