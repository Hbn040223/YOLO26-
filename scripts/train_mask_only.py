"""Train only the Car Mask head on the published fixed-grid stereo model."""
from __future__ import annotations

import argparse
from dataclasses import fields, replace
import json
import os
from pathlib import Path
import random
from types import SimpleNamespace
from zipfile import ZipFile

import torch

from diagnose_pred_roi import _target_groups
from evaluate_paired_nms import sha256
from infer import ROOT, load_published_model
from lightbackbone_stereo_yolo.data.frame_dataset import KittiFrameDataset, build_frame_records
from lightbackbone_stereo_yolo.detection import decode_anchorfree_predictions
from lightbackbone_stereo_yolo.geometry import generate_point_locations, point_levels
from lightbackbone_stereo_yolo.idisp_ab import project_disparity
from lightbackbone_stereo_yolo.mask import mask_loss
from lightbackbone_stereo_yolo.training import freeze_for_mask_only, prepare_mask_rois
from lightbackbone_stereo_yolo.training_control import append_jsonl_record
from lightbackbone_stereo_yolo.training_metrics import (
    greedy_match_paired_detections, mask_pixel_counts, summarize_mask_counts,
)


def write_json(path, content):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def dataset_for(records, bundle, *, load_images=True):
    norm = bundle["normalization"]
    return KittiFrameDataset(records, bundle["class_mapping"], norm["mean"], norm["std"],
                             unknown_class="skip", load_images=load_images)


def prepare_split(records, val_records, bundle, dataset_root, out):
    by_id = {record.frame_id: record for record in records}
    all_ids = sorted(by_id)
    val_ids = {record.frame_id for record in val_records}
    if len(all_ids) != 160 or len(val_ids) != 40 or set(all_ids) & val_ids:
        raise RuntimeError("expected the disjoint KITTI 160/40 protocol")
    manifest_path = out / "frames.json"
    rng = random.Random(17)
    if manifest_path.exists():
        split = json.loads(manifest_path.read_text())
    else:
        dev = sorted(rng.sample(all_ids, 16))
        train = sorted(set(all_ids) - set(dev))
        ranked = train[:]
        rng.shuffle(ranked)
        metadata = dataset_for([by_id[key] for key in ranked], bundle, load_images=False)
        covered = {}
        choices = []
        car_class = bundle["class_mapping"][26]
        with ZipFile(dataset_root / "data_scene_flow_calib.zip") as calibration:
            for sample in metadata:
                cars = (sample["classes"] == car_class).nonzero().flatten()
                if not cars.numel():
                    continue
                gt = sample["boxes6"][cars, :4]
                longest = (gt[:, 2:] - gt[:, :2]).amax(1)
                groups = _target_groups(sample, bundle["class_mapping"], calibration)
                near = any(groups[i]["depth_m"] is not None and 0 < groups[i]["depth_m"] < 10 for i in cars.tolist())
                record = {"near": near, "small": bool((longest <= 32).any()), "multi_car": len(cars) > 1}
                key = sample["frame_id"]
                for name, present in record.items():
                    if present:
                        covered.setdefault(name, key)
                choices.append((key, record))
                if len(covered) == 3 and len(choices) >= 8:
                    break
        if len(covered) != 3 or len(choices) < 8:
            raise RuntimeError("cannot choose eight frames with near, small and multiple Car targets")
        selected = list(dict.fromkeys(list(covered.values()) + [key for key, _ in choices]))[:8]
        split = {"mask_train_144": train, "mask_dev_16": dev, "overfit_8": selected,
                 "overfit_scene_coverage": {key: detail for key, detail in choices if key in selected}}
        write_json(manifest_path, split)
    train, dev, overfit = (set(split[k]) for k in ("mask_train_144", "mask_dev_16", "overfit_8"))
    if not (len(train) == 144 and len(dev) == 16 and len(overfit) == 8
            and not train & dev and train | dev == set(all_ids) and overfit < train):
        raise RuntimeError("mask split manifest is not a disjoint 144/16 with 8 drawn from train")
    return {name: [by_id[key] for key in split[name]] for name in ("mask_train_144", "mask_dev_16", "overfit_8")}


def build_archived_pairs(archive_path, replay_path, bundle, weights, val_split):
    archive = json.loads(archive_path.read_text())
    replay = json.loads(replay_path.read_text())
    for header in (archive, replay["source"]):
        if header["weights_sha256"] != sha256(weights) or header["split_sha256"] != sha256(val_split):
            raise RuntimeError("fixed pair archive does not use these weights and validation frames")
    if bundle["config"]["right_offset_mode"] != archive["right_offset_mode"]:
        raise RuntimeError("fixed pair archive has a different box decode mode")
    off, on = ({row["frame_id"]: row for row in next(
        value["frames"] for value in replay["settings"]
        if value["score_threshold"] == .40 and value["nms_iou"] == mode)} for mode in (None, .70))
    cache = {}
    for frame in archive["frames"]:
        key = frame["frame_id"]
        previous, current = off[key], on[key]
        common = set(previous["matched_gt_ids"]) & set(current["matched_gt_ids"])
        selected = [(target, dense) for target, dense in zip(current["matched_gt_ids"], current["matched_dense_ids"])
                    if target in common]
        cache[key] = {"targets": frame["gt_boxes6"], "target_classes": frame["gt_classes"],
                      "matches": [{"target": target, "dense": dense, "boxes6": frame["boxes6"][dense],
                                   "score": frame["scores"][dense], "class": frame["classes"][dense]}
                                  for target, dense in selected]}
    if len(cache) != 40 or sum(len(row["matches"]) for row in cache.values()) != 151:
        raise RuntimeError("unexpected fixed comparison cohort")
    return cache


def stereo_pairs(model, pyramid, device):
    dense = model.detector(pyramid)
    scores = {"objectness": dense.objectness[0], "class_logits": dense.class_logits[0], "box6": dense.box6[0]}
    return decode_anchorfree_predictions(
        scores, generate_point_locations(point_levels(pyramid), device), .40, nms_threshold=.70,
        pre_nms_topk=500, right_offset_mode=model.right_offset_mode, apply_nms=True, class_aware_nms=True,
    )


def to_cpu(crop):
    return replace(crop, **{field.name: getattr(crop, field.name).cpu()
                           for field in fields(crop) if isinstance(getattr(crop, field.name), torch.Tensor)})


def size_bin(longest):
    if longest <= 16:
        return "le16"
    if longest <= 32:
        return "16_32"
    if longest <= 64:
        return "32_64"
    return "gt64"


@torch.no_grad()
def build_features(model, dataset, device, car_class, *, live_pred=False, archived=None, original=False):
    results = {"gt": [], "paired_gt": [], "pred": [], "pred_gt_right": []}
    for index, sample in enumerate(dataset):
        frame = sample["frame_id"]
        cars = (sample["classes"] == car_class).nonzero().flatten()
        if not cars.numel() and archived is None:
            continue
        left, right = (sample[side][None].to(device) for side in ("left", "right"))
        hw = sample["masks"].shape[-2:]
        pyramid = model.encoder(left, right)
        boxes = sample["boxes6"]

        def collect(name, ids, boxes6, dense_ids=None, scores=None):
            if not ids.numel():
                return
            masks = sample["masks"][ids]
            roi = prepare_mask_rois(model.instance_head, pyramid.p2_left, boxes6.to(device), masks.to(device), hw)
            size = (boxes[ids, 2:4] - boxes[ids, :2]).amax(1)
            record = {k: roi[k].cpu() for k in ("features", "target", "valid")}
            crop = to_cpu(roi["crop"])
            record.update(frame=frame, size=[size_bin(float(side)) for side in size], crop=crop, hw=hw,
                          classes=sample["classes"][ids], targets=ids, boxes=boxes6.cpu(),
                          dense_ids=dense_ids, scores=scores)
            if original:
                record["masks"] = masks
            results[name].append(record)

        collect("gt", cars, boxes[cars])
        if live_pred:
            pairs = stereo_pairs(model, pyramid, device)
            match = greedy_match_paired_detections(pairs.boxes6, boxes.to(device), pairs.scores, .50,
                prediction_classes=pairs.classes, target_classes=sample["classes"].to(device))
            target = match.target_indices.cpu()
            keep = sample["classes"][target] == car_class
            pred = match.prediction_indices[keep.to(device)]
            target = target[keep]
            prediction = pairs.boxes6[pred].cpu()
            collect("pred", target, prediction, pairs.indices[pred].cpu(), pairs.scores[pred].cpu())
            collect("paired_gt", target, boxes[target])
            hybrid = torch.cat((prediction[:, :4], boxes[target, 4:]), 1)
            collect("pred_gt_right", target, hybrid)
        if archived is not None:
            known = archived[frame]
            torch.testing.assert_close(torch.tensor(known["targets"]).reshape(-1, 6), boxes, atol=0, rtol=0)
            if known["target_classes"] != sample["classes"].tolist():
                raise RuntimeError(f"{frame}: archive labels changed")
            rows = known["matches"]
            target = torch.tensor([row["target"] for row in rows], dtype=torch.long)
            prediction = torch.tensor([row["boxes6"] for row in rows]).reshape(-1, 6)
            collect("pred", target, prediction, torch.tensor([row["dense"] for row in rows], dtype=torch.long),
                    torch.tensor([row["score"] for row in rows]))
        if (index + 1) % 10 == 0:
            print(f"cached {index + 1}/{len(dataset)} stereo frames", flush=True)
    return results


def add_rows(rows, counts, entry):
    for index, target_id in enumerate(entry["targets"].tolist()):
        crop = entry["crop"]
        box = entry["boxes"][index]
        row = {key: int(value[index]) for key, value in counts.items()}
        row.update(frame=entry["frame"], target_id=target_id, size=entry["size"][index],
                   category=int(entry["classes"][index]), crop_width=float(crop.c[index]),
                   crop_offset=float(crop.c_l[index] - crop.c_r[index]),
                   left_width=float(box[2] - box[0]), right_width=float(box[5] - box[4]),
                   crop_to_left_width=float(crop.c[index] / (box[2] - box[0]).clamp_min(1e-3)))
        if entry["dense_ids"] is not None:
            row["dense_id"] = int(entry["dense_ids"][index])
            row["score"] = float(entry["scores"][index])
            row["boxes6"] = box.tolist()
            row["crop"] = {
                "c_l": float(crop.c_l[index]), "c_r": float(crop.c_r[index]),
                "c": float(crop.c[index]), "y0": float(crop.y0[index]),
                "h_crop": float(crop.h_crop[index]), "w_roi": int(crop.w_roi[index]),
                "h_roi": int(crop.h_roi[index]),
            }
        rows.append(row)


def summarize(rows):
    result = summarize_mask_counts({key: torch.tensor([row[key] for row in rows], dtype=torch.long)
                                    for key in ("tp", "fp", "fn", "valid_pixels")})
    return result


def grouped(rows, car_class):
    cars = [row for row in rows if row["category"] == car_class]
    return {
        "all": summarize(rows), "car": summarize(cars),
        "car_by_size": {name: summarize([row for row in cars if row["size"] == name])
                        for name in ("le16", "16_32", "32_64", "gt64")},
        "by_class": {str(category): summarize([row for row in rows if row["category"] == category])
                     for category in sorted({row["category"] for row in rows})},
    }


@torch.no_grad()
def evaluate_mask(model, entries, device, car_class, *, original=False, detail=False):
    model.instance_head.mask.eval()
    grid_rows, original_rows = [], []
    for entry in entries:
        logits = model.instance_head.mask(entry["features"].to(device)).cpu()
        add_rows(grid_rows, mask_pixel_counts(logits > 0, entry["target"], entry["valid"]), entry)
        if original:
            grid = entry["crop"]
            mask_crop = replace(grid, w_roi=torch.full_like(grid.w_roi, logits.shape[-1]),
                                h_roi=torch.full_like(grid.h_roi, logits.shape[-2]))
            mask, valid = project_disparity(SimpleNamespace(absolute=logits[:, 0].sigmoid(),
                                       valid=entry["valid"][:, 0]), mask_crop, entry["hw"])
            prediction = (mask > .50) & valid
            gt = entry["masks"][:, 0]
            add_rows(original_rows, mask_pixel_counts(prediction, gt, torch.ones_like(gt)), entry)
    result = {"crop_grid": grouped(grid_rows, car_class)}
    if original:
        result["original_image"] = grouped(original_rows, car_class)
    if detail:
        result["crop_rows"] = grid_rows
        result["original_rows"] = original_rows
    return result


def frozen_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            if not key.startswith("instance_head.mask.")}


def verify_frozen(model, expected):
    for key, value in frozen_state(model).items():
        if not torch.equal(value, expected[key]):
            raise RuntimeError(f"frozen state changed: {key}")
    for module in (model.encoder, model.detector, model.instance_head.idisp):
        if module.training or any(p.requires_grad or p.grad is not None for p in module.parameters()):
            raise RuntimeError("a frozen module is in training mode or received gradients")


def save(path, model, bundle, optimizer, epoch, step, metric):
    payload = {key: value for key, value in bundle.items() if key not in ("model", "optimizer")}
    payload.update(model=model.state_dict(), optimizer=optimizer.state_dict(),
                   mask_only={"supported_raw_classes": [26], "epoch": epoch, "steps": step, **metric})
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def update_mask(model, entry, optimizer, device):
    model.instance_head.mask.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model.instance_head.mask(entry["features"].to(device))
    target, valid = (entry[key].to(device) for key in ("target", "valid"))
    loss = mask_loss(logits, target.float(), valid)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.instance_head.mask.parameters(), 10., error_if_nonfinite=True)
    if not torch.isfinite(loss):
        raise RuntimeError("non-finite Mask loss")
    optimizer.step()
    return float(loss.detach()), float(norm)


def overfit_gate(metrics):
    return all(metrics[key] is not None and metrics[key] >= value
               for key, value in (("iou", .80), ("precision", .75), ("recall", .85)))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset-root", "train-json", "val-json", "openstereo-source", "candidate-archive", "replay", "output-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--weights", type=Path, default=ROOT / "models/stereo-yolo26-fixed56-seed17-e20.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stage", choices=("baseline", "overfit", "formal", "all"), default="all")
    return parser.parse_args()


def run_formal(args, model, bundle, split, frozen, heldout, car_class):
    out = args.output_dir
    print("Preparing pinned 16 dev frames and 144 train frames", flush=True)
    dev = build_features(model, dataset_for(split["mask_dev_16"], bundle), args.device, car_class, live_pred=True)
    train = build_features(model, dataset_for(split["mask_train_144"], bundle), args.device, car_class)["gt"]
    if not train or not dev["pred"] or not any((row["target"] & row["valid"]).any() for row in train):
        raise RuntimeError("missing valid Car training or Pred-RoI development samples")
    optimizer = torch.optim.AdamW(model.instance_head.mask.parameters(), lr=3e-4)
    best, last_significant, patience, steps = None, None, 0, 0
    rng = random.Random(17)
    for epoch in range(1, 31):
        order = list(range(len(train)))
        rng.shuffle(order)
        losses = []
        for index in order:
            loss, _ = update_mask(model, train[index], optimizer, args.device)
            losses.append(loss)
            steps += 1
        gt = evaluate_mask(model, dev["gt"], args.device, car_class)["crop_grid"]["car"]
        pred = evaluate_mask(model, dev["pred"], args.device, car_class)["crop_grid"]["car"]
        metric = {"dev_gt": gt, "dev_pred": pred, "train_loss": sum(losses) / len(losses)}
        row = {"epoch": epoch, "steps": steps, **metric}
        append_jsonl_record(out / "formal_metrics.jsonl", row)
        print(json.dumps(row), flush=True)
        save(out / "last.pt", model, bundle, optimizer, epoch, steps, metric)
        if gt["iou"] is not None and gt["iou"] >= .75 and pred["iou"] is not None:
            if best is None or pred["iou"] > best:
                best = pred["iou"]
                save(out / "best_mask.pt", model, bundle, optimizer, epoch, steps, metric)
        score = pred["iou"]
        if score is not None and (last_significant is None or score >= last_significant + .005):
            last_significant, patience = score, 0
        else:
            patience += 1
        if epoch >= 20 and patience >= 5:
            break
    verify_frozen(model, frozen)
    if best is None:
        write_json(out / "formal_gate.json", {"passed": False, "reason": "no checkpoint achieved dev GT IoU .75"})
        return
    chosen = torch.load(out / "best_mask.pt", map_location=args.device, weights_only=True)
    model.load_state_dict(chosen["model"], strict=True)
    risk = {k: evaluate_mask(model, dev[k], args.device, car_class)["crop_grid"]["car"]
            for k in ("paired_gt", "pred", "pred_gt_right")}
    write_json(out / "dev_crop_diagnostic.json", risk)
    gt_risk, pred_risk = (risk[k]["iou"] for k in ("paired_gt", "pred"))
    right_risk = risk["pred_gt_right"]["iou"]
    if (gt_risk is not None and pred_risk is not None and right_risk is not None
            and gt_risk >= .75 and gt_risk - pred_risk > .10):
        write_json(out / "formal_gate.json", {"passed": False, "reason": "dev crop gap requires mixed-training diagnosis",
                   "diagnostic": risk})
        return
    gt = evaluate_mask(model, heldout["gt"], args.device, car_class, original=True, detail=True)
    pred = evaluate_mask(model, heldout["pred"], args.device, car_class, original=True, detail=True)
    gt_primary, pred_primary = gt["crop_grid"]["car"], pred["crop_grid"]["car"]
    accepted = (gt_primary["iou"] is not None and gt_primary["iou"] >= .75
                and all(pred_primary[key] is not None and pred_primary[key] >= goal
                        for key, goal in (("iou", .65), ("precision", .75), ("recall", .85))))
    write_json(out / "formal_gate.json", {"passed": accepted, "checkpoint": chosen["mask_only"],
                                         "gt_roi": gt, "pred_roi": pred})
    if accepted:
        deploy = {key: value for key, value in chosen.items() if key != "optimizer"}
        deploy["foreground_mask_raw_classes"] = [26]
        temporary = out / "foreground_ready.pt.tmp"
        torch.save(deploy, temporary)
        os.replace(temporary, out / "foreground_ready.pt")
    print(f"formal gate: {accepted}", flush=True)


def main():
    args = parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(17)
    torch.backends.cudnn.benchmark = False
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    print(f"Loading fixed_56 E20 on {args.device}; two CPU threads", flush=True)
    model, bundle = load_published_model(args.weights, args.openstereo_source, args.device,
                                        score=.40, topk=500, apply_paired_nms=True, paired_nms_iou=.70)
    if bundle["epoch"] != 20 or model.instance_head.roi_sampling != "fixed_56":
        raise RuntimeError("expected the published E20 fixed grid checkpoint")
    car_class = bundle["class_mapping"][26]
    freeze_for_mask_only(model)
    frozen = frozen_state(model)
    records = build_frame_records(args.dataset_root, args.train_json, require_files=True)
    val = build_frame_records(args.dataset_root, args.val_json, require_files=True)
    split = prepare_split(records, val, bundle, args.dataset_root, out)
    print("Pinned splits 144/16/8 and official 40", flush=True)
    archive = build_archived_pairs(args.candidate_archive, args.replay, bundle, args.weights, args.val_json)
    heldout = build_features(model, dataset_for(val, bundle), args.device, car_class, archived=archive, original=True)
    baseline = {name: evaluate_mask(model, heldout[name], args.device, car_class, original=True, detail=True)
                for name in ("gt", "pred")}
    reproduced = baseline["pred"]["crop_grid"]["all"]
    if reproduced["iou_instances"] != 151 or abs(reproduced["iou"] - .3065) > .003:
        raise RuntimeError(f"historical crop-grid Mask IoU was not reproduced: {reproduced}")
    write_json(out / "baseline.json", baseline)
    print("E20 baseline: " + json.dumps({name: {grid: s[grid]["car"] for grid in ("crop_grid", "original_image")}
                                         for name, s in baseline.items()}), flush=True)
    if args.stage == "baseline":
        return
    if args.stage == "formal":
        report = json.loads((out / "overfit_gate.json").read_text())
        if not report["passed"]:
            raise RuntimeError("eight-frame prerequisite failed")
    else:
        entries = build_features(model, dataset_for(split["overfit_8"], bundle), args.device, car_class)["gt"]
        if len(entries) != 8 or not all((row["target"] & row["valid"]).any() for row in entries):
            raise RuntimeError("eight-frame manifest must supply eight nonempty Mask training batches")
        initial_mask = {key: tensor.clone() for key, tensor in model.instance_head.mask.state_dict().items()}
        optimizer = torch.optim.AdamW(model.instance_head.mask.parameters(), lr=3e-4)
        for step in range(401):
            if step:
                loss, grad_norm = update_mask(model, entries[(step-1) % len(entries)], optimizer, args.device)
                if step == 1:
                    if grad_norm <= 0:
                        raise RuntimeError("first Mask step received zero gradient")
                    verify_frozen(model, frozen)
                    write_json(out / "preflight.json", {"passed": True, "loss": loss, "grad_norm": grad_norm})
            if step % 50 != 0:
                continue
            mask = evaluate_mask(model, entries, args.device, car_class)["crop_grid"]["car"]
            progress = {"step": step, **mask}
            append_jsonl_record(out / "overfit_metrics.jsonl", progress)
            print("overfit: " + json.dumps(progress), flush=True)
        verify_frozen(model, frozen)
        passed = overfit_gate(mask)
        report = {"passed": passed, "final": progress, "frames": [row["frame"] for row in entries]}
        write_json(out / "overfit_gate.json", report)
        save(out / "overfit_last.pt", model, bundle, optimizer, 0, 400, report)
        print(f"eight-frame gate: {passed}", flush=True)
        if not passed or args.stage == "overfit":
            return
        model.instance_head.mask.load_state_dict(initial_mask, strict=True)
    run_formal(args, model, bundle, split, frozen, heldout, car_class)


if __name__ == "__main__":
    main()
