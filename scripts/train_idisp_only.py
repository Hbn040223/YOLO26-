"""Frozen B2a training loop for the two iDisp sampling modes."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader

from lightbackbone_stereo_yolo.data.frame_dataset import KittiFrameDataset, build_class_mapping, build_frame_records
from lightbackbone_stereo_yolo.idisp_ab import error_totals, original_foreground, project_disparity, summarize_error
from lightbackbone_stereo_yolo.lightstereo_encoder import LightStereoFeatureEncoder
from lightbackbone_stereo_yolo.model import StereoLocalInstanceHead
from lightbackbone_stereo_yolo.stereo_yolo26 import StereoYOLO26PanHead
from lightbackbone_stereo_yolo.training import (
    StereoYoloTrainingModel, freeze_for_idisp_only, load_idisp_frozen_state, training_forward_idisp_only,
)
from lightbackbone_stereo_yolo.training_control import append_jsonl_record


def frozen_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            if not key.startswith("instance_head.idisp.")}


def check_frozen_state(model, original):
    for key, value in frozen_state(model).items():
        if not torch.equal(value, original[key]):
            raise RuntimeError(f"frozen state changed: {key}")


def tensor_digest(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(tensor.contiguous().numpy().tobytes())
    return digest.hexdigest()


def write_json(path, record):
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save(path, args, model, optimizer, mapping, epoch, steps, best_total, best_epe):
    payload = {
        "config": vars(args), "class_mapping": mapping, "model": model.state_dict(),
        "optimizer": optimizer.state_dict(), "epoch": epoch, "steps": steps,
        "dmax_roi": args.dmax_roi, "best_val_loss": best_total, "best_gt_roi_epe": best_epe,
        "best_mask_iou": None, "best_mask_iou_status": "N/A: frozen B2a Mask",
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def frame_input(sample, device):
    return tuple(sample[side].unsqueeze(0).to(device) for side in ("left", "right"))


def forward(model, sample, args):
    return training_forward_idisp_only(
        model, *frame_input(sample, args.device), sample, args.dmax_roi, args.roi_margin,
    )


def training_mode(model):
    model.train()
    freeze_for_idisp_only(model)


def update(model, sample, args, optimizer):
    optimizer.zero_grad(set_to_none=True)
    values = forward(model, sample, args)
    loss = values["loss_total"]
    if not torch.isfinite(loss):
        raise RuntimeError(f"{sample['frame_id']}: non-finite iDisp loss")
    if not values["train_pixels"]:
        return None
    loss.backward()
    parameters = tuple(model.instance_head.idisp.parameters())
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip, error_if_nonfinite=True)
    optimizer.step()
    return {"loss": float(loss.detach()), "pixels": values["train_pixels"],
            "grad_norm": float(grad_norm)}


@torch.no_grad()
def validate(model, dataset, args):
    model.eval()
    losses, train_pixels = [], 0
    counts = {"pixels": 0, "epe_sum": 0., "d1_hits": 0}
    for index, sample in enumerate(dataset):
        if args.limit_val_batches and index >= args.limit_val_batches:
            break
        values = forward(model, sample, args)
        if "disparity" not in values:
            continue
        disparity = values["disparity"]
        if not torch.isfinite(disparity.absolute[disparity.valid]).all():
            raise RuntimeError(f"{sample['frame_id']}: non-finite output on valid pixels")
        if values["train_pixels"]:
            losses.append(float(values["loss_total"]))
            train_pixels += values["train_pixels"]
        truth, base = original_foreground(sample, args.device)
        prediction, valid = project_disparity(disparity, values["crop"], truth.shape[-2:])
        frame = error_totals(prediction, truth, base & valid)
        for key in counts:
            counts[key] += frame[key]
    return {
        **summarize_error(counts), "loss_total": sum(losses) / len(losses) if losses else None,
        "train_pixels": train_pixels, "valid_loss_frames": len(losses),
    }


def prepare_args(args, initial, resume):
    initial_config = initial["config"]
    stored = resume["config"] if resume is not None else None
    if initial_config.get("detector_variant") != "stereo_yolo26_pan" or not initial_config.get("one_to_one"):
        raise SystemExit("iDisp-only requires a Stereo-YOLO26 PAN one-to-one B2a source")
    if stored:
        if not stored.get("idisp_only"):
            raise SystemExit("iDisp-only resume requires an iDisp-only checkpoint")
        for key in ("roi_sampling", "dmax_roi"):
            value = getattr(args, key)
            if value not in (None, -1, stored[key]):
                raise SystemExit(f"{key} conflicts with the iDisp-only checkpoint")
            setattr(args, key, stored[key])
        for key in ("seed", "roi_margin", "lr", "weight_decay", "grad_clip", "idisp_calibration",
                    "idisp_fixed_frames", "idisp_max_steps", "idisp_eval_steps"):
            setattr(args, key, stored.get(key, getattr(args, key)))
        args.detector_variant = stored["detector_variant"]
        args.one_to_one = stored["one_to_one"]
    if args.detector_variant != "stereo_yolo26_pan" or not args.one_to_one:
        raise SystemExit("use --detector-variant stereo_yolo26_pan --one-to-one with new --idisp-only runs")
    if args.dmax_roi <= 0 or args.roi_sampling is None:
        raise SystemExit("iDisp-only requires a calibrated --dmax-roi and --roi-sampling")
    for key in ("dataset_root", "train_json", "val_json", "lightstereo_source", "checkpoint"):
        if Path(getattr(args, key)).resolve() != Path(initial_config[key]).resolve():
            raise SystemExit(f"{key} differs from the B2a experiment")
    if args.adapter_channels != int(initial_config["adapter_channels"]):
        raise SystemExit("adapter channels differ from B2a")
    if not args.idisp_calibration:
        raise SystemExit("supply --idisp-calibration with the selected prediction audit")
    calibration_path = Path(args.idisp_calibration).resolve()
    report = json.loads(calibration_path.read_text())
    if (report["mode"] != args.roi_sampling or report["selected_dmax_roi"] != args.dmax_roi
            or report["margin_image_px"] != args.roi_margin):
        raise SystemExit("selected calibration differs from sampling mode, Dmax or margin")
    args.idisp_calibration = str(calibration_path)
    args.idisp_calibration_sha256 = hashlib.sha256(calibration_path.read_bytes()).hexdigest()
    if stored and args.idisp_calibration_sha256 != stored["idisp_calibration_sha256"]:
        raise SystemExit("selected calibration changed since checkpoint")
    args.idisp_only = True
    args.right_offset_mode = initial_config.get("right_offset_mode", "legacy_clamp")
    args.det_weight, args.mask_weight, args.disp_weight = 0., 0., 1.
    args.best_mask_iou = "N/A"


def preflight(model, dataset, index_by_id, args, optimizer, original, out):
    if args.idisp_preflight_frame not in index_by_id:
        raise SystemExit("preflight requires a real --idisp-preflight-frame from the training split")
    pressure_id = args.idisp_preflight_frame
    ordinary = next(frame_id for frame_id in index_by_id if frame_id != pressure_id)
    results = []
    for frame_id in (ordinary, pressure_id):
        sample = dataset[index_by_id[frame_id]]
        before = {name: parameter.detach().clone() for name, parameter in model.instance_head.idisp.named_parameters()}
        if args.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(args.device)
        training_mode(model)
        result = update(model, sample, args, optimizer)
        if result is None or not result["grad_norm"] > 0:
            raise RuntimeError(f"{frame_id}: missing finite nonzero supervised gradient")
        if not any(not torch.equal(before[name], parameter.detach())
                   for name, parameter in model.instance_head.idisp.named_parameters()):
            raise RuntimeError(f"{frame_id}: iDisp parameters did not change")
        if not all(torch.isfinite(parameter).all() for parameter in model.instance_head.idisp.parameters()):
            raise RuntimeError(f"{frame_id}: non-finite iDisp parameters after step")
        with torch.no_grad():
            model.eval()
            result_after_step = forward(model, sample, args)["disparity"]
            if not torch.isfinite(result_after_step.absolute[result_after_step.valid]).all():
                raise RuntimeError(f"{frame_id}: non-finite iDisp output after step")
        check_frozen_state(model, original)
        results.append({"frame_id": frame_id, "instances": len(sample["boxes6"]), **result,
                        "peak_cuda_mib": torch.cuda.max_memory_allocated(args.device) / 2**20
                        if args.device.startswith("cuda") else None})
    write_json(out / "preflight.json", {"rows": results, "frozen_state_sha256": tensor_digest(original)})
    for result in results:
        print(json.dumps(result), flush=True)


def epoch_orders(out, frame_ids, args, resume):
    path = out / "epoch_order.json"
    if resume is None:
        orders = []
        for epoch in range(max(args.epochs, 40)):
            order = list(frame_ids)
            if not args.idisp_fixed_frames:
                random.Random(args.seed + epoch).shuffle(order)
            orders.append(order)
        write_json(path, {"epochs": orders})
    else:
        path = Path(resume["config"]["epoch_order_file"])
        orders = json.loads(path.read_text())["epochs"]
    if len(orders) < args.epochs or any(len(order) != len(frame_ids) or set(order) != set(frame_ids) for order in orders):
        raise SystemExit("epoch-order manifest does not match the split and epoch budget")
    args.epoch_order_file = str(path.resolve())
    args.epoch_order_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if resume and args.epoch_order_sha256 != resume["config"]["epoch_order_sha256"]:
        raise SystemExit("frame order changed since checkpoint")
    return orders


def run(args, resume):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    source = Path(resume["config"]["idisp_b2a_source"] if resume else (args.init_weights or ""))
    if not source.is_file():
        raise SystemExit("start iDisp-only training with --init-weights pointing to B2a")
    initial = torch.load(source, map_location="cpu", weights_only=False)
    prepare_args(args, initial, resume)
    args.idisp_b2a_source = str(source.resolve())
    args.initial_weights_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    if resume and args.initial_weights_sha256 != resume["config"]["initial_weights_sha256"]:
        raise SystemExit("frozen B2a source changed since checkpoint")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_records = build_frame_records(args.dataset_root, args.train_json)
    val_records = build_frame_records(args.dataset_root, args.val_json)
    mapping = build_class_mapping(train_records)
    if mapping != initial["class_mapping"] or (resume and mapping != resume["class_mapping"]):
        raise SystemExit("training label mapping differs from checkpoint")
    if args.idisp_fixed_frames:
        fixed = json.loads(Path(args.idisp_fixed_frames).read_text())["frame_ids"]
        by_id = {frame.frame_id: frame for frame in train_records}
        if len(fixed) != 8 or len(set(fixed)) != 8 or not set(fixed) <= by_id.keys():
            raise SystemExit("fixed manifest must select 8 different training frames")
        train_records = [by_id[frame_id] for frame_id in fixed]
        val_records = train_records
    train_data = KittiFrameDataset(train_records, mapping)
    val_data = KittiFrameDataset(val_records, mapping, unknown_class="skip")
    index_by_id = {frame.frame_id: i for i, frame in enumerate(train_records)}
    out = Path(args.output_dir)
    orders = epoch_orders(out, index_by_id, args, resume)

    encoder = LightStereoFeatureEncoder.from_checkpoint(args.lightstereo_source, args.checkpoint,
                                                       adapter_channels=args.adapter_channels)
    detector = StereoYOLO26PanHead(args.adapter_channels, len(mapping), one_to_one=True)
    head = StereoLocalInstanceHead(args.adapter_channels, args.dmax_roi, roi_sampling=args.roi_sampling)
    # Keep same-shaped modules identically initialized across different search widths.
    with torch.random.fork_rng(devices=[]):
        for name, module in head.idisp.named_modules():
            reset = getattr(module, "reset_parameters", None)
            if reset is not None:
                token = hashlib.sha256(f"{args.seed}:{name}".encode()).digest()[:8]
                torch.manual_seed(int.from_bytes(token, "little") % 2**63)
                reset()
    model = StereoYoloTrainingModel(encoder, detector, head)
    skipped = load_idisp_frozen_state(model, initial["model"])
    original = frozen_state(model)
    model.to(args.device)
    freeze_for_idisp_only(model)
    optimizer = torch.optim.AdamW(model.instance_head.idisp.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if resume:
        model.load_state_dict(resume["model"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        check_frozen_state(model, original)
    del initial
    write_json(out / "config.json", {"args": vars(args), "class_mapping": mapping,
                                    "skipped_state_keys": skipped, "frozen_state_sha256": tensor_digest(original)})
    print(f"idisp-only {args.roi_sampling} Dmax={args.dmax_roi} seed={args.seed} train={len(train_data)} val={len(val_data)}", flush=True)
    if args.preflight:
        preflight(model, train_data, index_by_id, args, optimizer, original, out)
        return

    start_epoch, steps = (resume["epoch"], resume["steps"]) if resume else (0, 0)
    best_total = resume["best_val_loss"] if resume else float("inf")
    best_epe = resume["best_gt_roi_epe"] if resume else float("inf")
    metrics_path = out / "metrics.jsonl"
    if args.idisp_max_steps and not resume:
        append_jsonl_record(metrics_path, {"kind": "step_validation", "step": 0, "val": validate(model, val_data, args)})
    for epoch in range(start_epoch, args.epochs):
        training_mode(model)
        losses, pixels, empty = [], 0, 0
        loader = DataLoader(train_data, batch_size=None, sampler=[index_by_id[key] for key in orders[epoch]],
                            num_workers=args.workers, pin_memory=args.device.startswith("cuda"))
        for index, sample in enumerate(loader):
            if args.limit_train_batches and index >= args.limit_train_batches:
                break
            step = update(model, sample, args, optimizer)
            if step is None:
                empty += 1
                continue
            steps += 1
            losses.append(step["loss"])
            pixels += step["pixels"]
            if args.log_interval and steps % args.log_interval == 0:
                print(f"epoch={epoch+1} step={steps} frame={sample['frame_id']} loss={step['loss']:.4f}", flush=True)
            if args.idisp_max_steps and steps % args.idisp_eval_steps == 0:
                val = validate(model, val_data, args)
                append_jsonl_record(metrics_path, {"kind": "step_validation", "step": steps, "val": val})
                print(f"step={steps} val_epe={val['epe']} val_d1={val['d1']}", flush=True)
                training_mode(model)
            if args.idisp_max_steps and steps >= args.idisp_max_steps:
                break
        if not losses:
            raise RuntimeError("training epoch had no valid iDisp updates")
        if args.idisp_max_steps and steps < args.idisp_max_steps:
            continue
        check_frozen_state(model, original)
        val = validate(model, val_data, args)
        best_loss_here = val["loss_total"] is not None and val["loss_total"] < best_total
        best_epe_here = val["epe"] is not None and val["epe"] < best_epe
        if best_loss_here:
            best_total = val["loss_total"]
        if best_epe_here:
            best_epe = val["epe"]
        append_jsonl_record(metrics_path, {
            "kind": "epoch", "epoch": epoch + 1, "step": steps, "val": val,
            "train_loss": sum(losses) / len(losses), "train_pixels": pixels, "empty_train_frames": empty,
            "best_mask_iou": "N/A",
        })
        print(f"epoch={epoch+1} steps={steps} val_epe={val['epe']} val_d1={val['d1']}", flush=True)
        for filename, enabled in (
            ("last.pt", True), ("best_total.pt", best_loss_here), ("best_gt_roi_epe.pt", best_epe_here),
            ("epoch_020.pt", epoch + 1 == 20 and not args.idisp_max_steps),
        ):
            if enabled:
                save(out / filename, args, model, optimizer, mapping, epoch + 1, steps, best_total, best_epe)
        if args.idisp_max_steps:
            break
    if args.idisp_max_steps and steps != args.idisp_max_steps:
        raise RuntimeError(f"fixed-frame run stopped at {steps} of {args.idisp_max_steps} requested steps")
