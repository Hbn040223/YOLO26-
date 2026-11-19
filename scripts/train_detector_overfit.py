"""Fixed-manifest detector-only overfit gate for a frozen LightStereo backbone."""
from __future__ import annotations
import argparse,json,os,random,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


# This comparator keeps the fixed initial precision floor from moving during training.
def is_better_overfit_checkpoint(candidate: dict, best: dict | None, precision_floor: float) -> bool:
    if candidate["paired_precision"] < precision_floor:
        return False
    if best is None:
        return True
    if candidate["paired_coverage"] != best["paired_coverage"]:
        return candidate["paired_coverage"] > best["paired_coverage"]
    return candidate["pre_nms_left_recall"] > best["pre_nms_left_recall"]



def summarize_overfit_metrics(frame_rows: list[dict]) -> dict:
    """Aggregate fixed-manifest detection metrics with all post-NMS pairs as precision denominator."""
    totals = {key: sum(int(row[key]) for row in frame_rows) for key in (
        "eligible_gt", "raw_left_hits", "pre_nms_left_hits", "paired_hits", "post_nms_pairs"
    )}
    denominator = max(totals["eligible_gt"], 1)
    counts = torch.tensor([row["post_nms_pairs"] for row in frame_rows], dtype=torch.float64)
    return {
        **totals,
        "raw_dense_left_recall": totals["raw_left_hits"] / denominator,
        "pre_nms_left_recall": totals["pre_nms_left_hits"] / denominator,
        "paired_coverage": totals["paired_hits"] / denominator,
        "paired_precision": totals["paired_hits"] / max(totals["post_nms_pairs"], 1),
        "post_nms_pair_count_p50": float(torch.quantile(counts, .5)) if counts.numel() else 0.0,
        "post_nms_pair_count_p90": float(torch.quantile(counts, .9)) if counts.numel() else 0.0,
    }



def count_gt_left_hits(iou: torch.Tensor, threshold: float = .5) -> int:
    """Count GTs with any candidate whose left-box IoU meets the recall threshold."""
    if not iou.numel() or not iou.shape[0]:
        return 0
    return int((iou.max(dim=0).values >= threshold).sum())

def _clip_valid_pairs(boxes6: torch.Tensor, image_height: int, image_width: int) -> tuple[torch.Tensor, int]:
    """Clip decoded pairs before recall auditing and drop non-finite or degenerate pairs."""
    finite = torch.isfinite(boxes6).all(dim=1)
    boxes6 = boxes6[finite].clone()
    if not boxes6.numel():
        return boxes6, int((~finite).sum())
    boxes6[:, 0].clamp_(0, image_width)
    boxes6[:, 2].clamp_(0, image_width)
    boxes6[:, 1].clamp_(0, image_height)
    boxes6[:, 3].clamp_(0, image_height)
    boxes6[:, 4].clamp_(0, image_width)
    boxes6[:, 5].clamp_(0, image_width)
    geometry = (boxes6[:, 2] > boxes6[:, 0]) & (boxes6[:, 3] > boxes6[:, 1]) & (boxes6[:, 5] > boxes6[:, 4])
    return boxes6[geometry], int((~finite).sum() + (~geometry).sum())


@torch.no_grad()
def evaluate_fixed8(model, dataset, device: torch.device, *, right_offset_mode="legacy_clamp") -> dict:
    """Evaluate raw dense geometry, pre-NMS recall, and deployed paired coverage on all eight fixed frames."""
    from diagnose_pred_roi import _decode_anchorfree, _frame_diagnosis, _iou
    from lightbackbone_stereo_yolo.geometry import decode_anchorfree_boxes6, generate_point_locations, point_levels
    model.eval()
    rows, invalid_pairs, dense_pairs = [], 0, 0
    for sample in dataset:
        left = sample["left"].unsqueeze(0).to(device)
        right = sample["right"].unsqueeze(0).to(device)
        pyramid = model.encoder(left, right)
        dense = model.detector(pyramid)
        locations = generate_point_locations(point_levels(pyramid), device)
        targets = sample["boxes6"].to(device)
        raw_encoded = dense.box6[0].reshape(locations.shape[0], 6)
        dense_pairs += int(raw_encoded.shape[0])
        decoded = decode_anchorfree_boxes6(raw_encoded, locations, right_offset_mode=right_offset_mode)
        raw_boxes, raw_invalid = _clip_valid_pairs(decoded, left.shape[-2], left.shape[-1])
        invalid_pairs += raw_invalid
        raw_hits = count_gt_left_hits(_iou(raw_boxes[:, :4], targets[:, :4]), .5)
        _, pre_nms, _ = _decode_anchorfree(dense, locations, .01, 0, right_offset_mode=right_offset_mode)
        pre_boxes, _ = _clip_valid_pairs(pre_nms.boxes6, left.shape[-2], left.shape[-1])
        pre_hits = count_gt_left_hits(_iou(pre_boxes[:, :4], targets[:, :4]), .5)
        deployed, _, _ = _decode_anchorfree(dense, locations, .25, 500, right_offset_mode=right_offset_mode)
        deployed_boxes, _ = _clip_valid_pairs(deployed.boxes6, left.shape[-2], left.shape[-1])
        post = _frame_diagnosis(deployed_boxes, torch.ones(deployed_boxes.shape[0], device=device), targets, .5)
        rows.append({"frame_id": sample["frame_id"], "eligible_gt": int(targets.shape[0]),
                     "raw_left_hits": raw_hits, "pre_nms_left_hits": pre_hits,
                     "paired_hits": post["formal_paired"], "post_nms_pairs": int(deployed_boxes.shape[0])})
    result = summarize_overfit_metrics(rows)
    result.update({"invalid_dense_pair_count": invalid_pairs, "dense_pair_count": dense_pairs,
                   "invalid_dense_pair_ratio": invalid_pairs / max(dense_pairs, 1), "per_frame": rows})
    return result


def _snapshot(model, optimizer, step: int, args: argparse.Namespace, frames: list[str], metrics: dict, out: Path, name: str) -> None:
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                "config": vars(args), "frames": frames, "metrics": metrics}, out / name)


def main():
    from lightbackbone_stereo_yolo.detection import BOX_SUPERVISION_MODES, box_supervision_mode_from_config
    from lightbackbone_stereo_yolo.geometry import (
        ASSIGNMENT_MODES, RIGHT_OFFSET_MODES, assignment_mode_from_config, right_offset_mode_from_config,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--right-offset-mode", choices=RIGHT_OFFSET_MODES)
    parser.add_argument("--assignment-mode", choices=ASSIGNMENT_MODES)
    parser.add_argument("--box-supervision-mode", choices=BOX_SUPERVISION_MODES)
    parser.add_argument("--detector-init-checkpoint", help="initialize from detector weights and inherit its right-offset mode")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--train-json", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--lightstereo-source", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    from lightbackbone_stereo_yolo.anchorfree_training import build_anchorfree_detection_targets
    from lightbackbone_stereo_yolo.data.frame_dataset import KittiFrameDataset, build_class_mapping, build_frame_records
    from lightbackbone_stereo_yolo.detection import anchorfree_detection_loss
    from lightbackbone_stereo_yolo.geometry import generate_point_locations, point_levels
    from lightbackbone_stereo_yolo.lightstereo_encoder import LightStereoFeatureEncoder
    from lightbackbone_stereo_yolo.stereo_yolo26 import StereoYOLO26PanHead
    from lightbackbone_stereo_yolo.training import StereoYoloTrainingModel, set_backbone_trainable

    wanted = {item["frame_id"] for item in json.load(open(args.manifest))["frames"]}
    all_records = build_frame_records(args.dataset_root, args.train_json)
    records = [record for record in all_records if record.frame_id in wanted]
    if len(records) != 8:
        raise RuntimeError("fixed manifest must resolve exactly eight frames")
    mapping = build_class_mapping(all_records)
    dataset = KittiFrameDataset(records, mapping)
    device = torch.device(args.device)
    encoder = LightStereoFeatureEncoder.from_checkpoint(args.lightstereo_source, args.checkpoint, adapter_channels=96)
    model = StereoYoloTrainingModel(encoder, StereoYOLO26PanHead(96, len(mapping)), torch.nn.Identity()).to(device)
    init_payload = None
    if args.detector_init_checkpoint:
        init_payload = torch.load(args.detector_init_checkpoint, map_location="cpu", weights_only=False)
        if init_payload.get("class_mapping", mapping) != mapping:
            raise ValueError("detector init checkpoint class mapping differs from fixed-8 data")
        detector_state = {key: tensor for key, tensor in init_payload["model"].items()
                          if not key.startswith("instance_head.")}
        model.load_state_dict(detector_state)
    stored_config = init_payload.get("config", {}) if init_payload is not None else {}
    args.right_offset_mode = right_offset_mode_from_config(stored_config, args.right_offset_mode)
    args.assignment_mode = assignment_mode_from_config(stored_config, args.assignment_mode)
    args.box_supervision_mode = box_supervision_mode_from_config(stored_config, args.box_supervision_mode)
    set_backbone_trainable(encoder, False)
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=args.lr, weight_decay=1e-4)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    initial = evaluate_fixed8(model, dataset, device, right_offset_mode=args.right_offset_mode)
    precision_floor = .9 * initial["paired_precision"]
    config = {"frames": [record.frame_id for record in records], "steps": args.steps, "seed": args.seed,
              "cpu_threads": args.cpu_threads, "precision_floor": precision_floor,
              "right_offset_mode": args.right_offset_mode, "assignment_mode": args.assignment_mode,
              "box_supervision_mode": args.box_supervision_mode, "metrics_clip_boxes": True,
              "deployment": {"score": .25, "pre_nms_topk": 500}, "pre_nms": {"score": .01, "topk": "all"}}
    (out / "config.json").write_text(json.dumps(config, indent=2))
    _snapshot(model, optimizer, 0, args, config["frames"], initial, out, "initial.pt")
    _snapshot(model, optimizer, 0, args, config["frames"], initial, out, "best_paired_coverage.pt")
    history, best, loss_window = [{"step": 0, "phase": "initial", **initial}], initial, []
    print(json.dumps(history[-1]), flush=True)

    for step in range(1, args.steps + 1):
        sample = dataset[(step - 1) % len(dataset)]
        model.train()
        model.encoder.backbone.eval()
        left = sample["left"].unsqueeze(0).to(device)
        right = sample["right"].unsqueeze(0).to(device)
        pyramid = model.encoder(left, right)
        dense = model.detector(pyramid)
        locations = generate_point_locations(point_levels(pyramid), device)
        targets = build_anchorfree_detection_targets(
            locations, sample["boxes6"].to(device), sample["classes"].to(device),
            assignment_mode=args.assignment_mode, image_hw=left.shape[-2:],
        )
        losses = anchorfree_detection_loss(
            {"objectness": dense.objectness[0], "class_logits": dense.class_logits[0], "box6": dense.box6[0]},
            targets, locations, right_offset_mode=args.right_offset_mode,
            box_supervision_mode=args.box_supervision_mode,
        )
        loss = losses["loss_obj"] + losses["loss_cls"] + losses["loss_box6"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_window.append({key: float(value.detach()) for key, value in losses.items() if key.startswith("loss_")})
        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate_fixed8(model, dataset, device, right_offset_mode=args.right_offset_mode)
            means = {key: sum(item[key] for item in loss_window) / len(loss_window) for key in loss_window[0]}
            record = {"step": step, "phase": "train", **means, **metrics}
            history.append(record)
            _snapshot(model, optimizer, step, args, config["frames"], record, out, "last.pt")
            if is_better_overfit_checkpoint(record, best, precision_floor):
                best = record
                _snapshot(model, optimizer, step, args, config["frames"], record, out, "best_paired_coverage.pt")
            (out / "metrics.json").write_text(json.dumps(history, indent=2))
            print(json.dumps(record), flush=True)
            loss_window = []


if __name__ == "__main__":
    main()
