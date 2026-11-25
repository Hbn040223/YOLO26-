"""GT-RoI supervised training entry for the LightStereo + Stereo-YOLO local model.

Default recipe: one GPU, one stereo pair per step, frozen LightStereo backbone,
new 1x1 adapters/detector/mask/iDisp heads trained from the first epoch, and an
optional tail unfreeze at a chosen epoch.  Validation reports detector losses,
Mask IoU and GT-RoI EPE/D1 without running decode, NMS or predicted-box RoIs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lightbackbone_stereo_yolo.data.frame_dataset import (  # noqa: E402
    LIGHTSTEREO_KITTI_MEAN,
    LIGHTSTEREO_KITTI_STD,
    KittiFrameDataset,
    build_class_mapping,
    build_frame_records,
    collate_frames,
)
from lightbackbone_stereo_yolo.detection import (  # noqa: E402
    BOX_SUPERVISION_MODES, ONE_TO_ONE_SCORE_TARGETS, StereoYoloHead, box_supervision_mode_from_config,
)
from lightbackbone_stereo_yolo.geometry import (  # noqa: E402
    ASSIGNMENT_MODES, RIGHT_OFFSET_MODES, assignment_mode_from_config, right_offset_mode_from_config,
)
from lightbackbone_stereo_yolo.stereo_yolo26 import StereoYOLO26PanHead, initialize_one_to_one_weights  # noqa: E402
from lightbackbone_stereo_yolo.anchorfree_training import (  # noqa: E402
    ONE_TO_ONE_ASSIGNMENT_MODES, one_to_one_assignment_mode_from_config, training_forward_anchorfree,
)
from lightbackbone_stereo_yolo.lightstereo_encoder import (  # noqa: E402
    ENCODER_BACKBONES, LightStereoFeatureEncoder, build_lightstereo_backbone, encoder_backbone_from_config,
)
from lightbackbone_stereo_yolo.model import StereoLocalInstanceHead, StereoYoloLocalModel  # noqa: E402
from lightbackbone_stereo_yolo.roi_selection import Near10Policy  # noqa: E402
from lightbackbone_stereo_yolo.roi import ROI_SAMPLING_MODES  # noqa: E402
from lightbackbone_stereo_yolo.training import (  # noqa: E402
    StereoYoloTrainingModel,
    build_optimizer,
    near10_selection_rank,
    set_backbone_trainable,
    training_forward,
    unfreeze_backbone_tail,
)
from lightbackbone_stereo_yolo.training_metrics import evaluate_pred_roi_frame, evaluate_anchorfree_detector_frame  # noqa: E402
from lightbackbone_stereo_yolo.training_control import DetectorCheckpointSelector, DetectorPlateauController, append_jsonl_record, percentile  # noqa: E402


LOSS_KEYS = ("loss_obj", "loss_cls", "loss_box6", "loss_mask", "loss_disp", "loss_total")
METRIC_KEYS = ("iou_sum", "iou_count", "epe_sum", "epe_count", "d1_hit", "d1_count")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, help="KITTI2015 root containing training/image_2 etc.")
    parser.add_argument("--train-json", required=True)
    parser.add_argument("--val-json", required=True)
    parser.add_argument("--lightstereo-source", required=True, help="OpenStereo source root")
    parser.add_argument("--checkpoint", help="LightStereo-S-KITTI.ckpt for a fresh 1.0× model")
    parser.add_argument("--encoder-backbone", choices=ENCODER_BACKBONES)
    parser.add_argument("--encoder-init", type=str, help="local 0.5× timm state dict")
    parser.add_argument("--near10-baseline-report", type=str, help="frozen 1.0× FP32 R2 report")
    parser.add_argument("--calibration-zip", type=str)
    parser.add_argument("--half-width-pilot", action="store_true", help="run 3–5 exploratory epochs")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "local_train"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr-scale", type=float, default=.1)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=10.)
    parser.add_argument("--adapter-channels", type=int, default=96)
    parser.add_argument("--pred-roi-metrics", action="store_true", help="compute prediction-driven RoI metrics during validation")
    parser.add_argument("--detector-car-cohort", help="validation candidate cache defining the fixed near-car cohort")
    parser.add_argument("--detector-metrics", action="store_true", help="decode detector-only validation metrics")
    parser.add_argument("--min-epochs-before-stop", type=int, default=30)
    parser.add_argument("--plateau-patience", type=int, default=5)
    parser.add_argument("--plateau-min-delta", type=float, default=.005)
    parser.add_argument("--detector-variant", choices=("baseline", "stereo_yolo26_pan"), default="baseline")
    parser.add_argument("--dmax-roi", type=int, default=-1, help="required for a fresh run; read from checkpoint on resume")
    parser.add_argument("--baseline-paired-precision", type=float, default=.1065, help="fixed Epoch-2 paired precision proxy for deploy checkpoint selection")
    parser.add_argument("--roi-margin", type=float, default=16.)
    parser.add_argument("--roi-sampling", choices=ROI_SAMPLING_MODES,
                        help="legacy native padding or the fixed 56x56 matching grid")
    parser.add_argument("--positive-iou", type=float, default=.5)
    parser.add_argument("--negative-iou", type=float, default=.4)
    parser.add_argument("--det-weight", type=float, default=1.)
    parser.add_argument("--mask-weight", type=float, default=1.)
    parser.add_argument("--disp-weight", type=float, default=1.)
    parser.add_argument("--unfreeze-after-epoch", type=int, default=-1, help="0-based epoch index that trains the backbone tail")
    parser.add_argument("--right-offset-mode", choices=RIGHT_OFFSET_MODES)
    parser.add_argument("--assignment-mode", choices=ASSIGNMENT_MODES)
    parser.add_argument("--box-supervision-mode", choices=BOX_SUPERVISION_MODES)
    parser.add_argument("--car-left-only-supervision", choices=("legacy", "off", "on"),
                        help="C2 controlled supervision of left-only repaired cars")
    parser.add_argument("--left-iou-weight", type=float, help="weight of left IoU loss on paired regression points")
    parser.add_argument("--repair-car-supervision", action="store_true", default=None,
                        help="train on all visible car left boxes, with paired losses only for usable right boxes")
    parser.add_argument("--one-to-one", action="store_true", default=None,
                        help="train an additional one-to-one stereo head and deploy it without NMS")
    parser.add_argument("--one-to-one-rank-weight", type=float, default=None,
                        help="opt-in weight for per-object duplicate ranking (default 0)")
    parser.add_argument("--one-to-one-assignment-mode", choices=ONE_TO_ONE_ASSIGNMENT_MODES)
    parser.add_argument("--one-to-one-score-target", choices=ONE_TO_ONE_SCORE_TARGETS)
    parser.add_argument("--init-weights", help="load a complete model state with a fresh optimizer")
    parser.add_argument("--idisp-only", action="store_true", help="freeze B2a and supervise only paired-RoI disparity")
    parser.add_argument("--idisp-calibration", help="selected per-mode prediction overflow audit JSON")
    parser.add_argument("--idisp-fixed-frames", help="8-frame overfit manifest")
    parser.add_argument("--idisp-max-steps", type=int, default=0)
    parser.add_argument("--idisp-eval-steps", type=int, default=50)
    parser.add_argument("--idisp-preflight-frame")
    parser.add_argument("--epoch-order-file", help="JSON with frame IDs for every training epoch")
    parser.add_argument("--resume", default="")
    parser.add_argument("--preflight", action="store_true", help="one forward/backward/step with finite checks, then exit")
    parser.add_argument("--limit-train-batches", type=int, default=0)
    parser.add_argument("--score-threshold", type=float, default=.40)
    parser.add_argument("--validation-paired-nms-iou", type=float, default=.70)
    parser.add_argument("--limit-val-batches", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    args = parser.parse_args()
    if args.epochs < 1 or args.workers < 0:
        raise SystemExit("epochs must be positive and workers non-negative")
    if args.min_epochs_before_stop < 0 or args.plateau_patience < 1 or args.plateau_min_delta < 0:
        raise SystemExit("invalid plateau settings")
    if args.grad_clip <= 0 or args.lr <= 0:
        raise SystemExit("grad-clip and lr must be positive")
    if not 0 <= args.negative_iou < args.positive_iou <= 1:
        raise SystemExit("require 0 <= negative-iou < positive-iou <= 1")
    if not (0 <= args.score_threshold <= 1 and 0 <= args.validation_paired_nms_iou <= 1):
        parser.error("score threshold and validation paired-NMS IoU must be in [0, 1]")
    if args.roi_margin < 0:
        raise SystemExit("roi-margin must be non-negative")
    if min(args.det_weight, args.mask_weight, args.disp_weight) < 0:
        raise SystemExit("loss weights must be non-negative")
    if not 0 < args.baseline_paired_precision <= 1:
        raise SystemExit("baseline-paired-precision must be in (0, 1]")
    return args


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def aligned_class_mapping(train_records, val_records) -> dict[int, int]:
    mapping = build_class_mapping(train_records)
    unseen = sorted({a.category_id for frame in val_records for a in frame.annotations} - set(mapping))
    if unseen:
        raise SystemExit(f"validation split has categories absent from training: {unseen}")
    return mapping


def read_epoch_orders(path: Path, frame_ids: list[str], epochs: int) -> tuple[list[list[int]], str]:
    raw = path.read_bytes()
    orders = json.loads(raw)["epochs"]
    indices = {frame_id: i for i, frame_id in enumerate(frame_ids)}
    if len(indices) != len(frame_ids) or len(orders) < epochs:
        raise ValueError("epoch-order file and current split must cover every training epoch")
    for order in orders:
        if len(order) != len(frame_ids) or set(order) != indices.keys():
            raise ValueError("every epoch must be a permutation of the training split")
    return [[indices[frame_id] for frame_id in order] for order in orders], hashlib.sha256(raw).hexdigest()


def make_loader(dataset: KittiFrameDataset, workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collate_frames,
        pin_memory=True,
        persistent_workers=workers > 0,
    )


def assert_finite_losses(output: dict[str, object]) -> None:
    for key in LOSS_KEYS:
        value = output[key]
        if not torch.isfinite(value):
            raise RuntimeError(f"non-finite training loss: {key}={value.item()}")


def assert_finite_gradients(model: torch.nn.Module) -> dict[str, float]:
    norm_squared = 0.
    checked = 0
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        checked += 1
        if not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"non-finite gradient: {name}")
        norm_squared += float(parameter.grad.detach().double().pow(2).sum().item())
    if checked == 0:
        raise RuntimeError("no parameter received a gradient")
    return {"grad_norm": math.sqrt(norm_squared), "grad_tensors": checked}


def assert_finite_parameters(model: torch.nn.Module) -> None:
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all():
            raise RuntimeError(f"non-finite parameter: {name}")


def forward_batch(model: StereoYoloTrainingModel, batch: dict[str, object], args: argparse.Namespace, collect_metrics: bool) -> dict[str, object]:
    target = batch["targets"][0]  # type: ignore[index]
    if args.detector_variant == "stereo_yolo26_pan":
        return training_forward_anchorfree(
            model.encoder, model.detector, model.instance_head,
            batch["left"].to(args.device, non_blocking=True),
            batch["right"].to(args.device, non_blocking=True),
            target, dmax_roi=args.dmax_roi, roi_margin=args.roi_margin,
            detection_weight=args.det_weight, mask_weight=args.mask_weight,
            disparity_weight=args.disp_weight, collect_metrics=collect_metrics,
            right_offset_mode=getattr(args, "right_offset_mode", None) or "legacy_clamp",
            assignment_mode=getattr(args, "assignment_mode", None) or "legacy",
            box_supervision_mode=getattr(args, "box_supervision_mode", None) or "joint",
            left_iou_weight=getattr(args, "left_iou_weight", None) or 0.,
            car_left_only_supervision=getattr(args, "car_left_only_supervision", None) or "legacy",
            one_to_one_rank_weight=getattr(args, "one_to_one_rank_weight", 0.),
            one_to_one_score_target=getattr(args, "one_to_one_score_target", None) or "floor",
            one_to_one_assignment_mode=getattr(args, "one_to_one_assignment_mode", None) or "legacy",
        )
    return training_forward(
        model.encoder,
        model.detector,
        model.instance_head,
        batch["left"].to(args.device, non_blocking=True),  # type: ignore[union-attr]
        batch["right"].to(args.device, non_blocking=True),  # type: ignore[union-attr]
        target,
        dmax_roi=args.dmax_roi,
        roi_margin=args.roi_margin,
        positive_iou=args.positive_iou,
        negative_iou=args.negative_iou,
        detection_weight=args.det_weight,
        mask_weight=args.mask_weight,
        disparity_weight=args.disp_weight,
        collect_metrics=collect_metrics,
    )


def run_epoch(
    model: StereoYoloTrainingModel,
    loader: DataLoader,
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer | None,
    limit_batches: int,
    collect_metrics: bool,
    log,
    *,
    car_cohort: dict | None = None,
    scheduler=None,
) -> dict[str, object]:
    totals = {key: 0. for key in LOSS_KEYS}
    pred_metrics = {"matched_count": 0., "target_count": 0., "iou_sum": 0., "iou_count": 0., "epe_sum": 0., "epe_count": 0., "d1_hit": 0., "d1_count": 0.}
    sparse_totals = {"one_to_one_positives": 0., "loss_one_to_one": 0., "loss_one_to_one_rank": 0.}
    assignment_totals = {}
    car_supervision_totals = {}
    car_totals = {}
    left_iou_total = 0.
    car_near_rows = []
    duplicate_predictions = 0
    detector_metrics = {"target_count": 0., "raw_dense_left_hits": 0., "pre_nms_left_hits_score_001": 0., "deployment_left_hits": 0., "paired_hits": 0., "post_nms_pairs": 0.}
    metrics = {key: 0. for key in METRIC_KEYS}
    post_nms_pair_counts: list[float] = []
    detection_size_groups = {}
    frames = 0
    if optimizer is not None:
        model.train()
        # The pretrained backbone keeps its inference BN statistics while fine-tuning.
        model.encoder.backbone.eval()
        if getattr(args, "encoder_backbone", "v2") == "v2-050":
            for name in ("fpn_layer2", "fpn_layer3", "fpn_layer4", "out_conv"):
                getattr(model.encoder.backbone, name).train()
    else:
        model.eval()
    for step, batch in enumerate(loader):
        if limit_batches and step >= limit_batches:
            break
        with torch.set_grad_enabled(optimizer is not None):
            output = forward_batch(model, batch, args, collect_metrics=collect_metrics)
            assert_finite_losses(output)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                output["loss_total"].backward()  # type: ignore[union-attr]
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                if not torch.isfinite(gradient_norm):
                    raise RuntimeError("non-finite gradient norm")
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
        frames += 1
        left_iou_total += float(output.get("loss_left_iou", 0.))
        for key in sparse_totals:
            if key in output:
                sparse_totals[key] += float(output[key])
        for key, count in output.get("one_to_one_assignment_counts", {}).items():
            assignment_totals[key] = assignment_totals.get(key, 0) + count
        for key, count in output.get("car_supervision_counts", {}).items():
            car_supervision_totals[key] = car_supervision_totals.get(key, 0) + count
        for key in LOSS_KEYS:
            totals[key] += float(output[key].detach().item())  # type: ignore[union-attr]
        for key in METRIC_KEYS:
            metrics[key] += float(output["metrics"][key])  # type: ignore[index]
        if optimizer is None and collect_metrics and getattr(args, "detector_metrics", False):
            if args.detector_variant != "stereo_yolo26_pan":
                raise RuntimeError("detector-only metrics require stereo_yolo26_pan")
            target = batch["targets"][0]
            near_car = None
            if car_cohort is not None:
                car_class, entries = car_cohort
                row = entries[target["frame_id"]]
                if target["boxes6"].tolist() != row["gt_boxes6"] or target["classes"].tolist() != row["gt_classes"]:
                    raise RuntimeError(f"car cohort labels changed for {target['frame_id']}")
                near_car = (car_class, [i for i, group in enumerate(row["gt_groups"]) if group["near"]])
            detector_result = evaluate_anchorfree_detector_frame(
                model.encoder, model.detector, batch["left"].to(args.device),
                batch["right"].to(args.device), target,
                right_offset_mode=getattr(args, "right_offset_mode", None) or "legacy_clamp",
                score_threshold=getattr(args, "score_threshold", .40),
                paired_nms_iou=getattr(args, "validation_paired_nms_iou", .70),
                car_cohort=near_car,
            )
            for key, count in detector_result.get("car_deployment_score_040", {}).items():
                car_totals[key] = car_totals.get(key, 0) + count
            car_near_rows.extend(
                {"frame_id": target["frame_id"], **row}
                for row in detector_result.get("car_deployment_near_states", ())
            )
            post_nms_pair_counts.append(float(detector_result["post_nms_pairs"]))
            duplicate_predictions += detector_result.get("paired_duplicate_predictions", 0)
            for key in detector_metrics:
                detector_metrics[key] += float(detector_result[key])
            for name, counts in detector_result.get("detection_size_groups", {}).items():
                total = detection_size_groups.setdefault(name, {key: 0 for key in counts})
                for key, value in counts.items():
                    total[key] += value
        if optimizer is None and collect_metrics and getattr(args, "pred_roi_metrics", False):
            pred = evaluate_pred_roi_frame(
                model.encoder, model.detector, model.instance_head,
                batch["left"].to(args.device), batch["right"].to(args.device),
                batch["targets"][0], args.dmax_roi, args.detector_variant,
                score_threshold=args.score_threshold, roi_margin=args.roi_margin,
                paired_nms_iou=getattr(args, "validation_paired_nms_iou", .70),
                right_offset_mode=getattr(args, "right_offset_mode", None) or "legacy_clamp",
            )
            for key in pred_metrics:
                pred_metrics[key] += float(pred.get(key, 0.))
        if optimizer is not None and args.log_interval and step % args.log_interval == 0:
            log(
                f"  step {step:04d} total={float(output['loss_total'].detach().item()):.4f} "  # type: ignore[union-attr]
                f"obj={float(output['loss_obj'].detach().item()):.4f} "  # type: ignore[union-attr]
                f"cls={float(output['loss_cls'].detach().item()):.4f} "  # type: ignore[union-attr]
                f"box6={float(output['loss_box6'].detach().item()):.4f} "  # type: ignore[union-attr]
                f"mask={float(output['loss_mask'].detach().item()):.4f} "  # type: ignore[union-attr]
                f"disp={float(output['loss_disp'].detach().item()):.4f}"  # type: ignore[union-attr]
            )
    if frames == 0:
        raise RuntimeError("no batches were consumed")
    result = {key: totals[key] / frames for key in LOSS_KEYS}
    if args.detector_variant == "stereo_yolo26_pan":
        result["loss_left_iou"] = left_iou_total / frames
    result["frames"] = float(frames)
    result["mask_iou"] = metrics["iou_sum"] / metrics["iou_count"] if metrics["iou_count"] else 0.
    result["epe"] = metrics["epe_sum"] / metrics["epe_count"] if metrics["epe_count"] else 0.
    result["d1"] = metrics["d1_hit"] / metrics["d1_count"] if metrics["d1_count"] else 0.
    result["pred_coverage"] = pred_metrics["matched_count"] / pred_metrics["target_count"] if pred_metrics["target_count"] else 0.
    result["pred_mask_iou"] = pred_metrics["iou_sum"] / pred_metrics["iou_count"] if pred_metrics["iou_count"] else 0.
    result["pred_epe"] = pred_metrics["epe_sum"] / pred_metrics["epe_count"] if pred_metrics["epe_count"] else 0.
    result["pred_d1"] = pred_metrics["d1_hit"] / pred_metrics["d1_count"] if pred_metrics["d1_count"] else 0.
    result["raw_dense_left_recall"] = detector_metrics["raw_dense_left_hits"] / detector_metrics["target_count"] if detector_metrics["target_count"] else 0.
    result["pre_nms_left_recall"] = detector_metrics["pre_nms_left_hits_score_001"] / detector_metrics["target_count"] if detector_metrics["target_count"] else 0.
    result["deployment_left_recall"] = detector_metrics["deployment_left_hits"] / detector_metrics["target_count"] if detector_metrics["target_count"] else 0.
    result["paired_coverage"] = detector_metrics["paired_hits"] / detector_metrics["target_count"] if detector_metrics["target_count"] else 0.
    result["post_nms_pairs_p50"] = percentile(post_nms_pair_counts, .5) if post_nms_pair_counts else 0.
    result["post_nms_pairs_p90"] = percentile(post_nms_pair_counts, .9) if post_nms_pair_counts else 0.
    result["paired_precision"] = detector_metrics["paired_hits"] / detector_metrics["post_nms_pairs"] if detector_metrics["post_nms_pairs"] else 0.
    result["post_nms_pairs"] = detector_metrics["post_nms_pairs"]
    if optimizer is None and collect_metrics and (getattr(args, "detector_metrics", False)
                                                  or getattr(args, "pred_roi_metrics", False)):
        result["validation_score_threshold"] = getattr(args, "score_threshold", .40)
        result["validation_paired_nms_iou"] = getattr(args, "validation_paired_nms_iou", .70)
        result["validation_class_aware"] = True
    result["paired_duplicate_predictions"] = duplicate_predictions
    if getattr(args, "one_to_one", False):
        result.update({key: value / frames for key, value in sparse_totals.items()})
    if car_supervision_totals:
        result["car_supervision_counts"] = car_supervision_totals
    if assignment_totals:
        result["one_to_one_assignment_counts"] = assignment_totals
    if car_totals:
        outputs = car_totals["outputs"]
        car_totals["precision"] = car_totals["paired_hits"] / outputs if outputs else 0.
        result["car_deployment_score_040"] = car_totals
        result["car_deployment_near_states"] = car_near_rows
    if detection_size_groups:
        result["detection_size_groups"] = detection_size_groups
    return result


def save_checkpoint(path: Path, model, optimizer, epoch, best, best_gt_roi_epe, args, class_mapping,
                    detector_plateau=None, detector_checkpoint_selector=None, *, extra=None) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_loss": best,
        "dmax_roi": args.dmax_roi,
        "best_gt_roi_epe": best_gt_roi_epe,
        "class_mapping": class_mapping,
        "detector_checkpoint_selector": detector_checkpoint_selector,
        "normalization": {"mean": list(LIGHTSTEREO_KITTI_MEAN), "std": list(LIGHTSTEREO_KITTI_STD)},
        "config": vars(args),
        "detector_plateau": detector_plateau,
        **(extra or {}),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA device requested but no GPU is visible")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train.log"
    metrics_path = output_dir / "metrics.jsonl"

    def log(message: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    if args.resume and args.init_weights:
        raise SystemExit("--init-weights and --resume are mutually exclusive")
    resume_payload = None
    if args.resume:
        resume_payload = torch.load(args.resume, map_location="cpu", weights_only=False)
        if args.dmax_roi <= 0:
            args.dmax_roi = int(resume_payload["dmax_roi"])
    stored_config = resume_payload.get("config", {}) if resume_payload is not None else {}
    args.encoder_backbone = (encoder_backbone_from_config(stored_config, args.encoder_backbone)
                             if resume_payload is not None else args.encoder_backbone or "v2")
    half = args.encoder_backbone == "v2-050"
    if half and args.idisp_only:
        raise SystemExit("half-width adaptation uses the joint training entry")
    if half and not args.preflight and not (args.calibration_zip and args.near10_baseline_report):
        raise SystemExit("half-width validation requires --calibration-zip and --near10-baseline-report")
    if half and not args.preflight:
        if args.half_width_pilot and not 3 <= args.epochs <= 5:
            raise SystemExit("half-width pilot expects 3–5 epochs")
        if not args.half_width_pilot and (args.epochs != 60 or args.limit_train_batches or args.limit_val_batches):
            raise SystemExit("formal half-width training requires --epochs 60 and the full datasets")
        if resume_payload is not None and bool(stored_config.get("half_width_pilot")) != args.half_width_pilot:
            raise SystemExit("pilot and formal training require separate initializations")
    if args.idisp_only or stored_config.get("idisp_only"):
        from train_idisp_only import run

        if args.idisp_max_steps < 0 or args.idisp_eval_steps <= 0:
            raise SystemExit("invalid iDisp overfit step settings")
        run(args, resume_payload)
        return
    stored_roi_sampling = stored_config.get("roi_sampling", "native_pad")
    if resume_payload is not None and args.roi_sampling not in (None, stored_roi_sampling):
        raise SystemExit("RoI sampling differs from resume checkpoint; use --init-weights to change geometry")
    if args.roi_sampling is None:
        args.roi_sampling = stored_roi_sampling
    stored_sparse = bool(stored_config.get("one_to_one", False))
    if resume_payload is not None and args.one_to_one not in (None, stored_sparse):
        raise SystemExit("one-to-one mode differs from resume checkpoint; use --init-weights to add the branch")
    if args.one_to_one is None:
        args.one_to_one = stored_sparse
    if args.one_to_one and args.detector_variant != "stereo_yolo26_pan":
        raise SystemExit("--one-to-one requires --detector-variant stereo_yolo26_pan")
    stored_score_target = stored_config.get("one_to_one_score_target", "floor")
    if resume_payload is not None and args.one_to_one_score_target not in (None, stored_score_target):
        raise SystemExit("one-to-one score target differs from resume checkpoint")
    if args.one_to_one_score_target is None:
        args.one_to_one_score_target = stored_score_target
    if args.one_to_one_score_target not in ONE_TO_ONE_SCORE_TARGETS:
        raise SystemExit("invalid one-to-one score target in checkpoint")
    if args.one_to_one_score_target != "floor" and not args.one_to_one:
        raise SystemExit("--one-to-one-score-target requires --one-to-one")
    stored_rank_weight = float(stored_config.get("one_to_one_rank_weight", 0.))
    if resume_payload is not None and args.one_to_one_rank_weight not in (None, stored_rank_weight):
        raise SystemExit("one-to-one ranking weight differs from resume checkpoint")
    if args.one_to_one_rank_weight is None:
        args.one_to_one_rank_weight = stored_rank_weight
    if not 0 <= args.one_to_one_rank_weight < float("inf") or (not args.one_to_one and args.one_to_one_rank_weight):
        raise SystemExit("one-to-one ranking weight must be finite, nonnegative, and requires --one-to-one")
    stored_sparse_assignment = one_to_one_assignment_mode_from_config(stored_config)
    if resume_payload is not None and args.one_to_one_assignment_mode not in (None, stored_sparse_assignment):
        raise SystemExit("one-to-one assignment mode differs from resume checkpoint")
    args.one_to_one_assignment_mode = one_to_one_assignment_mode_from_config(
        stored_config, args.one_to_one_assignment_mode,
    )
    if not args.one_to_one and args.one_to_one_assignment_mode != "legacy":
        raise SystemExit("global assignment requires --one-to-one")
    stored_mode = right_offset_mode_from_config(stored_config)
    if resume_payload is not None and args.right_offset_mode not in (None, stored_mode):
        raise SystemExit("right-offset mode differs from resume checkpoint")
    args.right_offset_mode = right_offset_mode_from_config(stored_config, args.right_offset_mode)
    stored_assignment = assignment_mode_from_config(stored_config)
    if resume_payload is not None and args.assignment_mode not in (None, stored_assignment):
        raise SystemExit("assignment mode differs from resume checkpoint")
    args.assignment_mode = assignment_mode_from_config(stored_config, args.assignment_mode)
    stored_repair = bool(stored_config.get("repair_car_supervision", False))
    if resume_payload is not None and args.repair_car_supervision not in (None, stored_repair):
        raise SystemExit("car supervision repair mode differs from resume checkpoint")
    if args.repair_car_supervision is None:
        args.repair_car_supervision = stored_repair
    if args.repair_car_supervision:
        if args.detector_variant != "stereo_yolo26_pan":
            raise SystemExit("--repair-car-supervision requires stereo_yolo26_pan")
        if args.box_supervision_mode not in (None, "split_left"):
            raise SystemExit("--repair-car-supervision requires split_left box supervision")
        args.box_supervision_mode = "split_left"
    stored_left_only = stored_config.get("car_left_only_supervision", "legacy")
    if resume_payload is not None and args.car_left_only_supervision not in (None, stored_left_only):
        raise SystemExit("car left-only supervision differs from resume checkpoint")
    if args.car_left_only_supervision is None:
        args.car_left_only_supervision = stored_left_only
    if args.car_left_only_supervision != "legacy" and (
        not args.repair_car_supervision or not args.one_to_one or args.one_to_one_assignment_mode == "legacy"
    ):
        raise SystemExit("C2 supervision requires repaired cars and global one-to-one assignment")
    stored_left_iou_weight = float(stored_config.get("left_iou_weight", 0.))
    if resume_payload is not None and args.left_iou_weight not in (None, stored_left_iou_weight):
        raise SystemExit("left IoU weight differs from resume checkpoint")
    if args.left_iou_weight is None:
        args.left_iou_weight = stored_left_iou_weight
    if not 0 <= args.left_iou_weight < float("inf"):
        raise SystemExit("--left-iou-weight must be finite and nonnegative")
    if args.left_iou_weight and args.detector_variant != "stereo_yolo26_pan":
        raise SystemExit("--left-iou-weight requires stereo_yolo26_pan")
    stored_box_mode = box_supervision_mode_from_config(stored_config)
    if resume_payload is not None and args.box_supervision_mode not in (None, stored_box_mode):
        raise SystemExit("box supervision mode differs from resume checkpoint")
    args.box_supervision_mode = box_supervision_mode_from_config(stored_config, args.box_supervision_mode)
    if args.box_supervision_mode != "joint" and args.detector_variant != "stereo_yolo26_pan":
        raise SystemExit("split_left requires stereo_yolo26_pan")
    if args.assignment_mode != "legacy" and args.detector_variant != "stereo_yolo26_pan":
        raise SystemExit("inside_rescue requires stereo_yolo26_pan")
    if args.dmax_roi <= 0:
        raise SystemExit("--dmax-roi is required for a fresh run (or resume from a checkpoint that stores it)")
    if args.adapter_channels < 1:
        raise SystemExit("adapter-channels must be positive")

    train_frames = build_frame_records(args.dataset_root, args.train_json)
    val_frames = build_frame_records(args.dataset_root, args.val_json)
    class_mapping = aligned_class_mapping(train_frames, val_frames)
    if resume_payload is not None and resume_payload.get("class_mapping") != class_mapping:
        raise SystemExit("resume checkpoint class mapping does not match the current training split")
    train_dataset = KittiFrameDataset(
        train_frames, class_mapping, repair_car_supervision=args.repair_car_supervision,
    )
    train_loader = make_loader(train_dataset, args.workers, shuffle=True)
    orders = None
    frame_ids = [frame.frame_id for frame in train_frames]
    args.epoch_order_file = args.epoch_order_file or stored_config.get("epoch_order_file")
    args.epoch_order_sha256 = None
    if args.epoch_order_file:
        args.epoch_order_file = str(Path(args.epoch_order_file).resolve())
        orders, args.epoch_order_sha256 = read_epoch_orders(Path(args.epoch_order_file), frame_ids, args.epochs)
    if resume_payload is not None and args.epoch_order_sha256 != stored_config.get("epoch_order_sha256"):
        raise SystemExit("epoch-order manifest differs from resume checkpoint")
    car_cohort = None
    args.detector_car_cohort = args.detector_car_cohort or stored_config.get("detector_car_cohort")
    if args.detector_car_cohort:
        if not args.detector_metrics or not args.one_to_one or 26 not in class_mapping:
            raise SystemExit("--detector-car-cohort requires one-to-one detector metrics for a car class")
        cohort_path = Path(args.detector_car_cohort).resolve()
        args.detector_car_cohort = str(cohort_path)
        cache = json.loads(cohort_path.read_text())
        entries = {row["frame_id"]: row for row in cache["frames"]}
        if (Path(cache["validation_manifest"]).resolve() != Path(args.val_json).resolve()
                or len(entries) != len(cache["frames"])
                or set(entries) != {frame.frame_id for frame in val_frames}):
            raise SystemExit("fixed car cohort must use the identical validation manifest and frames")
        car_class = class_mapping[26]
        if any(len(row["gt_groups"]) != len(row["gt_classes"]) or
               any(group["near"] and category != car_class
                   for group, category in zip(row["gt_groups"], row["gt_classes"]))
               for row in entries.values()):
            raise SystemExit("fixed near-car cohort contains invalid target identities")
        car_cohort = (car_class, entries)
    val_loader = make_loader(
        KittiFrameDataset(val_frames, class_mapping, unknown_class="skip"), args.workers, shuffle=False
    )

    if not half and args.checkpoint:
        encoder = LightStereoFeatureEncoder.from_checkpoint(
            args.lightstereo_source, args.checkpoint, adapter_channels=args.adapter_channels)
    else:
        if not half and resume_payload is None:
            raise SystemExit("fresh 1.0× training requires --checkpoint")
        backbone = build_lightstereo_backbone(args.lightstereo_source, encoder_backbone=args.encoder_backbone)
        if half and args.encoder_init and resume_payload is None:
            backbone.load_trunk_state(torch.load(args.encoder_init, map_location="cpu", weights_only=True))
        elif half and resume_payload is None and not args.preflight:
            raise SystemExit("fresh half-width training requires --encoder-init")
        encoder = LightStereoFeatureEncoder(backbone, args.adapter_channels)
    initial = (torch.load(args.init_weights, map_location="cpu", weights_only=half) if args.init_weights else None)
    template = resume_payload if resume_payload is not None else initial
    args.mask_head_variant = (template or {}).get("config", {}).get("mask_head_variant", "light")
    foreground_metadata = {}
    if half:
        if template is None or args.mask_head_variant != "context" or 26 not in template.get("foreground_mask_raw_classes", ()):
            raise SystemExit("half-width training requires a Car context Mask checkpoint")
        foreground_metadata = {key: template[key] for key in ("foreground_mask_raw_classes", "mask_threshold", "mask_evaluation")
                               if key in template}
        args.encoder_init_sha256 = stored_config.get("encoder_init_sha256")
        if args.encoder_init and resume_payload is None:
            args.encoder_init_sha256 = hashlib.sha256(Path(args.encoder_init).read_bytes()).hexdigest()
    detector = StereoYoloHead(args.adapter_channels, num_classes=len(class_mapping))
    if args.detector_variant == "stereo_yolo26_pan":
        detector = StereoYOLO26PanHead(args.adapter_channels, num_classes=len(class_mapping), one_to_one=args.one_to_one)
    instance_head = StereoLocalInstanceHead(args.adapter_channels, args.dmax_roi, roi_sampling=args.roi_sampling,
                                           mask_head_variant=args.mask_head_variant)
    precision_floor = .9 * args.baseline_paired_precision
    detector_selector = DetectorCheckpointSelector(precision_floor)
    model = StereoYoloTrainingModel(encoder, detector, instance_head).to(args.device)
    args.initial_weights_sha256 = stored_config.get("initial_weights_sha256")
    if args.init_weights:
        args.init_weights = str(Path(args.init_weights).resolve())
        init_path = Path(args.init_weights)
        args.initial_weights_sha256 = hashlib.sha256(init_path.read_bytes()).hexdigest()
        if initial.get("class_mapping", class_mapping) != class_mapping:
            raise SystemExit("initial checkpoint class mapping differs from the current split")
        if half:
            config = initial["config"]
            for key in ("adapter_channels", "detector_variant", "one_to_one", "right_offset_mode", "roi_sampling"):
                if config.get(key) != getattr(args, key):
                    raise SystemExit(f"incompatible downstream initialization: {key}")
            if initial["dmax_roi"] != args.dmax_roi or config.get("one_to_one_only", False):
                raise SystemExit("incompatible iDisp or detector-head layout")
            prefixes = ("detector.", "instance_head.mask.", "instance_head.idisp.")
            source = {key: value for key, value in initial["model"].items() if key.startswith(prefixes)}
            expected = {key: value for key, value in model.state_dict().items() if key.startswith(prefixes)}
            if set(source) != set(expected) or any(source[key].shape != value.shape for key, value in expected.items()):
                raise SystemExit("downstream checkpoint tensors do not match")
            model.load_state_dict(source, strict=False)
            log(f"loaded {len(source)} downstream tensors; FPN and adapters initialized afresh")
            del source, expected
        else:
            state = initialize_one_to_one_weights(initial["model"]) if args.one_to_one else initial["model"]
            model.load_state_dict(state)
            del state
        del initial, template
    set_backbone_trainable(encoder, half)
    optimizer = build_optimizer(model, args.lr, args.weight_decay, args.backbone_lr_scale,
                                train_fpn_at_head_lr=half)
    scheduler = None
    if half:
        steps_per_epoch = len(train_loader)
        args.train_sha256 = hashlib.sha256(Path(args.train_json).read_bytes()).hexdigest()
        if resume_payload is not None and (stored_config.get("train_sha256") != args.train_sha256
                                          or resume_payload["steps_per_epoch"] != steps_per_epoch):
            raise SystemExit("half-width resume requires the original training split and step count")
        warmup = 3 * steps_per_epoch
        total = 60 * steps_per_epoch
        def lr_scale(step):
            if step <= warmup:
                return .1 + .9 * step / warmup
            progress = min(1., (step - warmup) / (total - warmup))
            return .1 + .45 * (1 + math.cos(math.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)

    start_epoch, best, best_gt_roi_epe = 0, float("inf"), float("inf")
    detector_plateau = DetectorPlateauController(args.min_epochs_before_stop, args.plateau_patience, args.plateau_min_delta)
    start_epoch, best, best_gt_roi_epe = 0, float("inf"), float("inf")
    precision_floor = .9 * args.baseline_paired_precision
    detector_plateau = DetectorPlateauController(args.min_epochs_before_stop, args.plateau_patience, args.plateau_min_delta)
    detector_selector = DetectorCheckpointSelector(precision_floor)
    if resume_payload is not None:
        model.load_state_dict(resume_payload["model"])
        optimizer.load_state_dict(resume_payload["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(resume_payload["scheduler"])
        start_epoch = int(resume_payload["epoch"])
        best = float(resume_payload.get("best_val_loss", best))
        best_gt_roi_epe = float(resume_payload.get("best_gt_roi_epe", best_gt_roi_epe))
        if args.detector_metrics:
            saved = resume_payload.get("config", {})
            same_protocol = (saved.get("score_threshold") == args.score_threshold
                             and saved.get("validation_paired_nms_iou") == args.validation_paired_nms_iou)
            if same_protocol:
                if resume_payload.get("detector_plateau"):
                    detector_plateau = DetectorPlateauController.from_state_dict(resume_payload["detector_plateau"])
                if resume_payload.get("detector_checkpoint_selector"):
                    detector_selector = DetectorCheckpointSelector.from_state_dict(resume_payload["detector_checkpoint_selector"])
            else:
                log("Detector validation protocol changed. Starting new plateau and checkpoint-selection metrics.")

    write_json(output_dir / "config.json", {"args": vars(args), "class_mapping": class_mapping,
        "num_classes": len(class_mapping), "dmax_roi": args.dmax_roi,
        "detector_selection": {"baseline_paired_precision": args.baseline_paired_precision, "precision_floor": precision_floor},
        "normalization": {"mean": list(LIGHTSTEREO_KITTI_MEAN), "std": list(LIGHTSTEREO_KITTI_STD)},
        "train_frames": len(train_frames), "val_frames": len(val_frames)})
    log(f"train frames={len(train_frames)} val frames={len(val_frames)} classes={len(class_mapping)} mapping={class_mapping} dmax_roi={args.dmax_roi} device={args.device}")
    log("detector-only: backbone frozen/eval; adapters, fusion, PAN and 6D head trainable" if args.mask_weight == 0 and args.disp_weight == 0 else "full multi-task training")

    if args.detector_variant == "stereo_yolo26_pan":
        log(f"left box IoU loss weight={args.left_iou_weight:g}")
    if args.one_to_one:
        log(f"detector deploy head: one-to-one (no NMS), assignment={args.one_to_one_assignment_mode}, score_target={args.one_to_one_score_target}")

    if args.preflight:
        batch = next(iter(train_loader)); model.train(); model.encoder.backbone.eval()
        if half:
            for name in ("fpn_layer2", "fpn_layer3", "fpn_layer4", "out_conv"):
                getattr(model.encoder.backbone, name).train()
        output = forward_batch(model, batch, args, collect_metrics=True); assert_finite_losses(output)
        optimizer.zero_grad(set_to_none=True); output["loss_total"].backward()
        gradient_info = assert_finite_gradients(model); torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip); optimizer.step(); assert_finite_parameters(model)
        report = {"frame_id": batch["targets"][0]["frame_id"], "boxes": int(batch["targets"][0]["boxes6"].shape[0]), "losses": {key: float(output[key].detach().item()) for key in LOSS_KEYS}, **gradient_info}
        if args.one_to_one:
            report["one_to_one_positives"] = int(output["one_to_one_positives"])
            report["loss_one_to_one"] = float(output["loss_one_to_one"])
        if "one_to_one_assignment_counts" in output:
            report["one_to_one_assignment_counts"] = output["one_to_one_assignment_counts"]
        log(f"preflight passed: {json.dumps(report, ensure_ascii=False)}"); write_json(output_dir / "preflight.json", report); return

    near10_state = (resume_payload or {}).get("near10_selection", {})
    if half:
        from zipfile import ZipFile
        from benchmark_edge_stream import decoded_frames
        from evaluate_edge_near10 import evaluate_r2_training, matched_ids
        from infer import resolve_mask_threshold

        norm = {"mean": LIGHTSTEREO_KITTI_MEAN, "std": LIGHTSTEREO_KITTI_STD}
        reference_path = Path(args.near10_baseline_report)
        args.reference_sha256 = hashlib.sha256(reference_path.read_bytes()).hexdigest()
        args.val_sha256 = hashlib.sha256(Path(args.val_json).read_bytes()).hexdigest()
        args.calibration_sha256 = hashlib.sha256(Path(args.calibration_zip).read_bytes()).hexdigest()
        if resume_payload is not None and any(
                getattr(args, key) != stored_config.get(key)
                for key in ("reference_sha256", "val_sha256", "calibration_sha256", "lr", "backbone_lr_scale")):
            raise SystemExit("half-width resume requires the original validation inputs and learning rates")
        reference = json.loads(reference_path.read_text())
        if (reference["protocol"]["split_sha256"] != args.val_sha256
                or reference["protocol"]["calibration_sha256"] != args.calibration_sha256):
            raise SystemExit("frozen baseline and validation inputs differ")
        baseline = matched_ids(reference["frames"], "R2")
        if len(baseline["Car"]) != 18:
            raise SystemExit("frozen R2 baseline must identify the 18 matched near cars")
        r2_dataset = KittiFrameDataset(val_frames, class_mapping, unknown_class="skip", load_images=False)
        with ZipFile(args.calibration_zip) as archive:
            r2_frames = decoded_frames(r2_dataset, archive)
        r2_model = StereoYoloLocalModel(
            encoder, detector, instance_head, detector_variant=args.detector_variant,
            right_offset_mode=args.right_offset_mode, apply_paired_nms=True,
            foreground_class_indices=(class_mapping[26],), mask_threshold=resolve_mask_threshold(foreground_metadata),
            near10_policy=Near10Policy("range-dedup", class_mapping[26], 15., .50, .45))
    unfrozen = False
    for epoch in range(start_epoch, args.epochs):
        frame_order_sha256 = None
        if orders is not None:
            order = orders[epoch]
            train_loader = make_loader(Subset(train_dataset, order), args.workers, shuffle=False)
            frame_order_sha256 = hashlib.sha256("\n".join(frame_ids[i] for i in order).encode()).hexdigest()
        if not unfrozen and 0 <= args.unfreeze_after_epoch <= epoch:
            names = unfreeze_backbone_tail(model.encoder); unfrozen = True; log(f"epoch {epoch}: unfroze backbone tail {names}")
        started = time.time()
        train_metrics = run_epoch(model, train_loader, args, optimizer, args.limit_train_batches, False, log,
                                  scheduler=scheduler)
        with torch.no_grad():
            val_metrics = run_epoch(
                model, val_loader, args, None, args.limit_val_batches, True, log, car_cohort=car_cohort,
            )
        should_stop = args.detector_metrics and detector_plateau.update(epoch, val_metrics["paired_coverage"], val_metrics["pre_nms_left_recall"])
        selected_detector = args.detector_metrics and detector_selector.update(val_metrics["paired_coverage"], val_metrics["paired_precision"], val_metrics["pre_nms_left_recall"])
        state = detector_plateau.state_dict() if args.detector_metrics else None
        selection_state = detector_selector.state_dict() if args.detector_metrics else None
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics, "seconds": round(time.time() - started, 1), "detector_plateau": state, "detector_selection": {"baseline_paired_precision": args.baseline_paired_precision, "precision_floor": precision_floor, "precision_qualified": bool(args.detector_metrics and val_metrics["paired_precision"] >= precision_floor), "selector": selection_state}}
        if frame_order_sha256 is not None:
            record["frame_order_sha256"] = frame_order_sha256
        extras = None
        if half:
            results, frame_report = evaluate_r2_training(
                r2_model.eval(), r2_dataset, r2_frames, norm, class_mapping, torch.device(args.device), baseline)
            car = results["Car"]
            row = {"epoch": epoch + 1, **{key: car[key] for key in (
                "tp", "gt", "exactly_one_pair_targets", "new_missed_gt", "precision", "predicted")},
                "foreground_coverage": car["foreground"]["pixel_coverage"], "epe": car["disparity"]["epe"],
                "d1": car["disparity"]["d1"], "optimizer_steps": scheduler.last_epoch}
            rank, qualified = near10_selection_rank(row)
            previous = near10_state.get("rank")
            improved_r2 = previous is None or rank > tuple(previous)
            if improved_r2:
                near10_state = {"rank": rank, "epoch": epoch + 1, "qualified": qualified}
            extras = {**foreground_metadata, "scheduler": scheduler.state_dict(),
                      "steps_per_epoch": steps_per_epoch, "optimizer_steps": scheduler.last_epoch,
                      "near10_selection": near10_state, "near10_epoch": row}
            write_json(output_dir / f"r2-epoch-{epoch + 1:03}.json", {"classes": results, "frames": frame_report})
            append_jsonl_record(output_dir / "near10-epochs.jsonl", row)
            log(f"epoch {epoch + 1} R2 {json.dumps(row)}")
            if improved_r2:
                name = "best_pilot.pt" if args.half_width_pilot else "best_r2.pt" if qualified else "best_diagnostic.pt"
                save_checkpoint(output_dir / name, model, optimizer, epoch + 1, best, best_gt_roi_epe,
                                args, class_mapping, extra=extras)
            should_stop = (not args.half_width_pilot and epoch + 1 >= 30 and near10_state["qualified"]
                           and epoch + 1 - near10_state["epoch"] >= 10)
        append_jsonl_record(metrics_path, record)
        log(f"epoch {epoch} train total={train_metrics['loss_total']:.4f} obj={train_metrics['loss_obj']:.4f} cls={train_metrics['loss_cls']:.4f} box6={train_metrics['loss_box6']:.4f} | val total={val_metrics['loss_total']:.4f} raw={val_metrics['raw_dense_left_recall']:.4f} pre001={val_metrics['pre_nms_left_recall']:.4f} deploy_left={val_metrics['deployment_left_recall']:.4f} paired={val_metrics['paired_coverage']:.4f} precision={val_metrics['paired_precision']:.4f} pairs={val_metrics['post_nms_pairs']:.0f} plateau={detector_plateau.plateau_counter}")
        if "loss_left_iou" in train_metrics:
            log(f"epoch {epoch} unweighted left IoU loss train={train_metrics['loss_left_iou']:.4f} val={val_metrics['loss_left_iou']:.4f}")
        if "car_supervision_counts" in train_metrics:
            log(f"epoch {epoch} car-supervision {json.dumps(train_metrics['car_supervision_counts'])}")
        if "car_deployment_score_040" in val_metrics:
            log(f"epoch {epoch} car@.40 {json.dumps(val_metrics['car_deployment_score_040'])}")
        improved = val_metrics["loss_total"] < best; improved_epe = val_metrics["epe"] < best_gt_roi_epe
        if improved: best = val_metrics["loss_total"]
        if improved_epe: best_gt_roi_epe = val_metrics["epe"]
        if half:
            save_checkpoint(output_dir / "last.pt", model, optimizer, epoch + 1, best, best_gt_roi_epe,
                            args, class_mapping, extra=extras)
            if should_stop:
                log(f"epoch {epoch + 1}: R2 plateau stop")
                break
            continue
        save_checkpoint(output_dir / "last.pt", model, optimizer, epoch + 1, best, best_gt_roi_epe, args, class_mapping, state, selection_state)
        if improved: save_checkpoint(output_dir / "best.pt", model, optimizer, epoch + 1, best, best_gt_roi_epe, args, class_mapping, state, selection_state)
        if improved_epe: save_checkpoint(output_dir / "best_gt_roi_epe.pt", model, optimizer, epoch + 1, best, best_gt_roi_epe, args, class_mapping, state, selection_state)
        if selected_detector: save_checkpoint(output_dir / "best_paired_coverage.pt", model, optimizer, epoch + 1, best, best_gt_roi_epe, args, class_mapping, state, selection_state)
        if should_stop:
            log(f"epoch {epoch}: detector plateau stop after {detector_plateau.plateau_counter} epochs")
            break
if __name__ == "__main__":
    main()
