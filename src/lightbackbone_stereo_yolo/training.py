"""GT-RoI supervised training forward for the shared-feature stereo model.

The training path never decodes boxes or runs paired NMS.  It uses the same
pyramid twice: dense anchors for the detector, and every GT pair for the mask
and local-disparity branches.  This keeps the first training stage honest while
the detector is still untrained.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch
from torch import Tensor, nn

from .detection import assign_left_iou_targets, detection_loss, generate_grid_anchors
from .geometry import encode_boxes6
from .local_disparity import disparity_loss, overflow_mask
from .mask import mask_loss
from .roi import (
    align_disparity_roi,
    align_instance_mask_roi,
    align_left_mask_target,
)


BACKBONE_TAIL_MODULES = ("block4", "fpn_layer4", "fpn_layer3", "fpn_layer2", "out_conv")


def anchor_levels(pyramid: object) -> tuple[tuple[int, int, int, float], ...]:
    """The P3/P4/C5 grids used by inference, kept in one place for training."""
    return (
        (pyramid.p3_left.shape[-2], pyramid.p3_left.shape[-1], 8, 32.),
        (pyramid.p4_left.shape[-2], pyramid.p4_left.shape[-1], 16, 64.),
        (pyramid.c5_left.shape[-2], pyramid.c5_left.shape[-1], 32, 128.),
    )


def repeat_roi_features(pyramid: object, count: int) -> SimpleNamespace:
    """Expose one shared feature map per GT instance without copying memory."""
    if count < 1:
        raise ValueError("count must be positive")
    for name in ("p2_left", "p2_right", "p3_left", "p4_left"):
        if getattr(pyramid, name).shape[0] != 1:
            raise ValueError("GT-RoI training requires a single image per step")
    return SimpleNamespace(
        **{name: getattr(pyramid, name).expand(count, -1, -1, -1) for name in ("p2_left", "p2_right", "p3_left", "p4_left")}
    )


def build_dense_detection_targets(
    anchors: Tensor,
    boxes6: Tensor,
    classes: Tensor,
    positive_iou: float,
    negative_iou: float,
) -> dict[str, Tensor]:
    """Turn every GT pair of one frame into dense anchor targets."""
    assignment = assign_left_iou_targets(anchors, boxes6, positive_iou, negative_iou)
    positive = assignment.labels == 1
    box6_target = anchors.new_zeros((anchors.shape[0], 6))
    class_target = torch.zeros(anchors.shape[0], dtype=torch.long, device=anchors.device)
    if positive.any():
        matched = assignment.matched_indices[positive]
        box6_target[positive] = encode_boxes6(boxes6[matched], anchors[positive])
        class_target[positive] = classes[matched]
    return {"labels": assignment.labels, "classes": class_target, "box6": box6_target}


@dataclass(frozen=True)
class InstanceSupervision:
    """RoI-aligned targets and their joint validity masks."""

    mask_target: Tensor
    disparity_target: Tensor
    disparity_valid: Tensor
    mask_validity: Tensor


def roi_residual_target(disparity_target: Tensor, crop: object) -> Tensor:
    """`Di_roi = (Df - (cL - cR)) * Wroi / C`."""
    offset = (crop.c_l - crop.c_r)[:, None, None, None]
    scale = (crop.w_roi.to(disparity_target.dtype) / crop.c)[:, None, None, None]
    return (disparity_target - offset) * scale


def supervision_valid(
    disparity: Tensor, gt_valid: Tensor, spatial_valid: Tensor | None, foreground: Tensor,
    residual: Tensor | None = None, dmax_roi: int | None = None,
) -> Tensor:
    """Use one foreground/geometry predicate in calibration, training and evaluation."""
    valid = gt_valid.bool() & foreground.bool() & torch.isfinite(disparity) & (disparity > 0)
    if spatial_valid is not None:
        valid = valid & spatial_valid.bool()
    if dmax_roi is None:
        return valid
    if residual is None:
        raise ValueError("residual bins are required when a supervision search limit is set")
    return valid & ~overflow_mask(residual, dmax_roi)


def build_roi_disparity_targets(
    disparity: Tensor, disparity_valid: Tensor, masks: Tensor, crop: object, supervision_mode: str = "foreground",
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return image-domain target, GT/spatial validity, foreground validity and residual bins."""
    if supervision_mode not in ("foreground", "all_roi"):
        raise ValueError("supervision_mode must be 'foreground' or 'all_roi'")
    target, roi_valid = align_disparity_roi(disparity, disparity_valid, crop)
    foreground = align_instance_mask_roi(masks, crop) if supervision_mode == "foreground" else torch.ones_like(roi_valid)
    base_valid = supervision_valid(target, roi_valid, None, foreground)
    return target, roi_valid, base_valid, roi_residual_target(target, crop)


def build_instance_supervision(
    instance_output: object,
    boxes6: Tensor,
    masks: Tensor,
    disparity: Tensor,
    disparity_valid: Tensor,
    dmax_roi: int,
    supervision_mode: str = "foreground",
) -> InstanceSupervision:
    """Assemble mask and disparity targets on exactly the P2 RoI grid."""
    crop = instance_output.crop
    mask_target = align_left_mask_target(masks, boxes6, crop).float()
    disparity_target, disparity_roi_valid, base_valid, residual = build_roi_disparity_targets(
        disparity, disparity_valid, masks, crop, supervision_mode,
    )
    if torch.any(disparity_roi_valid & ~instance_output.validity[:, None]):
        raise RuntimeError("disparity validity must be contained by instance output validity")
    valid = base_valid & ~overflow_mask(residual, dmax_roi)
    return InstanceSupervision(mask_target, disparity_target, valid, instance_output.mask_validity)


def instance_metrics(
    instance_output: object,
    supervision: InstanceSupervision,
    masks: Tensor,
    dmax_roi: int,
) -> dict[str, float]:
    """GT-mask Mask IoU and GT-RoI EPE/D1 for one frame."""
    crop = instance_output.crop
    target_mask = supervision.mask_target >= .5
    mask_valid = supervision.mask_validity.expand_as(target_mask)
    predicted_mask = instance_output.mask_logits > 0
    intersection = ((predicted_mask & target_mask) & mask_valid).flatten(1).sum(1)
    union = ((predicted_mask | target_mask) & mask_valid).flatten(1).sum(1)
    iou = intersection.to(torch.float64) / union.clamp_min(1).to(torch.float64)
    measurable = union > 0

    error = (instance_output.disparity - supervision.disparity_target[:, 0]).abs()
    valid = supervision.disparity_valid[:, 0]
    d1 = (error > 3.) & (error > .05 * supervision.disparity_target[:, 0].abs()) & valid
    return {
        "iou_sum": float(iou[measurable].sum().item()),
        "iou_count": int(measurable.sum().item()),
        "epe_sum": float((error * valid).sum().item()),
        "epe_count": int(valid.sum().item()),
        "d1_hit": int(d1.sum().item()),
        "d1_count": int(valid.sum().item()),
    }


def training_forward(
    encoder: nn.Module,
    detector: nn.Module,
    instance_head: nn.Module,
    left: Tensor,
    right: Tensor,
    target: dict[str, Tensor],
    dmax_roi: int,
    roi_margin: float = 16.,
    positive_iou: float = .5,
    negative_iou: float = .4,
    detection_weight: float = 1.,
    mask_weight: float = 1.,
    disparity_weight: float = 1.,
    supervision_mode: str = "foreground",
    collect_metrics: bool = False,
) -> dict[str, object]:
    """One detector + GT-RoI mask + GT-RoI local-disparity training pass."""
    if left.shape[0] != 1:
        raise ValueError("GT-RoI training currently accepts one stereo pair per step")
    pyramid = encoder(left, right)
    dense = detector(pyramid)
    anchors = generate_grid_anchors(anchor_levels(pyramid), left.device)
    boxes6 = target["boxes6"].to(left.device)
    classes = target["classes"].to(left.device)
    detection_targets = build_dense_detection_targets(anchors, boxes6, classes, positive_iou, negative_iou)
    detection = detection_loss(
        {"objectness": dense.objectness[0], "class_logits": dense.class_logits[0], "box6": dense.box6[0]},
        detection_targets,
    )

    zero = dense.objectness.sum() * 0
    metrics = {"iou_sum": 0., "iou_count": 0, "epe_sum": 0., "epe_count": 0, "d1_hit": 0, "d1_count": 0}
    if boxes6.numel():
        roi_pyramid = repeat_roi_features(pyramid, boxes6.shape[0])
        instance_output = instance_head(roi_pyramid, boxes6, left.shape[-2], left.shape[-1], margin=roi_margin)
        supervision = build_instance_supervision(
            instance_output, boxes6, target["masks"].to(left.device),
            target["disparity"].to(left.device).unsqueeze(1),
            target["disparity_valid"].to(left.device).unsqueeze(1), dmax_roi, supervision_mode,
        )
        loss_mask = mask_loss(instance_output.mask_logits, supervision.mask_target, supervision.mask_validity)
        loss_disp = disparity_loss(
            instance_output.disparity, supervision.disparity_target[:, 0], supervision.disparity_valid[:, 0]
        )
        if collect_metrics:
            metrics = instance_metrics(instance_output, supervision, target["masks"].to(left.device), dmax_roi)
    else:
        loss_mask, loss_disp = zero, zero

    loss_detector = detection["loss_obj"] + detection["loss_cls"] + detection["loss_box6"]
    total = detection_weight * loss_detector + mask_weight * loss_mask + disparity_weight * loss_disp
    return {
        "loss_obj": detection["loss_obj"],
        "loss_cls": detection["loss_cls"],
        "loss_box6": detection["loss_box6"],
        "loss_mask": loss_mask,
        "loss_disp": loss_disp,
        "loss_total": total,
        "metrics": metrics,
    }


class StereoYoloTrainingModel(nn.Module):
    """Encoder, detector and GT-RoI instance head trained from one shared pyramid."""

    def __init__(self, encoder: nn.Module, detector: nn.Module, instance_head: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.detector = detector
        self.instance_head = instance_head


def load_idisp_frozen_state(model: StereoYoloTrainingModel, state: dict[str, Tensor]) -> list[str]:
    """Load the three unchanged modules without loading geometry-dependent iDisp weights."""
    for prefix, module in (
        ("encoder.", model.encoder), ("detector.", model.detector),
        ("instance_head.mask.", model.instance_head.mask),
    ):
        module.load_state_dict({name[len(prefix):]: value for name, value in state.items()
                                if name.startswith(prefix)}, strict=True)
    return sorted(name for name in state if name.startswith("instance_head.idisp."))


def freeze_for_mask_only(model: StereoYoloTrainingModel) -> None:
    for module in (model.encoder, model.detector, model.instance_head.idisp):
        module.requires_grad_(False).eval()
    model.instance_head.mask.requires_grad_(True)


def prepare_mask_rois(instance_head: nn.Module, p2_left: Tensor, boxes6: Tensor,
                      masks: Tensor, image_hw: tuple[int, int], margin: float = 16.) -> dict:
    """Prepare frozen features and supervision on the same shared crop."""
    left_roi, valid, crop = instance_head.sample_mask_roi(p2_left, boxes6, *image_hw, margin=margin)
    target = align_left_mask_target(masks, boxes6, crop)
    return {"features": left_roi, "target": target, "valid": valid, "crop": crop}


def training_forward_mask_only(model: StereoYoloTrainingModel, left: Tensor, right: Tensor, target: dict,
                               car_class: int, margin: float = 16.) -> dict | None:
    """Run the encoder and left Mask only for annotated Car instances."""
    if left.shape[0] != 1 or left.shape != right.shape:
        raise ValueError("Mask-only accepts one stereo frame")
    cars = (target["classes"] == car_class).nonzero().flatten()
    if not cars.numel():
        return None
    with torch.no_grad():
        pyramid = model.encoder(left, right)
        rois = prepare_mask_rois(model.instance_head, pyramid.p2_left,
                                target["boxes6"][cars].to(left.device), target["masks"][cars].to(left.device),
                                left.shape[-2:], margin)
    logits = model.instance_head.mask(rois["features"])
    loss = mask_loss(logits, rois["target"].float(), rois["valid"])
    return {**rois, "logits": logits, "loss": loss}


def freeze_for_idisp_only(model: StereoYoloTrainingModel) -> None:
    for module in (model.encoder, model.detector, model.instance_head.mask):
        module.requires_grad_(False).eval()


def training_forward_idisp_only(
    model: StereoYoloTrainingModel, left: Tensor, right: Tensor, target: dict,
    dmax_roi: int, roi_margin: float,
) -> dict:
    """Forward the encoder and GT-pair disparity head only."""
    if left.shape[0] != 1:
        raise ValueError("iDisp-only training accepts one stereo frame per step")
    boxes = target["boxes6"].to(left.device)
    if not boxes.numel():
        return {"loss_total": model.instance_head.idisp.temperature.raw.sum() * 0,
                "train_pixels": 0}
    with torch.no_grad():
        pyramid = model.encoder(left, right)
    disparity, crop = model.instance_head.forward_disparity(
        repeat_roi_features(pyramid, len(boxes)), boxes, *left.shape[-2:], margin=roi_margin,
    )
    absolute_gt, roi_valid, base_valid, residual = build_roi_disparity_targets(
        target["disparity"].to(left.device).unsqueeze(1),
        target["disparity_valid"].to(left.device).unsqueeze(1),
        target["masks"].to(left.device), crop,
    )
    if torch.any(roi_valid & ~disparity.valid[:, None]):
        raise RuntimeError("disparity validity must be contained by instance output validity")
    valid = base_valid & ~overflow_mask(residual, dmax_roi)
    return {
        "loss_total": disparity_loss(disparity.absolute, absolute_gt[:, 0], valid[:, 0]),
        "train_pixels": int(valid.sum().item()),
        "disparity": disparity,
        "crop": crop,
    }


def set_backbone_trainable(encoder: nn.Module, trainable: bool) -> None:
    for parameter in encoder.backbone.parameters():
        parameter.requires_grad = trainable


def unfreeze_backbone_tail(encoder: nn.Module) -> list[str]:
    """Re-enable the last strides plus the FPN taps used by P2/P3/P4."""
    unfrozen: list[str] = []
    for name in BACKBONE_TAIL_MODULES:
        module = getattr(encoder.backbone, name, None)
        if module is None:
            continue
        for parameter in module.parameters():
            parameter.requires_grad = True
        unfrozen.append(name)
    return unfrozen


def near10_selection_rank(car):
    qualified = (car["gt"] == 23 and car["tp"] >= 18 and car["exactly_one_pair_targets"] >= 17
                 and car["new_missed_gt"] == 0)
    quality = (car["foreground_coverage"], None if car["epe"] is None else -car["epe"],
               None if car["d1"] is None else -car["d1"])
    return ((qualified, -car["new_missed_gt"], car["tp"], car["exactly_one_pair_targets"],
             *(value if value is not None else -1e30 for value in quality)), qualified)


def build_optimizer(model: nn.Module, lr: float, weight_decay: float, backbone_lr_scale: float, *,
                    train_fpn_at_head_lr=False) -> torch.optim.Optimizer:
    """Separate the pretrained backbone from newly initialized heads."""
    backbone_ids = {id(parameter) for name, parameter in model.encoder.backbone.named_parameters()
                    if not train_fpn_at_head_lr or not name.startswith(("fpn_layer", "out_conv."))}
    backbone = [parameter for parameter in model.parameters() if id(parameter) in backbone_ids]
    heads = [parameter for parameter in model.parameters() if id(parameter) not in backbone_ids]
    return torch.optim.AdamW(
        [
            {"params": backbone, "lr": lr * backbone_lr_scale, "weight_decay": weight_decay},
            {"params": heads, "lr": lr, "weight_decay": weight_decay},
        ],
        lr=lr,
    )
