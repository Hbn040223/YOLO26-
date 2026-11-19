"""Joint losses for the feature-sharing stereo instance baseline."""
from __future__ import annotations
from torch import Tensor
from .local_disparity import disparity_loss
from .mask import mask_loss


def assemble_losses(detection: dict[str, Tensor], mask_logits: Tensor, mask_target: Tensor, mask_valid: Tensor, disparity: Tensor, disparity_target: Tensor, disparity_valid: Tensor, detection_weight: float = 1., mask_weight: float = 1., disparity_weight: float = 1.) -> dict[str, Tensor]:
    """Return separately normalized detector, mask, image-domain disparity and total loss."""
    loss_det = detection["loss_obj"] + detection["loss_cls"] + detection["loss_box6"]
    loss_mask = mask_loss(mask_logits, mask_target, mask_valid)
    loss_disp = disparity_loss(disparity, disparity_target, disparity_valid)
    return {"loss_det": loss_det, "loss_mask": loss_mask, "loss_disp": loss_disp, "loss_total": detection_weight * loss_det + mask_weight * loss_mask + disparity_weight * loss_disp}
