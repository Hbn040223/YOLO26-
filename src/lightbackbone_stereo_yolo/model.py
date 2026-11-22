"""Instance-level assembly for the baseline local mask and iDisp branches."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import Tensor, nn
from .local_disparity import RegularizedIDispHead, RoiDisparity
from .mask import ContextInstanceMaskHead, LightInstanceMaskHead, foreground_disparity_validity
from .roi import ROI_SAMPLING_MODES, CropMetadata, LeftRoIAlign, StereoRoIAlign, align_mask_spatial_valid, make_pair_crop_windows, sample_roi_feature
from .local_aggregation import AttentionPyramid
from .geometry import generate_point_locations, point_levels
from .detection import StereoYoloHead, decode_anchorfree_predictions, decode_dense_predictions, generate_grid_anchors


@dataclass(frozen=True)
class InstanceDisparityOutput:
    mask_logits: Tensor
    mask_validity: Tensor
    disparity: Tensor
    validity: Tensor
    peak: Tensor
    entropy: Tensor
    crop: CropMetadata


class StereoLocalInstanceHead(nn.Module):
    def __init__(
        self, channels: int, dmax_roi: int, hidden_channels: int = 32, *,
        roi_sampling: str = "native_pad", mask_head_variant: str = "light"
    ) -> None:
        super().__init__()
        if roi_sampling not in ROI_SAMPLING_MODES:
            raise ValueError(f"unknown RoI sampling mode: {roi_sampling}")
        self.roi_sampling = roi_sampling
        self.roi = StereoRoIAlign(stride=4)
        self.p3_roi = LeftRoIAlign(stride=8)
        self.p4_roi = LeftRoIAlign(stride=16)
        if mask_head_variant == "light":
            self.mask = LightInstanceMaskHead(channels)
        elif mask_head_variant == "context" and roi_sampling == "fixed_56":
            self.mask = ContextInstanceMaskHead(channels)
        else:
            raise ValueError(f"unsupported Mask head {mask_head_variant!r} for {roi_sampling!r}")
        self.idisp = RegularizedIDispHead(channels, dmax_roi, hidden_channels)

    def _disparity_forward(
        self, pyramid: object, boxes6: Tensor, image_height: int, image_width: int, margin: float
    ) -> tuple[RoiDisparity, CropMetadata, Tensor, Tensor]:
        crop = make_pair_crop_windows(boxes6, image_height, image_width, margin, roi_sampling=self.roi_sampling)
        left_roi, right_roi, spatial_valid, _ = self.roi(pyramid.p2_left, pyramid.p2_right, boxes6, crop)
        p3_roi, _ = self.p3_roi(pyramid.p3_left, crop)
        p4_roi, _ = self.p4_roi(pyramid.p4_left, crop)
        attention = AttentionPyramid(left_roi, p3_roi, p4_roi)
        disparity = self.idisp(left_roi, right_roi, attention, crop, spatial_valid)
        return disparity, crop, left_roi, spatial_valid

    def sample_mask_roi(self, p2_left: Tensor, boxes6: Tensor, image_height: int, image_width: int,
                        margin: float = 16.) -> tuple[Tensor, Tensor, CropMetadata]:
        """Sample only left P2 with the deployed stereo crop geometry."""
        crop = make_pair_crop_windows(boxes6, image_height, image_width, margin, roi_sampling=self.roi_sampling)
        features = p2_left.expand(boxes6.shape[0], -1, -1, -1)
        left_roi, spatial_valid = sample_roi_feature(features, crop.c_l, crop, self.roi.stride)
        return left_roi, align_mask_spatial_valid(spatial_valid), crop

    def forward_disparity(
        self, pyramid: object, boxes6: Tensor, image_height: int, image_width: int, margin: float = 16.
    ) -> tuple[RoiDisparity, CropMetadata]:
        disparity, crop, _, _ = self._disparity_forward(pyramid, boxes6, image_height, image_width, margin)
        return disparity, crop

    def forward(self, pyramid: object, boxes6: Tensor, image_height: int, image_width: int, margin: float = 16.) -> InstanceDisparityOutput:
        disparity, crop, left_roi, spatial_valid = self._disparity_forward(
            pyramid, boxes6, image_height, image_width, margin
        )
        return InstanceDisparityOutput(
            mask_logits=self.mask(left_roi),
            mask_validity=align_mask_spatial_valid(spatial_valid),
            disparity=disparity.absolute,
            validity=disparity.valid,
            peak=disparity.peak,
            entropy=disparity.entropy,
            crop=crop,
        )


@dataclass(frozen=True)
class StereoYoloLocalOutput:
    pair_ids: Tensor
    foreground_validity: Tensor
    boxes_left: Tensor
    boxes_right: Tensor
    scores: Tensor
    classes: Tensor
    mask_logits: Tensor
    mask_validity: Tensor
    disparity: Tensor
    validity: Tensor
    peak: Tensor
    entropy: Tensor


class StereoYoloLocalModel(nn.Module):
    """Batch-one inference assembly; empty detections bypass all local matching."""

    def __init__(
        self, encoder: nn.Module, detector: StereoYoloHead, instance_head: StereoLocalInstanceHead,
        score_threshold: float = .40, pre_nms_topk: int = 500, detector_variant: str = "baseline", *,
        right_offset_mode: str = "legacy_clamp", apply_paired_nms: bool | None = None,
        paired_nms_iou: float = .5, foreground_class_indices: tuple[int, ...] = (),
        mask_threshold: float = .50,
    ) -> None:
        super().__init__()
        self.encoder, self.detector, self.instance_head, self.score_threshold, self.pre_nms_topk, self.detector_variant = encoder, detector, instance_head, score_threshold, pre_nms_topk, detector_variant
        self.right_offset_mode = right_offset_mode
        if apply_paired_nms is not None and detector_variant != "stereo_yolo26_pan":
            raise ValueError("overriding paired NMS requires the stereo_yolo26_pan detector")
        if not 0 <= paired_nms_iou <= 1:
            raise ValueError("paired NMS IoU must be between 0 and 1")
        self.apply_paired_nms = apply_paired_nms
        self.paired_nms_iou = paired_nms_iou
        if foreground_class_indices and instance_head.roi_sampling != "fixed_56":
            raise ValueError("foreground disparity export requires the fixed_56 grid")
        self.foreground_class_indices = foreground_class_indices
        if not 0 < mask_threshold < 1:
            raise ValueError("mask threshold must be strictly between zero and one")
        self.mask_threshold = mask_threshold

    def forward(self, left: Tensor, right: Tensor) -> StereoYoloLocalOutput:
        if left.shape[0] != 1:
            raise ValueError("baseline inference currently accepts one stereo pair per call")
        pyramid = self.encoder(left, right)
        dense = self.detector(pyramid)
        if self.detector_variant == "stereo_yolo26_pan":
            locations = generate_point_locations(point_levels(pyramid), left.device)
            pair = decode_anchorfree_predictions(
                {"objectness": dense.objectness[0], "class_logits": dense.class_logits[0], "box6": dense.box6[0]},
                locations, self.score_threshold, nms_threshold=self.paired_nms_iou, pre_nms_topk=self.pre_nms_topk,
                right_offset_mode=self.right_offset_mode,
                apply_nms=(not getattr(self.detector, "one_to_one_enabled", False)
                           if self.apply_paired_nms is None else self.apply_paired_nms),
                class_aware_nms=True,
            )
        elif self.detector_variant == "baseline":
            levels = (
                (pyramid.p3_left.shape[-2], pyramid.p3_left.shape[-1], 8, 32.),
                (pyramid.p4_left.shape[-2], pyramid.p4_left.shape[-1], 16, 64.),
                (pyramid.c5_left.shape[-2], pyramid.c5_left.shape[-1], 32, 128.),
            )
            anchors = generate_grid_anchors(levels, left.device)
            pair = decode_dense_predictions(
                {"objectness": dense.objectness[0], "class_logits": dense.class_logits[0], "box6": dense.box6[0]},
                anchors, self.score_threshold, nms_threshold=self.paired_nms_iou, pre_nms_topk=self.pre_nms_topk,
            )
        else:
            raise ValueError("unsupported detector_variant")
        if pair.boxes6.numel() == 0:
            empty = left.new_empty((0,))
            empty_ids = left.new_empty((0,), dtype=torch.long)
            empty_boxes = left.new_empty((0, 4))
            empty_disparity = left.new_empty((0, 0, 0))
            empty_valid = left.new_empty((0, 0, 0), dtype=torch.bool)
            return StereoYoloLocalOutput(
                pair_ids=empty_ids, foreground_validity=empty_valid, boxes_left=empty_boxes, boxes_right=empty_boxes,
                scores=empty, classes=empty_ids, mask_logits=left.new_empty((0, 1, 28, 28)),
                mask_validity=left.new_empty((0, 1, 28, 28), dtype=torch.bool), disparity=empty_disparity,
                validity=empty_valid, peak=empty_disparity, entropy=empty_disparity,
            )
        # RoI branches expect one feature item per candidate; repeat only feature maps, not the encoder.
        count = pair.boxes6.shape[0]
        roi_pyramid = type("RoiPyramid", (), {name: getattr(pyramid, name).expand(count, -1, -1, -1) for name in ("p2_left", "p2_right", "p3_left", "p4_left")})()
        instance = self.instance_head(roi_pyramid, pair.boxes6, left.shape[-2], left.shape[-1])
        boxes_left = pair.boxes6[:, :4]
        boxes_right = torch.stack((pair.boxes6[:, 4], pair.boxes6[:, 1], pair.boxes6[:, 5], pair.boxes6[:, 3]), 1)
        foreground = torch.zeros_like(instance.validity)
        if self.foreground_class_indices:
            classes = pair.classes.new_tensor(self.foreground_class_indices)
            supported = (pair.classes[:, None] == classes).any(1)
            foreground = foreground_disparity_validity(
                instance.mask_logits, instance.mask_validity, instance.disparity, instance.validity, supported,
                mask_threshold=self.mask_threshold,
            )
        return StereoYoloLocalOutput(
            pair_ids=pair.indices, foreground_validity=foreground, boxes_left=boxes_left, boxes_right=boxes_right,
            scores=pair.scores, classes=pair.classes, mask_logits=instance.mask_logits,
            mask_validity=instance.mask_validity, disparity=instance.disparity, validity=instance.validity,
            peak=instance.peak, entropy=instance.entropy,
        )
