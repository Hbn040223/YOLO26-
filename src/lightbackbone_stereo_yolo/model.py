"""Instance-level assembly for the baseline local mask and iDisp branches."""
from __future__ import annotations
from dataclasses import dataclass, replace
import torch
from torch import Tensor, nn
from torch.profiler import record_function
from .local_disparity import RegularizedIDispHead, RoiDisparity
from .mask import ContextInstanceMaskHead, LightInstanceMaskHead, foreground_disparity_validity
from .roi import ROI_SAMPLING_MODES, CropMetadata, LeftRoIAlign, StereoRoIAlign, align_mask_spatial_valid, make_pair_crop_windows, sample_roi_feature
from .local_aggregation import AttentionPyramid
from .roi_selection import LeftOnlyDetections, RoiPairSelection, select_roi_pairs, subset_pairs, subset_crop, project_foreground, keep_foreground_pairs, _iou
from .roi_selection import Near10Policy, near10_conflicts, near10_geometry, near10_keep, near10_measure
from .geometry import generate_point_locations, point_levels
from .detection import (
    ContainmentEvent, PairDetections, StereoYoloHead, decode_anchorfree_predictions, decode_dense_predictions,
    generate_grid_anchors, suppress_contained_pairs,
)
from .stereo_input import StereoInputGeometry


def restore_box_rows(boxes: Tensor, crop_top_px: int) -> Tensor:
    return boxes + boxes.new_tensor((0, crop_top_px, 0, crop_top_px)) if crop_top_px else boxes


def restore_crop_rows(crop: CropMetadata | None, crop_top_px: int) -> CropMetadata | None:
    return replace(crop, y0=crop.y0 + crop_top_px) if crop is not None and crop_top_px else crop


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
    recenter_left_edge = False
    def __init__(
        self, channels: int, dmax_roi: int, hidden_channels: int = 32, *,
        roi_sampling: str = "native_pad", mask_head_variant: str = "light",
        idisp_cost_backend: str = "loop",
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
        self.idisp = RegularizedIDispHead(channels, dmax_roi, hidden_channels, cost_backend=idisp_cost_backend)

    def _disparity_forward(
        self, pyramid: object, boxes6: Tensor, image_height: int, image_width: int, margin: float, *,
        fit_vertical_extent: bool = False,
    ) -> tuple[RoiDisparity, CropMetadata, Tensor, Tensor]:
        with record_function("stereo.roi_sample"):
            crop = make_pair_crop_windows(boxes6, image_height, image_width, margin, roi_sampling=self.roi_sampling,
                                          fit_vertical_extent=fit_vertical_extent,
                                          recenter_left_edge=self.recenter_left_edge)
            left_roi, right_roi, spatial_valid, _ = self.roi(pyramid.p2_left, pyramid.p2_right, boxes6, crop)
            p3_roi, _ = self.p3_roi(pyramid.p3_left, crop)
            p4_roi, _ = self.p4_roi(pyramid.p4_left, crop)
        attention = AttentionPyramid(left_roi, p3_roi, p4_roi)
        if self.recenter_left_edge:
            columns = torch.arange(right_roi.shape[-1], device=right_roi.device) + .5
            x = crop.c_r[:, None] + columns * (crop.c / crop.w_roi)[:, None]
            right_valid = ((x >= 0) & (x < image_width))[:, None, None, :].expand_as(spatial_valid)
            disparity = self.idisp(left_roi, right_roi, attention, crop, spatial_valid,
                                   right_spatial_valid=right_valid)
        else:
            disparity = self.idisp(left_roi, right_roi, attention, crop, spatial_valid)
        return disparity, crop, left_roi, spatial_valid

    def sample_mask_roi(self, p2_left: Tensor, boxes6: Tensor, image_height: int, image_width: int,
                        margin: float = 16.) -> tuple[Tensor, Tensor, CropMetadata]:
        """Sample only left P2 with the deployed stereo crop geometry."""
        crop = make_pair_crop_windows(boxes6, image_height, image_width, margin, roi_sampling=self.roi_sampling)
        features = p2_left.expand(boxes6.shape[0], -1, -1, -1)
        left_roi, spatial_valid = sample_roi_feature(features, crop.c_l, crop, self.roi.stride)
        return left_roi, align_mask_spatial_valid(spatial_valid), crop

    def forward_selected(self, pyramid, pairs, image_height, image_width, mask_threshold,
                         *, fit_vertical_extent=False):
        """Reuse Mask and P2-left RoIs; build disparity only for kept pairs."""
        count = len(pairs.indices)
        hw = (image_height, image_width)
        if count < 2 or not bool((_iou(pairs.boxes6[:, :4], pairs.boxes6[:, :4]) >= .35).triu(1).any()):
            roi = type("RoiPyramid", (), {n:getattr(pyramid,n).expand(count,-1,-1,-1)
                for n in ("p2_left","p2_right","p3_left","p4_left")})()
            return pairs, self(roi, pairs.boxes6, *hw, fit_vertical_extent=fit_vertical_extent)
        crop = make_pair_crop_windows(pairs.boxes6, *hw, 16., roi_sampling=self.roi_sampling,
                                      fit_vertical_extent=fit_vertical_extent)
        left, spatial = sample_roi_feature(pyramid.p2_left.expand(count,-1,-1,-1), crop.c_l, crop, 4)
        logits = self.mask(left)
        mask_validity = align_mask_spatial_valid(spatial)
        masks = project_foreground(logits, mask_validity, crop, hw, mask_threshold)
        keep = keep_foreground_pairs(pairs, masks)
        pairs = subset_pairs(pairs, keep)
        crop = subset_crop(crop, keep)
        left, spatial = left[keep], spatial[keep]
        count = len(keep)
        right, _ = sample_roi_feature(pyramid.p2_right.expand(count,-1,-1,-1), crop.c_r, crop, 4)
        p3, _ = self.p3_roi(pyramid.p3_left.expand(count,-1,-1,-1), crop)
        p4, _ = self.p4_roi(pyramid.p4_left.expand(count,-1,-1,-1), crop)
        disparity = self.idisp(left, right, AttentionPyramid(left,p3,p4), crop, spatial)
        return pairs, InstanceDisparityOutput(logits[keep],mask_validity[keep],disparity.absolute,
            disparity.valid,disparity.peak,disparity.entropy,crop)

    def forward_disparity(
        self, pyramid: object, boxes6: Tensor, image_height: int, image_width: int, margin: float = 16.
    ) -> tuple[RoiDisparity, CropMetadata]:
        disparity, crop, _, _ = self._disparity_forward(pyramid, boxes6, image_height, image_width, margin)
        return disparity, crop

    def forward_near10(self, pyramid, pairs, image_hw, conflicts, policy, mask_threshold):
        """Compute early Masks only for candidates eligible for Car suppression."""
        suspects = torch.where(conflicts.any(0) | conflicts.any(1))[0]
        with record_function("stereo.roi_sample"):
            crop = make_pair_crop_windows(pairs.boxes6, *image_hw, 16., roi_sampling=self.roi_sampling)
            left, spatial = sample_roi_feature(pyramid.p2_left.expand(len(pairs.indices), -1, -1, -1),
                                               crop.c_l, crop, self.roi.stride)
        mask_validity = align_mask_spatial_valid(spatial)
        suspect_pairs = subset_pairs(pairs, suspects)
        with record_function("stereo.mask"):
            early = self.mask(left[suspects])
            masks = project_foreground(early, mask_validity[suspects], subset_crop(crop, suspects),
                                       image_hw, mask_threshold)
        with record_function("stereo.near10_dedup"):
            early_keep = near10_keep(suspect_pairs, conflicts[suspects][:, suspects], masks, policy)
        selected = torch.ones_like(pairs.indices, dtype=torch.bool)
        selected[suspects] = False
        late = torch.where(selected)[0]
        selected[suspects[early_keep]] = True
        keep = torch.where(selected)[0]
        logits = early.new_empty((len(keep), *early.shape[1:]))
        logits[torch.searchsorted(keep, suspects[early_keep])] = early[early_keep]
        if len(late):
            with record_function("stereo.mask"):
                logits[torch.searchsorted(keep, late)] = self.mask(left[late])
        crop = subset_crop(crop, keep)
        left, spatial = left[keep], spatial[keep]
        count = len(keep)
        with record_function("stereo.roi_sample"):
            right, _ = sample_roi_feature(pyramid.p2_right.expand(count, -1, -1, -1),
                                          crop.c_r, crop, self.roi.stride)
            p3, _ = self.p3_roi(pyramid.p3_left.expand(count, -1, -1, -1), crop)
            p4, _ = self.p4_roi(pyramid.p4_left.expand(count, -1, -1, -1), crop)
        disparity = self.idisp(left, right, AttentionPyramid(left, p3, p4), crop, spatial)
        return subset_pairs(pairs, keep), InstanceDisparityOutput(
            logits, mask_validity[keep], disparity.absolute, disparity.valid, disparity.peak, disparity.entropy, crop)

    def forward(
        self, pyramid: object, boxes6: Tensor, image_height: int, image_width: int, margin: float = 16., *,
        fit_vertical_extent: bool = False,
    ) -> InstanceDisparityOutput:
        disparity, crop, left_roi, spatial_valid = self._disparity_forward(
            pyramid, boxes6, image_height, image_width, margin, fit_vertical_extent=fit_vertical_extent,
        )
        with record_function("stereo.mask"):
            logits = self.mask(left_roi)
        return InstanceDisparityOutput(
            mask_logits=logits,
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
    left_only: LeftOnlyDetections | None = None
    crop: CropMetadata | None = None
    crop_top_px: int = 0
    image_hw: tuple[int, int] | None = None
    encoder_padded_hw: tuple[int, int] | None = None
    containment_events: tuple[ContainmentEvent, ...] = ()
    near10_distance_m: Tensor | None = None


class StereoYoloLocalModel(nn.Module):
    """Batch-one inference assembly; empty detections bypass all local matching."""

    def __init__(
        self, encoder: nn.Module, detector: StereoYoloHead, instance_head: StereoLocalInstanceHead,
        score_threshold: float = .40, pre_nms_topk: int = 500, detector_variant: str = "baseline", *,
        right_offset_mode: str = "legacy_clamp", apply_paired_nms: bool | None = None,
        paired_nms_iou: float = .5, foreground_class_indices: tuple[int, ...] = (),
        mask_threshold: float = .50, roi_selection: RoiPairSelection | None = None,
        crop_top_px: int = 0, containment_threshold: float | None = None,
        near10_policy: Near10Policy | None = None,
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
        if type(crop_top_px) is not int or crop_top_px < 0:
            raise ValueError("crop_top_px must be a non-negative integer")
        if containment_threshold is not None:
            if not 0 < containment_threshold <= 1:
                raise ValueError("containment threshold must be in (0, 1]")
            if detector_variant == "stereo_yolo26_pan" and apply_paired_nms is not True:
                raise ValueError("containment suppression requires explicit paired NMS")
            if roi_selection is not None:
                raise ValueError("containment suppression and near selection require separate runs")
        self.crop_top_px = crop_top_px
        self.containment_threshold = containment_threshold
        if foreground_class_indices and instance_head.roi_sampling != "fixed_56":
            raise ValueError("foreground disparity export requires the fixed_56 grid")
        self.foreground_class_indices = foreground_class_indices
        if not 0 < mask_threshold < 1:
            raise ValueError("mask threshold must be strictly between zero and one")
        self.mask_threshold = mask_threshold
        if roi_selection is not None:
            if detector_variant != "stereo_yolo26_pan" or instance_head.roi_sampling != "fixed_56":
                raise ValueError("near selection requires the PAN detector and fixed_56 RoIs")
            if score_threshold != .40 or pre_nms_topk != 500 or paired_nms_iou != .50 or apply_paired_nms is not True:
                raise ValueError("near selector requires score=.40, topk=500 and explicit paired NMS=.50")
            if roi_selection.car_class not in foreground_class_indices:
                raise ValueError("near selection requires trained Car foreground")
        self.roi_selection = roi_selection
        if near10_policy is not None:
            if (roi_selection is not None or containment_threshold is not None or crop_top_px
                    or detector_variant != "stereo_yolo26_pan" or apply_paired_nms is not True):
                raise ValueError("10 m policy requires full-frame PAN and explicit ordinary paired NMS")
            if near10_policy.mode == "range-dedup" and near10_policy.car_class not in foreground_class_indices:
                raise ValueError("range-dedup requires weights with Car foreground")
        self.near10_policy = near10_policy

    def forward(
        self, left: Tensor, right: Tensor, *, replay_pairs: PairDetections | None = None, route=None, calibration=None,
    ) -> StereoYoloLocalOutput:
        if left.ndim != 4 or left.shape != right.shape or left.shape[:2] != (1, 3):
            raise ValueError(f"expected equal single-pair RGB NCHW inputs, actual {left.shape}, {right.shape}")
        geometry = StereoInputGeometry(tuple(left.shape[-2:]), self.crop_top_px)
        crop = geometry.crop_top_px
        return self.forward_prepared(left[..., crop:, :], right[..., crop:, :], geometry,
                                     replay_pairs=replay_pairs, route=route, calibration=calibration)

    def forward_prepared(self, left, right, geometry: StereoInputGeometry, *, replay_pairs=None, route=None,
                         calibration=None):
        """Run the shared encoder on the already-cropped pair."""
        self._validate_prepared(left, geometry, right)
        pyramid = self.encoder(left, right)
        dense = self.detector(pyramid)
        return self.decode_and_refine(left, pyramid, dense, input_geometry=geometry, replay_pairs=replay_pairs,
                                      route=route, calibration=calibration)

    def _validate_prepared(self, reference, geometry, right=None):
        geometry.validate_prepared(reference, right)
        if geometry.crop_top_px != self.crop_top_px:
            raise ValueError(f"expected crop_top_px {self.crop_top_px}, actual {geometry.crop_top_px}")

    def decode_and_refine(
        self, left: Tensor, pyramid: object, dense: object, *, input_geometry: StereoInputGeometry,
        replay_pairs: PairDetections | None = None, route=None, calibration=None,
    ) -> StereoYoloLocalOutput:
        """Decode frontend tensors and run native RoI heads in cropped coordinates."""
        self._validate_prepared(left, input_geometry)
        image_hw = input_geometry.original_hw
        crop_top_px = input_geometry.crop_top_px
        cropped_hw = input_geometry.cropped_hw
        padding = getattr(pyramid, "metadata", None)
        encoder_padded_hw = getattr(padding, "padded_size", None)
        if encoder_padded_hw is not None and (
                encoder_padded_hw != input_geometry.padded_hw or getattr(padding, "original_size", cropped_hw) != cropped_hw):
            raise ValueError(f"expected frontend geometry {input_geometry.as_dict()}, actual {padding}")
        if replay_pairs is not None:
            if self.roi_selection is not None or self.containment_threshold is not None or self.near10_policy is not None:
                raise ValueError("fixed RoI replay requires ordinary instance dispatch")
            pair = replay_pairs
        elif self.detector_variant == "stereo_yolo26_pan":
            locations = generate_point_locations(point_levels(pyramid), left.device)
            pair = decode_anchorfree_predictions(
                {"objectness": dense.objectness[0], "class_logits": dense.class_logits[0], "box6": dense.box6[0]},
                locations, self.score_threshold, nms_threshold=self.paired_nms_iou, pre_nms_topk=self.pre_nms_topk,
                right_offset_mode=self.right_offset_mode,
                apply_nms=(not getattr(self.detector, "one_to_one_enabled", False)
                           if self.apply_paired_nms is None else self.apply_paired_nms),
                class_aware_nms=True,
                route=route,
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
        containment_events = ()
        if self.containment_threshold is not None:
            pair, containment_events = suppress_contained_pairs(pair, self.containment_threshold)
        left_only = None
        conflicts = None
        policy = self.near10_policy
        if policy is not None:
            if calibration is None:
                raise ValueError("10 m policy requires calibration for each frame")
            with record_function("stereo.near10_range"):
                geometry = near10_geometry(pair, cropped_hw, calibration, policy)
            if route is not None:
                route["range_far"] = subset_pairs(pair, geometry.far)
                route["range_unknown"] = subset_pairs(pair, geometry.left_only)
                route["range_invalid"] = subset_pairs(pair, geometry.invalid)
            if policy.mode != "baseline":
                fallback = subset_pairs(pair, geometry.left_only)
                left_only = LeftOnlyDetections(fallback.boxes6[:, :4], fallback.scores, fallback.indices,
                                               fallback.classes, fallback.scores.new_full(fallback.scores.shape, float("nan")))
                pair = subset_pairs(pair, geometry.admitted)
            if route is not None:
                route["range_kept"] = pair
            if policy.mode == "range-dedup" and len(pair.indices) > 1:
                with record_function("stereo.near10_dedup"):
                    conflicts = near10_conflicts(pair, geometry.disparity[geometry.admitted], policy)
        if self.roi_selection is not None:
            pair, left_only = select_roi_pairs(pair,
                {"objectness": dense.objectness[0], "class_logits": dense.class_logits[0], "box6": dense.box6[0]},
                locations, cropped_hw, self.roi_selection, self.right_offset_mode)
            if crop_top_px:
                left_only = replace(left_only, boxes=restore_box_rows(left_only.boxes, crop_top_px))
        if pair.boxes6.numel() == 0:
            if route is not None:
                route["idisp"] = pair
                if policy is not None:
                    route["near10_final"] = pair
            empty = left.new_empty((0,))
            empty_ids = left.new_empty((0,), dtype=torch.long)
            empty_boxes = left.new_empty((0, 4))
            empty_disparity = left.new_empty((0, 0, 0))
            empty_valid = left.new_empty((0, 0, 0), dtype=torch.bool)
            return StereoYoloLocalOutput(
                pair_ids=empty_ids, foreground_validity=empty_valid, boxes_left=empty_boxes, boxes_right=empty_boxes,
                scores=empty, classes=empty_ids, mask_logits=left.new_empty((0, 1, 28, 28)),
                mask_validity=left.new_empty((0, 1, 28, 28), dtype=torch.bool), disparity=empty_disparity,
                validity=empty_valid, peak=empty_disparity, entropy=empty_disparity, left_only=left_only,
                crop_top_px=crop_top_px, image_hw=image_hw, encoder_padded_hw=encoder_padded_hw,
                near10_distance_m=empty if policy is not None else None,
            )
        # RoI branches expect one feature item per candidate; repeat only feature maps, not the encoder.
        count = pair.boxes6.shape[0]
        roi_pyramid = type("RoiPyramid", (), {name: getattr(pyramid, name).expand(count, -1, -1, -1) for name in ("p2_left", "p2_right", "p3_left", "p4_left")})()
        roi_options = {"fit_vertical_extent": True} if crop_top_px else {}
        if conflicts is not None and bool(conflicts.any()):
            pair, instance = self.instance_head.forward_near10(
                pyramid, pair, cropped_hw, conflicts, policy, self.mask_threshold)
        elif self.roi_selection is None:
            instance = self.instance_head(roi_pyramid, pair.boxes6, *cropped_hw, **roi_options)
        else:
            pair, instance = self.instance_head.forward_selected(
                pyramid, pair, *cropped_hw, self.mask_threshold, **roi_options)
        if route is not None:
            route["idisp"] = pair
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
        distance = None
        if policy is not None:
            supported = (pair.classes[:, None] == pair.classes.new_tensor(self.foreground_class_indices)).any(1)
            validity = torch.where(supported[:, None, None], foreground, instance.validity)
            with record_function("stereo.near10_measure"):
                distance = near10_measure(instance.disparity, validity, calibration)
            keep = torch.where(torch.isfinite(distance) & (distance <= 10.))[0]
            if route is not None:
                route["near10_unknown"] = subset_pairs(pair, ~torch.isfinite(distance))
                route["near10_final"] = subset_pairs(pair, keep)
            pair = subset_pairs(pair, keep)
            distance = distance[keep]
            boxes_left, boxes_right, foreground = boxes_left[keep], boxes_right[keep], foreground[keep]
            instance = InstanceDisparityOutput(instance.mask_logits[keep], instance.mask_validity[keep],
                instance.disparity[keep], instance.validity[keep], instance.peak[keep], instance.entropy[keep],
                subset_crop(instance.crop, keep))
        return StereoYoloLocalOutput(
            pair_ids=pair.indices, foreground_validity=foreground,
            boxes_left=restore_box_rows(boxes_left, crop_top_px),
            boxes_right=restore_box_rows(boxes_right, crop_top_px),
            scores=pair.scores, classes=pair.classes, mask_logits=instance.mask_logits,
            mask_validity=instance.mask_validity, disparity=instance.disparity, validity=instance.validity,
            peak=instance.peak, entropy=instance.entropy, left_only=left_only,
            crop=restore_crop_rows(getattr(instance, "crop", None), crop_top_px),
            crop_top_px=crop_top_px, image_hw=image_hw,
            encoder_padded_hw=encoder_padded_hw, containment_events=containment_events,
            near10_distance_m=distance,
        )
