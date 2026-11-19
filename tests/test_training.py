"""Contracts for frame-level targets and the GT-RoI training forward."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from lightbackbone_stereo_yolo.data.frame_dataset import (
        build_class_mapping,
        build_frame_records,
    )
    from lightbackbone_stereo_yolo.detection import StereoYoloHead
    from lightbackbone_stereo_yolo.geometry import encode_boxes6
    from lightbackbone_stereo_yolo.local_disparity import overflow_mask, restore_absolute_disparity
    from lightbackbone_stereo_yolo.model import StereoLocalInstanceHead
    from lightbackbone_stereo_yolo.roi import CropMetadata, align_disparity_roi
    from lightbackbone_stereo_yolo.training import (
        build_dense_detection_targets,
        roi_residual_target,
        training_forward,
    )
except ImportError:
    build_class_mapping = build_frame_records = None
    StereoYoloHead = encode_boxes6 = generate_grid_anchors = None
    overflow_mask = restore_absolute_disparity = None
    StereoLocalInstanceHead = None
    CropMetadata = align_disparity_roi = None
    build_dense_detection_targets = roi_residual_target = training_forward = None


class FrameTargetContractTest(unittest.TestCase):
    def test_frame_manifest_groups_all_instances_and_maps_classes_contiguously(self) -> None:
        self.assertIsNotNone(build_frame_records)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in (
                "training/image_2/000001_10.png",
                "training/image_3/000001_10.png",
                "training/instance/000001_10.png",
                "training/disp_occ_0/000001_10.png",
                "rpn_annotations_v1/ignore_masks/000001_10.png",
            ):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            payload = {
                "images": [{
                    "id": 1,
                    "file_name": "training/image_2/000001_10.png",
                    "width": 10,
                    "height": 8,
                    "ignore_mask_file": "rpn_annotations_v1/ignore_masks/000001_10.png",
                }],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 26, "bbox": [1, 2, 3, 4], "source_instance_value": 11},
                    {"id": 2, "image_id": 1, "category_id": 24, "bbox": [4, 2, 2, 4], "source_instance_value": 12},
                ],
            }
            source = root / "split.json"
            source.write_text(json.dumps(payload))
            frames = build_frame_records(root, source)
            self.assertEqual(len(frames), 1)
            self.assertEqual(len(frames[0].annotations), 2)
            self.assertEqual(build_class_mapping(frames), {24: 0, 26: 1})

    def test_dense_targets_encode_only_assigned_positive_boxes(self) -> None:
        self.assertIsNotNone(build_dense_detection_targets)
        anchors = torch.tensor([[0., 0., 10., 10.], [20., 0., 30., 10.]])
        boxes6 = torch.tensor([[0., 0., 10., 10., -20., -10.]])
        classes = torch.tensor([1])
        targets = build_dense_detection_targets(anchors, boxes6, classes, .5, .4)
        expected = encode_boxes6(boxes6, anchors[:1])
        self.assertEqual(targets["labels"].tolist(), [1, 0])
        self.assertEqual(targets["classes"].tolist(), [1, 0])
        self.assertTrue(torch.allclose(targets["box6"][:1], expected))
        self.assertTrue(torch.equal(targets["box6"][1], torch.zeros(6)))


class RoiSupervisionContractTest(unittest.TestCase):
    @staticmethod
    def _metadata() -> "CropMetadata":
        return CropMetadata(
            c_l=torch.tensor([0.]),
            c_r=torch.tensor([-4.]),
            c=torch.tensor([8.]),
            y0=torch.tensor([0.]),
            h_crop=torch.tensor([8.]),
            w_roi=torch.tensor([2]),
            h_roi=torch.tensor([2]),
        )

    def test_disparity_is_cropped_on_the_p2_grid_and_inverts_the_restore_formula(self) -> None:
        self.assertIsNotNone(align_disparity_roi)
        disparity = torch.full((1, 1, 8, 8), 5.)
        valid = torch.ones_like(disparity, dtype=torch.bool)
        target, target_valid = align_disparity_roi(disparity, valid, self._metadata())
        self.assertEqual(target.shape, (1, 1, 2, 2))
        self.assertTrue(torch.equal(target, torch.full((1, 1, 2, 2), 5.)))
        self.assertTrue(target_valid.all())
        residual = roi_residual_target(target, self._metadata())
        restored = restore_absolute_disparity(residual, self._metadata())
        self.assertTrue(torch.allclose(restored, target, atol=1e-6))

    def test_overflow_excludes_pixels_outside_the_fixed_search_range(self) -> None:
        self.assertIsNotNone(overflow_mask)
        metadata = self._metadata()
        disparity = torch.full((1, 1, 8, 8), 44.)
        valid = torch.ones_like(disparity, dtype=torch.bool)
        target, _ = align_disparity_roi(disparity, valid, metadata)
        residual = roi_residual_target(target, metadata)
        self.assertTrue(torch.allclose(residual, torch.full_like(residual, 10.)))
        self.assertTrue(overflow_mask(residual, 1).all())
        self.assertFalse(overflow_mask(residual, 16).any())


class TrainingForwardContractTest(unittest.TestCase):
    def test_gt_roi_forward_produces_finite_losses_metrics_and_gradients(self) -> None:
        self.assertIsNotNone(training_forward)
        torch.manual_seed(3)

        class TinyEncoder(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.projection = nn.Parameter(torch.tensor(1.))

            def forward(self, left, right):
                base = self.projection * (left[:, :1] + right[:, :1])
                p2 = torch.cat((base, torch.zeros_like(base), torch.zeros_like(base), torch.zeros_like(base)), 1)
                return SimpleNamespace(
                    p2_left=p2, p2_right=p2,
                    p3_left=torch.zeros(1, 4, 8, 8, device=left.device),
                    p3_right=torch.zeros(1, 4, 8, 8, device=left.device),
                    p4_left=torch.zeros(1, 4, 4, 4, device=left.device),
                    p4_right=torch.zeros(1, 4, 4, 4, device=left.device),
                    c5_left=torch.zeros(1, 4, 2, 2, device=left.device),
                    c5_right=torch.zeros(1, 4, 2, 2, device=left.device),
                )

        encoder = TinyEncoder()
        detector = StereoYoloHead(4, num_classes=2)
        instance_head = StereoLocalInstanceHead(4, dmax_roi=2, hidden_channels=4)
        left = torch.randn(1, 3, 64, 64)
        right = torch.randn(1, 3, 64, 64)
        masks = torch.zeros(1, 1, 64, 64, dtype=torch.bool)
        masks[:, :, 8:32, 8:40] = True
        target = {
            "boxes6": torch.tensor([[8., 8., 40., 32., 2., 34.]]),
            "classes": torch.tensor([1]),
            "masks": masks,
            "disparity": torch.full((1, 64, 64), 4.),
            "disparity_valid": torch.ones(1, 64, 64, dtype=torch.bool),
        }
        output = training_forward(
            encoder, detector, instance_head, left, right, target,
            dmax_roi=2, collect_metrics=True,
        )
        for key in ("loss_obj", "loss_cls", "loss_box6", "loss_mask", "loss_disp", "loss_total"):
            self.assertTrue(torch.isfinite(output[key]), key)
        self.assertGreater(output["metrics"]["iou_count"], 0)
        self.assertGreater(output["metrics"]["epe_count"], 0)
        output["loss_total"].backward()
        for parameter in list(encoder.parameters()) + list(detector.parameters()) + list(instance_head.parameters()):
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())


if __name__ == "__main__":
    unittest.main()
