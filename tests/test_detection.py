"""Contracts for one-stage paired stereo detection."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from lightbackbone_stereo_yolo.detection import (
        assign_left_iou_targets,
        detection_loss,
        paired_nms,
        StereoYoloHead,
        decode_dense_predictions,
        generate_grid_anchors,
    )
except ImportError:
    assign_left_iou_targets = detection_loss = paired_nms = StereoYoloHead = decode_dense_predictions = generate_grid_anchors = None


class StereoDetectionContractTest(unittest.TestCase):
    def test_multiscale_left_grid_anchors_follow_feature_cells(self) -> None:
        self.assertIsNotNone(generate_grid_anchors)
        anchors = generate_grid_anchors(((2, 3, 8, 32.), (1, 2, 16, 64.)))
        self.assertEqual(anchors.shape, (8, 4))
        self.assertTrue(torch.equal(anchors[0], torch.tensor([-12., -12., 20., 20.])))
        self.assertTrue(torch.equal(anchors[-1], torch.tensor([-8., -24., 56., 40.])))

    def test_grid_anchors_reuse_the_cached_tensor_for_same_levels_and_device(self) -> None:
        """Would fail if every inference reconstructed static feature-grid anchors."""
        levels = ((2, 3, 8, 32.0), (1, 2, 16, 64.0))
        first = generate_grid_anchors(levels, torch.device("cpu"))
        second = generate_grid_anchors(levels, torch.device("cpu"))
        changed = generate_grid_anchors(((2, 4, 8, 32.0), (1, 2, 16, 64.0)), torch.device("cpu"))
        self.assertIs(first, second)
        self.assertTrue(torch.equal(first, second))
        self.assertIsNot(first, changed)
    def test_decode_filters_empty_candidates_without_breaking_pair_contract(self) -> None:
        self.assertIsNotNone(decode_dense_predictions)
        prediction = {
            "objectness": torch.tensor([-10., 10.]),
            "class_logits": torch.tensor([[1., 0.], [0., 3.]]),
            "box6": torch.zeros(2, 6),
        }
        anchors = torch.tensor([[0., 0., 10., 10.], [20., 0., 30., 10.]])
        decoded = decode_dense_predictions(prediction, anchors, score_threshold=.5)
        self.assertEqual(decoded.boxes6.shape, (1, 6))
        self.assertEqual(decoded.classes.tolist(), [1])

    def test_decode_limits_candidates_before_paired_nms(self) -> None:
        """Would fail if all threshold-passing anchors reach paired NMS."""
        prediction = {
            "objectness": torch.tensor([1.0, 3.0, 2.0]),
            "class_logits": torch.tensor([[3., 0.], [3., 0.], [3., 0.]]),
            "box6": torch.zeros(3, 6),
        }
        anchors = torch.tensor([[0., 0., 10., 10.], [20., 0., 30., 10.], [40., 0., 50., 10.]])
        decoded = decode_dense_predictions(prediction, anchors, score_threshold=.1, pre_nms_topk=2)
        self.assertEqual(decoded.boxes6.shape[0], 2)
        self.assertTrue(torch.equal(decoded.indices, torch.tensor([1, 2])))
    def test_concat_head_consumes_p3_p4_c5_without_a_second_backbone(self) -> None:
        self.assertIsNotNone(StereoYoloHead)
        pyramid = SimpleNamespace(
            p3_left=torch.randn(2, 8, 8, 10), p3_right=torch.randn(2, 8, 8, 10),
            p4_left=torch.randn(2, 8, 4, 5), p4_right=torch.randn(2, 8, 4, 5),
            c5_left=torch.randn(2, 8, 2, 3), c5_right=torch.randn(2, 8, 2, 3),
        )
        predictions = StereoYoloHead(8, num_classes=3)(pyramid)
        self.assertEqual(predictions.objectness.shape, (2, 106))
        self.assertEqual(predictions.class_logits.shape, (2, 106, 3))
        self.assertEqual(predictions.box6.shape, (2, 106, 6))
    def test_assignment_uses_left_iou_only_and_forces_best_anchor_positive(self) -> None:
        self.assertIsNotNone(assign_left_iou_targets)
        anchors = torch.tensor([[0., 0., 10., 10.], [20., 0., 30., 10.]])
        # Right box has no overlap with an implicit zero-disparity anchor. It must
        # still be positive because this detector has static left-view anchors.
        target = torch.tensor([[0., 0., 10., 10., -20., -10.]])
        assigned = assign_left_iou_targets(anchors, target, positive_iou=0.5, negative_iou=0.4)
        self.assertEqual(assigned.labels.tolist(), [1, 0])
        self.assertEqual(assigned.matched_indices.tolist(), [0, -1])

    def test_detection_loss_returns_three_trainable_terms(self) -> None:
        self.assertIsNotNone(detection_loss)
        prediction = {
            "objectness": torch.tensor([2.0, -2.0], requires_grad=True),
            "class_logits": torch.tensor([[0.2, 1.0], [1.0, 0.2]], requires_grad=True),
            "box6": torch.zeros(2, 6, requires_grad=True),
        }
        targets = {
            "labels": torch.tensor([1, 0]),
            "classes": torch.tensor([1, -1]),
            "box6": torch.zeros(2, 6),
        }
        losses = detection_loss(prediction, targets)
        self.assertEqual(set(losses), {"loss_obj", "loss_cls", "loss_box6"})
        sum(losses.values()).backward()
        self.assertIsNotNone(prediction["box6"].grad)

    def test_paired_nms_preserves_left_right_index_binding(self) -> None:
        self.assertIsNotNone(paired_nms)
        pairs = torch.tensor([
            [0., 0., 10., 10., -10., 0.],
            [1., 1., 11., 11., -9., 1.],
            [30., 0., 40., 10., 20., 30.],
        ])
        kept = paired_nms(pairs, torch.tensor([.9, .8, .7]), iou_threshold=.5)
        self.assertTrue(torch.equal(kept.indices, torch.tensor([0, 2])))
        self.assertTrue(torch.equal(kept.boxes6[:, 4], torch.tensor([-10., 20.])))


if __name__ == "__main__":
    unittest.main()
