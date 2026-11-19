"""Anchor-free loss and decoding contracts."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from lightbackbone_stereo_yolo.detection import anchorfree_detection_loss, decode_anchorfree_predictions
except ImportError:
    anchorfree_detection_loss = decode_anchorfree_predictions = None


class AnchorFreeDetectionTest(unittest.TestCase):
    def test_anchorfree_loss_handles_outside_fallback_without_box_regression(self) -> None:
        self.assertIsNotNone(anchorfree_detection_loss)
        prediction = {
            "objectness": torch.tensor([1., -1.], requires_grad=True),
            "class_logits": torch.tensor([[0., 2.], [2., 0.]], requires_grad=True),
            "box6": torch.zeros(2, 6, requires_grad=True),
        }
        targets = {
            "labels": torch.tensor([1, 0]),
            "classes": torch.tensor([1, 0]),
            "box6": torch.zeros(2, 6),
            "box_valid_mask": torch.tensor([False, False]),
        }
        locations = torch.tensor([[4., 4., 8.], [12., 4., 8.]])
        losses = anchorfree_detection_loss(prediction, targets, locations)
        self.assertTrue(torch.isfinite(sum(losses.values())))
        self.assertEqual(float(losses["loss_box6"]), 0.)
        sum(losses.values()).backward()


    def test_c1_left_iou_has_gradient_even_when_right_iou_is_lower(self) -> None:
        from lightbackbone_stereo_yolo.geometry import encode_anchorfree_boxes6

        points = torch.tensor([[20., 20., 8.]])
        expected = torch.tensor([[8., 8., 28., 28., 1., 21.]])
        estimate = torch.tensor([[10., 8., 26., 28., 1., 9.]])
        box_logits = encode_anchorfree_boxes6(estimate, points).requires_grad_()
        pred = {"objectness": torch.zeros(1), "class_logits": torch.zeros(1, 1), "box6": box_logits}
        target = {
            "labels": torch.ones(1, dtype=torch.long), "classes": torch.zeros(1, dtype=torch.long),
            "box_valid_mask": torch.ones(1, dtype=torch.bool), "box6": encode_anchorfree_boxes6(expected, points),
        }
        base = anchorfree_detection_loss(pred, target, points)
        c1 = anchorfree_detection_loss(pred, target, points, left_iou_weight=.5)
        torch.testing.assert_close(base["loss_pair_iou"], torch.tensor(.60))
        torch.testing.assert_close(c1["loss_pair_iou"], base["loss_pair_iou"])
        torch.testing.assert_close(c1["loss_left_iou"], torch.tensor(.20))
        extra = c1["loss_box6"] - base["loss_box6"]
        torch.testing.assert_close(extra, torch.tensor(.10))
        gradient, = torch.autograd.grad(extra, box_logits)
        self.assertGreater(float(gradient[:, :4].abs().sum()), 0.)
        torch.testing.assert_close(gradient[:, 4:], torch.zeros_like(gradient[:, 4:]))

    def test_c1_returns_connected_zero_when_no_paired_target_is_valid(self) -> None:
        box_logits = torch.zeros(1, 6, requires_grad=True)
        pred = {"objectness": torch.zeros(1), "class_logits": torch.zeros(1, 1), "box6": box_logits}
        target = {"labels": torch.tensor([1]), "classes": torch.tensor([0]),
                  "box6": torch.zeros(1, 6), "box_valid_mask": torch.zeros(1, dtype=torch.bool)}
        losses = anchorfree_detection_loss(pred, target, torch.tensor([[4., 4., 8.]]), left_iou_weight=.5)
        self.assertEqual(float(losses["loss_left_iou"]), 0.)
        losses["loss_box6"].backward()
        torch.testing.assert_close(box_logits.grad, torch.zeros_like(box_logits))


    def test_d_soft_iou_target_uses_joint_q_and_detaches_it(self) -> None:
        logits = torch.zeros(2, requires_grad=True)
        quality = torch.tensor([.8, .3], requires_grad=True)
        boxes = torch.zeros(2, 6, requires_grad=True)
        pred = {"objectness": logits, "class_logits": torch.zeros(2, 1), "box6": boxes}
        expected = {"labels": torch.ones(2, dtype=torch.long), "classes": torch.zeros(2, dtype=torch.long),
                    "box_valid_mask": torch.ones(2, dtype=torch.bool), "box6": boxes.detach()}
        points = torch.tensor([[4., 4., 8.], [12., 4., 8.]])
        loss = anchorfree_detection_loss(
            pred, expected, points, one_to_one_quality=quality, one_to_one_score_target="joint_iou",
        )["loss_obj"]
        dl, dq, db = torch.autograd.grad(loss, (logits, quality, boxes), allow_unused=True)
        torch.testing.assert_close(dl, torch.tensor([-.15, .10]))
        self.assertIsNone(dq)
        self.assertIsNone(db)
        base = anchorfree_detection_loss(pred, expected, points, one_to_one_quality=quality)["loss_obj"]
        base_grad, = torch.autograd.grad(base, logits)
        torch.testing.assert_close(base_grad, torch.tensor([-.20, -.075]))

    def test_anchorfree_decode_reuses_paired_nms(self) -> None:
        self.assertIsNotNone(decode_anchorfree_predictions)
        prediction = {
            "objectness": torch.tensor([10., -10.]),
            "class_logits": torch.tensor([[0., 4.], [4., 0.]]),
            "box6": torch.zeros(2, 6),
        }
        locations = torch.tensor([[20., 20., 8.], [40., 20., 8.]])
        result = decode_anchorfree_predictions(prediction, locations, score_threshold=.5)
        self.assertEqual(result.boxes6.shape, (1, 6))
        self.assertEqual(result.classes.tolist(), [1])
        self.assertGreaterEqual(float(result.boxes6[0, 4]), 0.)


if __name__ == "__main__":
    unittest.main()

