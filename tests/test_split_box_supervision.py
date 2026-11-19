"""Left-box supervision remains available when right labels degenerate."""
import unittest

import torch

from lightbackbone_stereo_yolo.anchorfree_training import build_anchorfree_detection_targets
from lightbackbone_stereo_yolo.detection import anchorfree_detection_loss


class SplitBoxSupervisionTest(unittest.TestCase):
    def setUp(self):
        self.points = torch.tensor([[12., 12., 8.], [20., 12., 8.]])
        self.cls = torch.tensor([0])

    @staticmethod
    def logits(count):
        return {"objectness": torch.zeros(count), "class_logits": torch.zeros(count, 1),
                "box6": torch.zeros(count, 6, requires_grad=True)}

    def test_single_pixel_right_box_enables_only_left_box_gradient(self):
        gt = torch.tensor([[6., 6., 16., 18., 9., 9.]])
        target = build_anchorfree_detection_targets(self.points, gt, self.cls)
        self.assertEqual(target["left_box_valid_mask"].tolist(), [True, False])
        self.assertEqual(target["box_valid_mask"].tolist(), [False, False])
        logits = self.logits(2)
        baseline = anchorfree_detection_loss(logits, target, self.points, right_offset_mode="signed")
        split = anchorfree_detection_loss(logits, target, self.points, right_offset_mode="signed",
                                          box_supervision_mode="split_left")
        self.assertEqual(float(baseline["loss_box6"]), 0.)
        self.assertGreater(float(split["loss_left"]), 0.)
        for name in ("loss_disparity", "loss_right_width", "loss_pair_iou"):
            self.assertEqual(float(split[name]), 0.)
        for name in ("loss_obj", "loss_cls"):
            self.assertEqual(float(split[name]), float(baseline[name]))
        split["loss_box6"].backward()
        self.assertTrue((logits["box6"].grad[0, :4] != 0).all())
        self.assertEqual(logits["box6"].grad[:, 4:].count_nonzero(), 0)
        self.assertEqual(logits["box6"].grad[1].count_nonzero(), 0)

    def test_paired_valid_targets_preserve_all_existing_losses(self):
        gt = torch.tensor([[6., 6., 16., 18., 5., 14.]])
        target = build_anchorfree_detection_targets(self.points, gt, self.cls)
        self.assertEqual(target["left_box_valid_mask"].tolist(), [True, False])
        self.assertEqual(target["box_valid_mask"].tolist(), [True, False])
        logits = self.logits(2)
        baseline = anchorfree_detection_loss(logits, target, self.points, right_offset_mode="signed")
        split = anchorfree_detection_loss(logits, target, self.points, right_offset_mode="signed",
                                          box_supervision_mode="split_left")
        for name, value in baseline.items():
            torch.testing.assert_close(value, split[name], atol=0., rtol=0.)

    def test_outside_classification_fallback_remains_without_box_gradient(self):
        gt = torch.tensor([[13., 13., 14., 14., 8., 9.]])
        target = build_anchorfree_detection_targets(self.points, gt, self.cls)
        self.assertEqual(int((target["labels"] == 1).sum()), 1)
        self.assertFalse(target["left_box_valid_mask"].any())
        self.assertFalse(target["box_valid_mask"].any())
        split = anchorfree_detection_loss(self.logits(2), target, self.points,
                                          box_supervision_mode="split_left")
        self.assertEqual(float(split["loss_box6"]), 0.)
