"""Paired detection audit and signed right-offset contracts."""
import unittest
from types import SimpleNamespace

import torch

from scripts.diagnose_pred_roi import _frame_diagnosis, _state_transitions
from lightbackbone_stereo_yolo.detection import anchorfree_detection_loss, decode_anchorfree_predictions
from lightbackbone_stereo_yolo.geometry import decode_anchorfree_boxes6, encode_anchorfree_boxes6


class DetectorOffsetAuditTest(unittest.TestCase):
    def setUp(self):
        self.point = torch.tensor([[20., 20., 8.]])
        self.gt = torch.tensor([[16., 16., 36., 28., 19., 27.]])
        self.encoded = encode_anchorfree_boxes6(self.gt, self.point)

    def test_diagnostic_center_error_uses_x2(self):
        pred = torch.tensor([[16., 16., 32., 28., 19., 27.]])
        result = _frame_diagnosis(pred, torch.tensor([.8]), self.gt, .5)
        self.assertEqual(result["dR_errors"], [-2.])

    def test_left_selected_right_pass_and_formal_match_are_separate(self):
        pred = torch.tensor([[16., 16., 36., 28., 39., 47.],
                             [18., 16., 36., 28., 19., 27.]])
        result = _frame_diagnosis(pred, torch.tensor([.9, .8]), self.gt, .5)
        self.assertEqual(result["right_pass_given_left"], 0)
        self.assertEqual(result["formal_paired"], 1)

    def test_duplicate_frame_id_is_rejected_in_transitions(self):
        records = [{"frame_id": "a", "statuses": [{"formal_paired": False}]}] * 2
        with self.assertRaises(ValueError):
            _state_transitions({"per_frame": records}, {"per_frame": records})

    def test_signed_right_offset_round_trip(self):
        legacy = decode_anchorfree_boxes6(self.encoded, self.point)
        torch.testing.assert_close(legacy[0, 4:], torch.tensor([16., 24.]))
        encoded = self.encoded.clone().requires_grad_()
        signed = decode_anchorfree_boxes6(encoded, self.point, right_offset_mode="signed")
        torch.testing.assert_close(signed, self.gt, atol=1e-4, rtol=1e-5)
        signed[0, 4:].mean().backward()
        self.assertLess(float(encoded.grad[0, 4]), 0.)

    def test_signed_iou_loss_uses_original_gt_coordinates(self):
        # Supply a legacy-shifted encoded target on purpose to prove which
        # representation provides the geometry for the signed-mode IoU term.
        clipped = self.encoded.clone()
        clipped[:, 4] = 0.
        target = {"labels": torch.tensor([1]), "classes": torch.tensor([0]),
                  "box6": clipped, "matched_boxes6": self.gt, "box_valid_mask": torch.tensor([True])}
        logits = {"objectness": torch.ones(1), "class_logits": torch.zeros(1, 1)}
        correct = anchorfree_detection_loss(dict(logits, box6=self.encoded), target, self.point, right_offset_mode="signed")
        shifted = anchorfree_detection_loss(dict(logits, box6=clipped), target, self.point, right_offset_mode="signed")
        self.assertAlmostEqual(float(correct["loss_pair_iou"]), 0., places=5)
        self.assertAlmostEqual(float(shifted["loss_pair_iou"]), 6. / 11., places=5)

    def test_each_candidate_stage_counts_left_and_same_candidate_joint_hits(self):
        from scripts.diagnose_pred_roi import audit_frame

        wide_gt = torch.tensor([[16., 16., 36., 28., 22., 30.]])
        points = self.point.repeat(2, 1)
        wrong_right = torch.tensor([[16., 16., 36., 28., 7., 15.]])
        enc = encode_anchorfree_boxes6(torch.cat((wrong_right, wide_gt)), points)
        dense = SimpleNamespace(box6=enc[None], objectness=torch.tensor([[4., -4.]]),
                                class_logits=torch.zeros(1, 2, 1))
        result = audit_frame(dense, points, wide_gt, .25, 500, right_offset_mode="signed")
        self.assertEqual(result["stage_hits"]["raw"], {"left": 1, "joint": 1})
        for stage in ("score", "topk", "nms"):
            self.assertEqual(result["stage_hits"][stage], {"left": 1, "joint": 0})
        self.assertEqual(result["formal_paired"], 0)

    def test_finite_logit_that_decodes_to_infinity_is_excluded(self):
        overflow = self.encoded.clone()
        overflow[:, 4] = 1000.
        preds = {"objectness": torch.ones(1), "class_logits": torch.zeros(1, 1), "box6": overflow}
        pairs = decode_anchorfree_predictions(preds, self.point, right_offset_mode="signed")
        self.assertEqual(pairs.boxes6.shape, (0, 6))
