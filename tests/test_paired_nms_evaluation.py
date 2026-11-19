"""Regression tests for deterministic, pair-preserving NMS and its on/off audit."""
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

from evaluate_paired_nms import evaluate
from lightbackbone_stereo_yolo.detection import anchorfree_candidate_indices, paired_nms
from lightbackbone_stereo_yolo.training_metrics import greedy_match_paired_detections


class PairedNmsEvaluationTest(unittest.TestCase):
    def test_stable_class_aware_nms_preserves_single_view_overlap(self):
        boxes = torch.tensor([[0., 0., 4., 4., 0., 4.],
                              [0., 0., 4., 4., 0., 4.],
                              [0., 0., 4., 4., 0., 4.],
                              [0., 0., 4., 4., 8., 12.],
                              [0., 0., 4., 4., 0., 2.]])
        scores = torch.full((5,), .5)
        classes = torch.tensor([0, 0, 1, 0, 0])
        kept = paired_nms(boxes, scores, .5, classes=classes)
        # Row 4 is exactly at the threshold, and equality survives.
        self.assertEqual(kept.indices.tolist(), [0, 2, 3, 4])
        torch.testing.assert_close(kept.boxes6, boxes[kept.indices])
        torch.testing.assert_close(kept.classes, classes[kept.indices])
        for count in (0, 1):
            self.assertEqual(paired_nms(boxes[:count], scores[:count]).indices.tolist(), list(range(count)))
        _, topk = anchorfree_candidate_indices(scores, .5, 2)
        self.assertEqual(topk.tolist(), [0, 1])

    def test_paired_match_cannot_mix_ground_truth_views_or_classes(self):
        gt = torch.tensor([[0., 0., 4., 4., 0., 4.], [8., 0., 12., 4., 8., 12.]])
        mixed = torch.tensor([[0., 0., 4., 4., 8., 12.]])
        self.assertEqual(greedy_match_paired_detections(mixed, gt, torch.tensor([.9])).matched_count, 0)
        self.assertEqual(greedy_match_paired_detections(gt, gt, torch.tensor([.9, .8])).matched_count, 2)
        wrong_class = greedy_match_paired_detections(
            gt, gt, torch.tensor([.9, .8]),
            prediction_classes=torch.tensor([1, 0]), target_classes=torch.tensor([0, 1]),
        )
        self.assertEqual(wrong_class.matched_count, 0)

    def test_same_candidate_pool_is_used_for_every_iou(self):
        box = [0., 0., 4., 4., 0., 4.]
        cache = {"schema": "anchorfree-pre-score-v1", "frames": [
            {"frame_id": "a", "boxes6": [None, box, box, box], "scores": [.99, .7, .7, .15],
             "classes": [0, 0, 0, 1], "gt_boxes6": [box], "gt_classes": [0]},
            {"frame_id": "b", "boxes6": [], "scores": [], "classes": [], "gt_boxes6": [], "gt_classes": []},
        ]}
        report = evaluate(cache, [.25], [.5], 3, .5, bootstrap=20)
        before, after = report["settings"]
        self.assertEqual(before["frames"][0]["pre_nms_dense_ids"], [1, 2])
        self.assertEqual(after["frames"][0]["pre_nms_dense_ids"], [1, 2])
        self.assertEqual(after["frames"][0]["kept_dense_ids"], [1])
        self.assertEqual((before["pairs"], before["hits"], before["duplicates"]), (2, 1, 1))
        self.assertEqual((after["pairs"], after["hits"], after["duplicates"]), (1, 1, 0))
        self.assertEqual(after["lost_gt_count"], 0)
        self.assertEqual(sum(report["score_histogram"]["eligible_for_same_gt"]), 2)
        self.assertIn("coverage", after["paired_bootstrap_delta_95ci"])


if __name__ == "__main__":
    unittest.main()
