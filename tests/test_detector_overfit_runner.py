"""Contract tests for the fixed-manifest detector-overfit runner."""
from __future__ import annotations

import importlib.util
import unittest
import torch
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "detector_overfit", ROOT / "scripts" / "train_detector_overfit.py"
)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class DetectorOverfitCheckpointTest(unittest.TestCase):
    def test_checkpoint_selection_requires_fixed_precision_floor_before_coverage(self):
        best = {"paired_coverage": 0.40, "paired_precision": 0.12, "pre_nms_left_recall": 0.50}
        lower_precision = {"paired_coverage": 0.90, "paired_precision": 0.08, "pre_nms_left_recall": 0.90}
        eligible = {"paired_coverage": 0.41, "paired_precision": 0.10, "pre_nms_left_recall": 0.51}
        self.assertFalse(runner.is_better_overfit_checkpoint(lower_precision, best, precision_floor=0.09))
        self.assertTrue(runner.is_better_overfit_checkpoint(eligible, best, precision_floor=0.09))

    def test_checkpoint_tie_breaks_on_pre_nms_left_recall(self):
        best = {"paired_coverage": 0.40, "paired_precision": 0.12, "pre_nms_left_recall": 0.50}
        candidate = {"paired_coverage": 0.40, "paired_precision": 0.10, "pre_nms_left_recall": 0.51}
        self.assertTrue(runner.is_better_overfit_checkpoint(candidate, best, precision_floor=0.09))


    def test_summary_uses_all_gt_and_all_post_nms_pairs(self):
        summary = runner.summarize_overfit_metrics([
            {"eligible_gt": 2, "raw_left_hits": 2, "pre_nms_left_hits": 1, "paired_hits": 1, "post_nms_pairs": 3},
            {"eligible_gt": 3, "raw_left_hits": 2, "pre_nms_left_hits": 2, "paired_hits": 2, "post_nms_pairs": 5},
        ])
        self.assertEqual(summary["eligible_gt"], 5)
        self.assertAlmostEqual(summary["raw_dense_left_recall"], 0.8)
        self.assertAlmostEqual(summary["pre_nms_left_recall"], 0.6)
        self.assertAlmostEqual(summary["paired_coverage"], 0.6)
        self.assertAlmostEqual(summary["paired_precision"], 3 / 8)
        self.assertEqual(summary["post_nms_pair_count_p50"], 4.0)

    def test_vectorized_left_hit_counts_each_gt_with_any_qualifying_candidate(self):
        iou = torch.tensor([[.7, .2], [.4, .8], [.1, .3]])
        self.assertEqual(runner.count_gt_left_hits(iou, threshold=.5), 2)
        self.assertEqual(runner.count_gt_left_hits(torch.empty(0, 2), threshold=.5), 0)
if __name__ == "__main__":
    unittest.main()
