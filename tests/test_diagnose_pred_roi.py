"""CLI threshold parsing contract for Pred-RoI diagnosis."""
import importlib.util
import unittest
from pathlib import Path
import torch
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("diagnose",ROOT/"scripts"/"diagnose_pred_roi.py")
diagnose=importlib.util.module_from_spec(spec); spec.loader.exec_module(diagnose)


class ThresholdParsingTest(unittest.TestCase):
    def test_thresholds_include_primary_score_and_are_sorted_unique(self):
        self.assertEqual(diagnose.parse_thresholds("0.25,0.01,0.25,0.10"),[0.01,0.1,0.25])



    def test_topk_parser_accepts_zero_as_no_truncation(self):
        self.assertEqual(diagnose.parse_topks("500,1000,0"), [500, 1000, 0])

    def test_sweep_settings_cover_each_threshold_and_topk_pair(self):
        self.assertEqual(
            diagnose.sweep_settings([0.05, 0.25], [500, 0]),
            [(0.05, 500), (0.05, 0), (0.25, 500), (0.25, 0)],
        )

    def test_zero_topk_keeps_all_score_filtered_candidates(self):
        dense = SimpleNamespace(
            objectness=torch.zeros(1, 1, 1, 2),
            class_logits=torch.zeros(1, 1, 1, 2),
            box6=torch.zeros(1, 6, 1, 2),
        )
        _, _, stages = diagnose._decode_anchorfree(
            dense, torch.tensor([[4., 4., 8.], [12., 4., 8.]]), .1, 0
        )
        self.assertEqual(stages["score_filtered"], 2)
        self.assertEqual(stages["pre_nms_topk"], 2)
        self.assertFalse(stages["topk_saturated"])
if __name__=="__main__":
    unittest.main()
