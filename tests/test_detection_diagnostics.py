"""Left-first versus joint paired-detection diagnostic contracts."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from lightbackbone_stereo_yolo.training_metrics import analyze_paired_detections


class DetectionDiagnosticTest(unittest.TestCase):
    def test_left_hit_with_wrong_right_box_is_separated_from_formal_pair_coverage(self):
        targets=torch.tensor([[0.,0.,20.,20.,-10.,10.]])
        predictions=torch.tensor([[0.,0.,20.,20.,20.,40.]])
        result=analyze_paired_detections(predictions,targets,torch.tensor([.9]))
        self.assertEqual(result["left_matched_count"],1)
        self.assertEqual(result["right_pass_given_left_count"],0)
        self.assertEqual(result["paired_matched_count"],0)
        self.assertEqual(result["dR_error_signed"],[-30.0])
        self.assertEqual(result["wR_error_signed"],[0.0])

    def test_empty_predictions_have_zero_counts(self):
        result=analyze_paired_detections(torch.empty(0,6),torch.tensor([[0.,0.,20.,20.,-10.,10.]]),torch.empty(0))
        self.assertEqual(result["left_matched_count"],0)
        self.assertEqual(result["paired_matched_count"],0)



    def test_left_miss_audit_separates_nms_suppression(self):
        from lightbackbone_stereo_yolo.training_metrics import audit_left_misses
        target = torch.tensor([[0., 0., 10., 10., -5., 5.]])
        result = audit_left_misses(target, torch.tensor([[0., 0., 10., 10., -5., 5.]]), torch.tensor([.1]), torch.empty(0, 6), torch.empty(0))
        self.assertEqual(result["nms_suppressed"], 1)
        self.assertEqual(result["below_reference_score"], 0)
        self.assertEqual(result["no_pre_nms_left_match"], 0)
if __name__=="__main__":
    unittest.main()
