"""Pred-RoI paired matching contracts."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
import torch
PROJECT_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT_ROOT/"src"))
try:
    from lightbackbone_stereo_yolo.training_metrics import greedy_match_paired_detections
except ImportError:
    greedy_match_paired_detections=None


class PairedMatchingTest(unittest.TestCase):
    def test_greedy_matching_uses_each_prediction_and_gt_at_most_once(self):
        self.assertIsNotNone(greedy_match_paired_detections)
        predictions=torch.tensor([[0.,0.,10.,10.,-5.,5.],[20.,0.,30.,10.,15.,25.]])
        targets=torch.tensor([[0.,0.,10.,10.,-5.,5.],[1.,0.,11.,10.,-4.,6.]])
        matches=greedy_match_paired_detections(predictions,targets,torch.tensor([.9,.8]),minimum_quality=.5)
        self.assertEqual(matches.prediction_indices.tolist(),[0])
        self.assertEqual(matches.target_indices.tolist(),[0])
        self.assertEqual(matches.matched_count,1)
        self.assertEqual(matches.target_count,2)



    def test_fixed_roi_metrics_use_common_instance_mask_pixels(self):
        from lightbackbone_stereo_yolo.training_metrics import fixed_roi_disparity_metrics
        gt = torch.tensor([[[10., 20.]]])
        pred = torch.tensor([[[11., 99.]]])
        target = torch.tensor([[[10., 20.]]])
        instance = torch.tensor([[[True, True]]])
        gt_crop = torch.tensor([[[True, True]]])
        pred_crop = torch.tensor([[[True, False]]])
        result = fixed_roi_disparity_metrics(gt, pred, target, instance, gt_crop, pred_crop)
        self.assertEqual(result["common_pixel_count"], 1)
        self.assertAlmostEqual(result["gt_epe"], 0.)
        self.assertAlmostEqual(result["pred_epe"], 1.)
        self.assertAlmostEqual(result["delta_epe"], 1.)
        self.assertAlmostEqual(result["mask_coverage"], .5)

    def test_fixed_roi_metric_marks_empty_common_crop(self):
        from lightbackbone_stereo_yolo.training_metrics import fixed_roi_disparity_metrics
        z = torch.ones(1, 1, 1)
        result = fixed_roi_disparity_metrics(z, z, z, torch.ones_like(z, dtype=torch.bool), torch.ones_like(z, dtype=torch.bool), torch.zeros_like(z, dtype=torch.bool))
        self.assertEqual(result["common_pixel_count"], 0)
        self.assertEqual(result["empty_instances"], 1)
        self.assertEqual(result["mask_coverage"], 0.)
if __name__=="__main__":
    unittest.main()

