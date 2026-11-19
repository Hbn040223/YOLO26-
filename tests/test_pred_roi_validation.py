"""Pred-RoI validation helper contracts."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
PROJECT_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT_ROOT/"src"))
try:
    from lightbackbone_stereo_yolo.training_metrics import evaluate_pred_roi_frame
    from lightbackbone_stereo_yolo.detection import DenseStereoPredictions
    from lightbackbone_stereo_yolo.geometry import encode_anchorfree_boxes6
    from lightbackbone_stereo_yolo.model import StereoLocalInstanceHead
except ImportError:
    evaluate_pred_roi_frame=DenseStereoPredictions=encode_anchorfree_boxes6=StereoLocalInstanceHead=None


class PredRoiValidationTest(unittest.TestCase):
    def test_empty_predictions_report_zero_coverage_without_roi_work(self):
        self.assertIsNotNone(evaluate_pred_roi_frame)
        class Encoder(nn.Module):
            def forward(self,left,right):
                z=lambda h,w: torch.zeros(1,4,h,w)
                return SimpleNamespace(p2_left=z(16,16),p2_right=z(16,16),p3_left=z(8,8),p3_right=z(8,8),p4_left=z(4,4),p4_right=z(4,4),c5_left=z(2,2),c5_right=z(2,2))
        class Detector(nn.Module):
            def forward(self,pyramid):
                return DenseStereoPredictions(torch.full((1,84),-20.),torch.zeros(1,84,2),torch.zeros(1,84,6))
        target={"boxes6":torch.tensor([[8.,8.,40.,32.,2.,34.]]),"classes":torch.tensor([1]),"masks":torch.ones(1,1,64,64,dtype=torch.bool),"disparity":torch.ones(1,64,64),"disparity_valid":torch.ones(1,64,64,dtype=torch.bool)}
        result=evaluate_pred_roi_frame(Encoder(),Detector(),nn.Identity(),torch.randn(1,3,64,64),torch.randn(1,3,64,64),target,2,detector_variant="stereo_yolo26_pan",score_threshold=.5)
        self.assertEqual(result["matched_count"],0)
        self.assertEqual(result["target_count"],1)
        self.assertEqual(result["coverage"],0.)


    def test_matched_second_target_reuses_frame_disparity_map(self):
        """A GT index selects an instance mask, never a frame-level disparity map."""
        self.assertIsNotNone(StereoLocalInstanceHead)
        class Encoder(nn.Module):
            def forward(self,left,right):
                z=lambda h,w: torch.zeros(1,4,h,w)
                return SimpleNamespace(p2_left=z(16,16),p2_right=z(16,16),p3_left=z(8,8),p3_right=z(8,8),p4_left=z(4,4),p4_right=z(4,4),c5_left=z(2,2),c5_right=z(2,2))
        class Detector(nn.Module):
            def forward(self,pyramid):
                logits=torch.full((1,84),-20.); logits[0,0]=20.
                classes=torch.zeros(1,84,2); classes[0,0,1]=4.
                boxes=torch.zeros(1,84,6)
                target_pair=torch.tensor([[0.,0.,8.,8.,0.,8.]])
                boxes[0,0]=encode_anchorfree_boxes6(target_pair,torch.tensor([[4.,4.,8.]]))[0]
                return DenseStereoPredictions(logits,classes,boxes)
        masks=torch.zeros(2,1,64,64,dtype=torch.bool)
        masks[0,:,20:29,20:29]=True
        masks[1,:,:9,:9]=True
        target={"boxes6":torch.tensor([[20.,20.,28.,28.,20.,28.],[0.,0.,8.,8.,0.,8.]]),"classes":torch.tensor([0,1]),"masks":masks,"disparity":torch.ones(1,64,64),"disparity_valid":torch.ones(1,64,64,dtype=torch.bool)}
        result=evaluate_pred_roi_frame(Encoder(),Detector(),StereoLocalInstanceHead(4,2),torch.randn(1,3,64,64),torch.randn(1,3,64,64),target,2,detector_variant="stereo_yolo26_pan",score_threshold=.5)
        self.assertEqual(result["matched_count"],1)
        self.assertEqual(result["target_count"],2)
if __name__=="__main__":
    unittest.main()

