"""Anchor-free inference assembly contract."""
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
    from lightbackbone_stereo_yolo.detection import DenseStereoPredictions
    from lightbackbone_stereo_yolo.model import StereoLocalInstanceHead, StereoYoloLocalModel
except ImportError:
    DenseStereoPredictions=StereoLocalInstanceHead=StereoYoloLocalModel=None


class AnchorFreeInferenceTest(unittest.TestCase):
    def test_anchorfree_variant_decodes_a_pair_then_runs_existing_roi_heads(self):
        self.assertIsNotNone(StereoYoloLocalModel)
        class Encoder(nn.Module):
            def forward(self,left,right):
                return SimpleNamespace(
                    p2_left=torch.randn(1,4,16,16),p2_right=torch.randn(1,4,16,16),
                    p3_left=torch.randn(1,4,8,8),p3_right=torch.randn(1,4,8,8),
                    p4_left=torch.randn(1,4,4,4),p4_right=torch.randn(1,4,4,4),
                    c5_left=torch.randn(1,4,2,2),c5_right=torch.randn(1,4,2,2))
        class Detector(nn.Module):
            def forward(self,pyramid):
                count=84
                logits=torch.full((1,count),-20.); logits[0,0]=20.
                classes=torch.zeros(1,count,2); classes[0,0,1]=4.
                return DenseStereoPredictions(logits,classes,torch.zeros(1,count,6))
        model=StereoYoloLocalModel(Encoder(),Detector(),StereoLocalInstanceHead(4,1),score_threshold=.5,detector_variant="stereo_yolo26_pan")
        output=model(torch.randn(1,3,64,64),torch.randn(1,3,64,64))
        self.assertEqual(output.boxes_left.shape,(1,4))
        self.assertEqual(output.classes.tolist(),[1])
        self.assertEqual(output.disparity.shape[0],1)


if __name__=="__main__":
    unittest.main()

