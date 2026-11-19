"""No-redundant-decode contract for the anchor-free inference branch."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from torch import nn
PROJECT_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT_ROOT/"src"))
from lightbackbone_stereo_yolo.detection import DenseStereoPredictions
from lightbackbone_stereo_yolo.model import StereoYoloLocalModel


class NoRedundantDecodeTest(unittest.TestCase):
    def test_anchorfree_variant_never_generates_baseline_anchors(self):
        class Encoder(nn.Module):
            def forward(self,left,right):
                z=lambda h,w: torch.randn(1,4,h,w)
                return SimpleNamespace(p2_left=z(16,16),p2_right=z(16,16),p3_left=z(8,8),p3_right=z(8,8),p4_left=z(4,4),p4_right=z(4,4),c5_left=z(2,2),c5_right=z(2,2))
        class Detector(nn.Module):
            def forward(self,pyramid):
                return DenseStereoPredictions(torch.full((1,84),-20.),torch.zeros(1,84,2),torch.zeros(1,84,6))
        model=StereoYoloLocalModel(Encoder(),Detector(),nn.Identity(),detector_variant="stereo_yolo26_pan")
        with patch("lightbackbone_stereo_yolo.model.generate_grid_anchors", side_effect=AssertionError("baseline decode ran")):
            output=model(torch.randn(1,3,64,64),torch.randn(1,3,64,64))
        self.assertEqual(output.boxes_left.shape,(0,4))


if __name__=="__main__":
    unittest.main()

