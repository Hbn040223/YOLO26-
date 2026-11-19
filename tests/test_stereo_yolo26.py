"""TDD contracts for the LightStereo-backed Stereo-YOLO26 detector."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from lightbackbone_stereo_yolo.stereo_yolo26 import SPPFLite, StereoFusion
except ModuleNotFoundError:
    SPPFLite = StereoFusion = None


class StereoYOLO26PrimitiveTest(unittest.TestCase):
    def test_stereo_fusion_preserves_shape_and_uses_right_view(self) -> None:
        self.assertIsNotNone(StereoFusion)
        torch.manual_seed(7)
        left = torch.randn(2, 8, 4, 5)
        right = torch.randn(2, 8, 4, 5)
        fusion = StereoFusion(8)
        self.assertEqual(fusion(left, right).shape, left.shape)
        self.assertFalse(torch.equal(fusion(left, right), fusion(left, right + 1.0)))

    def test_sppf_lite_preserves_nchw_shape(self) -> None:
        self.assertIsNotNone(SPPFLite)
        deep = torch.randn(2, 8, 2, 3)
        self.assertEqual(SPPFLite(8)(deep).shape, deep.shape)


if __name__ == "__main__":
    unittest.main()

