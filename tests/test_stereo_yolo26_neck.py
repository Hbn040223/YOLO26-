"""PAN-only neck contracts for the Stereo-YOLO26 detector."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from lightbackbone_stereo_yolo.stereo_yolo26 import StereoYOLO26PanNeck
except ImportError:
    StereoYOLO26PanNeck = None


class StereoYOLO26PanNeckTest(unittest.TestCase):
    def test_pan_neck_consumes_only_detection_scales_and_preserves_ceil_shapes(self) -> None:
        self.assertIsNotNone(StereoYOLO26PanNeck)
        pyramid = SimpleNamespace(
            p3_left=torch.randn(2, 8, 8, 10),
            p3_right=torch.randn(2, 8, 8, 10),
            p4_left=torch.randn(2, 8, 4, 5),
            p4_right=torch.randn(2, 8, 4, 5),
            c5_left=torch.randn(2, 8, 2, 3),
            c5_right=torch.randn(2, 8, 2, 3),
        )
        n3, n4, n5 = StereoYOLO26PanNeck(8)(pyramid)
        self.assertEqual(n3.shape, (2, 8, 8, 10))
        self.assertEqual(n4.shape, (2, 8, 4, 5))
        self.assertEqual(n5.shape, (2, 8, 2, 3))


if __name__ == "__main__":
    unittest.main()

