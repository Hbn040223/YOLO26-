"""Local 2D regularization contracts."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
import torch
from torch import nn
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
try:
    from lightbackbone_stereo_yolo.local_aggregation import AttentionPyramid, LocalAggregation
except ImportError:
    AttentionPyramid = LocalAggregation = None


class LocalAggregationContractTest(unittest.TestCase):
    def test_local_unet_returns_finite_logits_and_zeros_invalid_spatial_pixels(self) -> None:
        self.assertIsNotNone(LocalAggregation)
        attention = AttentionPyramid(
            p2=torch.randn(2, 8, 8, 12), p3=torch.randn(2, 8, 4, 6), p4=torch.randn(2, 8, 2, 3)
        )
        valid = torch.ones(2, 1, 8, 12, dtype=torch.bool)
        valid[0, :, :, 10:] = False
        model = LocalAggregation(dlocal=5, p2_channels=8, p3_channels=8, p4_channels=8, hidden_channels=8)
        refined = model(torch.randn(2, 5, 8, 12), attention, valid)
        self.assertEqual(refined.shape, (2, 5, 8, 12))
        self.assertTrue(torch.isfinite(refined).all())
        self.assertTrue((refined[0, :, :, 10:] == 0).all())
        self.assertFalse(any(isinstance(module, nn.BatchNorm2d) for module in model.modules()))


if __name__ == "__main__":
    unittest.main()
