"""Decoupled Stereo-YOLO26 head contracts."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from lightbackbone_stereo_yolo.stereo_yolo26 import StereoYOLO26PanHead
except ImportError:
    StereoYOLO26PanHead = None


class StereoYOLO26PanHeadTest(unittest.TestCase):
    def test_decoupled_head_outputs_flattened_p3_p4_p5_predictions_with_gradients(self) -> None:
        self.assertIsNotNone(StereoYOLO26PanHead)
        pyramid = SimpleNamespace(
            p3_left=torch.randn(2, 8, 8, 10, requires_grad=True),
            p3_right=torch.randn(2, 8, 8, 10, requires_grad=True),
            p4_left=torch.randn(2, 8, 4, 5, requires_grad=True),
            p4_right=torch.randn(2, 8, 4, 5, requires_grad=True),
            c5_left=torch.randn(2, 8, 2, 3, requires_grad=True),
            c5_right=torch.randn(2, 8, 2, 3, requires_grad=True),
        )
        head = StereoYOLO26PanHead(8, num_classes=3)
        output = head(pyramid)
        self.assertEqual(output.objectness.shape, (2, 106))
        self.assertEqual(output.class_logits.shape, (2, 106, 3))
        self.assertEqual(output.box6.shape, (2, 106, 6))
        loss = output.objectness.square().mean() + output.class_logits.square().mean() + output.box6.square().mean()
        loss.backward()
        self.assertTrue(all(parameter.grad is not None for parameter in head.parameters()))
        self.assertTrue(torch.isfinite(pyramid.p3_left.grad).all())


if __name__ == "__main__":
    unittest.main()

