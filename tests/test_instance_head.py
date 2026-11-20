"""Instance-level mask and regularized-iDisp integration contract."""
from __future__ import annotations
import sys
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
try:
    from lightbackbone_stereo_yolo.detection import DenseStereoPredictions
    from lightbackbone_stereo_yolo.model import StereoLocalInstanceHead, StereoYoloLocalModel
    from lightbackbone_stereo_yolo import roi
except ImportError:
    DenseStereoPredictions = StereoLocalInstanceHead = StereoYoloLocalModel = roi = None


class InstanceHeadContractTest(unittest.TestCase):
    def _encoder(self):
        class Encoder(nn.Module):
            def forward(self, left, right):
                return SimpleNamespace(
                    p2_left=torch.randn(1, 4, 16, 16), p2_right=torch.randn(1, 4, 16, 16),
                    p3_left=torch.randn(1, 4, 8, 8), p3_right=torch.randn(1, 4, 8, 8),
                    p4_left=torch.randn(1, 4, 4, 4), p4_right=torch.randn(1, 4, 4, 4),
                    c5_left=torch.randn(1, 4, 2, 2), c5_right=torch.randn(1, 4, 2, 2),
                )
        return Encoder()

    def test_end_to_end_empty_detection_short_circuits_instance_branches(self) -> None:
        self.assertIsNotNone(StereoYoloLocalModel)
        class EmptyDetector(nn.Module):
            def forward(self, pyramid):
                count = 8 * 8 + 4 * 4 + 2 * 2
                return DenseStereoPredictions(torch.full((1, count), -20.), torch.zeros(1, count, 2), torch.zeros(1, count, 6))
        model = StereoYoloLocalModel(self._encoder(), EmptyDetector(), StereoLocalInstanceHead(4, 1), score_threshold=.5)
        output = model(torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64))
        self.assertEqual(output.boxes_left.shape, (0, 4))
        self.assertEqual(output.disparity.shape, (0, 0, 0))
        self.assertEqual(output.pair_ids.shape, (0,))
        self.assertEqual(output.pair_ids.dtype, torch.long)
        self.assertEqual(output.foreground_validity.shape, output.disparity.shape)

    def test_end_to_end_candidate_reaches_mask_and_disparity_outputs(self) -> None:
        self.assertIsNotNone(StereoYoloLocalModel)
        class OneDetector(nn.Module):
            def forward(self, pyramid):
                count = 8 * 8 + 4 * 4 + 2 * 2
                logits = torch.full((1, count), -20.)
                logits[0, 0] = 20.
                return DenseStereoPredictions(logits, torch.zeros(1, count, 2), torch.zeros(1, count, 6))
        model = StereoYoloLocalModel(self._encoder(), OneDetector(), StereoLocalInstanceHead(4, 1), score_threshold=.5)
        output = model(torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64))
        self.assertEqual(output.boxes_left.shape, (1, 4))
        self.assertEqual(output.mask_logits.shape, (1, 1, 28, 28))
        self.assertEqual(output.mask_validity.shape, (1, 1, 28, 28))
        self.assertEqual(output.disparity.shape[0], 1)
        self.assertEqual(output.pair_ids.tolist(), [0])
        self.assertFalse(output.foreground_validity.any())

    def test_instance_head_returns_left_mask_and_image_disparity_only(self) -> None:
        self.assertIsNotNone(StereoLocalInstanceHead)
        pyramid = SimpleNamespace(
            p2_left=torch.randn(1, 4, 16, 16), p2_right=torch.randn(1, 4, 16, 16),
            p3_left=torch.randn(1, 4, 8, 8), p4_left=torch.randn(1, 4, 4, 4),
        )
        output = StereoLocalInstanceHead(channels=4, dmax_roi=1)(
            pyramid, torch.tensor([[8., 8., 40., 32., 2., 34.]]), image_height=64, image_width=64
        )
        self.assertEqual(output.mask_logits.shape, (1, 1, 28, 28))
        self.assertEqual(output.mask_validity.shape, (1, 1, 28, 28))
        self.assertEqual(output.disparity.shape, (1, 14, 16))
        self.assertFalse(hasattr(output, "depth"))
        self.assertFalse(hasattr(output, "boxes3d"))

    def test_instance_head_performs_only_four_real_roi_samples(self) -> None:
        """Would fail if left attention reuses the paired sampler and discards fake right crops."""
        self.assertIsNotNone(roi)
        pyramid = SimpleNamespace(
            p2_left=torch.randn(1, 4, 16, 16), p2_right=torch.randn(1, 4, 16, 16),
            p3_left=torch.randn(1, 4, 8, 8), p4_left=torch.randn(1, 4, 4, 4),
        )
        head = StereoLocalInstanceHead(channels=4, dmax_roi=1, hidden_channels=4)
        with patch("lightbackbone_stereo_yolo.roi.sample_roi_feature", wraps=roi.sample_roi_feature) as sampled:
            head(pyramid, torch.tensor([[8., 8., 40., 32., 2., 34.]]), image_height=64, image_width=64)
        self.assertEqual(sampled.call_count, 4)


if __name__ == "__main__":
    unittest.main()
