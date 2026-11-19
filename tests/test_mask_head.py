"""Left-only instance-mask contracts."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
import torch
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
try:
    from lightbackbone_stereo_yolo.mask import LightInstanceMaskHead, mask_loss
except ImportError:
    LightInstanceMaskHead = mask_loss = None


class MaskHeadContractTest(unittest.TestCase):
    def test_left_roi_produces_one_28_square_logit_mask(self) -> None:
        self.assertIsNotNone(LightInstanceMaskHead)
        logits = LightInstanceMaskHead(8)(torch.randn(3, 8, 6, 11))
        self.assertEqual(logits.shape, (3, 1, 28, 28))

    def test_invalid_pixels_do_not_contribute_to_mask_loss(self) -> None:
        self.assertIsNotNone(mask_loss)
        logits = torch.zeros(1, 1, 2, 2, requires_grad=True)
        target = torch.tensor([[[[1., 0.], [1., 0.]]]])
        valid = torch.tensor([[[[1., 1.], [0., 0.]]]])
        self.assertAlmostEqual(mask_loss(logits, target, valid).item(), 0.693147, places=5)


if __name__ == "__main__":
    unittest.main()
