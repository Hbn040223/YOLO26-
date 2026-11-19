"""RoI-bin Dmax calibration contracts."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lightbackbone_stereo_yolo.calibration import derive_global_dmax


class CalibrationContractTest(unittest.TestCase):
    def test_global_dmax_keeps_roi_bin_quantile_and_uses_max_instance_margin(self) -> None:
        result = derive_global_dmax(
            torch.tensor([1., 2., 7.]),
            torch.tensor([.25, .5]),
            quantile=1.,
            margin_img_px=4.,
        )
        self.assertEqual(result.roi_residual_q_bins, 7.)
        self.assertEqual(result.margin_roi_global_bins, 2.)
        self.assertEqual(result.dmax_roi, 9)


if __name__ == "__main__":
    unittest.main()
