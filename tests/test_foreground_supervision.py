"""Foreground iDisp supervision contracts."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lightbackbone_stereo_yolo.roi import CropMetadata
from lightbackbone_stereo_yolo.training import build_instance_supervision, instance_metrics


class ForegroundSupervisionContractTest(unittest.TestCase):
    @staticmethod
    def _instance_output(validity: torch.Tensor | None = None) -> SimpleNamespace:
        metadata = CropMetadata(
            c_l=torch.tensor([0.]), c_r=torch.tensor([0.]), c=torch.tensor([8.]),
            y0=torch.tensor([0.]), h_crop=torch.tensor([8.]),
            w_roi=torch.tensor([2]), h_roi=torch.tensor([2]),
        )
        if validity is None:
            validity = torch.ones(1, 1, 2, 2, dtype=torch.bool)
        return SimpleNamespace(
            crop=metadata,
            validity=validity,
            mask_validity=torch.ones(1, 1, 28, 28, dtype=torch.bool),
            disparity=torch.zeros(1, 2, 2),
            mask_logits=torch.zeros(1, 28, 28),
        )

    @staticmethod
    def _foreground_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        boxes6 = torch.tensor([[0., 0., 8., 8., 0., 8.]])
        masks = torch.zeros(1, 1, 8, 8, dtype=torch.bool)
        masks[:, :, :4, :4] = True
        disparity = torch.full((1, 1, 8, 8), 4.)
        valid = torch.ones_like(disparity, dtype=torch.bool)
        return boxes6, masks, disparity, valid

    def test_foreground_and_all_roi_switch_loss_and_metric_validity_together(self) -> None:
        boxes6, masks, disparity, valid = self._foreground_inputs()
        output = self._instance_output()
        foreground = build_instance_supervision(output, boxes6, masks, disparity, valid, 8, supervision_mode="foreground")
        all_roi = build_instance_supervision(output, boxes6, masks, disparity, valid, 8, supervision_mode="all_roi")
        self.assertEqual(int(foreground.disparity_valid.sum()), 1)
        self.assertEqual(int(all_roi.disparity_valid.sum()), 4)
        self.assertEqual(instance_metrics(output, foreground, masks, 8)["epe_count"], 1)
        self.assertEqual(instance_metrics(output, all_roi, masks, 8)["epe_count"], 4)

    def test_valid_disparity_must_be_contained_by_instance_output_validity(self) -> None:
        boxes6, masks, disparity, valid = self._foreground_inputs()
        invalid_output = self._instance_output(torch.zeros(1, 1, 2, 2, dtype=torch.bool))
        with self.assertRaisesRegex(RuntimeError, "validity"):
            build_instance_supervision(
                invalid_output, boxes6, masks, disparity, valid, 8, supervision_mode="foreground"
            )


if __name__ == "__main__":
    unittest.main()
