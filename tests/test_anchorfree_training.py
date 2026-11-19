"""Training integration contracts for the Stereo-YOLO26 anchor-free variant."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from lightbackbone_stereo_yolo.anchorfree_training import (
        build_anchorfree_detection_targets,
        training_forward_anchorfree,
    )
    from lightbackbone_stereo_yolo.model import StereoLocalInstanceHead
    from lightbackbone_stereo_yolo.stereo_yolo26 import StereoYOLO26PanHead
except ImportError:
    build_anchorfree_detection_targets = training_forward_anchorfree = None
    StereoLocalInstanceHead = StereoYOLO26PanHead = None


class AnchorFreeTrainingContractTest(unittest.TestCase):
    def test_anchorfree_targets_keep_forced_fallback_finite_and_box_invalid(self) -> None:
        self.assertIsNotNone(build_anchorfree_detection_targets)
        locations = torch.tensor([[4., 4., 8.], [12., 4., 8.]])
        boxes6 = torch.tensor([[5., 1., 11., 7., -15., -9.]])
        targets = build_anchorfree_detection_targets(locations, boxes6, torch.tensor([1]))
        self.assertTrue((targets["labels"] == 1).any())
        self.assertTrue(torch.isfinite(targets["box6"]).all())
        self.assertFalse(targets["box_valid_mask"][targets["labels"] == 1].any())

    def test_anchorfree_gt_roi_forward_produces_finite_losses_and_diagnostics(self) -> None:
        self.assertIsNotNone(training_forward_anchorfree)

        class TinyEncoder(nn.Module):
            def forward(self, left, right):
                base = left[:, :1] + right[:, :1]
                p2 = base.repeat(1, 4, 1, 1)
                return SimpleNamespace(
                    p2_left=p2, p2_right=p2,
                    p3_left=torch.zeros(1, 4, 8, 8), p3_right=torch.zeros(1, 4, 8, 8),
                    p4_left=torch.zeros(1, 4, 4, 4), p4_right=torch.zeros(1, 4, 4, 4),
                    c5_left=torch.zeros(1, 4, 2, 2), c5_right=torch.zeros(1, 4, 2, 2),
                )

        encoder = TinyEncoder()
        detector = StereoYOLO26PanHead(4, 2)
        instance_head = StereoLocalInstanceHead(4, dmax_roi=2, hidden_channels=4)
        left = torch.randn(1, 3, 64, 64)
        right = torch.randn(1, 3, 64, 64)
        masks = torch.zeros(1, 1, 64, 64, dtype=torch.bool)
        masks[:, :, 8:32, 8:40] = True
        target = {
            "boxes6": torch.tensor([[8., 8., 40., 32., 2., 34.]]),
            "classes": torch.tensor([1]),
            "masks": masks,
            "disparity": torch.full((1, 64, 64), 4.),
            "disparity_valid": torch.ones(1, 64, 64, dtype=torch.bool),
        }
        output = training_forward_anchorfree(
            encoder, detector, instance_head, left, right, target, dmax_roi=2, collect_metrics=True,
        )
        for key in ("loss_obj", "loss_cls", "loss_box6", "loss_left", "loss_disparity", "loss_right_width", "loss_pair_iou", "loss_total"):
            self.assertTrue(torch.isfinite(output[key]), key)
        output["loss_total"].backward()



    def test_detector_only_weights_skip_instance_head(self) -> None:
        self.assertIsNotNone(training_forward_anchorfree)
        class TinyEncoder(nn.Module):
            def forward(self, left, right):
                p2 = left[:, :1].repeat(1, 4, 1, 1)
                return SimpleNamespace(p2_left=p2, p2_right=p2,
                    p3_left=torch.zeros(1, 4, 8, 8), p3_right=torch.zeros(1, 4, 8, 8),
                    p4_left=torch.zeros(1, 4, 4, 4), p4_right=torch.zeros(1, 4, 4, 4),
                    c5_left=torch.zeros(1, 4, 2, 2), c5_right=torch.zeros(1, 4, 2, 2))
        class ForbiddenInstanceHead(nn.Module):
            def forward(self, *args, **kwargs):
                raise AssertionError("detector-only training must not invoke the instance head")
        target = {"boxes6": torch.tensor([[8., 8., 40., 32., 2., 34.]]), "classes": torch.tensor([1])}
        output = training_forward_anchorfree(TinyEncoder(), StereoYOLO26PanHead(4, 2), ForbiddenInstanceHead(),
            torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64), target, dmax_roi=2,
            mask_weight=0., disparity_weight=0.)
        self.assertEqual(float(output["loss_mask"]), 0.)
        self.assertEqual(float(output["loss_disp"]), 0.)
if __name__ == "__main__":
    unittest.main()

