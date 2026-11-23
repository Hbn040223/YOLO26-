"""Mask-only training must update only the Mask and use the published metric."""
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from lightbackbone_stereo_yolo.model import StereoLocalInstanceHead
from lightbackbone_stereo_yolo.training import (
    StereoYoloTrainingModel, freeze_for_mask_only, training_forward_mask_only,
)
from lightbackbone_stereo_yolo.training_metrics import mask_pixel_counts, summarize_mask_counts


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 4, stride=4)
        self.bn = nn.BatchNorm2d(4)

    def forward(self, left, right):
        return SimpleNamespace(p2_left=self.bn(self.conv(left)))


class BypassedDetector(nn.Module):
    def forward(self, *_):
        raise AssertionError("Mask-only must bypass the detector")


class MaskOnlyTest(unittest.TestCase):
    def test_train_step_updates_only_mask_with_mapped_car_target(self):
        torch.manual_seed(17)
        model = StereoYoloTrainingModel(TinyEncoder(), BypassedDetector(), StereoLocalInstanceHead(4, 1))
        model.train()
        freeze_for_mask_only(model)
        snapshot = {k: v.clone() for k, v in model.state_dict().items()}
        image = torch.rand(1, 3, 64, 64)
        masks = torch.zeros(2, 1, 64, 64, dtype=torch.bool)
        masks[1, :, 12:35, 18:45] = True
        target = {
            "masks": masks, "classes": torch.tensor([0, 1]),
            "boxes6": torch.tensor([[8., 8., 32., 32., 4., 28.], [16., 10., 46., 36., 12., 42.]]),
        }
        step = training_forward_mask_only(model, image, image, target, 1)
        self.assertEqual(tuple(step["target"].shape), (1, 1, 28, 28))
        self.assertGreater(int(step["target"].sum()), 0)
        step["loss"].backward()
        optimizer = torch.optim.SGD(model.instance_head.mask.parameters(), lr=.1)
        optimizer.step()
        for module in (model.encoder, model.detector, model.instance_head.idisp):
            self.assertFalse(module.training)
            self.assertTrue(all(p.grad is None for p in module.parameters()))
        changed = [k for k, v in model.state_dict().items() if not torch.equal(v, snapshot[k])]
        self.assertTrue(changed)
        self.assertTrue(all(k.startswith("instance_head.mask.") for k in changed))

    def test_mask_metrics_respect_validity_and_exclude_empty_union(self):
        truth = torch.tensor([[[1, 1, 0, 0]], [[0, 0, 0, 0]]], dtype=torch.bool)
        prediction = torch.tensor([[[1, 0, 1, 1]], [[0, 0, 0, 0]]], dtype=torch.bool)
        valid = torch.tensor([[[1, 1, 1, 0]], [[1, 1, 1, 0]]], dtype=torch.bool)
        counts = mask_pixel_counts(prediction, truth, valid)
        results = summarize_mask_counts(counts)
        self.assertEqual(results["instances"], 2)
        self.assertEqual(results["iou_instances"], 1)
        self.assertAlmostEqual(results["iou"], 1 / 3)
        self.assertEqual(results["precision"], .5)
        self.assertEqual(results["recall"], .5)


class ContextMaskContract(unittest.TestCase):
    def test_fixed_grid_context_head_runs_on_shared_left_roi(self):
        from lightbackbone_stereo_yolo.mask import ContextInstanceMaskHead
        head = StereoLocalInstanceHead(4, 1, roi_sampling="fixed_56", mask_head_variant="context")
        self.assertIsInstance(head.mask, ContextInstanceMaskHead)
        features = torch.rand(2, 4, 56, 56)
        result = head.mask(features)
        self.assertEqual(tuple(result.shape), (2, 1, 28, 28))
        result.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.mask.parameters()))


class ForegroundReproducibilityTest(unittest.TestCase):
    def test_instance_bce_dice_ignores_empty_instances(self):
        from lightbackbone_stereo_yolo.mask import mask_loss

        logits = torch.zeros(2, 1, 1, 2, requires_grad=True)
        truth = torch.tensor([[[[1., 0.]]], [[[0., 0.]]]])
        valid = torch.tensor([[[[True, True]]], [[[False, False]]]])
        loss = mask_loss(logits, truth, valid, dice_weight=1.)
        self.assertAlmostEqual(float(loss), .693147 + 1 / 3, places=6)
        loss.backward()
        self.assertTrue(torch.equal(logits.grad[1], torch.zeros_like(logits.grad[1])))

    def test_runtime_threshold_preserves_legacy_fold_before_resize(self):
        from math import log
        from lightbackbone_stereo_yolo.mask import foreground_disparity_validity

        logits = torch.tensor([[[[-.3, .05]]]])
        valid = torch.ones_like(logits, dtype=torch.bool)
        disparity = torch.ones((1, 2, 3))
        support = torch.tensor([True])
        expected = foreground_disparity_validity(
            logits - log(.4 / .6), valid, disparity, disparity.bool(), support)
        actual = foreground_disparity_validity(
            logits, valid, disparity, disparity.bool(), support, mask_threshold=.4)
        self.assertTrue(torch.equal(expected, actual))
        self.assertFalse(torch.equal(actual, foreground_disparity_validity(
            logits, valid, disparity, disparity.bool(), support, mask_threshold=.5)))

    def test_legacy_bundle_threshold_and_explicit_override(self):
        from scripts.infer import resolve_mask_threshold

        legacy = {"mask_evaluation": {"threshold_before_logit_calibration": .4}}
        self.assertEqual(resolve_mask_threshold(legacy), .5)
        self.assertAlmostEqual(resolve_mask_threshold(legacy, .4), .5)
        self.assertEqual(resolve_mask_threshold({"mask_threshold": .4}), .4)

    def test_development_override_is_recorded_without_passing_gate(self):
        from pathlib import Path
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from train_mask_only import development_gate

        strict = development_gate(.7383, .7677, .75, False)
        override = development_gate(.7383, .7677, .75, True)
        self.assertFalse(strict["eligible_for_evaluation"])
        self.assertTrue(override["eligible_for_evaluation"])
        self.assertFalse(override["dev_gate_passed"])
        self.assertTrue(override["gate_override"])
        self.assertEqual(override["dev_gt_iou"], .7383)
        self.assertFalse(development_gate(None, .80, .75, True)["eligible_for_evaluation"])
