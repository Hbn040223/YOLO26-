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
