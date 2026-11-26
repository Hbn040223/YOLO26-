import sys
from pathlib import Path
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_fixed_pair_features import (
    automatic_pair, correlation_maps, edge_band, left_right_consistency, region_result, valid_box,
)
from lightbackbone_stereo_yolo.local_disparity import LocalCostVolume


class FixedPairFeatureTest(unittest.TestCase):
    def test_four_pixel_band_includes_image_boundary(self):
        image = torch.ones((12, 12), dtype=torch.bool)
        band = edge_band(image)
        self.assertEqual(int(band.sum()), 12 * 12 - 4 * 4)
        self.assertFalse(band[4:8, 4:8].any())

    def test_correlation_masks_unavailable_bins(self):
        values = torch.tensor([[[[.9, 0.]], [[.5, .9]], [[.1, .8]]]])
        valid = torch.tensor([[[[True, False]], [[True, False]], [[False, False]]]])
        maps = correlation_maps(values, valid, .1)
        torch.testing.assert_close(maps["peak_margin"][0], torch.tensor([[[.4, 0.]]]))
        self.assertEqual(maps["entropy"][1].tolist(), [[[True, False]]])
        self.assertTrue(torch.isfinite(maps["entropy"][0]).all())

    def test_common_disparity_pixels_and_per_model_coverage(self):
        truth = torch.tensor([[10., 12.]])
        support = torch.ones_like(truth, dtype=torch.bool)
        first = (truth, support, support, {}, None)
        second = (truth + 1, torch.tensor([[True, False]]), support, {}, None)
        result = region_result(truth, support, first, second)
        self.assertEqual(result["1.0x"]["valid_pixels"], 2)
        self.assertEqual(result["0.5x"]["valid_pixels"], 1)
        self.assertEqual(result["0.5x"]["disparity"]["epe"], 1.)
        self.assertEqual(result["1.0x"]["disparity"]["pixels"], 1)
        self.assertFalse(valid_box(None, (20, 20)))
        self.assertFalse(valid_box([0, 0, 21, 20], (20, 20)))

    def test_automatic_match_needs_overlap_in_both_views(self):
        boxes = torch.tensor([[0., 0., 5., 5.]])
        none = automatic_pair(torch.tensor([12]), torch.tensor([0]), boxes, boxes, None,
                              0, [0, 0, 5, 5], [10, 0, 15, 5])
        self.assertIsNone(none)

    def test_left_right_consistency_checks_the_selected_reverse_column(self):
        spatial = torch.ones((1, 1, 1, 4), dtype=torch.bool)
        cost = torch.zeros((1, 3, 1, 4))
        left = LocalCostVolume(cost, torch.zeros_like(cost, dtype=torch.bool), spatial)
        right = LocalCostVolume(cost, torch.zeros_like(cost, dtype=torch.bool), spatial)
        left.valid_bins[0, 2, 0, 2] = True  # x=2 -> x=1
        right.valid_bins[0, 2, 0, 1] = True  # x=1 -> x=0
        consistency, valid = left_right_consistency(left, right, 1)
        self.assertTrue(valid[0, 0, 2])
        self.assertEqual(float(consistency[0, 0, 2]), 0.)
