"""Fixed matching-grid geometry, supervision, and full RoI-head coverage."""
from types import SimpleNamespace
import unittest

import torch

from lightbackbone_stereo_yolo.local_disparity import restore_absolute_disparity
from lightbackbone_stereo_yolo.model import StereoLocalInstanceHead
from lightbackbone_stereo_yolo.roi import (
    align_disparity_roi, align_instance_mask_roi, make_pair_crop_windows,
    resample_roi_map_to_crop, sample_roi_feature,
)
from lightbackbone_stereo_yolo.training import roi_residual_target


class FixedRoISamplingTest(unittest.TestCase):
    def setUp(self):
        self.boxes = torch.tensor([[32., 16., 88., 80., 20., 76.],
                                   [32., 16., 144., 112., 20., 132.]])

    def test_fixed_grid_is_sampled_directly_in_both_views(self):
        crop = make_pair_crop_windows(self.boxes, 128, 192, 0., roi_sampling="fixed_56")
        horizontal_ramp = torch.arange(48, dtype=torch.float32).expand(2, 1, 32, 48) * 4
        left, valid = sample_roi_feature(horizontal_ramp, crop.c_l, crop, 4)
        right, _ = sample_roi_feature(horizontal_ramp, crop.c_r, crop, 4)
        self.assertEqual(tuple(left.shape), (2, 1, 56, 56))
        self.assertTrue(valid.all())
        torch.testing.assert_close(left[:, 0, 0, 0], torch.tensor([32.5, 33.]))
        torch.testing.assert_close(left[:, 0, 0, -1], torch.tensor([87.5, 143.]))
        torch.testing.assert_close(right[:, 0, 0, 0], torch.tensor([20.5, 21.]))

    def test_fixed_grid_supervision_and_disparity_units_use_same_geometry(self):
        crop = make_pair_crop_windows(self.boxes, 128, 192, 0., roi_sampling="fixed_56")
        label = torch.full((1, 1, 128, 192), 20.)
        target, valid = align_disparity_roi(label, label > 0, crop)
        mask = align_instance_mask_roi(label > 0, crop)
        self.assertEqual(tuple(target.shape), (2, 1, 56, 56))
        self.assertTrue(valid.all() and mask.all())
        bins = roi_residual_target(target, crop)
        torch.testing.assert_close(bins[:, 0, 0, 0], torch.tensor([8., 4.]))
        absolute = restore_absolute_disparity(bins[:, 0], crop)
        torch.testing.assert_close(absolute, target[:, 0])
        restored_on_same_grid, on_grid = resample_roi_map_to_crop(absolute, crop, crop)
        self.assertTrue(on_grid.all())
        torch.testing.assert_close(restored_on_same_grid, absolute)

    def test_fixed_head_aggregates_all_pairs_in_one_batch(self):
        pyramid = SimpleNamespace(
            p2_left=torch.rand(2, 4, 32, 48), p2_right=torch.rand(2, 4, 32, 48),
            p3_left=torch.rand(2, 4, 16, 24), p4_left=torch.rand(2, 4, 8, 12),
        )
        head = StereoLocalInstanceHead(4, 2, hidden_channels=4, roi_sampling="fixed_56").eval()
        seen = []

        def trace(_module, inputs):
            cost, attention, _valid = inputs
            seen.append((tuple(cost.shape), attention.p3.shape[-2:], attention.p4.shape[-2:]))

        handle = head.idisp.aggregation.register_forward_pre_hook(trace)
        try:
            with torch.no_grad():
                output = head(pyramid, self.boxes, 128, 192, margin=0.)
        finally:
            handle.remove()
        self.assertEqual(seen, [((2, 5, 56, 56), (28, 28), (14, 14))])
        self.assertEqual(tuple(output.disparity.shape), (2, 56, 56))
        self.assertEqual(tuple(output.mask_logits.shape), (2, 1, 28, 28))
        self.assertTrue(output.validity.all())
        self.assertTrue(torch.isfinite(output.disparity).all())
