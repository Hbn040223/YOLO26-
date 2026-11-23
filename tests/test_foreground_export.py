"""Foreground output uses one valid Mask and array row per stereo pair."""
import unittest
import torch

from lightbackbone_stereo_yolo.mask import foreground_disparity_validity


class ForegroundExportTest(unittest.TestCase):
    def test_threshold_support_and_empty_masks_stay_per_instance(self):
        mask = torch.tensor([
            [[[2., -2.], [2., -2.]]],
            [[[-2., -2.], [-2., -2.]]],
            [[[2., 2.], [2., 2.]]],
            [[[-2., 2.], [-2., 2.]]],
        ])
        disparity = torch.arange(64, dtype=torch.float32).reshape(4, 4, 4)
        disparity[0, 0, 0] = float("nan")
        valid = torch.ones_like(disparity, dtype=torch.bool)
        valid[0, 1, 0] = False
        mask_valid = torch.ones_like(mask, dtype=torch.bool)
        mask_valid[3, 0, 0, 1] = False
        foreground = foreground_disparity_validity(
            mask, mask_valid, disparity, valid, torch.tensor([True, True, False, True])
        )
        self.assertEqual(foreground.shape, disparity.shape)
        self.assertFalse(foreground[0, :, 2:].any())
        self.assertFalse(foreground[0, 0, 0])
        self.assertFalse(foreground[0, 1, 0])
        self.assertTrue(foreground[0, 2:, :2].all())
        self.assertFalse(foreground[1:3].any())
        self.assertFalse(foreground[3, :2, 2:].any())
        self.assertTrue(foreground[3, 2:, 2:].all())
