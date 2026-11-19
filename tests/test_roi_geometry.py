"""Continuous paired crop geometry contracts."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
import torch
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
try:
    from lightbackbone_stereo_yolo.roi import LeftRoIAlign, StereoRoIAlign, align_left_attention_pyramid, align_left_mask_target, align_mask_spatial_valid, make_pair_crop_windows, sample_roi_feature
except ImportError:
    LeftRoIAlign = StereoRoIAlign = align_left_attention_pyramid = align_left_mask_target = align_mask_spatial_valid = make_pair_crop_windows = sample_roi_feature = None


class CropMetadataContractTest(unittest.TestCase):
    def test_pair_crop_windows_include_shared_vertical_margin(self) -> None:
        """Would fail if the paired crop keeps only horizontal geometry."""
        pairs = torch.tensor([[20.0, 10.0, 50.0, 30.0, 12.0, 42.0]])
        metadata = make_pair_crop_windows(
            pairs, image_height=64, image_width=100, margin=4.0
        )
        self.assertEqual(metadata.y0.tolist(), [6.0])
        self.assertEqual(metadata.h_crop.tolist(), [28.0])
        self.assertEqual(metadata.h_roi.tolist(), [7])
        self.assertEqual(metadata.w_roi.tolist(), [10])

    def test_pair_crop_windows_clamp_shared_vertical_margin_at_edges(self) -> None:
        """Would fail if a crop can sample above or below its source image."""
        pairs = torch.tensor([[1.0, 2.0, 25.0, 20.0, 0.0, 28.0]])
        metadata = make_pair_crop_windows(
            pairs, image_height=40, image_width=100, margin=8.0
        )
        self.assertEqual(metadata.y0.tolist(), [0.0])
        self.assertEqual(metadata.h_crop.tolist(), [34.0])
        self.assertLessEqual(float(metadata.y0[0] + metadata.h_crop[0]), 40.0)

    def test_mask_target_uses_left_crop_geometry_and_returns_binary_28_square(self) -> None:
        self.assertIsNotNone(align_left_mask_target)
        pairs = torch.tensor([[8.0, 8.0, 40.0, 32.0, 2.0, 34.0]])
        metadata = make_pair_crop_windows(pairs, image_height=64, image_width=64, margin=4.0)
        mask = torch.zeros(1, 1, 64, 64); mask[:, :, 8:32, 8:40] = 1
        target = align_left_mask_target(mask, pairs, metadata)
        self.assertEqual(target.shape, (1, 1, 28, 28))
        self.assertTrue(((target == 0) | (target == 1)).all())

    def test_mask_target_keeps_vertical_context_margin(self) -> None:
        """Would fail if mask supervision silently reverts to the raw box y-range."""
        pairs = torch.tensor([[16.0, 16.0, 48.0, 32.0, 8.0, 40.0]])
        metadata = make_pair_crop_windows(pairs, image_height=64, image_width=64, margin=8.0)
        mask = torch.zeros(1, 1, 64, 64)
        mask[:, :, 16:32, 16:48] = 1
        target = align_left_mask_target(mask, pairs, metadata)
        self.assertFalse(target[:, :, :6].any())
        self.assertTrue(target[:, :, 10:18, 8:20].all())
        self.assertFalse(target[:, :, 22:].any())

    def test_mask_validity_resamples_padded_roi_positions_to_mask_grid(self) -> None:
        """Would fail if padded feature columns became supervised mask pixels."""
        self.assertIsNotNone(align_mask_spatial_valid)
        spatial_valid = torch.zeros(1, 1, 4, 8, dtype=torch.bool)
        spatial_valid[:, :, :, :4] = True
        mask_valid = align_mask_spatial_valid(spatial_valid, output_size=28)
        self.assertEqual(mask_valid.shape, (1, 1, 28, 28))
        self.assertTrue(mask_valid[:, :, :, :14].all())
        self.assertFalse(mask_valid[:, :, :, 14:].any())
    def test_attention_pyramid_aligns_original_crop_at_each_scale(self) -> None:
        self.assertIsNotNone(align_left_attention_pyramid)
        pairs = torch.tensor([[8.0, 8.0, 40.0, 32.0, 2.0, 34.0]])
        metadata = make_pair_crop_windows(pairs, image_height=64, image_width=64, margin=4.0)
        p2, _ = sample_roi_feature(torch.randn(1, 4, 16, 16), metadata.c_l, metadata, stride=4)
        pyramid = align_left_attention_pyramid(p2, torch.randn(1, 4, 8, 8), torch.randn(1, 4, 4, 4), metadata)
        self.assertEqual(pyramid.p2.shape, (1, 4, 8, 10))
        self.assertEqual(pyramid.p3.shape, (1, 4, 4, 5))
        self.assertEqual(pyramid.p4.shape, (1, 4, 2, 3))

    def test_batched_sampling_preserves_each_instances_native_grid_scale(self) -> None:
        """Would fail if both RoIs are sampled using the batch-maximum width."""
        self.assertIsNotNone(sample_roi_feature)
        pairs = torch.tensor([
            [16.0, 8.0, 84.0, 40.0, 10.0, 78.0],
            [16.0, 8.0, 184.0, 40.0, 10.0, 178.0],
        ])
        metadata = make_pair_crop_windows(pairs, image_height=64, image_width=256, margin=16.0)
        feature = torch.arange(2 * 16 * 64, dtype=torch.float32).reshape(2, 1, 16, 64)
        sampled, valid = sample_roi_feature(feature, metadata.c_l, metadata, stride=4)
        self.assertEqual(sampled.shape, (2, 1, 16, 50))
        self.assertEqual(valid.shape, (2, 1, 16, 50))
        self.assertEqual(valid[0].sum().item(), 16 * 25)
        self.assertEqual(valid[1].sum().item(), 16 * 50)
        first_meta = make_pair_crop_windows(pairs[:1], image_height=64, image_width=256, margin=16.0)
        single, single_valid = sample_roi_feature(feature[:1], first_meta.c_l, first_meta, stride=4)
        self.assertTrue(torch.equal(valid[:1, :, :, :25], single_valid))
        self.assertTrue(torch.allclose(sampled[:1, :, :, :25], single, atol=1e-6))
    def test_p2_align_uses_shared_metadata_with_native_variable_width(self) -> None:
        self.assertIsNotNone(StereoRoIAlign)
        pairs = torch.tensor([[8.0, 8.0, 40.0, 32.0, 2.0, 34.0]])
        metadata = make_pair_crop_windows(pairs, image_height=64, image_width=64, margin=4.0)
        left = torch.randn(1, 3, 16, 16)
        right = torch.randn(1, 3, 16, 16)
        left_roi, right_roi, valid, returned = StereoRoIAlign()(left, right, pairs, metadata)
        self.assertEqual(left_roi.shape, (1, 3, 8, 10))
        self.assertEqual(right_roi.shape, left_roi.shape)
        self.assertTrue(valid.all())
        self.assertIs(returned, metadata)

    def test_attention_scales_use_their_own_feature_stride(self) -> None:
        pairs = torch.tensor([[8.0, 8.0, 40.0, 32.0, 2.0, 34.0]])
        metadata = make_pair_crop_windows(pairs, image_height=64, image_width=64, margin=4.0)
        p3 = torch.randn(1, 3, 8, 8)
        aligned, valid = LeftRoIAlign(stride=8)(p3, metadata)
        self.assertEqual(aligned.shape, (1, 3, 4, 5))
        self.assertTrue(valid.all())

    def test_border_safe_windows_keep_both_boxes_and_native_stride4_width(self) -> None:
        self.assertIsNotNone(make_pair_crop_windows)
        pairs = torch.tensor([[1.0, 10.0, 25.0, 30.0, 0.0, 28.0]])
        metadata = make_pair_crop_windows(pairs, image_height=64, image_width=100, margin=8.0)
        self.assertEqual(metadata.c_l.tolist(), [0.0])
        self.assertEqual(metadata.c_r.tolist(), [0.0])
        self.assertEqual(metadata.c.tolist(), [44.0])
        self.assertEqual(metadata.w_roi.tolist(), [11])
        self.assertGreaterEqual(metadata.c_l[0], 0)
        self.assertLessEqual(pairs[0, 2], metadata.c_l[0] + metadata.c[0])
        self.assertLessEqual(pairs[0, 5], metadata.c_r[0] + metadata.c[0])



    def test_roi_map_resampling_preserves_values_on_identical_crop_geometry(self) -> None:
        from lightbackbone_stereo_yolo.roi import resample_roi_map_to_crop
        boxes = torch.tensor([[8., 8., 24., 24., 4., 20.]])
        metadata = make_pair_crop_windows(boxes, 32, 32, margin=0.)
        source = torch.tensor([[[1., 2., 3., 4.], [5., 6., 7., 8.], [9., 10., 11., 12.], [13., 14., 15., 16.]]])
        sampled, valid = resample_roi_map_to_crop(source, metadata, metadata)
        self.assertTrue(torch.allclose(sampled, source))
        self.assertTrue(valid.all())
if __name__ == "__main__":
    unittest.main()
