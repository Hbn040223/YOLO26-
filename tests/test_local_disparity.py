"""RoI-only residual-disparity contracts."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
import torch
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from lightbackbone_stereo_yolo.local_disparity import LearnableTemperature, LocalCostVolume, RegularizedIDispHead, ResidualSearch, build_local_cost_volume, disparity_loss, overflow_mask, restore_absolute_disparity
from lightbackbone_stereo_yolo.roi import CropMetadata, make_pair_crop_windows
from lightbackbone_stereo_yolo.local_aggregation import AttentionPyramid


class LocalDisparityContractTest(unittest.TestCase):
    def test_fixed_grid_mask_shortcut_and_forced_fp32(self):
        head = RegularizedIDispHead(4, 1, 4).eval()
        meta = make_pair_crop_windows(torch.tensor([[8., 8., 24., 24., 6., 22.]]), 64, 64, 0,
                                      roi_sampling="fixed_56")
        left = torch.randn(1, 4, 56, 56)
        attention = AttentionPyramid(left, left[..., ::2, ::2], left[..., ::4, ::4])
        valid = torch.ones(1, 1, 56, 56, dtype=torch.bool)
        args = left, left, attention, meta, valid
        with torch.no_grad():
            reference = head(*args)
            head.skip_fixed_spatial_mask = True
            torch.testing.assert_close(head(*args).absolute, reference.absolute, rtol=0, atol=0)
            head.force_fp32_aggregation = True
            with torch.autocast("cpu", dtype=torch.bfloat16):
                actual = head(*args)
        torch.testing.assert_close(actual.absolute, reference.absolute, rtol=0, atol=0)
    def test_bmm_matches_loop(self) -> None:
        rng = torch.Generator().manual_seed(17)
        for count in (1, 2, 6, 9):
            for width in (5, 56):
                with self.subTest(count=count, width=width):
                    shape = (count, 96, 7, width)
                    left = torch.randn(shape, generator=rng)
                    right = torch.randn(shape, generator=rng)
                    valid = torch.rand((count, 1, 7, width), generator=rng) > .2
                    valid[:, :, 0] = False
                    args = (left, right, ResidualSearch(16), valid, LearnableTemperature())
                    reference = build_local_cost_volume(*args)
                    result = build_local_cost_volume(*args, backend="bmm")
                    self.assertTrue(torch.equal(reference.valid_bins, result.valid_bins))
                    self.assertTrue(torch.equal(reference.spatial_valid, result.spatial_valid))
                    torch.testing.assert_close(result.cost, reference.cost, atol=1e-5, rtol=1e-5)

    def test_regularized_head_returns_image_domain_disparity_and_confidence(self) -> None:
        self.assertIsNotNone(RegularizedIDispHead)
        metadata = CropMetadata(c_l=torch.tensor([4.]), c_r=torch.tensor([0.]), c=torch.tensor([16.]), y0=torch.tensor([0.]), h_crop=torch.tensor([8.]), w_roi=torch.tensor([4]), h_roi=torch.tensor([2]))
        attention = AttentionPyramid(torch.randn(1, 4, 2, 4), torch.randn(1, 4, 1, 2), torch.randn(1, 4, 1, 1))
        output = RegularizedIDispHead(channels=4, dmax_roi=1, hidden_channels=4)(torch.randn(1, 4, 2, 4), torch.randn(1, 4, 2, 4), attention, metadata, torch.ones(1, 1, 2, 4, dtype=torch.bool))
        self.assertEqual(output.absolute.shape, (1, 2, 4))
        self.assertEqual(output.peak.shape, (1, 2, 4))
        self.assertEqual(output.entropy.shape, (1, 2, 4))
    def test_image_domain_loss_ignores_overflow_pixels(self) -> None:
        self.assertIsNotNone(disparity_loss)
        predicted = torch.tensor([[[12., 20.]]])
        target = torch.tensor([[[10., 100.]]])
        valid = torch.tensor([[[True, False]]])
        self.assertAlmostEqual(disparity_loss(predicted, target, valid).item(), 1.5, places=5)
        self.assertTrue(torch.equal(overflow_mask(torch.tensor([[[-3., 4.]]]), 3), torch.tensor([[[False, True]]])))

    def test_disparity_loss_macro_averages_nonempty_instances(self) -> None:
        """A large instance must not outweigh a small instance solely by pixel count."""
        self.assertIsNotNone(disparity_loss)
        predicted = torch.tensor([[[0., 0., 0.]], [[2., 2., 2.]]])
        target = torch.zeros_like(predicted)
        valid = torch.tensor([[[True, False, False]], [[True, True, True]]])
        # Smooth-L1 values are 0 and 1.5, so the macro instance mean is .75.
        self.assertAlmostEqual(disparity_loss(predicted, target, valid).item(), .75, places=5)

    def test_disparity_loss_excludes_empty_instances_from_its_denominator(self) -> None:
        self.assertIsNotNone(disparity_loss)
        predicted = torch.tensor([[[2.]], [[100.]]])
        target = torch.zeros_like(predicted)
        valid = torch.tensor([[[True]], [[False]]])
        self.assertAlmostEqual(disparity_loss(predicted, target, valid).item(), 1.5, places=5)

    def test_temperature_starts_near_point_one_and_stays_positive(self) -> None:
        self.assertIsNotNone(LearnableTemperature)
        temperature = LearnableTemperature()
        self.assertAlmostEqual(temperature().item(), 0.1, places=4)
        self.assertGreater(temperature().item(), 0.0)

    def test_cosine_volume_keeps_out_of_range_bins_neutral_and_invalid(self) -> None:
        self.assertIsNotNone(LocalCostVolume)
        left = torch.tensor([[[[1., 1.]], [[0., 0.]]]])
        right = left.clone()
        volume = build_local_cost_volume(left, right, ResidualSearch(1), torch.ones(1, 1, 1, 2, dtype=torch.bool), LearnableTemperature(initial=1.0))
        self.assertIsInstance(volume, LocalCostVolume)
        self.assertEqual(volume.cost.shape, (1, 3, 1, 2))
        self.assertEqual(volume.cost[0, 2, 0, 0].item(), 0.0)
        self.assertFalse(volume.valid_bins[0, 2, 0, 0].item())
        self.assertAlmostEqual(volume.cost[0, 1, 0, 0].item(), 1.0, places=5)

    def test_cost_logits_apply_positive_temperature(self) -> None:
        left = torch.tensor([[[[1.]], [[0.]]]])
        volume = build_local_cost_volume(left, left, ResidualSearch(0), torch.ones(1, 1, 1, 1, dtype=torch.bool), LearnableTemperature())
        self.assertAlmostEqual(volume.cost.item(), 10.0, places=3)

    def test_cost_volume_excludes_padded_spatial_positions(self) -> None:
        """Would fail if a zero-padded RoI column is treated as a real match location."""
        left = torch.tensor([[[[1., 1., 0.]], [[0., 0., 0.]]]])
        valid = torch.tensor([[[[True, True, False]]]])
        volume = build_local_cost_volume(
            left, left, ResidualSearch(1), valid, LearnableTemperature()
        )
        self.assertFalse(volume.valid_bins[0, :, 0, 2].any())
        self.assertTrue((volume.cost[0, :, 0, 2] == 0).all())
        self.assertFalse(volume.spatial_valid[0, 0, 0, 2].item())

    def test_regularized_head_matches_single_and_mixed_native_roi_batch(self) -> None:
        """Would fail if padded columns alter GroupNorm or decoder interpolation."""
        torch.manual_seed(7)
        head = RegularizedIDispHead(channels=4, dmax_roi=1, hidden_channels=4).eval()
        left = torch.randn(2, 4, 4, 8)
        right = torch.randn(2, 4, 4, 8)
        left[0, :, :, 4:] = 0
        right[0, :, :, 4:] = 0
        valid = torch.tensor([
            [[[True, True, True, True, False, False, False, False]] * 4],
            [[[True, True, True, True, True, True, True, True]] * 4],
        ])
        metadata = CropMetadata(
            c_l=torch.tensor([4., 4.]), c_r=torch.tensor([0., 0.]), c=torch.tensor([16., 32.]),
            y0=torch.tensor([0., 0.]), h_crop=torch.tensor([16., 16.]),
            w_roi=torch.tensor([4, 8]), h_roi=torch.tensor([4, 4]),
        )
        attention = AttentionPyramid(
            left, torch.randn(2, 4, 2, 4), torch.randn(2, 4, 1, 2)
        )
        with torch.no_grad():
            batched = head(left, right, attention, metadata, valid)
            single = head(
                left[:1, :, :, :4], right[:1, :, :, :4],
                AttentionPyramid(attention.p2[:1, :, :, :4], attention.p3[:1, :, :, :2], attention.p4[:1, :, :, :1]),
                CropMetadata(
                    c_l=metadata.c_l[:1], c_r=metadata.c_r[:1], c=metadata.c[:1],
                    y0=metadata.y0[:1], h_crop=metadata.h_crop[:1], w_roi=metadata.w_roi[:1], h_roi=metadata.h_roi[:1],
                ),
                valid[:1, :, :, :4],
            )
        self.assertLess((batched.absolute[0, :, :4] - single.absolute[0]).abs().max().item(), 1e-4)
    def test_restoration_uses_only_crop_metadata(self) -> None:
        self.assertIsNotNone(restore_absolute_disparity)
        metadata = CropMetadata(c_l=torch.tensor([8.]), c_r=torch.tensor([4.]), c=torch.tensor([40.]), y0=torch.tensor([0.]), h_crop=torch.tensor([8.]), w_roi=torch.tensor([10]), h_roi=torch.tensor([2]))
        restored = restore_absolute_disparity(torch.full((1, 2, 10), 2.0), metadata)
        self.assertTrue(torch.equal(restored, torch.full((1, 2, 10), 12.0)))

    def test_dynamic_q_recovers_constant_image_disparity(self) -> None:
        boxes = torch.tensor([[40., 10., 80., 30., 20., 60.]])
        metadata = make_pair_crop_windows(boxes, 64, 128, margin=0.)
        q = metadata.c / metadata.w_roi.float()
        restored = restore_absolute_disparity(torch.full((1, 1, int(metadata.w_roi.item())), 2.), metadata)
        self.assertTrue(torch.allclose(restored, (metadata.c_l - metadata.c_r + 2 * q)[:, None, None]))

    def test_constant_feature_match_peaks_at_expected_local_offset(self) -> None:
        left = torch.zeros(1, 2, 1, 4); right = torch.zeros_like(left)
        left[0, 0, 0, 1] = 1.; right[0, 0, 0, 0] = 1.
        volume = build_local_cost_volume(left, right, ResidualSearch(1), torch.ones(1, 1, 1, 4, dtype=torch.bool), LearnableTemperature(initial=1.0))
        self.assertEqual(int(volume.cost[0, :, 0, 1].argmax()), 2)

    def test_different_crop_widths_restore_same_image_disparity(self) -> None:
        first = CropMetadata(torch.tensor([20.]), torch.tensor([0.]), torch.tensor([40.]), torch.tensor([0.]), torch.tensor([8.]), torch.tensor([10]), torch.tensor([2]))
        second = CropMetadata(torch.tensor([20.]), torch.tensor([0.]), torch.tensor([52.]), torch.tensor([0.]), torch.tensor([8.]), torch.tensor([13]), torch.tensor([2]))
        target = torch.tensor(28.)
        r1 = (target - 20.) / (first.c / first.w_roi.float())
        r2 = (target - 20.) / (second.c / second.w_roi.float())
        self.assertAlmostEqual(restore_absolute_disparity(r1[:, None, None], first).item(), 28., places=5)
        self.assertAlmostEqual(restore_absolute_disparity(r2[:, None, None], second).item(), 28., places=5)

    def test_boundary_clamp_uses_actual_offset_and_dynamic_q(self) -> None:
        metadata = make_pair_crop_windows(torch.tensor([[3., 8., 23., 28., 0., 20.]]), 64, 128, margin=16.)
        restored = restore_absolute_disparity(torch.ones(1, 1, 1), metadata)
        expected = (metadata.c_l - metadata.c_r + metadata.c / metadata.w_roi.float())[:, None, None]
        self.assertTrue(torch.allclose(restored, expected))
if __name__ == "__main__":
    unittest.main()
