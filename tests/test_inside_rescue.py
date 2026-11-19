"""Anchor-free inside rescue and fallback assignment contracts."""
import unittest

import torch

from lightbackbone_stereo_yolo.geometry import assign_anchorfree_left_targets, decode_anchorfree_boxes6


class InsideRescueTest(unittest.TestCase):
    def test_cross_scale_inside_rescue_has_no_extra_classification_point(self):
        loc = torch.tensor([[4., 4., 8.], [8., 8., 16.]])
        gt = torch.tensor([[6., 6., 10., 10., 2., 6.]])
        labels = torch.tensor([1])
        old = assign_anchorfree_left_targets(loc, gt, labels)
        self.assertEqual(old.box_valid_mask.tolist(), [False, False])
        actual = assign_anchorfree_left_targets(loc, gt, labels, assignment_mode="inside_rescue", image_hw=(20, 20))
        self.assertEqual(actual.labels.tolist(), [0, 1])
        self.assertEqual(actual.matched_indices.tolist(), [-1, 0])
        self.assertEqual(actual.box_valid_mask.tolist(), [False, True])
        boxes = decode_anchorfree_boxes6(actual.box6[1:], loc[1:], right_offset_mode="signed")
        torch.testing.assert_close(boxes, gt, atol=1e-4, rtol=1e-5)

    def test_same_scale_inside_rescue_precedes_outside_fallback(self):
        loc = torch.tensor([[20., 20., 8.], [4., 4., 8.]])
        gt = torch.tensor([[0., 0., 58., 60., 0., 50.]])
        result = assign_anchorfree_left_targets(loc, gt, torch.tensor([0]), center_radius_in_strides=.25,
                                                assignment_mode="inside_rescue", image_hw=(64, 64))
        self.assertEqual(result.matched_indices.tolist(), [0, -1])
        self.assertEqual(result.box_valid_mask.tolist(), [True, False])

    def test_last_resort_does_not_steal_a_positive_point(self):
        loc = torch.tensor([[4., 4., 8.]])
        gt = torch.tensor([[2., 2., 6., 6., 1., 5.], [8., 8., 10., 10., 6., 8.]])
        result = assign_anchorfree_left_targets(loc, gt, torch.tensor([0, 1]),
                                                assignment_mode="inside_rescue", image_hw=(12, 12))
        self.assertEqual(result.matched_indices.tolist(), [0])
        self.assertEqual(result.box_valid_mask.tolist(), [True])

    def test_boxless_gt_gets_only_an_in_image_classification_fallback(self):
        loc = torch.tensor([[4., 4., 8.], [8., 8., 8.]])
        gt = torch.tensor([[4.75, 7., 5.5, 8., 2., 3.]])
        result = assign_anchorfree_left_targets(loc, gt, torch.tensor([0]),
                                                assignment_mode="inside_rescue", image_hw=(9, 6))
        self.assertEqual(result.matched_indices.tolist(), [0, -1])
        self.assertEqual(result.box_valid_mask.tolist(), [False, False])
        self.assertTrue(bool(torch.isfinite(result.box6).all()))

    def test_degenerate_right_box_does_not_acquire_regression(self):
        loc = torch.tensor([[4., 4., 8.], [8., 8., 16.]])
        gt = torch.tensor([[6., 6., 10., 10., 3., 3.]])
        result = assign_anchorfree_left_targets(loc, gt, torch.tensor([0]),
                                                assignment_mode="inside_rescue", image_hw=(20, 20))
        self.assertEqual(sum(result.box_valid_mask.tolist()), 0)
        self.assertEqual(sum(result.labels.eq(1).tolist()), 1)

    def test_degenerate_right_width_preserves_normal_classification_positives(self):
        loc = torch.tensor([[4., 4., 8.], [12., 4., 8.]])
        gt = torch.tensor([[2., 2., 14., 8., 3., 3.]])
        classes = torch.tensor([0])
        old = assign_anchorfree_left_targets(loc, gt, classes)
        result = assign_anchorfree_left_targets(loc, gt, classes, assignment_mode="inside_rescue", image_hw=(20, 20))
        self.assertEqual(old.labels.tolist(), [1, 1])
        self.assertEqual(result.labels.tolist(), [1, 1])
        self.assertEqual(result.box_valid_mask.tolist(), [False, False])
