"""Anchor-free paired 6D geometry contracts."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from lightbackbone_stereo_yolo.geometry import (
        assign_anchorfree_left_targets,
        decode_anchorfree_boxes6,
        encode_anchorfree_boxes6,
        generate_point_locations,
        point_levels,
    )
except ImportError:
    assign_anchorfree_left_targets = decode_anchorfree_boxes6 = encode_anchorfree_boxes6 = None
    generate_point_locations = point_levels = None


class AnchorFreeStereoGeometryTest(unittest.TestCase):
    def test_point_levels_and_locations_follow_p3_p4_c5_flatten_order(self) -> None:
        self.assertIsNotNone(point_levels)
        pyramid = SimpleNamespace(
            p3_left=torch.empty(1, 8, 2, 3),
            p4_left=torch.empty(1, 8, 1, 2),
            c5_left=torch.empty(1, 8, 1, 1),
        )
        levels = point_levels(pyramid)
        self.assertEqual(levels, ((2, 3, 8), (1, 2, 16), (1, 1, 32)))
        locations = generate_point_locations(levels, torch.device("cpu"))
        self.assertEqual(locations.shape, (9, 3))
        self.assertTrue(torch.equal(locations[0], torch.tensor([4., 4., 8.])))
        self.assertTrue(torch.equal(locations[6], torch.tensor([8., 8., 16.])))
        self.assertTrue(torch.equal(locations[-1], torch.tensor([16., 16., 32.])))

    def test_anchorfree_pair_encoding_round_trips_a_physical_pair(self) -> None:
        self.assertIsNotNone(encode_anchorfree_boxes6)
        target = torch.tensor([[20., 30., 60., 70., 4., 44.]])
        location = torch.tensor([[40., 48., 8.]])
        encoded = encode_anchorfree_boxes6(target, location)
        decoded = decode_anchorfree_boxes6(encoded, location)
        relative_error = (decoded - target).abs() / target.abs().clamp_min(1.)
        self.assertLess(float(relative_error.max()), 1e-4)

    def test_anchorfree_assignment_uses_left_geometry_and_forces_a_match(self) -> None:
        self.assertIsNotNone(assign_anchorfree_left_targets)
        locations = torch.tensor([[4., 4., 8.], [12., 4., 8.], [100., 100., 16.]])
        boxes6 = torch.tensor([[5., 1., 11., 7., -15., -9.]])
        classes = torch.tensor([2])
        assigned = assign_anchorfree_left_targets(locations, boxes6, classes, center_radius_in_strides=2.5)
        self.assertTrue((assigned.labels == 1).any())
        self.assertTrue((assigned.matched_indices == 0).any())
        self.assertTrue(torch.isfinite(assigned.box6).all())
        self.assertFalse(assigned.box_valid_mask[assigned.labels == 1].any())

    def test_empty_anchorfree_assignment_is_all_negative(self) -> None:
        self.assertIsNotNone(assign_anchorfree_left_targets)
        locations = torch.tensor([[4., 4., 8.], [12., 4., 8.]])
        assigned = assign_anchorfree_left_targets(locations, torch.empty(0, 6), torch.empty(0, dtype=torch.long))
        self.assertTrue(torch.equal(assigned.labels, torch.zeros(2, dtype=torch.long)))


if __name__ == "__main__":
    unittest.main()

