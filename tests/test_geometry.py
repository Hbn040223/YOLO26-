"""Stereo 6D box geometry contracts."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch

try:
    from lightbackbone_stereo_yolo.geometry import decode_boxes6, encode_boxes6, pair_boxes_to_lr
except ModuleNotFoundError:
    decode_boxes6 = encode_boxes6 = pair_boxes_to_lr = None


class StereoBoxGeometryTest(unittest.TestCase):
    def test_six_term_encoding_round_trips_and_keeps_shared_vertical_bounds(self) -> None:
        anchors = torch.tensor([[10.0, 20.0, 50.0, 60.0], [80.0, 10.0, 120.0, 30.0]])
        targets = torch.tensor([
            [12.0, 21.0, 55.0, 63.0, -4.0, 39.0],
            [78.0, 12.0, 122.0, 31.0, 50.0, 93.0],
        ])
        self.assertIsNotNone(encode_boxes6)
        encoded = encode_boxes6(targets, anchors)
        self.assertEqual(encoded.shape, (2, 6))
        self.assertTrue(torch.allclose(decode_boxes6(encoded, anchors), targets, atol=1e-5))

    def test_pair_boxes_expand_to_aligned_left_and_right_four_boxes(self) -> None:
        pairs = torch.tensor([[10.0, 20.0, 50.0, 60.0, -4.0, 39.0]])
        left, right = pair_boxes_to_lr(pairs)
        self.assertTrue(torch.equal(left, torch.tensor([[10.0, 20.0, 50.0, 60.0]])))
        self.assertTrue(torch.equal(right, torch.tensor([[-4.0, 20.0, 39.0, 60.0]])))


if __name__ == "__main__":
    unittest.main()
