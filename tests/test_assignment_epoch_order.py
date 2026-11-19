"""Prescribed frame orders can be applied consistently across assignment runs."""
import json
import tempfile
import unittest
from pathlib import Path

from scripts import train


class AssignmentEpochOrderTest(unittest.TestCase):
    def test_manifest_maps_each_epoch_to_dataset_indices_and_preserves_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "order.json"
            path.write_text(json.dumps({"epochs": [["b", "a"], ["a", "b"]]}))
            actual, digest = train.read_epoch_orders(path, ["a", "b"], 2)
            self.assertEqual(actual, [[1, 0], [0, 1]])
            self.assertEqual(len(digest), 64)

    def test_manifest_rejects_duplicate_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "order.json"
            path.write_text(json.dumps({"epochs": [["a", "a"]]}))
            with self.assertRaisesRegex(ValueError, "permutation"):
                train.read_epoch_orders(path, ["a", "b"], 1)
