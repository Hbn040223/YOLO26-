"""Regression checks for strip-only one-to-one deployment bundles."""
from types import SimpleNamespace
import unittest

import torch

from scripts.export_one_to_one import prune_one_to_many
from lightbackbone_stereo_yolo.stereo_yolo26 import StereoYOLO26PanHead


class OneToOneDeploymentTest(unittest.TestCase):
    def test_strip_one_branch_keeps_predictions_and_foreground_metadata(self):
        torch.manual_seed(17)
        full = StereoYOLO26PanHead(8, 2, one_to_one=True).eval()
        state = {"detector." + key: value for key, value in full.state_dict().items()}
        bundle = {
            "model": state, "optimizer": {"state": {}},
            "config": {"detector_variant": "stereo_yolo26_pan", "one_to_one": True},
            "foreground_mask_raw_classes": [26], "mask_threshold": .4,
            "mask_gate": {"evaluation_only": True},
        }
        exported = prune_one_to_many(bundle)
        self.assertNotIn("optimizer", exported)
        self.assertTrue(exported["config"]["one_to_one_only"])
        self.assertNotIn("one_to_one_only", bundle["config"])
        for key in ("foreground_mask_raw_classes", "mask_threshold", "mask_gate"):
            self.assertEqual(exported[key], bundle[key])
        retained = {key for key in state if not key.startswith("detector.levels.")}
        self.assertEqual(set(exported["model"]), retained)
        for key in retained:
            self.assertIs(exported["model"][key], state[key])

        pruned = StereoYOLO26PanHead(8, 2, one_to_one=True, one_to_one_only=True).eval()
        pruned.load_state_dict({key[len("detector."):]: value for key, value in exported["model"].items()},
                               strict=True)
        self.assertIsNone(pruned.levels)
        self.assertIsNotNone(full.levels)
        pyramid = SimpleNamespace(**{
            f"{level}_{side}": torch.rand(1, 8, n, n)
            for level, n in (("p3", 8), ("p4", 4), ("c5", 2))
            for side in ("left", "right")
        })
        with torch.inference_mode():
            before, after = full(pyramid), pruned(pyramid)
            for key in ("objectness", "class_logits", "box6"):
                self.assertTrue(torch.equal(getattr(before, key), getattr(after, key)))
            with self.assertRaisesRegex(RuntimeError, "deployment-only"):
                pruned.forward_training(pyramid)

    def test_rejects_missing_or_incomplete_one_to_one_branch(self):
        config = {"detector_variant": "stereo_yolo26_pan", "one_to_one": True}
        tensor = torch.ones(1)
        with self.assertRaises(ValueError):
            prune_one_to_many({"config": {**config, "one_to_one": False}, "model": {}})
        with self.assertRaises(ValueError):
            prune_one_to_many({"config": config, "model": {
                "detector.levels.0.stem.weight": tensor,
                "detector.levels.1.stem.weight": tensor,
                "detector.one_to_one_levels.0.stem.weight": tensor,
            }})
