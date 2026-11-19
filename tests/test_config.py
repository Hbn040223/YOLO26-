"""Configuration contract for the feature-only Stereo-YOLO experiment."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from lightbackbone_stereo_yolo.config import ExperimentConfig
except ModuleNotFoundError:
    ExperimentConfig = None


class ExperimentConfigContractTest(unittest.TestCase):
    def test_feature_only_configuration_exposes_required_constraints(self) -> None:
        self.assertIsNotNone(
            ExperimentConfig,
            "ExperimentConfig must define the feature-only experiment contract",
        )
        config = ExperimentConfig(
            runtime_python="environment/python/bin/python",
            dataset_root="/path/to/KITTI2015",
            lightstereo_source="references/OpenStereo",
            lightstereo_checkpoint="weights/lightstereo_s_sceneflow.ckpt",
        )
        self.assertEqual(config.p2_stride, 4)
        self.assertEqual(config.residual_quantile, 0.995)
        self.assertEqual(config.pre_nms_topk, 500)
        self.assertEqual(config.roi_margin, 16.0)
        self.assertFalse(config.allow_global_cost_volume)
        self.assertFalse(config.allow_global_disparity)
        self.assertFalse(config.allow_3d_outputs)

    def test_rejects_any_forbidden_execution_path(self) -> None:
        self.assertIsNotNone(ExperimentConfig)
        with self.assertRaises(ValueError):
            ExperimentConfig(
                runtime_python="environment/python/bin/python",
                dataset_root="/path/to/KITTI2015",
                lightstereo_source="references/OpenStereo",
                lightstereo_checkpoint="weights/lightstereo_s_sceneflow.ckpt",
                allow_global_disparity=True,
            )


if __name__ == "__main__":
    unittest.main()
