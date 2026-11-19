"""Detector validation plateau-stop contracts."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lightbackbone_stereo_yolo.training_control import DetectorPlateauController


class DetectorPlateauTest(unittest.TestCase):
    def test_does_not_stop_before_epoch_30_then_stops_after_five_flat_epochs(self):
        state = DetectorPlateauController(min_epoch=30, patience=5, min_delta=.005)
        for epoch in range(30):
            self.assertFalse(state.update(epoch, .40, .50))
        for epoch in range(30, 34):
            self.assertFalse(state.update(epoch, .40, .50))
        self.assertTrue(state.update(34, .40, .50))
        self.assertEqual(state.stop_reason, "plateau")

    def test_either_metric_improvement_resets_patience_and_restores_state(self):
        state = DetectorPlateauController(min_epoch=30, patience=5, min_delta=.005)
        state.update(0, .40, .50)
        state.update(30, .40, .50)
        state.update(31, .40, .50)
        self.assertEqual(state.plateau_counter, 2)
        self.assertFalse(state.update(32, .406, .50))
        self.assertEqual(state.plateau_counter, 0)
        restored = DetectorPlateauController.from_state_dict(state.state_dict())
        self.assertEqual(restored.best_paired_coverage, .406)
        self.assertEqual(restored.plateau_counter, 0)
        self.assertFalse(restored.update(33, .406, .50))


if __name__ == "__main__":
    unittest.main()
