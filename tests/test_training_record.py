"""Metrics-record serialization contracts."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lightbackbone_stereo_yolo.training_control import (DetectorCheckpointSelector, append_jsonl_record, is_precision_qualified, percentile)


class TrainingRecordTest(unittest.TestCase):
    def test_append_jsonl_record_writes_one_parseable_object_per_physical_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            append_jsonl_record(path, {"epoch": 0})
            append_jsonl_record(path, {"epoch": 1})
            self.assertEqual(
                [json.loads(line) for line in path.read_text().splitlines()],
                [{"epoch": 0}, {"epoch": 1}],
            )


    def test_precision_qualification_is_an_eligibility_gate(self):
        self.assertFalse(is_precision_qualified(.80, .09584, .09585))
        self.assertTrue(is_precision_qualified(.50, .09585, .09585))

    def test_checkpoint_selector_rejects_higher_coverage_below_precision_floor(self):
        selector = DetectorCheckpointSelector(precision_floor=.09585)
        self.assertTrue(selector.update(.59, .1025, .76))
        self.assertFalse(selector.update(.60, .0853, .80))
        self.assertEqual(selector.best_paired_coverage, .59)


    def test_percentile_uses_linear_interpolation_for_per_frame_candidate_counts(self):
        self.assertEqual(percentile([1., 2., 3., 4.], .5), 2.5)
        self.assertAlmostEqual(percentile([1., 2., 3., 4.], .9), 3.7)

if __name__ == "__main__":
    unittest.main()
