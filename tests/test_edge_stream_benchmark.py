from pathlib import Path
import sys
from time import sleep
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


class StreamTest(unittest.TestCase):
    def test_replay_preserves_order_and_output_count(self):
        from benchmark_edge_stream import run_stream
        report = run_stream([("a", None), ("b", None)], lambda frame: None,
                            rate_fps=100, duration_s=.035)
        self.assertEqual([row["frame_id"] for row in report["rows"]], ["a", "b", "a", "b"])
        self.assertEqual([row["sequence_id"] for row in report["rows"]], list(range(4)))
        self.assertEqual(report["submitted"], report["completed"] + report["in_flight_at_end"])

    def test_latency_includes_wait_for_a_busy_consumer(self):
        from benchmark_edge_stream import run_stream
        report = run_stream([("a", None)], lambda frame: sleep(.02),
                            rate_fps=100, duration_s=.045)
        self.assertGreater(report["rows"][1]["queue_ms"], 0.)
        self.assertGreater(report["timeouts"], 0)
        self.assertGreater(report["in_flight_at_end"], 0)
