from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from benchmark_idisp_fixed_pairs import benchmark_loads, ranked_indices


class FixedPairBenchmarkTest(unittest.TestCase):
    def test_nested_rank_with_id_tiebreaker_and_missing_loads(self):
        ids = [10, 9, 8, 7, 6, 5, 4, 3, 2]
        order = ranked_indices(ids, [0.9, 0.8, 0.9, 0.7, 0.8, 0.6, 0.6, 0.5, 0.4])
        ranked = [ids[index] for index in order]
        self.assertEqual(ranked[:6], [8, 10, 6, 9, 7, 4])
        self.assertEqual(len(set(ranked)), 9)
        with self.assertRaises(ValueError):
            ranked_indices([1, 1], [.5, .5])
        rows, summary = benchmark_loads([], {"cached": self.fail}, torch.device("cpu"), 1, 2, {})
        self.assertEqual(rows, [])
        self.assertEqual(summary, {"cached": {}})

    def test_zero_skips_call_and_order_is_paired(self):
        calls = []
        rows, summary = benchmark_loads([0, 6, 9], {"cached": calls.append}, torch.device("cpu"), 0, 2, {})
        self.assertEqual(calls, [6, 9, 9, 6])
        zero_rows = [row for row in rows if row["N"] == 0]
        self.assertTrue(all(row["wall_ms"] == 0 and row["paired_delta_ms"] is None for row in zero_rows))
        self.assertIn("delta_9_minus_6", summary["cached"])
