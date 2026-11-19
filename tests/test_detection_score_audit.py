"""Fixed-weight score audit must distinguish geometry, score factors and rank."""
import unittest

import torch

from lightbackbone_stereo_yolo.score_diagnostics import audit_score_candidates, budget_keep_indices


class DetectionScoreAuditTest(unittest.TestCase):
    def test_low_score_blockers_are_partitioned_using_same_candidate_pair_geometry(self):
        gt = torch.tensor([[0., 0., 10., 10., -1., 9.],
                           [20., 0., 30., 10., 19., 29.],
                           [40., 0., 50., 10., 39., 49.]])
        candidates = gt.clone()
        candidates[0, 4:] += 20.
        obj = torch.tensor([.1, .24, .40])
        probs = torch.tensor([[.9,.1], [.99,.01], [.4,.6]])
        rows = audit_score_candidates(candidates, obj, probs, gt, torch.zeros(3, dtype=torch.long))
        self.assertTrue(all(row["left_filtered_by_score"] for row in rows))
        self.assertEqual([row["joint_score_blocker"] for row in rows],
                         ["no_joint_geometry", "objectness_below_threshold", "product_below_threshold"])
        self.assertEqual(rows[2]["best_joint_score"]["pred_class"], 1)
        self.assertAlmostEqual(rows[2]["best_joint_score"]["gt_class_probability"], .4)
        self.assertAlmostEqual(rows[2]["best_joint_score"]["final_score"], .24)

    def test_bad_right_joint_geometry_can_outrank_a_good_pair(self):
        gt = torch.tensor([[20., 0., 30., 10., 19., 29.]])
        boxes = gt.repeat(2, 1)
        boxes[0, 4:] += 18.
        rows = audit_score_candidates(boxes, torch.tensor([.9,.2]), torch.ones(2,1), gt, torch.zeros(1,dtype=torch.long))
        row = rows[0]
        self.assertEqual(row["best_left_score"]["dense_index"], 0)
        self.assertEqual(row["best_joint_score"]["dense_index"], 1)
        self.assertEqual(row["best_joint_geometry"]["dense_index"], 1)
        self.assertTrue(row["bad_right_outranks_joint"])
        self.assertFalse(row["left_filtered_by_score"])
        self.assertEqual(row["joint_score_blocker"], "objectness_below_threshold")

    def test_budget_respects_exact_total_and_deterministic_ties(self):
        frames = [torch.tensor([.5, .1]), torch.tensor([.8, .5, .4]), torch.empty(0)]
        picked = budget_keep_indices(frames, 2)
        self.assertEqual([x.tolist() for x in picked], [[0], [0], []])
        self.assertEqual([x.tolist() for x in budget_keep_indices(frames, 0)], [[], [], []])
