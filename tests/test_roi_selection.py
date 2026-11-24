import unittest
import torch
from lightbackbone_stereo_yolo.detection import PairDetections
from lightbackbone_stereo_yolo.roi_selection import RoiPairSelection, keep_foreground_pairs

class RoiSelectionTest(unittest.TestCase):
    def test_invalid_calibration_fails_explicitly(self):
        with self.assertRaises(ValueError):
            RoiPairSelection(1, 0., 0.)

    def test_foreground_tie_preserves_dense_ids_and_classes(self):
        p=PairDetections(torch.tensor([[0.,0.,10.,10.,0.,10.]]).repeat(3,1),
                         torch.tensor([.84,.22,.84]),torch.tensor([15,8,19]),torch.tensor([1,1,2]))
        masks=torch.ones(3,4,4,dtype=torch.bool)
        keep=keep_foreground_pairs(p,masks)
        self.assertEqual(keep.tolist(),[0,2])

    def test_empty_and_disjoint_masks_preserve_candidates(self):
        p=PairDetections(torch.tensor([[0.,0.,10.,10.,0.,10.]]).repeat(2,1),
                         torch.ones(2),torch.arange(2),torch.ones(2,dtype=torch.long))
        masks=torch.zeros(2,4,4,dtype=torch.bool)
        self.assertEqual(keep_foreground_pairs(p,masks).tolist(),[0,1])

class Near10Test(unittest.TestCase):
    def pairs(self, boxes):
        count = len(boxes)
        return PairDetections(torch.tensor(boxes, dtype=torch.float32),
                              torch.linspace(.9, .8, count), torch.arange(101, 101 + count),
                              torch.ones(count, dtype=torch.long))

    def test_far_candidate_does_not_remove_unknown_left_observation(self):
        from lightbackbone_stereo_yolo.roi_selection import Near10Policy, near10_geometry
        pairs = self.pairs([[40, 10, 60, 30, 38, 58], [40, 10, 60, 30, 40, 60],
                            [108, 10, 118, 30, 60, 70]])
        geometry = near10_geometry(pairs, (100, 100), (390., 0.), Near10Policy("range", 1))
        self.assertEqual(pairs.indices[geometry.far].tolist(), [101])
        self.assertEqual(pairs.indices[geometry.left_only].tolist(), [102])
        self.assertEqual(pairs.indices[geometry.invalid].tolist(), [103])

    def test_duplicate_requires_both_views_and_foreground(self):
        from lightbackbone_stereo_yolo.roi_selection import Near10Policy, near10_conflicts, near10_keep
        policy = Near10Policy("range-dedup", 1)
        pairs = self.pairs([[100, 20, 120, 40, 0, 20], [107, 20, 127, 40, 30, 50],
                            [100, 20, 120, 40, 0, 20]])
        conflicts = near10_conflicts(pairs, torch.tensor([100., 77., 100.]), policy)
        masks = torch.ones(3, 4, 4, dtype=torch.bool)
        self.assertEqual(pairs.indices[near10_keep(pairs, conflicts, masks, policy)].tolist(), [101, 102])
        masks[2] = False
        self.assertEqual(pairs.indices[near10_keep(pairs, conflicts, masks, policy)].tolist(), [101, 102, 103])

    def test_range_preserves_adjacent_near_pairs(self):
        from lightbackbone_stereo_yolo.roi_selection import Near10Policy, near10_geometry
        pairs = self.pairs([[60, 20, 80, 40, 20, 40], [67, 20, 87, 40, 27, 47]])
        geometry = near10_geometry(pairs, (100, 100), (390., 0.), Near10Policy("range", 1))
        self.assertEqual(pairs.indices[geometry.admitted].tolist(), [101, 102])

    def test_measured_ten_meter_boundary_and_unknown(self):
        from lightbackbone_stereo_yolo.roi_selection import near10_measure
        disparity = torch.tensor([[[40., 40.]], [[39., 39.]], [[38., 38.]], [[40., 40.]]])
        valid = torch.ones_like(disparity, dtype=torch.bool)
        valid[-1] = False
        distance = near10_measure(disparity, valid, (390., 1.))
        self.assertEqual(distance[0].item(), 10.)
        self.assertGreater(distance[1].item(), 10.)
        self.assertTrue(distance[-1].isnan())

if __name__=="__main__":
    unittest.main()
