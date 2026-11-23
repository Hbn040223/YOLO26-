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

if __name__=="__main__":
    unittest.main()
