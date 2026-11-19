import unittest

import torch

from lightbackbone_stereo_yolo.detection import PairDetections
from lightbackbone_stereo_yolo.roi_budget import RectifiedStereoCalibration, coarse_depth_m, select_roi_pairs


class RoiBudgetTests(unittest.TestCase):
    def setUp(self):
        disparities = torch.tensor([12., 12., 12., 12., 52.])
        left = torch.tensor([[60., 10., 100., 40.]]).expand(5, -1)
        boxes = torch.cat((left, left[:, [0, 2]] - disparities[:, None]), dim=1)
        self.pairs = PairDetections(boxes, torch.tensor([.9, .8, .7, .6, .5]),
                                    torch.arange(10, 15), torch.arange(5))
        self.calibration = RectifiedStereoCalibration(384., 4.)

    def test_depth_uses_box_centres_and_principal_point_correction(self):
        torch.testing.assert_close(coarse_depth_m(self.pairs.boxes6, self.calibration),
                                   torch.tensor([48., 48., 48., 48., 8.]))

    def test_reservation_replaces_tail_and_keeps_pair_fields_aligned(self):
        result = select_roi_pairs(self.pairs, max_pairs=3, near_quota=2, calibration=self.calibration)
        rows = torch.tensor([0, 1, 4])
        torch.testing.assert_close(result.boxes6, self.pairs.boxes6[rows])
        torch.testing.assert_close(result.scores, self.pairs.scores[rows])
        self.assertEqual(result.indices.tolist(), [10, 11, 14])
        self.assertEqual(result.classes.tolist(), [0, 1, 4])

    def test_missing_or_invalid_calibration_uses_original_order(self):
        for calibration in (None, RectifiedStereoCalibration(-384., 4.),
                            RectifiedStereoCalibration(384., float("nan"))):
            selected = select_roi_pairs(self.pairs, max_pairs=3, near_quota=2, calibration=calibration)
            self.assertEqual(selected.indices.tolist(), [10, 11, 12])

    def test_unlimited_and_over_budget_return_original(self):
        for cap in (None, 5, 100):
            self.assertIs(select_roi_pairs(self.pairs, max_pairs=cap), self.pairs)
        empty = select_roi_pairs(self.pairs, max_pairs=0)
        self.assertEqual(tuple(empty.boxes6.shape), (0, 6))
        self.assertIs(select_roi_pairs(empty, max_pairs=4, near_quota=2), empty)

    def test_bad_disparity_cannot_reserve_and_classes_are_optional(self):
        boxes = self.pairs.boxes6.clone()
        boxes[-1, 4:] = boxes[-1, [0, 2]]
        pairs = PairDetections(boxes, self.pairs.scores, self.pairs.indices)
        result = select_roi_pairs(pairs, max_pairs=3, near_quota=1, calibration=self.calibration)
        self.assertEqual(result.indices.tolist(), [10, 11, 12])
        self.assertIsNone(result.classes)


if __name__ == "__main__":
    unittest.main()
