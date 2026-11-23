"""Fallback keeps usable left observations out of the disparity batch."""
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from lightbackbone_stereo_yolo.detection import DenseStereoPredictions
from lightbackbone_stereo_yolo.geometry import encode_anchorfree_boxes6, generate_point_locations
from lightbackbone_stereo_yolo.model import StereoLocalInstanceHead, StereoYoloLocalModel
from lightbackbone_stereo_yolo.roi_selection import RoiPairSelection


class TinyEncoder(nn.Module):
    def forward(self, left, right):
        return SimpleNamespace(**{name: left.new_zeros((1, 4, size, size))
                                 for name, size in (("p2_left",16), ("p2_right",16), ("p3_left",8),
                                                    ("p4_left",4), ("c5_left",2))})


class FixedDetector(nn.Module):
    def __init__(self, examples):
        super().__init__()
        loc = generate_point_locations(((8,8,8),(4,4,16),(2,2,32)))
        obj = torch.full((1, len(loc)), -30.)
        boxes = torch.zeros((1, len(loc), 6))
        for index, score, endpoints, invalid in examples:
            obj[0,index] = score
            boxes[0,index] = encode_anchorfree_boxes6(torch.tensor([endpoints]), loc[index:index+1])[0]
            if invalid:
                boxes[0,index,4] = float("nan")
        self.predictions = DenseStereoPredictions(obj, torch.zeros(1,len(loc),1), boxes)

    def forward(self, pyramid):
        return self.predictions


class LeftOnlyTest(unittest.TestCase):
    def test_invalid_right_keeps_only_left_and_empty_disparity_batch(self):
        source = FixedDetector([(36, 4., [30.,30.,42.,43.,30.,42.], True)])
        model = StereoYoloLocalModel(TinyEncoder(), source, StereoLocalInstanceHead(4,1,roi_sampling="fixed_56"),
                                     detector_variant="stereo_yolo26_pan",apply_paired_nms=True,
                                     foreground_class_indices=(0,),roi_selection=RoiPairSelection(0,60.,0.))
        image = torch.zeros(1,3,64,64)
        calls = []
        with model.instance_head.idisp.register_forward_hook(lambda *args: calls.append(1)):
            result = model(image,image)
        self.assertEqual(calls, [])
        self.assertEqual(result.pair_ids.tolist(), [])
        self.assertEqual(result.left_only.ids.tolist(), [36])
        self.assertTrue(torch.isnan(result.left_only.distance_m).all())
        torch.testing.assert_close(result.left_only.boxes, torch.tensor([[30.,30.,42.,43.]]))

    def test_overlapping_unknown_duplicate_and_known_far_do_not_join_fallback(self):
        source = FixedDetector([
            (36, 5., [30.,30.,42.,43.,22.,34.], False),
            (37, 4., [31.,30.,47.,43.,31.,47.], True),
            (19, 4., [21.,16.,35.,26.,19.,33.], False),
            (56, 4., [0.,52.,8.,63.,0.,8.], False),
        ])
        model = StereoYoloLocalModel(TinyEncoder(), source, StereoLocalInstanceHead(4,1,roi_sampling="fixed_56"),
                                     detector_variant="stereo_yolo26_pan",apply_paired_nms=True,
                                     foreground_class_indices=(0,),roi_selection=RoiPairSelection(0,60.,0.))
        image = torch.zeros(1,3,64,64)
        result = model(image,image)
        self.assertEqual(result.pair_ids.tolist(), [36])
        self.assertEqual(result.left_only.ids.tolist(), [56])
        self.assertEqual(result.disparity.shape[0],1)
