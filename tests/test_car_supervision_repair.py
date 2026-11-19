"""Camera-pair validity must not erase visible car supervision."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image
import torch
from torch import nn

from lightbackbone_stereo_yolo.anchorfree_training import (
    build_anchorfree_detection_targets,
    training_forward_anchorfree,
)
from lightbackbone_stereo_yolo.data.frame_dataset import KittiFrameDataset, KittiFrameRecord
from lightbackbone_stereo_yolo.data.kitti import KittiManifestRecord
from lightbackbone_stereo_yolo.detection import DenseStereoPredictions, anchorfree_detection_loss


class CarSupervisionRepairTest(unittest.TestCase):
    def test_dataset_retains_left_only_cars_and_ignores_other_classes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            instance = np.zeros((16, 40), dtype=np.uint16)
            disparity = np.zeros_like(instance)
            ignore = np.zeros(instance.shape, dtype=np.uint8)
            for i, x in enumerate((2, 10, 18, 26), 1):
                instance[2:8, x:x + 6] = i
            disparity[2:8, 2:8] = 512
            disparity[3, 20] = 512
            disparity[2:8, 26:32] = 256
            ignore[4, 36] = 255
            paths = tuple(root / name for name in ("left.png", "right.png", "instance.png", "disp.png", "ignore.png"))
            for path, values in zip(paths[2:], (instance, disparity, ignore)):
                Image.fromarray(values).save(path)
            anns = tuple(
                KittiManifestRecord("frame", *paths, category, i, (float(x), 2., x + 5., 7.))
                for i, (category, x) in enumerate(((26, 2), (26, 10), (26, 18), (24, 26)), 1)
            )
            frame = KittiFrameRecord("frame", *paths, anns)
            mapping = {24: 0, 26: 1}
            legacy = KittiFrameDataset([frame], mapping, load_images=False)[0]
            self.assertEqual(len(legacy["boxes6"]), 3)
            dataset = KittiFrameDataset([frame], mapping, load_images=False)
            dataset.repair_car_supervision = True
            sample = dataset[0]
            self.assertEqual(sample.get("detection_classes", torch.empty(0)).tolist(), [1, 1, 1])
            self.assertEqual(sample["classes"].tolist(), [1])
            self.assertEqual(sample["masks"].shape[0], 1)
            torch.testing.assert_close(sample["detection_boxes6"][:, :4], torch.tensor([
                [2., 2., 7., 7.], [10., 2., 15., 7.], [18., 2., 23., 7.],
            ]))
            mask = sample["detection_ignore_mask"]
            self.assertTrue(mask[4, 28])
            self.assertTrue(mask[4, 36])
            self.assertFalse(mask[4, 12])
            self.assertFalse(legacy.get("detection_ignore_mask") is not None)

    def test_ignore_only_removes_background_and_left_only_gradient_stays_left(self):
        locations = torch.tensor([[4., 4., 8.], [12., 4., 8.], [20., 4., 8.]])
        left_only = torch.tensor([[2., 2., 7., 7., 4., 4.]])
        ignored = torch.zeros(16, 24, dtype=torch.bool)
        ignored[4, 4:13] = True
        targets = build_anchorfree_detection_targets(
            locations, left_only, torch.tensor([1]), image_hw=(16, 24), ignore_mask=ignored,
        )
        self.assertEqual(targets["labels"].tolist(), [1, -1, 0])
        self.assertEqual(targets["left_box_valid_mask"].tolist(), [True, False, False])
        self.assertFalse(targets["box_valid_mask"].any())
        predictions = {
            "objectness": torch.zeros(3, requires_grad=True),
            "class_logits": torch.zeros(3, 2, requires_grad=True),
            "box6": torch.zeros(3, 6, requires_grad=True),
        }
        losses = anchorfree_detection_loss(predictions, targets, locations, box_supervision_mode="split_left")
        (losses["loss_obj"] + losses["loss_cls"] + losses["loss_box6"]).backward()
        self.assertLess(float(predictions["objectness"].grad[0]), 0.)
        self.assertEqual(float(predictions["objectness"].grad[1]), 0.)
        self.assertGreater(float(predictions["box6"].grad[0, :4].abs().sum()), 0.)
        self.assertEqual(float(predictions["box6"].grad[:, 4:].abs().sum()), 0.)


    def test_c2_sparse_assignment_reserves_paired_points_then_left_only(self):
        from lightbackbone_stereo_yolo.anchorfree_training import build_global_one_to_one_targets
        from lightbackbone_stereo_yolo.geometry import encode_anchorfree_boxes6

        points = torch.tensor([[12., 12., 8.], [20., 12., 8.], [32., 12., 8.]])
        boxes = torch.tensor([[8., 8., 24., 20., 2., 18.], [8., 8., 23., 19., 15., 15.]])
        trial = torch.tensor([[8., 8., 24., 20., 2., 18.], [8., 8., 23., 19., 1., 21.], [28., 8., 36., 20., 2., 18.]])
        pred = {"box6": encode_anchorfree_boxes6(trial, points), "class_logits": torch.zeros(3, 2)}
        kwargs = {"image_hw": (40, 40), "right_offset_mode": "signed",
                  "assignment_mode": "global_b2_cross_scale"}
        off, old_quality, _ = build_global_one_to_one_targets(pred, boxes, torch.tensor([0, 1]), points, **kwargs)
        on, new_quality, counts = build_global_one_to_one_targets(
            pred, boxes, torch.tensor([0, 1]), points, supervise_left_only=True, **kwargs,
        )
        self.assertEqual(off["matched_indices"].tolist(), [0, -1, -1])
        self.assertEqual(on["matched_indices"].tolist(), [0, 1, -1])
        self.assertEqual(on["box_valid_mask"].tolist(), [True, False, False])
        self.assertEqual(on["left_box_valid_mask"].tolist(), [True, True, False])
        torch.testing.assert_close(old_quality, new_quality)
        self.assertEqual(counts["left_only_selected"], 1)

    def test_c2_dense_control_keeps_pair_positions_and_ignores_unlabelled_left_only_points(self):
        from lightbackbone_stereo_yolo.anchorfree_training import build_car_supervision_targets

        points = torch.tensor([[4., 4., 8.], [12., 4., 8.], [20., 4., 8.], [28., 4., 8.]])
        boxes = torch.tensor([[2., 2., 7., 7., 0., 5.], [10., 2., 15., 7., 12., 12.],
                              [18., 2., 23., 7., 20., 20.]])
        ignore = torch.zeros(20, 32, dtype=torch.bool)
        ignore[4, 20] = True
        args = (points, boxes, torch.tensor([0, 0, 0]))
        kwargs = {"assignment_mode": "legacy", "image_hw": (20, 32), "ignore_mask": ignore}
        off = build_car_supervision_targets(*args, **kwargs, supervise_left_only=False)
        on = build_car_supervision_targets(*args, **kwargs, supervise_left_only=True)
        self.assertEqual(off["labels"].tolist(), [1, -1, -1, 0])
        self.assertEqual(on["labels"].tolist(), [1, 1, -1, 0])
        self.assertEqual(off["box_valid_mask"].tolist(), [True, False, False, False])
        torch.testing.assert_close(off["box_valid_mask"], on["box_valid_mask"])

    def test_c2_sparse_left_only_backpropagates_left_and_class_but_no_paired_objectness(self):
        from lightbackbone_stereo_yolo.anchorfree_training import one_to_one_duplicate_loss

        points = torch.tensor([[12., 12., 8.], [20., 12., 8.]])
        pred = {"box6": torch.zeros(2, 6, requires_grad=True),
                "class_logits": torch.zeros(2, 2, requires_grad=True),
                "objectness": torch.zeros(2, requires_grad=True)}
        one = {
            "labels": torch.tensor([1, 0]), "box6": torch.ones(2, 6),
            "classes": torch.tensor([1, 0]), "box_valid_mask": torch.zeros(2, dtype=torch.bool),
            "left_box_valid_mask": torch.tensor([True, False]), "matched_indices": torch.tensor([0, -1]),
        }
        many = {
            "labels": torch.tensor([1, 1]), "box_valid_mask": torch.tensor([False, True]),
            "left_box_valid_mask": torch.tensor([True, True]), "matched_indices": torch.tensor([0, 0]),
        }
        losses = anchorfree_detection_loss(
            pred, one, points, box_supervision_mode="split_left", one_to_one_quality=torch.zeros(2),
        )
        rank = one_to_one_duplicate_loss(pred["objectness"], many, one)
        self.assertEqual(float(rank), 0.)
        self.assertEqual(float(losses["loss_pair_iou"]), 0.)
        (losses["loss_box6"] + losses["loss_obj"] + losses["loss_cls"] + rank).backward()
        self.assertEqual(float(pred["objectness"].grad[0]), 0.)
        self.assertGreater(float(pred["objectness"].grad[1]), 0.)
        self.assertGreater(float(pred["class_logits"].grad[0].abs().sum()), 0.)
        self.assertGreater(float(pred["box6"].grad[0, :4].abs().sum()), 0.)
        self.assertEqual(float(pred["box6"].grad[:, 4:].abs().sum()), 0.)

    def test_normal_detector_forward_consumes_repaired_fields(self):
        class TinyEncoder(nn.Module):
            def forward(self, left, right):
                tap = left[:, :1, :1, :1]
                return SimpleNamespace(p3_left=tap, p4_left=tap, c5_left=tap)

        class TinyDetector(nn.Module):
            def forward(self, pyramid):
                return DenseStereoPredictions(
                    torch.zeros(1, 3), torch.zeros(1, 3, 2), torch.zeros(1, 3, 6),
                )

        image = torch.zeros(1, 3, 32, 32)
        target = {
            "boxes6": torch.empty(0, 6),
            "classes": torch.empty(0, dtype=torch.long),
            "detection_boxes6": torch.tensor([[2., 2., 7., 7., 4., 4.]]),
            "detection_classes": torch.tensor([1]),
            "detection_ignore_mask": torch.zeros(32, 32, dtype=torch.bool),
        }
        output = training_forward_anchorfree(
            TinyEncoder(), TinyDetector(), None, image, image, target,
            dmax_roi=2, mask_weight=0., disparity_weight=0., box_supervision_mode="split_left",
        )
        self.assertGreater(float(output["loss_left"]), 0.)
        self.assertGreater(float(output["loss_cls"]), 0.)
        self.assertEqual(float(output["loss_disparity"]), 0.)
        self.assertEqual(float(output["loss_pair_iou"]), 0.)
