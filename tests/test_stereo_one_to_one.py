from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from lightbackbone_stereo_yolo.anchorfree_training import (
    build_anchorfree_detection_targets, training_forward_anchorfree,
)
from lightbackbone_stereo_yolo.detection import decode_anchorfree_predictions
from lightbackbone_stereo_yolo.geometry import encode_anchorfree_boxes6
from lightbackbone_stereo_yolo.model import StereoYoloLocalModel
from lightbackbone_stereo_yolo.stereo_yolo26 import StereoYOLO26PanHead, initialize_one_to_one_weights


def pyramid():
    return SimpleNamespace(**{
        f"{level}_{side}": torch.randn(1, 8, size, size)
        for level, size in (("p3", 8), ("p4", 4), ("c5", 2))
        for side in ("left", "right")
    })


class StereoOneToOneTest(unittest.TestCase):
    def test_one_to_one_has_separate_heads_and_one_shared_neck(self):
        self.assertIn("one_to_one", StereoYOLO26PanHead.__init__.__code__.co_varnames)
        detector = StereoYOLO26PanHead(8, 2, one_to_one=True)
        traces = []
        handles = [
            module.register_forward_hook(lambda _, args, output, key=key: traces.append(key))
            for key, module in (("neck", detector.neck), ("many", detector.levels[0]),
                                ("one", detector.one_to_one_levels[0]))
        ]
        many, one = detector.forward_training(pyramid())
        self.assertEqual(traces, ["neck", "many", "one"])
        torch.testing.assert_close(many.box6, one.box6)
        one.objectness.sum().backward()
        self.assertTrue(any(p.grad is not None for p in detector.one_to_one_levels.parameters()))
        self.assertTrue(all(p.grad is None for p in detector.neck.parameters()))
        traces.clear()
        detector.eval()(pyramid())
        self.assertEqual(traces, ["neck", "one"])
        for handle in handles:
            handle.remove()

    def test_assignment_keeps_best_pair_for_each_target_and_demotes_duplicates(self):
        from lightbackbone_stereo_yolo import anchorfree_training
        self.assertTrue(hasattr(anchorfree_training, "build_one_to_one_targets"))
        points = torch.tensor([[12.,12.,8.], [20.,12.,8.], [52.,12.,8.],
                               [60.,12.,8.], [76.,12.,8.]])
        boxes = torch.tensor([[8.,8.,28.,28.,1.,21.], [48.,8.,68.,28.,44.,64.],
                              [72.,8.,84.,28.,76.,76.]])
        labels = torch.tensor([0, 1, 0])
        base = build_anchorfree_detection_targets(points, boxes, labels, image_hw=(96,96))
        trial = boxes[[0,0,1,1,2]].clone()
        trial[0, 4:] -= 24
        trial[3, 4:] -= 24
        dense = {"box6": encode_anchorfree_boxes6(trial, points),
                 "class_logits": torch.zeros(5,2), "objectness": torch.zeros(5)}
        targets, q = anchorfree_training.build_one_to_one_targets(
            dense, base, points, right_offset_mode="signed",
        )
        self.assertEqual(targets["labels"].tolist(), [0,1,1,0,-1])
        self.assertEqual(targets["matched_indices"].tolist(), [-1,0,1,-1,-1])
        self.assertEqual(base["labels"].tolist(), [1,1,1,1,1])
        torch.testing.assert_close(q, torch.tensor([0.,1.,1.,0.,0.]), atol=1e-5, rtol=0.)


    def test_global_assignment_reassigns_a_shared_point_to_cover_both_targets(self):
        from lightbackbone_stereo_yolo import anchorfree_training

        points = torch.tensor([[12., 12., 8.], [20., 12., 8.], [28., 12., 8.]])
        boxes = torch.tensor([[8., 8., 24., 20., 2., 18.], [9., 8., 15., 20., 5., 11.]])
        pred = {"box6": encode_anchorfree_boxes6(boxes[[1, 0, 0]], points),
                "class_logits": torch.zeros(3, 2)}
        target, quality, counts = anchorfree_training.build_global_one_to_one_targets(
            pred, boxes, torch.tensor([0, 1]), points, image_hw=(40, 40), right_offset_mode="signed",
        )
        self.assertEqual(target["matched_indices"].tolist(), [1, 0, -1])
        self.assertEqual(target["classes"][target["labels"] == 1].tolist(), [1, 0])
        torch.testing.assert_close(target["matched_boxes6"][:2], boxes[[1, 0]])
        torch.testing.assert_close(quality, torch.tensor([1., 1., 0.]), atol=1e-5, rtol=0)
        self.assertEqual(counts["full_pool_conflicts"], 0)

    def test_global_assignment_reopens_full_pool_when_top12_leaves_a_target_unmatched(self):
        from lightbackbone_stereo_yolo import anchorfree_training

        points = torch.tensor([(x, y, 8.) for y in (4., 12., 20.) for x in (4., 12., 20., 28., 36.)][:13])
        boxes = torch.tensor([[0., 0., 40., 40., -8., 32.]]).expand(14, -1)
        pred = {"box6": encode_anchorfree_boxes6(boxes[:13], points),
                "class_logits": torch.zeros(13, 1)}
        kwargs = {"image_hw": (40, 40), "right_offset_mode": "signed"}
        inputs = (pred, boxes, torch.zeros(14, dtype=torch.long), points)
        target, _, counts = anchorfree_training.build_global_one_to_one_targets(*inputs, **kwargs)
        again, _, _ = anchorfree_training.build_global_one_to_one_targets(*inputs, **kwargs)
        self.assertEqual(int((target["labels"] == 1).sum()), 13)
        self.assertEqual(target["matched_indices"].unique().numel(), 13)
        self.assertEqual(counts["top12_unmatched"], 2)
        self.assertEqual(counts["top12_recovered"], 1)
        self.assertEqual(counts["full_pool_conflicts"], 1)
        torch.testing.assert_close(again["matched_indices"], target["matched_indices"])

    def test_global_assignment_handles_empty_annotations_ignore_and_decode_overflow(self):
        from lightbackbone_stereo_yolo import anchorfree_training

        points = torch.tensor([[4., 4., 8.], [12., 4., 8.], [20., 4., 8.]])
        boxes = torch.tensor([[0., 0., 16., 8., -1., 15.]])
        dense_boxes = encode_anchorfree_boxes6(boxes.expand(3, -1), points)
        dense_boxes[1, 0] = 1000.  # Finite encoding, non-finite decoded coordinate.
        pred = {"box6": dense_boxes, "class_logits": torch.zeros(3, 1)}
        ignore = torch.zeros(16, 24, dtype=torch.bool)
        ignore[4, 4] = True
        kwargs = {"image_hw": (16, 24), "ignore_mask": ignore, "right_offset_mode": "signed"}
        target, quality, counts = anchorfree_training.build_global_one_to_one_targets(
            pred, boxes, torch.tensor([0]), points, **kwargs,
        )
        self.assertEqual(target["labels"].tolist(), [-1, -1, 0])
        self.assertEqual(counts["no_legal"], 1)
        self.assertFalse(quality.any())
        empty, _, counts = anchorfree_training.build_global_one_to_one_targets(
            pred, boxes[:0], torch.empty(0, dtype=torch.long), points, **kwargs,
        )
        self.assertEqual(empty["labels"].tolist(), [-1, 0, 0])
        self.assertEqual(counts["no_legal"], 0)


    def test_b2_cross_scale_rescues_missing_primary_level_without_duplicate_targets(self):
        from lightbackbone_stereo_yolo import anchorfree_training

        points = torch.tensor([[32., 32., 32.], [32., 32., 16.]])
        boxes = torch.tensor([[24., 24., 46., 46., 20., 42.]])
        pred = {"box6": encode_anchorfree_boxes6(boxes.expand(2, -1), points),
                "class_logits": torch.zeros(2, 1)}
        args = (pred, boxes, torch.tensor([0]), points)
        kwargs = {"image_hw": (64, 64), "right_offset_mode": "signed"}
        _, _, b1 = anchorfree_training.build_global_one_to_one_targets(*args, **kwargs)
        target, quality, b2 = anchorfree_training.build_global_one_to_one_targets(
            *args, **kwargs, assignment_mode="global_b2_cross_scale",
        )
        self.assertEqual(b1["no_legal"], 1)
        self.assertEqual(b2["no_legal"], 0)
        self.assertEqual(int((target["labels"] == 1).sum()), 1)
        self.assertEqual(b2["selected_p3"], 0)
        self.assertEqual(b2["selected_p4"] + b2["selected_p5"], 1)
        torch.testing.assert_close(quality.max(), torch.tensor(1.), atol=1e-5, rtol=0.)

    def test_b2_quality_changes_the_winner_while_keeping_the_same_legal_pool(self):
        from lightbackbone_stereo_yolo import anchorfree_training

        points = torch.tensor([[12., 12., 8.], [20., 12., 8.], [20., 12., 16.]])
        boxes = torch.tensor([[8., 8., 28., 28., 1., 21.]])
        predictions = torch.tensor([
            [8., 8., 20., 28., 1., 21.],
            [8., 8., 28., 28., 1., 10.],
            [8., 8., 28., 28., 1., 21.],
        ])
        pred = {"box6": encode_anchorfree_boxes6(predictions, points),
                "class_logits": torch.zeros(3, 2)}
        args = (pred, boxes, torch.tensor([0]), points)
        kwargs = {"image_hw": (40, 40), "right_offset_mode": "signed"}
        b1, _, _ = anchorfree_training.build_global_one_to_one_targets(*args, **kwargs)
        b2, quality, _ = anchorfree_training.build_global_one_to_one_targets(
            *args, **kwargs, assignment_mode="global_b2_quality",
        )
        self.assertEqual(b1["labels"].tolist(), [1, 0, -1])
        self.assertEqual(b2["labels"].tolist(), [0, 1, -1])
        torch.testing.assert_close(quality, torch.tensor([0., .45, 0.]), atol=1e-5, rtol=0.)

    def test_b2_quality_breaks_exact_zero_ties_by_center_then_dense_id(self):
        from lightbackbone_stereo_yolo import anchorfree_training

        points = torch.tensor([[12., 12., 8.], [16., 20., 8.], [24., 20., 8.]])
        boxes = torch.tensor([[8., 8., 32., 32., 1., 25.]])
        # Every left and right IoU is tiny and the target class probability is
        # exactly zero. This keeps the primary assignment score tied.
        pred = {"box6": torch.zeros(3, 6),
                "class_logits": torch.tensor([[-1000., 0.]]).expand(3, -1)}
        target, _, _ = anchorfree_training.build_global_one_to_one_targets(
            pred, boxes, torch.tensor([0]), points, image_hw=(40, 40),
            right_offset_mode="signed", assignment_mode="global_b2_quality",
        )
        self.assertEqual(target["labels"].tolist(), [0, 1, 0])

    def test_duplicate_ranking_uses_the_independent_winners_target(self):
        from lightbackbone_stereo_yolo import anchorfree_training

        logits = torch.zeros(3, requires_grad=True)
        many = {"matched_indices": torch.tensor([0, 0, 1]), "labels": torch.ones(3).long(),
                "box_valid_mask": torch.ones(3).bool(), "left_box_valid_mask": torch.ones(3).bool()}
        one = {"labels": torch.tensor([1, 0, 0]), "matched_indices": torch.tensor([1, -1, -1])}
        loss = anchorfree_training.one_to_one_duplicate_loss(logits, many, one)
        loss.backward()
        self.assertEqual(logits.grad.tolist(), [-1., 0., 1.])

    def test_sparse_inference_bypasses_suppression_and_keeps_pair_ids(self):
        self.assertIn("apply_nms", decode_anchorfree_predictions.__code__.co_varnames)
        points = torch.tensor([[12.,12.,8.], [20.,12.,8.]])
        boxes = torch.tensor([[8.,8.,28.,28.,1.,21.], [8.,8.,28.,28.,1.,21.]])
        dense = {"box6": encode_anchorfree_boxes6(boxes,points),
                 "class_logits": torch.tensor([[0.,4.],[4.,0.]]), "objectness": torch.tensor([2.,4.])}
        with patch("lightbackbone_stereo_yolo.detection.paired_nms", side_effect=AssertionError):
            kept = decode_anchorfree_predictions(dense, points, .25, apply_nms=False)
        self.assertEqual(kept.indices.tolist(), [1,0])
        self.assertEqual(kept.classes.tolist(), [0,1])
        torch.testing.assert_close(kept.boxes6, boxes[kept.indices])

    def test_duplicate_ranking_pushes_winner_above_hardest_sibling(self):
        from lightbackbone_stereo_yolo import anchorfree_training
        self.assertTrue(hasattr(anchorfree_training, "one_to_one_duplicate_loss"))
        logits = torch.tensor([0.8, 0.9, 0.1, 0.7, 5.0], requires_grad=True)
        many = {"matched_indices": torch.tensor([0, 0, 0, 1, -1]),
                "labels": torch.tensor([1, 1, 1, 1, 0]),
                "box_valid_mask": torch.tensor([1, 1, 1, 1, 0]).bool(),
                "left_box_valid_mask": torch.tensor([1, 1, 1, 1, 0]).bool()}
        one = {"labels": torch.tensor([1, 0, 0, 1, 0])}
        loss = anchorfree_training.one_to_one_duplicate_loss(logits, many, one)
        torch.testing.assert_close(loss, torch.tensor(1.1))
        loss.backward()
        self.assertEqual(logits.grad.tolist(), [-1., 1., 0., 0., 0.])

    def test_legacy_checkpoint_initializes_sparse_with_independent_parameters(self):
        legacy = torch.nn.ModuleDict({"detector": StereoYOLO26PanHead(8, 2)})
        sparse = torch.nn.ModuleDict({"detector": StereoYOLO26PanHead(8, 2, one_to_one=True)})
        sparse.load_state_dict(initialize_one_to_one_weights(legacy.state_dict()), strict=True)
        dense_weight = next(sparse["detector"].levels.parameters())
        sparse_weight = next(sparse["detector"].one_to_one_levels.parameters())
        torch.testing.assert_close(dense_weight, sparse_weight)
        self.assertNotEqual(dense_weight.data_ptr(), sparse_weight.data_ptr())

    def test_full_inference_passes_one_to_one_pairs_to_roi_without_second_decode(self):
        from lightbackbone_stereo_yolo.detection import DenseStereoPredictions

        class Encoder(torch.nn.Module):
            def forward(self, left, right):
                out = pyramid()
                out.p2_left = out.p2_right = torch.ones(1,8,16,16)
                return out

        class Detector(torch.nn.Module):
            one_to_one_enabled = True

            def forward(self, features):
                obj = torch.full((1,84), -20.)
                obj[0,9] = 8.
                cls = torch.zeros(1,84,2)
                cls[0,9,1] = 10.
                box = torch.zeros(1,84,6)
                box[0,9] = encode_anchorfree_boxes6(torch.tensor([[8.,8.,28.,28.,1.,21.]]),
                                                    torch.tensor([[12.,12.,8.]]))[0]
                return DenseStereoPredictions(obj,cls,box)

        class Instance(torch.nn.Module):
            def forward(self, pyramid, boxes, height, width):
                self.received = boxes
                z = boxes.new_zeros((boxes.shape[0],1,1))
                return SimpleNamespace(mask_logits=z,mask_validity=z.bool(),disparity=z,
                                       validity=z.bool(),peak=z,entropy=z)

        instance = Instance()
        model = StereoYoloLocalModel(Encoder(), Detector(), instance, score_threshold=.25,
                                     detector_variant="stereo_yolo26_pan")
        image = torch.zeros(1,3,64,64)
        with patch("lightbackbone_stereo_yolo.detection.paired_nms", side_effect=AssertionError):
            result = model.eval()(image,image)
        torch.testing.assert_close(instance.received,torch.tensor([[8.,8.,28.,28.,1.,21.]]))
        torch.testing.assert_close(result.boxes_right,torch.tensor([[1.,8.,21.,28.]]))
        self.assertEqual(result.classes.tolist(),[1])

    def test_training_path_uses_both_branches_with_one_encoder_pass(self):
        self.assertIn("one_to_one", StereoYOLO26PanHead.__init__.__code__.co_varnames)

        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, left, right):
                self.calls += 1
                return pyramid()

        enc = Encoder()
        detector = StereoYOLO26PanHead(8, 2, one_to_one=True)
        images = torch.zeros(1,3,64,64)
        output = training_forward_anchorfree(
            enc, detector, torch.nn.Identity(), images, images,
            {"boxes6": torch.tensor([[8.,8.,28.,28.,1.,21.]]), "classes": torch.tensor([0])},
            28, mask_weight=0., disparity_weight=0., one_to_one_assignment_mode="global_b1",
        )
        self.assertEqual(output["one_to_one_assignment_counts"]["no_legal"], 0)
        self.assertEqual(enc.calls,1)
        self.assertEqual(int(output["one_to_one_positives"]),1)
        output["loss_total"].backward()
        for layers in (detector.levels, detector.one_to_one_levels):
            self.assertTrue(all(p.grad is not None for p in layers.parameters()))


if __name__ == "__main__":
    unittest.main()

