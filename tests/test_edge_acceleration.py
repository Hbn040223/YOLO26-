"""Regression checks for zero-default stereo acceleration."""
from pathlib import Path
from contextlib import contextmanager
import sys
from types import SimpleNamespace
import unittest

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lightbackbone_stereo_yolo.detection import PairDetections
from lightbackbone_stereo_yolo.model import StereoYoloLocalModel


class EdgeAccelerationTest(unittest.TestCase):
    def test_rgb_is_cropped_once_before_shared_backbone_for_all_deployment_shapes(self):
        import numpy as np
        from PIL import Image
        from torch.nn import functional as F
        from lightbackbone_stereo_yolo.detection import DenseStereoPredictions
        from lightbackbone_stereo_yolo.lightstereo_encoder import LightStereoFeatureEncoder
        from lightbackbone_stereo_yolo.stereo_input import prepare_rectified_images

        class Backbone(nn.Module):
            output_channels = (3, 3, 3, 3)

            def forward(self, x):
                return tuple(F.avg_pool2d(x, stride) for stride in (4, 8, 16, 32))

        class Detector(nn.Module):
            def forward(self, pyramid):
                count = sum(getattr(pyramid, name).shape[-1] * getattr(pyramid, name).shape[-2]
                            for name in ("p3_left", "p4_left", "c5_left"))
                return DenseStereoPredictions(torch.full((1, count), -20.), torch.zeros(1, count, 2),
                                              torch.zeros(1, count, 6))

        rows = np.arange(375, dtype=np.int32)[:, None, None]
        channels = np.array([1, 2, 3])[None, None, :]
        left = np.broadcast_to((rows // channels) % 256, (375, 1242, 3)).astype(np.uint8)
        right = 255 - left
        normalization = {"mean": (0.2, 0.3, 0.4), "std": (0.5, 0.6, 0.7)}
        backbone = Backbone()
        model = StereoYoloLocalModel(LightStereoFeatureEncoder(backbone, 4), Detector(), nn.Identity(),
                                     detector_variant="stereo_yolo26_pan", apply_paired_nms=True)
        seen = []
        hook = backbone.register_forward_pre_hook(lambda _, args: seen.append(args[0].clone()))
        try:
            with torch.inference_mode():
                for crop, padded in ((0, 384), (64, 320), (96, 288), (128, 256)):
                    model.crop_top_px = crop
                    seen.clear()
                    l, r, geometry = prepare_rectified_images(
                        Image.fromarray(left), Image.fromarray(right), normalization, crop)
                    result = model.forward_prepared(l, r, geometry)
                    self.assertEqual(len(seen), 1)
                    self.assertEqual(tuple(seen[0].shape), (2, 3, padded, 1248))
                    self.assertEqual(result.image_hw, (375, 1242))
                    self.assertEqual(result.pair_ids.numel(), 0)
                    expected = torch.tensor(np.stack([left[crop:], right[crop:]])).permute(0, 3, 1, 2).float() / 255.
                    expected = ((expected - torch.tensor(normalization["mean"])[None, :, None, None])
                                / torch.tensor(normalization["std"])[None, :, None, None])
                    torch.testing.assert_close(seen[0][..., :375-crop, :1242], expected, atol=0, rtol=0)
                for wrong in (torch.zeros(1, 3, 375, 1242), l[..., 128:, :], l[:, :1]):
                    with self.assertRaisesRegex(ValueError, "expected.*actual"):
                        model.forward_prepared(wrong, r, geometry)
        finally:
            hook.remove()

    def test_native_nms_profiles_overlaps_transfer_and_greedy_without_changing_pair_ids(self):
        from lightbackbone_stereo_yolo.detection import paired_nms

        boxes = torch.tensor([[0., 0., 20., 10., 0., 20.], [0., 0., 20., 10., 0., 20.]])
        scores = torch.tensor([.8, .9])
        observed = []

        @contextmanager
        def measure(name):
            observed.append(name)
            yield

        output = paired_nms(boxes, scores, profile=measure)
        self.assertEqual(observed, ["paired_iou", "decision_mask_d2h", "greedy_cpu"])
        self.assertEqual(output.indices.tolist(), paired_nms(boxes, scores).indices.tolist())

    def test_containment_requires_the_same_strictly_larger_higher_scoring_pair(self):
        from lightbackbone_stereo_yolo.detection import suppress_contained_pairs

        large = [0, 0, 20, 10, 0, 20]
        contained_at_threshold = [11, 0, 21, 10, 11, 21]
        cases = [
            ([large, contained_at_threshold], [.9, .8], [1, 1], [0]),
            ([large, [11, 0, 21, 10, 19, 29]], [.9, .8], [1, 1], [0, 1]),
            ([[0, 0, 20, 10, 30, 50], [30, 0, 50, 10, 0, 20],
              contained_at_threshold], [.9, .85, .8], [1, 1, 1], [0, 1, 2]),
            ([large, contained_at_threshold], [.9, .8], [1, 2], [0, 1]),
            ([large, contained_at_threshold], [.9, .9], [1, 1], [0, 1]),
            ([large, [11, 7, 18, 7, 11, 18]], [.9, .8], [1, 1], [0, 1]),
            ([large, [11, 1, 18, 8, float("nan"), 18]], [.9, .8], [1, 1], [0, 1]),
            ([], [], [], []),
        ]
        for boxes, scores, classes, expected in cases:
            with self.subTest(boxes=boxes, scores=scores, classes=classes):
                ids = torch.arange(len(scores)) + 100
                pairs = PairDetections(torch.tensor(boxes, dtype=torch.float32).reshape(-1, 6),
                                       torch.tensor(scores), ids, torch.tensor(classes, dtype=torch.long))
                kept, events = suppress_contained_pairs(pairs, .90)
                self.assertEqual(kept.indices.tolist(), [100 + index for index in expected])
                self.assertEqual(kept.scores.tolist(), pairs.scores[expected].tolist())
                self.assertEqual(kept.classes.tolist(), pairs.classes[expected].tolist())
                if len(expected) < len(scores):
                    self.assertEqual((events[0].suppressor_id, events[0].suppressed_id), (100, 101))
                    self.assertAlmostEqual(events[0].left_coverage, .9)
                    self.assertAlmostEqual(events[0].right_coverage, .9)
                else:
                    self.assertFalse(events)

    def test_crop_validates_before_encoding(self):
        image = torch.zeros((1, 3, 64, 64))
        for crop in (-1, 64, 0.5, True):
            with self.subTest(crop=crop), self.assertRaisesRegex(ValueError, "crop_top_px"):
                model = StereoYoloLocalModel(nn.Identity(), nn.Identity(), nn.Identity(), crop_top_px=crop)
                model(image, image)

    def test_crop_shifts_only_vertical_original_coordinates(self):
        from lightbackbone_stereo_yolo.roi import make_pair_crop_windows
        from lightbackbone_stereo_yolo.model import restore_box_rows, restore_crop_rows

        boxes = torch.tensor([[20., 0., 40., 15., 16., 36.]])
        metadata = make_pair_crop_windows(boxes, 48, 64, 16., roi_sampling="fixed_56")
        restored = restore_crop_rows(metadata, 16)
        self.assertEqual(restored.y0.tolist(), [16.])
        self.assertEqual(restored.c_l.tolist(), metadata.c_l.tolist())
        self.assertEqual(restored.c_r.tolist(), metadata.c_r.tolist())
        self.assertEqual(restored.h_crop.tolist(), metadata.h_crop.tolist())
        self.assertEqual(restore_box_rows(boxes[:, :4], 16).tolist(), [[20., 16., 40., 31.]])
        self.assertIs(restore_box_rows(boxes[:, :4], 0)._base, boxes)
        self.assertEqual(restore_box_rows(torch.empty(0, 4), 16).shape, (0, 4))

    def test_fixed_roi_fits_a_truncated_image_when_margin_exceeds_remaining_height(self):
        from lightbackbone_stereo_yolo.roi import make_pair_crop_windows

        boxes = torch.tensor([[20., -4., 40., 50., 16., 36.]])
        crop = make_pair_crop_windows(boxes, 48, 64, 16., roi_sampling="fixed_56",
                                      fit_vertical_extent=True)
        self.assertEqual(crop.y0.tolist(), [0.])
        self.assertEqual(crop.h_crop.tolist(), [48.])
        self.assertEqual(crop.h_roi.tolist(), [56])
        self.assertEqual(crop.c_l.tolist(), [4.])
        self.assertEqual(crop.c_r.tolist(), [0.])
        with self.assertRaisesRegex(ValueError, "taller than"):
            make_pair_crop_windows(boxes, 48, 64, 16., roi_sampling="fixed_56")

    def test_locked_roi_replay_preserves_input_ids_metadata_and_pair_count(self):
        class Detector(nn.Module):
            def forward(self, pyramid):
                self.calls += 1
                return None

        class Instance(nn.Module):
            def forward(self, pyramid, boxes, height, width):
                self.count = len(boxes)
                z = boxes.new_zeros((self.count, 1, 1))
                return SimpleNamespace(mask_logits=z, mask_validity=z.bool(), disparity=z,
                                       validity=z.bool(), peak=z, entropy=z)

        class Encoder(nn.Module):
            def forward(self, left, right):
                return SimpleNamespace(**{name: left for name in ("p2_left", "p2_right", "p3_left", "p4_left")})

        detector = Detector()
        detector.calls = 0
        instance = Instance()
        model = StereoYoloLocalModel(Encoder(), detector, instance, detector_variant="stereo_yolo26_pan")
        image = torch.zeros(1, 3, 64, 64)
        pair = PairDetections(torch.tensor([[6., 0., 32., 26., 2., 28.]]).expand(6, -1),
                              torch.ones(6), torch.arange(10, 16), torch.ones(6, dtype=torch.long))
        output = model(image, image, replay_pairs=pair)
        self.assertEqual(detector.calls, 1)
        self.assertEqual(instance.count, 6)
        self.assertEqual(output.pair_ids.tolist(), list(range(10, 16)))

    def test_export_frontend_passes_dense_outputs_and_only_needed_feature_maps(self):
        from lightbackbone_stereo_yolo.edge_deployment import FRONTEND_OUTPUTS, StaticStereoFrontend

        class Encoder(nn.Module):
            def forward(self, left, right):
                return SimpleNamespace(p2_left=left, p2_right=right, p3_left=left * 3, p4_left=left * 4,
                                       c5_left=left * 5)

        class Detector(nn.Module):
            def forward(self, pyramid):
                return SimpleNamespace(objectness=pyramid.p2_left, class_logits=pyramid.p2_right,
                                       box6=pyramid.p3_left)

        frontend = StaticStereoFrontend(Encoder(), Detector())
        left = torch.ones(1, 3, 8, 8)
        right = left * 2
        tensors = dict(zip(FRONTEND_OUTPUTS, frontend(left, right)))
        self.assertEqual(list(tensors), ["objectness", "class_logits", "box6", "p2_left",
                                         "p2_right", "p3_left", "p4_left", "c5_left"])
        torch.testing.assert_close(tensors["box6"], left * 3)
        torch.testing.assert_close(tensors["c5_left"], left * 5)

    def test_crop_and_containment_reach_the_real_roi_path(self):
        from lightbackbone_stereo_yolo.detection import DenseStereoPredictions
        from lightbackbone_stereo_yolo.geometry import encode_anchorfree_boxes6
        from lightbackbone_stereo_yolo.model import StereoLocalInstanceHead

        class Encoder(nn.Module):
            def forward(self, left, right):
                self.received = left.clone(), right.clone()
                features = {f"{level}_{side}": torch.ones(1, 4, size, size)
                            for level, size in (("p2", 16), ("p3", 8), ("p4", 4), ("c5", 2))
                            for side in ("left", "right")}
                hw = tuple(left.shape[-2:])
                return SimpleNamespace(**features, metadata=SimpleNamespace(
                    original_size=hw, padded_size=tuple(size + (-size) % 32 for size in hw)))

        class Detector(nn.Module):
            one_to_one_enabled = True

            def forward(self, pyramid):
                obj = torch.full((1, 84), -20.)
                obj[0, [9, 10]] = torch.tensor([8., 6.])
                classes = torch.zeros(1, 84, 2)
                classes[0, [9, 10], 1] = 10.
                boxes = torch.zeros(1, 84, 6)
                boxes[0, [9, 10]] = encode_anchorfree_boxes6(
                    torch.tensor([[6., 0., 32., 26., 2., 28.], [16., 8., 24., 16., 12., 20.]]),
                    torch.tensor([[12., 12., 8.], [20., 12., 8.]]),
                )
                return DenseStereoPredictions(obj, classes, boxes)

        encoder = Encoder()
        head = StereoLocalInstanceHead(4, 1, hidden_channels=4, roi_sampling="fixed_56")
        model = StereoYoloLocalModel(encoder, Detector(), head, detector_variant="stereo_yolo26_pan",
                                     apply_paired_nms=True).eval()
        left = torch.arange(80.).view(1, 1, 80, 1).expand(1, 3, 80, 64)
        right = left + 100
        seen = []
        hook = head.register_forward_pre_hook(lambda _, args: seen.append((args[1].clone(), args[2:])))
        try:
            with torch.inference_mode():
                zero = model(left, right)
                self.assertEqual(zero.pair_ids.tolist(), [9, 10])
                model.containment_threshold = .90
                uncropped_one = model(left, right)
                model.crop_top_px = 16
                result = model(left, right)
                from lightbackbone_stereo_yolo.stereo_input import StereoInputGeometry
                prepared = model.forward_prepared(left[..., 16:, :], right[..., 16:, :],
                                                   StereoInputGeometry((80, 64), 16))
        finally:
            hook.remove()
        torch.testing.assert_close(encoder.received[0], left[..., 16:, :])
        torch.testing.assert_close(encoder.received[1], right[..., 16:, :])
        self.assertEqual(result.pair_ids.tolist(), [9])
        self.assertEqual(result.containment_events[0].suppressed_id, 10)
        self.assertEqual(seen[-1][1], (64, 64))
        torch.testing.assert_close(seen[-1][0], seen[0][0][:1])
        torch.testing.assert_close(result.boxes_right, zero.boxes_right[:1] + torch.tensor([0., 16., 0., 16.]))
        torch.testing.assert_close(result.crop.y0, zero.crop.y0[:1] + 16)
        torch.testing.assert_close(result.disparity, uncropped_one.disparity)
        for name in ("pair_ids", "boxes_left", "boxes_right", "disparity", "validity"):
            torch.testing.assert_close(getattr(prepared, name), getattr(result, name), atol=0, rtol=0)
        torch.testing.assert_close(prepared.crop.y0, result.crop.y0, atol=0, rtol=0)
        self.assertEqual(result.image_hw, (80, 64))

    def test_manifest_rejects_previous_schema_and_mixed_nchw_bindings(self):
        from lightbackbone_stereo_yolo.edge_deployment import FRONTEND_SCHEMA, manifest_geometry

        manifest = {"frontend_schema": FRONTEND_SCHEMA, "crop_top_px": 128, "original_hw": [375, 1242],
                    "input_hw": [247, 1242], "encoder_padded_hw": [256, 1248],
                    "bindings": {name: [1, 3, 247, 1242] for name in ("left", "right")}}
        self.assertEqual(manifest_geometry(manifest).cropped_nchw, (1, 3, 247, 1242))
        with self.assertRaisesRegex(ValueError, "expected frontend schema"):
            manifest_geometry({**manifest, "frontend_schema": 1})
        for wrong in ([1, 3, 375, 1242], [1, 3, 119, 1242], [1, 1, 247, 1242]):
            with self.assertRaisesRegex(ValueError, "expected manifest bindings.*actual"):
                manifest_geometry({**manifest, "bindings": {**manifest["bindings"], "right": wrong}})

    def test_tensorrt_runner_rejects_shape_before_using_buffers(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from benchmark_edge_tensorrt import TensorRTFrontend

        runner = object.__new__(TensorRTFrontend)
        runner.input_shape = (1, 3, 247, 1242)
        prepared = torch.empty(runner.input_shape)
        for actual in ((1, 3, 375, 1242), (1, 3, 119, 1242), (1, 1, 247, 1242)):
            with self.assertRaisesRegex(ValueError, "expected TensorRT NCHW.*actual"):
                runner(prepared, torch.empty(actual))
