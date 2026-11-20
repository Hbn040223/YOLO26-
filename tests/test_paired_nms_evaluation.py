"""Regression tests for deterministic, pair-preserving NMS and its on/off audit."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

from evaluate_paired_nms import evaluate
from benchmark_paired_nms import benchmark
from lightbackbone_stereo_yolo import detection
from lightbackbone_stereo_yolo.detection import anchorfree_candidate_indices, paired_nms
from lightbackbone_stereo_yolo.training_metrics import greedy_match_paired_detections


def legacy_paired_nms_indices(boxes, scores, threshold, classes):
    order = scores.argsort(descending=True, stable=True)
    right = boxes[:, [4, 1, 5, 3]]
    keep = []
    while order.numel():
        keep.append(order[0])
        if order.numel() == 1:
            break
        first, rest = order[:1], order[1:]
        overlaps = torch.minimum(
            detection._iou(boxes[first, :4], boxes[rest, :4]),
            detection._iou(right[first], right[rest]),
        )[0]
        survives = overlaps <= threshold
        if classes is not None:
            survives |= classes[rest] != classes[first]
        order = rest[survives]
    return torch.stack(keep) if keep else order


class PairedNmsEvaluationTest(unittest.TestCase):
    def test_stable_class_aware_nms_preserves_single_view_overlap(self):
        boxes = torch.tensor([[0., 0., 4., 4., 0., 4.],
                              [0., 0., 4., 4., 0., 4.],
                              [0., 0., 4., 4., 0., 4.],
                              [0., 0., 4., 4., 8., 12.],
                              [0., 0., 4., 4., 0., 2.]])
        scores = torch.full((5,), .5)
        classes = torch.tensor([0, 0, 1, 0, 0])
        kept = paired_nms(boxes, scores, .5, classes=classes)
        # Row 4 is exactly at the threshold, and equality survives.
        self.assertEqual(kept.indices.tolist(), [0, 2, 3, 4])
        torch.testing.assert_close(kept.boxes6, boxes[kept.indices])
        torch.testing.assert_close(kept.classes, classes[kept.indices])
        for count in (0, 1):
            self.assertEqual(paired_nms(boxes[:count], scores[:count]).indices.tolist(), list(range(count)))
        _, topk = anchorfree_candidate_indices(scores, .5, 2)
        self.assertEqual(topk.tolist(), [0, 1])

    def test_matrix_iou_replaces_per_candidate_overlap_calls(self):
        shifts = torch.arange(12).float() * 10
        boxes = torch.stack((shifts, shifts, shifts + 4, shifts + 4, shifts, shifts + 4), dim=1)
        scores = torch.ones(len(boxes))
        with patch.object(detection, "_iou", wraps=detection._iou) as overlaps:
            result = paired_nms(boxes, scores)
        self.assertEqual(result.indices.tolist(), list(range(len(boxes))))
        self.assertEqual(overlaps.call_count, 2)

    def test_matrix_nms_preserves_legacy_order_and_aligned_fields(self):
        rng = torch.Generator().manual_seed(17)
        # Use clustered boxes, tied scores and right offsets to exercise suppression.
        x, y, w, h, offset, w_right = torch.rand((6, 58), generator=rng)
        x, y, w, h, offset, w_right = x * 24, y * 24, w * 32, h * 32, offset * 10, w_right * 32
        boxes = torch.stack((x, y, x + w, y + h, x - offset, x - offset + w_right), dim=1)
        boxes[8:16] = boxes[:8]
        boxes[-1, 0] = float("nan")
        scores = torch.randint(0, 8, (len(boxes),), generator=rng).float()
        classes = torch.randint(0, 3, (len(boxes),), generator=rng)
        devices = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)
        for device in devices:
            b, s, c = boxes.to(device), scores.to(device), classes.to(device)
            for class_ids in (None, c):
                for threshold in (0., .5, .6, .7, 1.):
                    expected = legacy_paired_nms_indices(b, s, threshold, class_ids)
                    result = paired_nms(b, s, threshold, classes=class_ids)
                    torch.testing.assert_close(result.indices, expected, rtol=0, atol=0)
                    torch.testing.assert_close(result.boxes6, b[expected], rtol=0, atol=0, equal_nan=True)
                    torch.testing.assert_close(result.scores, s[expected], rtol=0, atol=0)
                    if class_ids is None:
                        self.assertIsNone(result.classes)
                    else:
                        torch.testing.assert_close(result.classes, class_ids[expected], rtol=0, atol=0)

    def test_large_candidate_pool_preserves_nms_across_matrix_blocks(self):
        boxes = torch.tensor([[0., 0., 4., 4., 0., 4.]]).repeat(1050, 1)
        boxes[-2:, [0, 2, 4, 5]] += 10
        result = paired_nms(boxes, torch.ones(len(boxes)))
        self.assertEqual(result.indices.tolist(), [0, 1048])

    def test_paired_match_cannot_mix_ground_truth_views_or_classes(self):
        gt = torch.tensor([[0., 0., 4., 4., 0., 4.], [8., 0., 12., 4., 8., 12.]])
        mixed = torch.tensor([[0., 0., 4., 4., 8., 12.]])
        self.assertEqual(greedy_match_paired_detections(mixed, gt, torch.tensor([.9])).matched_count, 0)
        self.assertEqual(greedy_match_paired_detections(gt, gt, torch.tensor([.9, .8])).matched_count, 2)
        wrong_class = greedy_match_paired_detections(
            gt, gt, torch.tensor([.9, .8]),
            prediction_classes=torch.tensor([1, 0]), target_classes=torch.tensor([0, 1]),
        )
        self.assertEqual(wrong_class.matched_count, 0)

    def test_same_candidate_pool_is_used_for_every_iou(self):
        box = [0., 0., 4., 4., 0., 4.]
        cache = {"schema": "anchorfree-pre-score-v1", "frames": [
            {"frame_id": "a", "boxes6": [None, box, box, box], "scores": [.99, .7, .7, .15],
             "classes": [0, 0, 0, 1], "gt_boxes6": [box], "gt_classes": [0]},
            {"frame_id": "b", "boxes6": [], "scores": [], "classes": [], "gt_boxes6": [], "gt_classes": []},
        ]}
        report = evaluate(cache, [.25], [.5], 3, .5, bootstrap=20)
        before, after = report["settings"]
        self.assertEqual(before["frames"][0]["pre_nms_dense_ids"], [1, 2])
        self.assertEqual(after["frames"][0]["pre_nms_dense_ids"], [1, 2])
        self.assertEqual(after["frames"][0]["kept_dense_ids"], [1])
        self.assertEqual((before["pairs"], before["hits"], before["duplicates"]), (2, 1, 1))
        self.assertEqual((after["pairs"], after["hits"], after["duplicates"]), (1, 1, 0))
        self.assertEqual(after["lost_gt_count"], 0)
        self.assertEqual(sum(report["score_histogram"]["eligible_for_same_gt"]), 2)
        self.assertIn("coverage", after["paired_bootstrap_delta_95ci"])

    def test_duplicate_ownership_is_independent_of_greedy_tp_assignment(self):
        gt_a = [0., 0., 10., 10., 0., 10.]
        gt_b = [2., 0., 12., 10., 2., 12.]
        cache = {"schema": "anchorfree-pre-score-v1", "frames": [{
            "frame_id": "overlapping", "boxes6": [gt_a, gt_a], "scores": [.9, .8],
            "classes": [0, 0], "gt_boxes6": [gt_a, gt_b], "gt_classes": [0, 0],
        }]}
        off, on = evaluate(cache, [.40], [.5], 500, .5, bootstrap=0)["settings"]
        # Greedy one-to-one can match both GTs. Both candidates prefer GT A.
        self.assertEqual(off["hits"], 2)
        self.assertEqual(off["duplicates"], 1)
        self.assertEqual(off["overlap_gt_duplicates"], 1)
        self.assertEqual(on["duplicates"], 0)
        self.assertEqual(on["lost_gt_count"], 1)

    def test_pre_nms_single_view_recall_survives_paired_suppression(self):
        gt = [0., 0., 10., 10., 0., 10.]
        left_only = [0., 0., 10., 10., 18., 28.]
        cache = {"schema": "anchorfree-pre-score-v1", "frames": [{
            "frame_id": "left_only", "boxes6": [left_only, left_only], "scores": [.9, .8],
            "classes": [0, 0], "gt_boxes6": [gt], "gt_classes": [0],
        }]}
        off, on = evaluate(cache, [.40], [.5], 500, .5, bootstrap=0)["settings"]
        for setting in (off, on):
            self.assertEqual(setting["pre_nms_left_hits"], 1)
            self.assertEqual(setting["pre_nms_right_hits"], 0)
            self.assertEqual(setting["hits"], 0)
        self.assertEqual(on["frames"][0]["pre_nms_dense_ids"], [0, 1])
        self.assertEqual(on["frames"][0]["kept_dense_ids"], [0])

    def test_benchmark_pairs_runs_and_handles_empty_instance_path(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.detector = torch.nn.Identity()
                self.instance_head = torch.nn.Identity()
                self.apply_paired_nms = None
                self.calls = []

            def forward(self, left, right):
                detected = self.detector(left)
                self.calls.append(self.apply_paired_nms)
                count = 0 if self.apply_paired_nms else 2
                if count:
                    self.instance_head(detected)
                return SimpleNamespace(scores=left.new_zeros(count))

        model = Model().eval()
        image = torch.zeros(3, 4, 4)
        data = [{"frame_id": "a", "left": image, "right": image}]
        report = benchmark(model, data, "cpu", warmup=1, repeats=2)
        self.assertEqual(model.calls, [False, True, False, True, True, False])
        self.assertIsNone(model.apply_paired_nms)
        self.assertEqual(report["paired_timing_samples"], 2)
        self.assertEqual(report["summary"]["off"]["pairs_per_forward"]["p50"], 2)
        self.assertEqual(report["summary"]["on"]["instance_head_ms"]["p95"], 0.)
        self.assertEqual(report["summary"]["off"]["paired_nms_ms"]["p95"], 0.)
        self.assertEqual(report["summary"]["on_minus_off"]["pairs"]["p50"], -2)
        self.assertGreater(report["summary"]["off"]["end_to_end_ms"]["p50"], 0.)
        self.assertFalse(model.detector._forward_hooks)
        self.assertFalse(model.instance_head._forward_pre_hooks)


if __name__ == "__main__":
    unittest.main()
