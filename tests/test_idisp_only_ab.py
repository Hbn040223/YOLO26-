import unittest
from types import SimpleNamespace

import torch
from torch import nn

from lightbackbone_stereo_yolo.model import StereoLocalInstanceHead
from lightbackbone_stereo_yolo.roi import make_pair_crop_windows
from lightbackbone_stereo_yolo.training import (
    StereoYoloTrainingModel, freeze_for_idisp_only, load_idisp_frozen_state,
    supervision_valid, training_forward_idisp_only,
)


class IDispOnlyTests(unittest.TestCase):
    def test_shared_mask_keeps_overflow_for_evaluation_and_includes_boundary_in_training(self):
        disp = torch.tensor([[[[1., 2., 3., 4., float("nan")]]]])
        valid = torch.ones_like(disp, dtype=torch.bool)
        foreground = torch.tensor([[[[True, True, True, False, True]]]])
        residual = torch.tensor([[[[0., 2., 3., 0., 0.]]]])
        base = supervision_valid(disp, valid, valid, foreground)
        train = supervision_valid(disp, valid, valid, foreground, residual, 2)
        self.assertEqual(base.flatten().tolist(), [True, True, True, False, False])
        self.assertEqual(train.flatten().tolist(), [True, True, False, False, False])

    def test_checkpoint_reuses_frozen_weights_across_dmax_but_keeps_new_idisp(self):
        init = StereoYoloTrainingModel(nn.BatchNorm2d(4), nn.Linear(1, 1), StereoLocalInstanceHead(4, 2))
        dst = StereoYoloTrainingModel(nn.BatchNorm2d(4), nn.Linear(1, 1), StereoLocalInstanceHead(4, 3))
        fresh = dst.instance_head.idisp.aggregation.stem[0].weight.clone()
        skipped = load_idisp_frozen_state(dst, init.state_dict())
        self.assertTrue(skipped and all(key.startswith("instance_head.idisp.") for key in skipped))
        torch.testing.assert_close(dst.detector.weight, init.detector.weight)
        torch.testing.assert_close(dst.instance_head.mask.layers[0].weight, init.instance_head.mask.layers[0].weight)
        torch.testing.assert_close(dst.instance_head.idisp.aggregation.stem[0].weight, fresh)

    def test_step_trains_only_idisp_and_skips_detector_and_mask_forward(self):
        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.adapter = nn.BatchNorm2d(4)

            def forward(self, left, right):
                p2 = self.adapter(left)[..., ::4, ::4]
                return SimpleNamespace(p2_left=p2, p2_right=self.adapter(right)[..., ::4, ::4],
                                       p3_left=p2[..., ::2, ::2], p4_left=p2[..., ::4, ::4])

        model = StereoYoloTrainingModel(Encoder(), nn.Identity(), StereoLocalInstanceHead(4, 2, 4))
        model.train()
        freeze_for_idisp_only(model)
        self.assertFalse(model.encoder.training or model.detector.training or model.instance_head.mask.training)
        hooks = [m.register_forward_pre_hook(lambda *_: self.fail("unused branch ran"))
                 for m in (model.detector, model.instance_head.mask)]
        disparity = torch.full((1, 64, 64), 5.)
        target = {"boxes6": torch.tensor([[12., 12., 44., 44., 8., 40.]]),
                  "masks": torch.ones((1, 1, 64, 64), dtype=torch.bool),
                  "disparity": disparity, "disparity_valid": disparity > 0}
        try:
            values = training_forward_idisp_only(model, torch.rand(1, 4, 64, 64),
                                                 torch.rand(1, 4, 64, 64), target, 2, 0.)
            self.assertGreater(values["train_pixels"], 0)
            values["loss_total"].backward()
        finally:
            for hook in hooks:
                hook.remove()
        self.assertTrue(any(p.grad is not None and torch.linalg.vector_norm(p.grad) > 0
                            for p in model.instance_head.idisp.parameters()))
        self.assertTrue(all(p.grad is None for p in model.encoder.parameters()))
        self.assertEqual(int(model.encoder.adapter.num_batches_tracked), 0)


class IDispEvaluationTests(unittest.TestCase):
    def test_image_projection_respects_each_native_width(self):
        from lightbackbone_stereo_yolo.idisp_ab import project_disparity

        crop = make_pair_crop_windows(torch.tensor([[4., 4., 12., 12., 0., 8.],
                                                     [4., 4., 20., 12., 0., 16.]]), 24, 24, 0.)
        values = torch.tensor([[[10., 10., 0., 0.], [10., 10., 0., 0.]],
                               [[12., 12., 12., 12.], [12., 12., 12., 12.]]])
        result, valid = project_disparity(SimpleNamespace(absolute=values, valid=values > 0), crop, (24, 24))
        self.assertEqual(valid.flatten(1).sum(1).tolist(), [64, 128])
        torch.testing.assert_close(result[0][valid[0]], torch.full((64,), 10.))
        torch.testing.assert_close(result[1][valid[1]], torch.full((128,), 12.))

    def test_paired_bootstrap_reuses_frame_weights(self):
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from evaluate_idisp_sampling_ab import paired_bootstrap

        runs = [{"seed": 17, "frames": []}]
        for i, pixels in enumerate((2, 5, 7)):
            arms = {"native_pad": {"common": {"pixels": pixels, "epe_sum": 2.*pixels, "d1_hits": 0}},
                    "fixed_56": {"common": {"pixels": pixels, "epe_sum": 2.1*pixels, "d1_hits": 0}}}
            runs[0]["frames"].append({"frame_id": str(i), "pred": arms, "gt": arms})
        result = paired_bootstrap(runs, repetitions=64)["pred"]
        self.assertAlmostEqual(result["relative_epe_increase"]["one_sided_95_upper"], .05)
        self.assertEqual(result["absolute_d1_increase"]["one_sided_95_upper"], 0)


class IDispCheckpointTests(unittest.TestCase):
    def test_checkpoint_round_trip_preserves_optimizer_progress_and_fixed_geometry(self):
        import json
        from pathlib import Path
        import sys
        from tempfile import TemporaryDirectory

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        from train_idisp_only import prepare_args, save

        with TemporaryDirectory() as directory:
            root = Path(directory)
            calibration = root / "fixed_56_pred.json"
            calibration.write_text(json.dumps({"mode": "fixed_56", "selected_dmax_roi": 3, "margin_image_px": 16.}))
            names = ("dataset_root", "train_json", "val_json", "lightstereo_source", "checkpoint")
            paths = {name: str(root / name) for name in names}
            cfg = {"detector_variant": "stereo_yolo26_pan", "one_to_one": True, "adapter_channels": 4, **paths}
            initial = {"config": cfg}
            args = SimpleNamespace(**cfg, roi_sampling="fixed_56", dmax_roi=3, roi_margin=16.,
                                   idisp_calibration=str(calibration), seed=17, lr=3e-4,
                                   weight_decay=1e-4, grad_clip=10., idisp_fixed_frames=None,
                                   idisp_max_steps=0, idisp_eval_steps=50)
            prepare_args(args, initial, None)
            model = StereoYoloTrainingModel(nn.Identity(), nn.Identity(), StereoLocalInstanceHead(4, 3, roi_sampling="fixed_56"))
            optimizer = torch.optim.AdamW(model.instance_head.idisp.parameters(), lr=args.lr)
            model.instance_head.idisp.temperature.raw.square().backward()
            optimizer.step()
            path = root / "last.pt"
            save(path, args, model, optimizer, {26: 0}, 7, 123, 2., 3.)
            restored = torch.load(path, map_location="cpu", weights_only=False)

            conflicting = SimpleNamespace(**{**vars(args), "roi_sampling": "native_pad"})
            with self.assertRaises(SystemExit):
                prepare_args(conflicting, initial, restored)
            resumed = SimpleNamespace(**{**vars(args), "roi_sampling": None, "dmax_roi": -1})
            prepare_args(resumed, initial, restored)
            self.assertEqual((resumed.roi_sampling, resumed.dmax_roi, restored["epoch"], restored["steps"]),
                             ("fixed_56", 3, 7, 123))
            self.assertEqual((restored["best_val_loss"], restored["best_gt_roi_epe"]), (2., 3.))
            self.assertTrue(restored["optimizer"]["state"])
            self.assertIsNone(restored["best_mask_iou"])
