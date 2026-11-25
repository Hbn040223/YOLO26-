"""Contract tests for the strict feature-only LightStereo wrapper."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None
    nn = None

try:
    from lightbackbone_stereo_yolo.lightstereo_encoder import (
        LightStereoFeatureEncoder,
        load_backbone_state,
    )
except ModuleNotFoundError:
    LightStereoFeatureEncoder = None
    load_backbone_state = None


@unittest.skipIf(torch is None, "PyTorch is required for encoder contract tests")
class LightStereoFeatureEncoderContractTest(unittest.TestCase):
    def test_unknown_encoder_variant_is_rejected_before_import(self):
        from lightbackbone_stereo_yolo.lightstereo_encoder import build_lightstereo_backbone
        with self.assertRaisesRegex(ValueError, "encoder backbone"):
            build_lightstereo_backbone("unused", encoder_backbone="unknown")

    def test_shared_encoder_returns_eight_separated_native_scale_features(self) -> None:
        class TinyLightStereoBackbone(nn.Module):
            output_channels = (4, 6, 8, 10)

            def __init__(self) -> None:
                super().__init__()
                self.calls = 0

            def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
                self.calls += 1
                batch, _, height, width = images.shape
                return [
                    torch.ones(batch, 4, height // 4, width // 4),
                    torch.ones(batch, 6, height // 8, width // 8),
                    torch.ones(batch, 8, height // 16, width // 16),
                    torch.ones(batch, 10, height // 32, width // 32),
                ]

        self.assertIsNotNone(LightStereoFeatureEncoder)
        backbone = TinyLightStereoBackbone()
        encoder = LightStereoFeatureEncoder(backbone, adapter_channels=12)
        pyramid = encoder(torch.randn(2, 3, 35, 67), torch.randn(2, 3, 35, 67))

        self.assertEqual(backbone.calls, 1, "left/right must share one encoder invocation")
        for name in (
            "p2_left", "p2_right", "p3_left", "p3_right",
            "p4_left", "p4_right", "c5_left", "c5_right",
        ):
            self.assertTrue(hasattr(pyramid, name), name)
            self.assertEqual(getattr(pyramid, name).shape[:2], (2, 12), name)
        self.assertEqual(pyramid.p2_left.shape[-2:], (9, 17))
        self.assertEqual(pyramid.metadata.p2_stride, 4)
        self.assertEqual(pyramid.metadata.padded_size, (64, 96))

    def test_public_wrapper_exposes_no_global_matching_or_disparity_module(self) -> None:
        class TinyLightStereoBackbone(nn.Module):
            output_channels = (4, 6, 8, 10)

            def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
                batch, _, height, width = images.shape
                return [
                    torch.ones(batch, channels, height // stride, width // stride)
                    for channels, stride in ((4, 4), (6, 8), (8, 16), (10, 32))
                ]

        encoder = LightStereoFeatureEncoder(TinyLightStereoBackbone(), adapter_channels=8)
        for forbidden in ("cost_volume", "cost_aggregation", "aggregation", "disparity", "disp_head"):
            self.assertFalse(hasattr(encoder, forbidden), forbidden)

    def test_default_p2_is_declared_as_backbone_fused_feature(self) -> None:
        class TinyLightStereoBackbone(nn.Module):
            output_channels = (4, 6, 8, 10)

            def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
                batch, _, height, width = images.shape
                return [
                    torch.ones(batch, channels, height // stride, width // stride)
                    for channels, stride in ((4, 4), (6, 8), (8, 16), (10, 32))
                ]

        encoder = LightStereoFeatureEncoder(TinyLightStereoBackbone(), adapter_channels=8)
        self.assertEqual(encoder.p2_semantics, "backbone_fpn_fused")

    def test_checkpoint_loader_requires_complete_backbone_namespace(self) -> None:
        self.assertIsNotNone(load_backbone_state)

        class TinyBackbone(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.projection = nn.Conv2d(1, 1, kernel_size=1)

        reference = TinyBackbone()
        payload = {
            "model_state": {
                f"backbone.{key}": value.detach().clone()
                for key, value in reference.state_dict().items()
            }
        }
        loaded = TinyBackbone()
        load_backbone_state(loaded, payload)
        self.assertEqual(set(loaded.state_dict()), set(reference.state_dict()))

        incomplete = {"model_state": {"backbone.projection.weight": reference.projection.weight}}
        with self.assertRaises(RuntimeError):
            load_backbone_state(TinyBackbone(), incomplete)


if __name__ == "__main__":
    unittest.main()
