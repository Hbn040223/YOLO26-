"""Strict feature-only access to LightStereo's shared image backbone.

This module deliberately imports only ``lightstereo.backbone.Backbone``.  It
does not construct a LightStereo model, a cost volume, an aggregation network,
or a disparity head.
"""
from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence
from unittest.mock import patch

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class PaddingMetadata:
    """Image geometry retained after encoder padding."""

    original_size: tuple[int, int]
    padded_size: tuple[int, int]
    p2_stride: int = 4


@dataclass(frozen=True)
class StereoFeaturePyramid:
    """Separated, view-aligned LightStereo features at four native scales."""

    p2_left: Tensor
    p2_right: Tensor
    p3_left: Tensor
    p3_right: Tensor
    p4_left: Tensor
    p4_right: Tensor
    c5_left: Tensor
    c5_right: Tensor
    metadata: PaddingMetadata


def _unwrap_checkpoint(payload: object) -> Mapping[str, Tensor]:
    if not isinstance(payload, Mapping):
        raise RuntimeError("LightStereo checkpoint must contain a state mapping")
    for key in ("model_state", "state_dict", "model", "model_state_dict"):
        nested = payload.get(key)
        if isinstance(nested, Mapping):
            payload = nested
            break
    if not isinstance(payload, Mapping):
        raise RuntimeError("LightStereo checkpoint state is not a mapping")
    return payload  # type: ignore[return-value]


def load_backbone_state(backbone: nn.Module, payload: object) -> None:
    """Load exactly the ``backbone.`` namespace, rejecting partial checkpoints."""

    source_state = _unwrap_checkpoint(payload)
    normalized = {
        (str(key)[len("module."):] if str(key).startswith("module.") else str(key)): value
        for key, value in source_state.items()
    }
    backbone_state = {
        key[len("backbone."):]: value
        for key, value in normalized.items()
        if key.startswith("backbone.")
    }
    if not backbone_state:
        raise RuntimeError("Checkpoint has no LightStereo backbone namespace")
    # strict=True explicitly makes both missing and unexpected encoder keys fatal.
    backbone.load_state_dict(backbone_state, strict=True)


def _register_openstereo_packages(source: Path) -> None:
    """Expose only OpenStereo's package path without importing its stereo model."""

    source = source.resolve()
    for name in ("stereo", "stereo.modeling"):
        package_path = str(source / Path(*name.split(".")))
        existing = sys.modules.get(name)
        if existing is None:
            package = types.ModuleType(name)
            package.__path__ = [package_path]
            package.__package__ = name
            sys.modules[name] = package
        elif package_path not in list(getattr(existing, "__path__", ())):
            raise RuntimeError("A different OpenStereo source is already imported")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))


def _register_lightstereo_source(source_root: str | Path) -> None:

    source = Path(source_root)
    referenced = source / "references" / "OpenStereo"
    if referenced.is_dir():
        source = referenced
    if not (source / "stereo" / "modeling" / "models" / "lightstereo" / "backbone.py").is_file():
        raise FileNotFoundError(f"OpenStereo LightStereo source not found under {source}")
    _register_openstereo_packages(source)


def build_lightstereo_backbone(source_root: str | Path) -> nn.Module:
    """Construct the official feature backbone without downloading weights."""
    _register_lightstereo_source(source_root)
    import timm
    from stereo.modeling.models.lightstereo.backbone import Backbone

    create_model = timm.create_model

    def no_download(*args: object, **kwargs: object) -> nn.Module:
        kwargs["pretrained"] = False
        return create_model(*args, **kwargs)

    with patch.object(timm, "create_model", side_effect=no_download):
        return Backbone("MobileNetv2")


class MobileNetV3SmallFpn(nn.Module):
    """MobileNetV3-Small through stride 32 with the LightStereo top-down FPN."""

    output_channels = (16, 24, 48, 96)

    def __init__(self, mobilenet, fpn_layer, p2_norm):
        super().__init__()
        self.stem = nn.Sequential(mobilenet.conv_stem, mobilenet.bn1, mobilenet.act1)
        self.blocks = mobilenet.blocks[:5]
        self.fpn_layer4 = fpn_layer(96, 48)
        self.fpn_layer3 = fpn_layer(48, 24)
        self.fpn_layer2 = fpn_layer(24, 16)
        self.out_conv = p2_norm

    def forward(self, images):
        c2 = self.blocks[0](self.stem(images))
        c3 = self.blocks[1](c2)
        c4 = self.blocks[3](self.blocks[2](c3))
        c5 = self.blocks[4](c4)
        p4 = self.fpn_layer4(c5, c4)
        p3 = self.fpn_layer3(p4, c3)
        p2 = self.out_conv(self.fpn_layer2(p3, c2))
        return p2, p3, p4, c5


def build_lightstereo_v3small(source_root: str | Path, *, pretrained: bool = False) -> nn.Module:
    """Build the V3 speed prototype with the repository's pinned TIMM layout."""
    _register_lightstereo_source(source_root)
    import timm
    from stereo.modeling.common.basic_block_2d import BasicConv2d
    from stereo.modeling.models.lightstereo.backbone import FPNLayer

    mobilenet = timm.create_model("mobilenetv3_small_100", features_only=True, pretrained=pretrained)
    p2_norm = BasicConv2d(16, 16, kernel_size=3, padding=1, padding_mode="replicate",
                          norm_layer=nn.InstanceNorm2d)
    return MobileNetV3SmallFpn(mobilenet, FPNLayer, p2_norm)


def load_lightstereo_backbone(source_root: str | Path, checkpoint: str | Path) -> nn.Module:
    """Construct and strictly load only the official MobileNetV2 feature backbone."""
    backbone = build_lightstereo_backbone(source_root)
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:  # compatibility with older torch versions
        payload = torch.load(checkpoint, map_location="cpu")
    load_backbone_state(backbone, payload)
    return backbone


class LightStereoFeatureEncoder(nn.Module):
    """One shared full-image encoder, with no stereo matching at image scale."""

    _STRIDES = (4, 8, 16, 32)
    # OpenStereo Backbone returns its FPN-fused 1/4 feature as features[0].
    p2_semantics = "backbone_fpn_fused"

    def __init__(self, backbone: nn.Module, adapter_channels: int = 96) -> None:
        super().__init__()
        if adapter_channels < 1:
            raise ValueError("adapter_channels must be positive")
        input_channels = tuple(getattr(backbone, "output_channels", ()))
        if len(input_channels) != 4:
            raise ValueError("LightStereo backbone must expose four output channel counts")
        self.backbone = backbone
        self.adapters = nn.ModuleList(
            nn.Conv2d(int(channels), adapter_channels, kernel_size=1)
            for channels in input_channels
        )

    @classmethod
    def from_checkpoint(
        cls,
        source_root: str | Path,
        checkpoint: str | Path,
        adapter_channels: int = 96,
    ) -> "LightStereoFeatureEncoder":
        return cls(load_lightstereo_backbone(source_root, checkpoint), adapter_channels)

    @staticmethod
    def _padded_size(height: int, width: int) -> tuple[int, int]:
        return height + (-height) % 32, width + (-width) % 32

    @staticmethod
    def _crop_native(feature: Tensor, height: int, width: int, stride: int) -> Tensor:
        return feature[..., : (height + stride - 1) // stride, : (width + stride - 1) // stride]

    def forward(self, left: Tensor, right: Tensor) -> StereoFeaturePyramid:
        if left.ndim != 4 or right.ndim != 4 or left.shape != right.shape:
            raise ValueError("left and right must be equal-shape NCHW image tensors")
        height, width = left.shape[-2:]
        padded_height, padded_width = self._padded_size(height, width)
        pad = (0, padded_width - width, 0, padded_height - height)
        # Concatenation causes one shared backbone invocation while preserving views.
        packed = torch.cat((F.pad(left, pad, mode="replicate"), F.pad(right, pad, mode="replicate")), dim=0)
        features: Sequence[Tensor] = self.backbone(packed)
        if len(features) != 4:
            raise RuntimeError("LightStereo backbone must return [P2, P3, P4, C5]")
        batch_size = left.shape[0]
        outputs: list[Tensor] = []
        for feature, adapter, stride in zip(features, self.adapters, self._STRIDES):
            adapted = self._crop_native(adapter(feature), height, width, stride)
            left_feature, right_feature = adapted.split(batch_size, dim=0)
            outputs.extend((left_feature, right_feature))
        metadata = PaddingMetadata(
            original_size=(height, width),
            padded_size=(padded_height, padded_width),
        )
        return StereoFeaturePyramid(*outputs, metadata=metadata)
