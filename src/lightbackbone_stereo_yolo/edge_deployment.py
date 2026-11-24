"""Tensor-only boundary for the static stereo encoder and detection frontend."""
from types import SimpleNamespace

from torch import nn

from .detection import DenseStereoPredictions
from .lightstereo_encoder import PaddingMetadata
from .stereo_input import StereoInputGeometry


DENSE_OUTPUTS = ("objectness", "class_logits", "box6")
FEATURE_OUTPUTS = ("p2_left", "p2_right", "p3_left", "p4_left", "c5_left")
FRONTEND_OUTPUTS = DENSE_OUTPUTS + FEATURE_OUTPUTS
FRONTEND_SCHEMA = 2


def manifest_geometry(manifest):
    if manifest.get("frontend_schema") != FRONTEND_SCHEMA:
        raise ValueError(f"expected frontend schema {FRONTEND_SCHEMA}, actual {manifest.get('frontend_schema')}")
    geometry = StereoInputGeometry(tuple(manifest["original_hw"]), manifest["crop_top_px"])
    expected = {"input_hw": list(geometry.cropped_hw), "encoder_padded_hw": list(geometry.padded_hw),
                "bindings": {name: list(geometry.cropped_nchw) for name in ("left", "right")}}
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"expected manifest {key} {value}, actual {manifest.get(key)}")
    return geometry


class StaticStereoFrontend(nn.Module):
    """Consume a cropped pair and emit tensors for native decode and RoI dispatch."""

    def __init__(self, encoder, detector):
        super().__init__()
        self.encoder = encoder
        self.detector = detector

    def forward(self, left, right):
        pyramid = self.encoder(left, right)
        dense = self.detector(pyramid)
        return tuple(getattr(dense, name) for name in DENSE_OUTPUTS) + tuple(
            getattr(pyramid, name) for name in FEATURE_OUTPUTS)


def unpack_frontend_outputs(outputs, cropped_hw, padded_hw):
    tensors = dict(zip(FRONTEND_OUTPUTS, outputs))
    dense = DenseStereoPredictions(**{name: tensors[name] for name in DENSE_OUTPUTS})
    pyramid = SimpleNamespace(
        **{name: tensors[name] for name in FEATURE_OUTPUTS},
        metadata=PaddingMetadata(original_size=cropped_hw, padded_size=padded_hw),
    )
    return pyramid, dense
