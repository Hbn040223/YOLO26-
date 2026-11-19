"""Explicit configuration for the local Stereo-YOLO experiment."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration that prevents unsupported global and 3D execution paths."""

    runtime_python: str
    dataset_root: str
    lightstereo_source: str
    lightstereo_checkpoint: str
    p2_stride: int = 4
    residual_quantile: float = 0.995
    roi_margin: float = 16.0
    pre_nms_topk: int = 500
    allow_global_cost_volume: bool = False
    allow_global_disparity: bool = False
    allow_3d_outputs: bool = False

    def __post_init__(self) -> None:
        if self.p2_stride != 4:
            raise ValueError("P2 must remain the native stride-4 LightStereo feature tap")
        if not 0.0 < self.residual_quantile < 1.0:
            raise ValueError("residual_quantile must be strictly between zero and one")
        if self.roi_margin < 0:
            raise ValueError("roi_margin must be non-negative")
        if self.pre_nms_topk < 1:
            raise ValueError("pre_nms_topk must be positive")
        if self.allow_global_cost_volume or self.allow_global_disparity or self.allow_3d_outputs:
            raise ValueError("global cost/disparity and 3D outputs are outside this experiment scope")

