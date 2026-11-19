"""Stateful validation controls for detector-only training."""
from __future__ import annotations
import json
import math

def append_jsonl_record(path: Path, payload: dict) -> None:
    """Append exactly one JSON object terminated by one physical newline."""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
from pathlib import Path
from dataclasses import dataclass



def is_precision_qualified(paired_coverage: float, paired_precision: float, precision_floor: float) -> bool:
    """Coverage is selectable only when its paired precision meets the fixed floor."""
    del paired_coverage
    return paired_precision >= precision_floor


def percentile(values: list[float], quantile: float) -> float:
    """Linear percentile used for per-frame post-NMS candidate counts."""
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)

@dataclass
class DetectorCheckpointSelector:
    """Select the best deployable detector checkpoint under a fixed precision floor."""
    precision_floor: float
    best_paired_coverage: float = float("-inf")
    best_paired_precision: float = float("-inf")
    best_pre_nms_recall: float = float("-inf")

    def update(self, paired_coverage: float, paired_precision: float, pre_nms_recall: float) -> bool:
        if not is_precision_qualified(paired_coverage, paired_precision, self.precision_floor):
            return False
        candidate = (paired_coverage, paired_precision, pre_nms_recall)
        current = (self.best_paired_coverage, self.best_paired_precision, self.best_pre_nms_recall)
        if candidate > current:
            self.best_paired_coverage, self.best_paired_precision, self.best_pre_nms_recall = candidate
            return True
        return False

    def state_dict(self) -> dict:
        return {"precision_floor": self.precision_floor, "best_paired_coverage": self.best_paired_coverage,
                "best_paired_precision": self.best_paired_precision, "best_pre_nms_recall": self.best_pre_nms_recall}

    @classmethod
    def from_state_dict(cls, state: dict) -> "DetectorCheckpointSelector":
        return cls(**state)

@dataclass
class DetectorPlateauController:
    min_epoch: int = 30
    patience: int = 5
    min_delta: float = .005
    best_paired_coverage: float = float("-inf")
    best_pre_nms_recall: float = float("-inf")
    plateau_counter: int = 0
    last_improvement_epoch: int = -1
    stop_reason: str = ""

    def update(self, epoch: int, paired_coverage: float, pre_nms_recall: float) -> bool:
        paired_improved = paired_coverage >= self.best_paired_coverage + self.min_delta
        recall_improved = pre_nms_recall >= self.best_pre_nms_recall + self.min_delta
        if paired_improved:
            self.best_paired_coverage = paired_coverage
        if recall_improved:
            self.best_pre_nms_recall = pre_nms_recall
        if paired_improved or recall_improved:
            self.plateau_counter = 0
            self.last_improvement_epoch = epoch
        elif epoch >= self.min_epoch:
            self.plateau_counter += 1
        if epoch >= self.min_epoch and self.plateau_counter >= self.patience:
            self.stop_reason = "plateau"
            return True
        return False

    def state_dict(self) -> dict:
        return {"min_epoch": self.min_epoch, "patience": self.patience, "min_delta": self.min_delta,
                "best_paired_coverage": self.best_paired_coverage, "best_pre_nms_recall": self.best_pre_nms_recall,
                "plateau_counter": self.plateau_counter, "last_improvement_epoch": self.last_improvement_epoch,
                "stop_reason": self.stop_reason}

    @classmethod
    def from_state_dict(cls, state: dict) -> "DetectorPlateauController":
        return cls(**state)
