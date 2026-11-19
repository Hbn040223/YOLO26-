"""Frame-level KITTI2015 supervision for joint detector/mask/local disparity training.

One sample is one stereo pair: both images, every GT instance of that frame, the
frame disparity map and the ignore mask.  This is the training counterpart of
``KittiInstanceDataset``, which repeats one image pair per annotation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from .kitti import KittiManifestRecord, build_kitti_manifest


# LightStereo KITTI config: NormalizeImage MEAN/STD.  The pretrained backbone was
# fitted with these statistics, so training must not feed raw [0, 1] RGB.
LIGHTSTEREO_KITTI_MEAN = (0.485, 0.456, 0.406)
LIGHTSTEREO_KITTI_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class KittiFrameRecord:
    """All annotations that belong to one stereo frame."""

    frame_id: str
    left_image: Path
    right_image: Path
    instance_map: Path
    disparity: Path
    ignore_mask: Path
    annotations: tuple[KittiManifestRecord, ...]


def build_frame_records(root: str | Path, annotation_json: str | Path) -> list[KittiFrameRecord]:
    """Group the validated per-instance manifest by frame without re-reading assets."""
    grouped: dict[str, list[KittiManifestRecord]] = {}
    for record in build_kitti_manifest(root, annotation_json):
        grouped.setdefault(record.frame_id, []).append(record)
    frames: list[KittiFrameRecord] = []
    for frame_id in sorted(grouped):
        annotations = tuple(grouped[frame_id])
        first = annotations[0]
        frames.append(
            KittiFrameRecord(
                frame_id=frame_id,
                left_image=first.left_image,
                right_image=first.right_image,
                instance_map=first.instance_map,
                disparity=first.disparity,
                ignore_mask=first.ignore_mask,
                annotations=annotations,
            )
        )
    return frames


def build_class_mapping(records: Iterable[KittiFrameRecord]) -> dict[int, int]:
    """Map the raw KITTI category ids of one split to contiguous 0..K-1."""
    raw_ids = sorted({annotation.category_id for frame in records for annotation in frame.annotations})
    return {raw: index for index, raw in enumerate(raw_ids)}


def normalized_rgb(path: Path, mean: tuple[float, float, float], std: tuple[float, float, float]) -> Tensor:
    """Load RGB and apply the exact LightStereo-KITTI normalization."""
    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.
    image = torch.from_numpy(array).permute(2, 0, 1)
    mean_tensor = torch.tensor(mean, dtype=torch.float32)[:, None, None]
    std_tensor = torch.tensor(std, dtype=torch.float32)[:, None, None]
    return (image - mean_tensor) / std_tensor


def _paired_boxes(mask: Tensor, valid: Tensor, disparity: Tensor) -> Tensor | None:
    """Left mask box plus the right box projected by valid disparity."""
    if not valid.any():
        return None
    y, x = mask.nonzero(as_tuple=True)
    left = torch.stack((x.min(), y.min(), x.max(), y.max())).to(disparity.dtype)
    vy, vx = valid.nonzero(as_tuple=True)
    right_x = vx.to(disparity.dtype) - disparity[valid]
    return torch.cat((left, torch.stack((right_x.min(), right_x.max()))))


class KittiFrameDataset(Dataset):
    """One entry per stereo frame, with variable-length instance supervision."""

    def __init__(
        self,
        records: list[KittiFrameRecord],
        class_mapping: dict[int, int],
        mean: tuple[float, float, float] = LIGHTSTEREO_KITTI_MEAN,
        std: tuple[float, float, float] = LIGHTSTEREO_KITTI_STD,
        unknown_class: str = "error",
        load_images: bool = True,
        *,
        repair_car_supervision: bool = False,
    ) -> None:
        if unknown_class not in ("error", "skip"):
            raise ValueError("unknown_class must be 'error' or 'skip'")
        if len(mean) != 3 or len(std) != 3:
            raise ValueError("mean and std must contain three channel values")
        self.records = records
        self.class_mapping = dict(class_mapping)
        self.mean = tuple(float(value) for value in mean)
        self.std = tuple(float(value) for value in std)
        self.unknown_class = unknown_class
        self.load_images = load_images
        self.repair_car_supervision = repair_car_supervision
        if repair_car_supervision and 26 not in self.class_mapping:
            raise ValueError("car supervision repair requires KITTI category 26")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        instance = torch.from_numpy(np.asarray(Image.open(record.instance_map), dtype=np.int64))
        disparity = torch.from_numpy(np.asarray(Image.open(record.disparity), dtype=np.float32)) / 256.
        ignore = torch.from_numpy(np.asarray(Image.open(record.ignore_mask), dtype=np.uint8) > 0)
        disparity_valid_image = torch.isfinite(disparity) & (disparity > 0) & ~ignore

        boxes: list[Tensor] = []
        masks: list[Tensor] = []
        classes: list[int] = []
        detection_boxes: list[Tensor] = []
        detection_ignore = ignore.clone() if self.repair_car_supervision else None
        for annotation in record.annotations:
            if detection_ignore is not None and annotation.category_id != 26:
                detection_ignore |= instance == annotation.instance_value
                continue
            if annotation.category_id not in self.class_mapping:
                if self.unknown_class == "skip":
                    continue
                raise ValueError(f"{record.frame_id}: unseen category_id {annotation.category_id}")
            mask = instance == annotation.instance_value
            valid = mask & disparity_valid_image
            boxes6 = _paired_boxes(mask, valid, disparity)
            if detection_ignore is not None:
                if boxes6 is None:
                    y, x = mask.nonzero(as_tuple=True)
                    if not x.numel():
                        continue
                    left = torch.stack((x.min(), y.min(), x.max(), y.max())).to(disparity.dtype)
                else:
                    left = boxes6[:4]
                if not ((left[2:] - left[:2]) > 0).all():
                    detection_ignore |= mask
                    continue
                if boxes6 is None or not (boxes6[5] - boxes6[4] > 1e-3):
                    # Equal right edges denote a left-only target. The existing
                    # paired-regression mask rejects this zero-width sentinel.
                    detection_boxes.append(torch.cat((left, left[[0, 2]].mean().repeat(2))))
                    continue
                detection_boxes.append(boxes6)
            if boxes6 is None:
                continue
            boxes.append(boxes6)
            masks.append(mask)
            classes.append(self.class_mapping[annotation.category_id])

        height, width = instance.shape
        if boxes:
            boxes6_tensor = torch.stack(boxes)
            mask_tensor = torch.stack(masks)[:, None]
            class_tensor = torch.tensor(classes, dtype=torch.long)
        else:
            boxes6_tensor = torch.empty((0, 6), dtype=torch.float32)
            mask_tensor = torch.empty((0, 1, height, width), dtype=torch.bool)
            class_tensor = torch.empty((0,), dtype=torch.long)

        sample: dict[str, object] = {
            "boxes6": boxes6_tensor,
            "masks": mask_tensor,
            "classes": class_tensor,
            "disparity": disparity[None],
            "disparity_valid": disparity_valid_image[None],
            "frame_id": record.frame_id,
        }
        if detection_ignore is not None:
            sample["detection_boxes6"] = torch.stack(detection_boxes) if detection_boxes else disparity.new_empty((0, 6))
            sample["detection_classes"] = torch.full((len(detection_boxes),), self.class_mapping[26], dtype=torch.long)
            sample["detection_ignore_mask"] = detection_ignore
        if self.load_images:
            sample["left"] = normalized_rgb(record.left_image, self.mean, self.std)
            sample["right"] = normalized_rgb(record.right_image, self.mean, self.std)
        return sample


def collate_frames(samples: list[dict[str, object]]) -> dict[str, object]:
    """Stack images and keep each frame's variable-length targets intact."""
    if not samples:
        raise ValueError("cannot collate an empty sample list")
    batch: dict[str, object] = {
        "targets": [{key: value for key, value in sample.items() if key not in ("left", "right")} for sample in samples]
    }
    if "left" in samples[0]:
        batch["left"] = torch.stack([sample["left"] for sample in samples])  # type: ignore[list-item]
        batch["right"] = torch.stack([sample["right"] for sample in samples])  # type: ignore[list-item]
    return batch
