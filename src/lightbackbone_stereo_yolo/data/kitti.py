"""Deterministic stereo label geometry for KITTI2015."""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from PIL import Image


@dataclass(frozen=True)
class KittiManifestRecord:
    frame_id: str
    left_image: Path
    right_image: Path
    instance_map: Path
    disparity: Path
    ignore_mask: Path
    category_id: int
    instance_value: int
    left_box: tuple[float, float, float, float]


@dataclass(frozen=True)
class KittiInstanceTarget:
    boxes6: Tensor
    mask: Tensor
    valid_disparity: Tensor
    category_id: int


def build_kitti_manifest(
    root: str | Path, annotation_json: str | Path, *, require_files: bool = False,
) -> list[KittiManifestRecord]:
    """Validate supplied COCO-style RPN split records against KITTI2015 assets."""
    root = Path(root)
    payload = json.loads(Path(annotation_json).read_text())
    images = {image["id"]: image for image in payload["images"]}
    records: list[KittiManifestRecord] = []
    for annotation in payload["annotations"]:
        image = images[annotation["image_id"]]
        left = root / image["file_name"]
        stem = left.stem
        paths = (left, root / "training/image_3" / left.name, root / "training/instance" / left.name, root / "training/disp_occ_0" / left.name, root / image["ignore_mask_file"])
        missing = [path for path in paths if not path.is_file()]
        if missing:
            if require_files:
                raise FileNotFoundError(f"{stem}: missing KITTI assets: {', '.join(map(str, missing))}")
            continue
        x, y, width, height = annotation["bbox"]
        records.append(KittiManifestRecord(stem, *paths, int(annotation["category_id"]), int(annotation["source_instance_value"]), (float(x), float(y), float(x + width), float(y + height))))
    return records


def project_right_box(left_instance_mask: Tensor, disparity: Tensor) -> Tensor:
    """Tight right box of valid left-instance pixels under xR=xL-Df."""
    valid = left_instance_mask.bool() & torch.isfinite(disparity) & (disparity > 0)
    if not valid.any():
        raise ValueError("instance has no valid disparity pixels for right-box projection")
    y, x = valid.nonzero(as_tuple=True)
    right_x = x.to(disparity.dtype) - disparity[valid]
    return torch.stack((right_x.min(), y.min().to(disparity.dtype), right_x.max(), y.max().to(disparity.dtype)))


def build_instance_target(instance_map: Tensor, disparity: Tensor, instance_value: int, category_id: int) -> KittiInstanceTarget:
    """Create a trainable paired-box target; invalid geometry is rejected."""
    mask = instance_map == instance_value
    valid_disparity = mask & torch.isfinite(disparity) & (disparity > 0)
    if not valid_disparity.any():
        raise ValueError("instance has no valid stereo disparity")
    y, x = mask.nonzero(as_tuple=True)
    left = torch.stack((x.min(), y.min(), x.max(), y.max())).to(disparity.dtype)
    right = project_right_box(mask, disparity)
    boxes6 = torch.cat((left, right[[0, 2]]))
    return KittiInstanceTarget(boxes6, mask, valid_disparity, category_id)


class KittiInstanceDataset(Dataset):
    """One entry per supplied RPN annotation instance, rejecting invalid geometry."""

    def __init__(self, records: list[KittiManifestRecord]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _rgb(path: Path) -> Tensor:
        return torch.from_numpy(np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)).permute(2, 0, 1) / 255.

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        record = self.records[index]
        instance = torch.from_numpy(np.asarray(Image.open(record.instance_map), dtype=np.int64))
        disparity = torch.from_numpy(np.asarray(Image.open(record.disparity), dtype=np.float32)) / 256.
        ignore = torch.from_numpy(np.asarray(Image.open(record.ignore_mask), dtype=np.uint8) > 0)
        target = build_instance_target(instance, disparity, record.instance_value, record.category_id)
        valid = target.valid_disparity & ~ignore
        if not valid.any():
            raise ValueError(f"{record.frame_id}: no non-ignored valid disparity pixels")
        return {"left": self._rgb(record.left_image), "right": self._rgb(record.right_image), "boxes6": target.boxes6, "mask": target.mask, "disparity": disparity, "valid_disparity": valid, "ignore": ignore, "category": torch.tensor(target.category_id)}


def collate_instances(samples: list[dict[str, Tensor]]) -> dict[str, object]:
    """Stack equal-size image pairs while keeping variable instance supervision intact."""
    if not samples:
        raise ValueError("cannot collate an empty sample list")
    return {"left": torch.stack([sample["left"] for sample in samples]), "right": torch.stack([sample["right"] for sample in samples]), "targets": [{key: value for key, value in sample.items() if key not in ("left", "right")} for sample in samples]}
