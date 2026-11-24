"""Source and runtime route evidence for edge experiments."""
from contextlib import contextmanager
import hashlib
from pathlib import Path
import subprocess
from unittest.mock import patch

from lightbackbone_stereo_yolo import detection, lightstereo_encoder, model as inference_model


def git_value(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def source_info(root, entry, source, weights):
    source = Path(source).resolve()
    embedded = source / "references" / "OpenStereo"
    if embedded.is_dir():
        source = embedded
    return {
        "commit": git_value(root, "rev-parse", "HEAD"),
        "dirty": bool(git_value(root, "status", "--porcelain")),
        "entry": str(Path(entry).resolve()),
        "model_module": str(Path(inference_model.__file__).resolve()),
        "encoder_module": str(Path(lightstereo_encoder.__file__).resolve()),
        "openstereo_commit": git_value(source, "rev-parse", "HEAD"),
        "weights_sha256": digest(weights),
    }


@contextmanager
def capture_route(model):
    route = {"backbone_inputs": [], "nms_calls": [], "idisp_inputs": []}
    backbone = model.encoder.backbone.register_forward_pre_hook(
        lambda _, args: route["backbone_inputs"].append(tuple(args[0].shape)))
    idisp = model.instance_head.idisp.register_forward_pre_hook(
        lambda _, args: route["idisp_inputs"].append(tuple(args[0].shape)))
    suppress = detection.paired_nms

    def paired_nms(boxes, scores, iou_threshold=.5, *, classes=None, **kwargs):
        route["nms_calls"].append({"pairs": len(boxes), "iou": iou_threshold, "class_aware": classes is not None})
        return suppress(boxes, scores, iou_threshold, classes=classes, **kwargs)

    try:
        with patch.object(detection, "paired_nms", paired_nms):
            yield route
    finally:
        backbone.remove()
        idisp.remove()


def serialize_route(route):
    pairs = ("pre_nms", "post_nms", "idisp")
    return {**{key: value for key, value in route.items() if key not in (*pairs, "dense")},
            **{key + "_pair_ids": route[key].indices.tolist() for key in pairs if key in route}}
