"""Run the published fixed_56 model on a rectified stereo pair."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lightbackbone_stereo_yolo.data.frame_dataset import normalized_rgb
from lightbackbone_stereo_yolo.lightstereo_encoder import LightStereoFeatureEncoder, build_lightstereo_backbone
from lightbackbone_stereo_yolo.model import StereoLocalInstanceHead, StereoYoloLocalModel
from lightbackbone_stereo_yolo.stereo_yolo26 import StereoYOLO26PanHead


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openstereo-source", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=ROOT / "models/stereo-yolo26-fixed56-seed17-e20.pt")
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--score", type=float, default=.40)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    bundle = torch.load(args.weights, map_location="cpu", weights_only=True)
    config = bundle["config"]
    channels = config["adapter_channels"]
    encoder = LightStereoFeatureEncoder(build_lightstereo_backbone(args.openstereo_source), channels)
    detector = StereoYOLO26PanHead(channels, len(bundle["class_mapping"]), one_to_one=config["one_to_one"])
    instance_head = StereoLocalInstanceHead(channels, bundle["dmax_roi"], roi_sampling=config["roi_sampling"])
    model = StereoYoloLocalModel(
        encoder, detector, instance_head, score_threshold=args.score, pre_nms_topk=500,
        detector_variant=config["detector_variant"], right_offset_mode=config["right_offset_mode"],
    )
    model.load_state_dict(bundle["model"], strict=True)
    model.to(args.device).eval()
    normalization = bundle["normalization"]
    left, right = (
        normalized_rgb(path, normalization["mean"], normalization["std"])[None].to(args.device)
        for path in (args.left, args.right)
    )
    crops = []
    hook = instance_head.register_forward_hook(lambda _module, _args, value: crops.append(value.crop))
    try:
        with torch.inference_mode():
            result = model(left, right)
            arrays = {name: getattr(result, name).cpu().numpy() for name in (
                "boxes_left", "boxes_right", "scores", "classes", "disparity", "validity", "mask_validity",
            )}
            arrays["mask_probability"] = result.mask_logits.sigmoid().cpu().numpy()
    finally:
        hook.remove()
    for name in ("c_l", "c_r", "c", "y0", "h_crop", "w_roi", "h_roi"):
        arrays[name] = getattr(crops[0], name).cpu().numpy() if crops else np.empty(0)
    arrays["image_hw"] = np.asarray(left.shape[-2:])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    print(f"{len(result.scores)} paired detections, disparity shape {tuple(result.disparity.shape)}")
    print(args.output)


if __name__ == "__main__":
    main()
