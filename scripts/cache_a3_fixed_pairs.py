"""Cache frozen stereo detections and official one-to-one matches on one split."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from diagnose_pred_roi import _dense_inputs, _rectified_calibration, _target_groups
from lightbackbone_stereo_yolo.data.frame_dataset import KittiFrameDataset, build_frame_records
from lightbackbone_stereo_yolo.detection import decode_anchorfree_predictions
from lightbackbone_stereo_yolo.geometry import generate_point_locations, point_levels, right_offset_mode_from_config
from lightbackbone_stereo_yolo.lightstereo_encoder import LightStereoFeatureEncoder
from lightbackbone_stereo_yolo.stereo_yolo26 import StereoYOLO26PanHead
from lightbackbone_stereo_yolo.training_metrics import greedy_match_paired_detections


def decode_from_pyramid(detector, pyramid, device, *, score, topk, right_offset_mode):
    return decode_anchorfree_predictions(
        _dense_inputs(detector(pyramid)), generate_point_locations(point_levels(pyramid), device), score,
        pre_nms_topk=topk, right_offset_mode=right_offset_mode, apply_nms=not detector.one_to_one_enabled,
    )


def pack_detection_frame(frame_id, image_hw, pairs, gt, groups, calibration):
    match = greedy_match_paired_detections(pairs.boxes6, gt, pairs.scores, .5)
    picked = match.prediction_indices
    return {
        "frame_id": frame_id, "image_hw": list(image_hw),
        "all_boxes6": pairs.boxes6.tolist(), "all_scores": pairs.scores.tolist(),
        "all_dense_ids": pairs.indices.tolist(), "all_classes": pairs.classes.tolist(),
        "target_indices": match.target_indices.tolist(),
        "prediction_rows": picked.tolist(), "prediction_indices": pairs.indices[picked].tolist(),
        "boxes6": pairs.boxes6[picked].tolist(), "scores": pairs.scores[picked].tolist(),
        "gt_boxes6": gt.tolist(), "gt_groups": groups, "calibration": calibration,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split-json", type=Path, help="Defaults to the checkpoint validation split")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--score", type=float, default=.25)
    parser.add_argument("--topk", type=int, default=500)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config, mapping, state = payload["config"], payload["class_mapping"], payload["model"]
    split = (args.split_json or Path(config["val_json"])).resolve()
    dataset = KittiFrameDataset(build_frame_records(config["dataset_root"], split), mapping, unknown_class="skip")
    encoder = LightStereoFeatureEncoder.from_checkpoint(
        config["lightstereo_source"], config["checkpoint"], adapter_channels=config["adapter_channels"],
    )
    detector = StereoYOLO26PanHead(config["adapter_channels"], len(mapping), one_to_one=config.get("one_to_one", False))
    encoder.load_state_dict({key[len("encoder."):]: value for key, value in state.items() if key.startswith("encoder.")})
    detector.load_state_dict({key[len("detector."):]: value for key, value in state.items() if key.startswith("detector.")})
    encoder.to(device).eval()
    detector.to(device).eval()
    right_offset_mode = right_offset_mode_from_config(config)
    frames = []
    with ZipFile(Path(config["dataset_root"]) / "data_scene_flow_calib.zip") as calibrations, torch.inference_mode():
        for index in range(len(dataset)):
            sample = dataset[index]
            frame_id = sample["frame_id"]
            try:
                calibration = _rectified_calibration(calibrations, frame_id)
                groups = _target_groups(sample, mapping, calibrations, calibration=calibration)
            except (KeyError, ValueError):
                calibration = None
                groups = [{"depth_m": None} for _ in sample["classes"]]
            for group, category in zip(groups, sample["classes"].tolist()):
                group["car"] = category == mapping.get(26)
                group["near_car"] = group["car"] and group["depth_m"] is not None and group["depth_m"] < 10.
            left, right = (sample[side].unsqueeze(0).to(device) for side in ("left", "right"))
            pairs = decode_from_pyramid(
                detector, encoder(left, right), device, score=args.score, topk=args.topk,
                right_offset_mode=right_offset_mode,
            )
            frames.append(pack_detection_frame(
                frame_id, left.shape[-2:], pairs, sample["boxes6"].to(device), groups, calibration,
            ))
            if (index + 1) % 10 == 0:
                print(f"cached {index + 1}/{len(dataset)} frames", flush=True)
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "manifest": str(split), "manifest_sha256": hashlib.sha256(split.read_bytes()).hexdigest(),
        "score": args.score, "topk": args.topk, "minimum_joint_iou": .5,
        "apply_paired_nms": not detector.one_to_one_enabled, "right_offset_mode": right_offset_mode,
        "matched_count": sum(len(frame["target_indices"]) for frame in frames),
        "candidate_count": sum(len(frame["all_boxes6"]) for frame in frames),
        "all_frames": frames,
        "frames": [{key: frame[key] for key in ("frame_id", "boxes6", "scores", "target_indices", "prediction_indices")}
                   for frame in frames if frame["target_indices"]],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"cached {len(frames)} frames, {result['candidate_count']} candidates, {result['matched_count']} matches")


if __name__ == "__main__":
    main()
