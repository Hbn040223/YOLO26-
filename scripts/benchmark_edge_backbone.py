"""Compare V2 and the V3-Small prototype under identical frozen RoI loads."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter

import torch

from benchmark_edge_acceleration import encoder_intervals, register_encoder_boundaries, revision, summarize, synchronize
from evaluate_paired_nms import sha256, validation_dataset, write_json
from infer import ROOT, load_published_model
from lightbackbone_stereo_yolo.detection import PairDetections
from lightbackbone_stereo_yolo.lightstereo_encoder import (
    LightStereoFeatureEncoder, build_lightstereo_backbone, build_lightstereo_v3small, encoder_backbone_from_config,
)
from lightbackbone_stereo_yolo.stereo_input import load_rectified_pair


def frozen_load(pairs, count):
    """Cycle ranked baseline proposals for controlled 6/9/50-pair loads."""
    if count is None or not pairs.scores.numel():
        return pairs
    indices = torch.arange(count, device=pairs.scores.device) % len(pairs.scores)
    return PairDetections(pairs.boxes6[indices], pairs.scores[indices], pairs.indices[indices], pairs.classes[indices])


def device_pairs(output):
    boxes = torch.cat((output.boxes_left, output.boxes_right[:, [0, 2]]), dim=1)
    if output.crop_top_px:
        boxes = boxes - boxes.new_tensor((0, output.crop_top_px, 0, output.crop_top_px, 0, 0))
    return PairDetections(boxes, output.scores, output.pair_ids, output.classes)


def elapsed(device, call):
    synchronize(device)
    start = perf_counter()
    result = call()
    synchronize(device)
    return (perf_counter() - start) * 1000, result


def profile_encoder(encoder, left, right, device):
    marks = {}

    def stamp(name):
        synchronize(device)
        marks[name] = perf_counter()

    with ExitStack() as stack:
        register_encoder_boundaries(stack, encoder, stamp)
        features = encoder(left, right)
    return {**encoder_intervals(lambda first, last: (marks[last] - marks[first]) * 1000),
            "encoder_padded_hw": features.metadata.padded_size}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=ROOT / "models/stereo-yolo26-fixed56-seed17-e20.pt")
    parser.add_argument("--openstereo-source", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--crop-top-px", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--power-mode", required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-frames", type=int, help="limit the split for a smoke test")
    parser.add_argument("--profile", action="store_true", help="time backbone, FPN and adapters in a separate pass")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--pretrained-v3", action="store_true", help="load TIMM MobileNetV3-Small pretrained blocks")
    parser.add_argument("--candidate-backbone", choices=("v3-small", "v2-050"), default="v3-small")
    parser.add_argument("--candidate-weights", type=Path, help="trained checkpoint for the requested candidate")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1 or (args.max_frames is not None and args.max_frames < 1):
        parser.error("invalid number of warmups, repeats, or frames")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    model, bundle = load_published_model(args.weights, args.openstereo_source, device, apply_paired_nms=True,
                                         crop_top_px=args.crop_top_px)
    if encoder_backbone_from_config(bundle["config"]) != "v2":
        parser.error("--weights must reference the 1.0× baseline")
    arm_name = "B1_V2Half" if args.candidate_backbone == "v2-050" else "B1_V3Small"
    if args.candidate_weights:
        candidate, candidate_bundle = load_published_model(args.candidate_weights, args.openstereo_source, device,
            apply_paired_nms=True, crop_top_px=args.crop_top_px)
        if encoder_backbone_from_config(candidate_bundle["config"]) != args.candidate_backbone:
            parser.error("candidate weights do not match --candidate-backbone")
    else:
        trunk = (build_lightstereo_backbone(args.openstereo_source, encoder_backbone="v2-050")
                 if args.candidate_backbone == "v2-050" else
                 build_lightstereo_v3small(args.openstereo_source, pretrained=args.pretrained_v3))
        encoder = LightStereoFeatureEncoder(trunk, bundle["config"]["adapter_channels"]).to(device).eval()
    records = validation_dataset(args.dataset_root, args.split_json, bundle).records[:args.max_frames]
    if not records:
        parser.error("validation split contains no frames")
    normal = bundle["normalization"]
    rows = []
    profiles = []
    for frame, record in enumerate(records):
        left, right, geometry = load_rectified_pair(record.left_image, record.right_image, normal, args.crop_top_px)
        left, right = left.to(device), right.to(device)
        baseline = model.forward_prepared(left, right, geometry)
        pairs = device_pairs(baseline)
        loads = (0, 1, 3, 6) if args.candidate_backbone == "v2-050" else (6, 9, 50)
        for load, count in [("observed_c0", None), *((f"pairs_{n}", n) for n in loads)]:
            replay = frozen_load(pairs, count)
            if count and not len(replay.indices):
                continue
            for repeat in range(-args.warmup if not frame else 0, args.repeats):
                order = ["B0_V2", arm_name]
                if (frame + repeat) % 2:
                    order.reverse()
                row = {"frame_id": record.frame_id, "load": load, "repeat": repeat,
                       "baseline_pair_ids": replay.indices.tolist(), "encoder_padded_hw": baseline.encoder_padded_hw}
                for arm in order:
                    runner = candidate if args.candidate_weights and arm == arm_name else model
                    original = runner.encoder
                    if not args.candidate_weights and arm == arm_name:
                        runner.encoder = encoder
                    try:
                        forward_ms, output = elapsed(device, lambda: runner.forward_prepared(
                            left, right, geometry, replay_pairs=replay))
                        encoder_ms, _ = elapsed(device, lambda: runner.encoder(left, right))
                    finally:
                        runner.encoder = original
                    if not torch.equal(output.pair_ids, replay.indices):
                        raise RuntimeError(f"{record.frame_id}: {arm} changed the replay pair IDs")
                    row[arm] = {"forward_ms": forward_ms, "encoder_ms": encoder_ms, "pairs": len(output.pair_ids)}
                if repeat >= 0:
                    rows.append(row)
        if args.profile:
            profiles.append({"frame_id": record.frame_id, **{
                arm: profile_encoder(features, left, right, device) for arm, features in (
                    ("B0_V2", model.encoder), (arm_name, candidate.encoder if args.candidate_weights else encoder))
            }})
    write_json(args.output, {
        "commit": revision(ROOT), "openstereo_commit": revision(args.openstereo_source),
        "weights_sha256": sha256(args.weights), "split_sha256": sha256(args.split_json),
        "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "power_mode": args.power_mode, "crop_top_px": args.crop_top_px, "seed": args.seed, "input_path": "prepared",
        "v3_pretrained_blocks": args.pretrained_v3,
        "candidate_backbone": args.candidate_backbone,
        "candidate_weights_sha256": sha256(args.candidate_weights) if args.candidate_weights else None,
        "replay_boundary": "after dense head, before decode and RoI dispatch; decode and NMS omitted in both arms",
        "load_source": "per-frame V2 C0; fixed counts cycle actual C0 RoIs when needed",
        "summary": {load: {arm: summarize([row[arm] for row in rows if row["load"] == load])
                           for arm in ("B0_V2", arm_name)}
                    for load in dict.fromkeys(row["load"] for row in rows)},
        "timings": rows,
        "encoder_profiles": profiles,
        "encoder_profile_summary": {
            arm: summarize([row[arm] for row in profiles]) for arm in ("B0_V2", arm_name)
        } if profiles else {},
    })


if __name__ == "__main__":
    main()
