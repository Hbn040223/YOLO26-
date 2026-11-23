"""Export a one-to-one inference checkpoint without the training-only head."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch


def prune_one_to_many(bundle: dict) -> dict:
    config = bundle["config"]
    if config.get("detector_variant") != "stereo_yolo26_pan" or not config.get("one_to_one"):
        raise ValueError("export requires a stereo_yolo26_pan checkpoint with one-to-one enabled")
    if config.get("one_to_one_only"):
        raise ValueError("checkpoint is already a one-to-one deployment export")

    state = bundle["model"]
    many_prefix, one_prefix = "detector.levels.", "detector.one_to_one_levels."
    many = {key[len(many_prefix):]: tensor for key, tensor in state.items() if key.startswith(many_prefix)}
    one = {key[len(one_prefix):]: tensor for key, tensor in state.items() if key.startswith(one_prefix)}
    if not many or many.keys() != one.keys() or any(many[key].shape != one[key].shape for key in many):
        raise ValueError("checkpoint must contain matching complete one-to-many and one-to-one branches")

    return {
        **{key: value for key, value in bundle.items() if key != "optimizer"},
        "config": {**config, "one_to_one_only": True},
        "model": {key: tensor for key, tensor in state.items() if not key.startswith(many_prefix)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.weights.resolve() == args.output.resolve():
        parser.error("choose a separate output path for the deployment checkpoint")
    if args.output.exists():
        parser.error("the output path already exists")
    bundle = torch.load(args.weights, map_location="cpu", weights_only=True)
    try:
        deployment = prune_one_to_many(bundle)
    except ValueError as error:
        parser.error(str(error))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=args.output.parent, suffix=".pt", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(deployment, temporary)
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)

    removed = len(bundle["model"]) - len(deployment["model"])
    print(f"Exported {args.output} after removing {removed} one-to-many tensors.")
    print(f"Checkpoint bytes: {args.weights.stat().st_size:,} -> {args.output.stat().st_size:,}")


if __name__ == "__main__":
    main()
