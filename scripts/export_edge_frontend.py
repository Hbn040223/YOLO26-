"""Export a static V2 stereo frontend and optionally build two TensorRT engines."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

# The Windows ONNX DLL and PyTorch DLLs can conflict when Torch is loaded first.
try:
    import onnx
except ImportError as exc:
    onnx = None
    ONNX_IMPORT_ERROR = exc

import torch

from benchmark_edge_acceleration import revision
from evaluate_paired_nms import sha256, write_json
from infer import ROOT, load_published_model
from lightbackbone_stereo_yolo.edge_deployment import FRONTEND_OUTPUTS, StaticStereoFrontend
from lightbackbone_stereo_yolo.lightstereo_encoder import LightStereoFeatureEncoder


def build_engines(trtexec, onnx):
    help_text = subprocess.check_output([trtexec, "--help"], text=True, stderr=subprocess.STDOUT)
    if "--fp16" not in help_text:
        raise RuntimeError("this builder needs TensorRT 8.6–10.x trtexec with --fp16 support")
    artifacts = {}
    for precision in ("fp32", "fp16"):
        engine = onnx.with_suffix(f".{precision}.plan")
        layers = onnx.with_suffix(f".{precision}.layers.json")
        log = onnx.with_suffix(f".{precision}.build.log")
        command = [trtexec, f"--onnx={onnx}", f"--saveEngine={engine}", f"--exportLayerInfo={layers}",
                   "--profilingVerbosity=detailed", "--dumpLayerInfo", "--skipInference", "--noTF32"]
        if precision == "fp16":
            command.append("--fp16")
        with log.open("w", encoding="utf-8") as handle:
            process = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False)
        if process.returncode:
            raise RuntimeError(f"{precision} TensorRT build failed, see {log}")
        if not engine.is_file() or not layers.is_file():
            raise RuntimeError(f"{precision} TensorRT build did not emit the engine and layer records, see {log}")
        artifacts[precision] = {"engine": engine.name, "sha256": sha256(engine),
                                "layer_info": layers.name, "log": log.name}
    return artifacts


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=ROOT / "models/stereo-yolo26-fixed56-seed17-e20.pt")
    parser.add_argument("--openstereo-source", type=Path, required=True)
    parser.add_argument("--height", type=int, default=375, help="original image height")
    parser.add_argument("--width", type=int, default=1242)
    parser.add_argument("--crop-top-px", type=int, default=0)
    parser.add_argument("--trtexec", help="build FP32 and FP16 engines with this TensorRT 8.6–10.x executable")
    parser.add_argument("--output", type=Path, required=True, help="static .onnx file")
    args = parser.parse_args()
    if args.width <= 0 or not 0 <= args.crop_top_px < args.height or args.output.suffix != ".onnx":
        parser.error("positive image dimensions, valid crop and .onnx output required")
    if onnx is None:
        parser.error(f"the onnx package could not be loaded: {ONNX_IMPORT_ERROR}")
    torch.set_num_threads(2)
    hw = (args.height - args.crop_top_px, args.width)
    padded_hw = LightStereoFeatureEncoder._padded_size(*hw)
    model, bundle = load_published_model(args.weights, args.openstereo_source, "cpu")
    frontend = StaticStereoFrontend(model.encoder, model.detector).eval()
    pair = (torch.zeros(1, 3, *hw), torch.zeros(1, 3, *hw))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        frontend, pair, str(args.output), input_names=("left", "right"), output_names=FRONTEND_OUTPUTS,
        opset_version=17,
    )
    onnx.checker.check_model(str(args.output))
    write_json(args.output.with_suffix(".json"), {
        "commit": revision(ROOT), "openstereo_commit": revision(args.openstereo_source),
        "weights_sha256": sha256(args.weights), "onnx_sha256": sha256(args.output),
        "class_mapping": bundle["class_mapping"], "normalization": bundle["normalization"],
        "original_hw": [args.height, args.width], "input_hw": hw, "encoder_padded_hw": padded_hw,
        "crop_top_px": args.crop_top_px, "outputs": FRONTEND_OUTPUTS, "opset": 17,
        "engines": build_engines(args.trtexec, args.output) if args.trtexec else {},
    })


if __name__ == "__main__":
    main()
