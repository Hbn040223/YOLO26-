# Stereo-YOLO26 + LightStereo fixed_56

This release contains the trained **fixed_56** instance disparity model. The stereo YOLO26 head emits linked left and right boxes. Shared LightStereo features feed a fixed 56×56 P2 grid per box pair. The local disparity head uses 33 residual bins (Dmax = 16). Left P3 and P4 supply context on 28×28 and 14×14 grids.

## Checkpoint

**models/stereo-yolo26-fixed56-seed17-e20.pt** carries the full network state from seed 17, epoch 20. LightStereo features, stereo detections, and the Mask head use frozen B2a weights. The local iDisp head was trained on KITTI. The bundle serves inference and contains no optimizer state. Verify it from the repository root with `sha256sum -c models/SHA256SUMS`.

## Inference

The tested environment used Python 3.8, PyTorch 2.1.2, torchvision 0.16.2, timm 0.4.12, NumPy 1.24.4 and Pillow 9.5.0. Install PyTorch and torchvision for your device, then install the remaining dependencies and check out the tested OpenStereo source revision.

```bash
python -m pip install timm==0.4.12 numpy==1.24.4 Pillow==9.5.0
git clone https://github.com/XiandaGuo/OpenStereo.git
git -C OpenStereo checkout 23d71c92e33ad1f80dfc42bf29f5c6a914d38769
python scripts/infer.py --openstereo-source OpenStereo \
  --left /path/to/left.png --right /path/to/right.png \
  --device cuda:0 --output disparity.npz
```

Use a rectified left/right pair with equal image dimensions. The released checkpoint already provides the backbone weights. Inference defaults to score 0.40 and pre-top-k 500 with the one-to-one detector. The NPZ contains aligned `boxes_left`, `boxes_right`, scores, classes, the per-pair absolute disparity in original-image pixels, the disparity validity mask, and the continuous crop geometry. Mask probabilities are in `mask_probability`. Given a disparity array index `(v,u)`, its left-image grid center is `(c_l + (u+0.5)*c/w_roi, y0 + (v+0.5)*h_crop/h_roi)`.

Append `--paired-nms --paired-nms-iou .5` to try class-aware stereo suppression. With the flag omitted, the published one-to-one model still skips NMS. When enabled, a lower-scoring pair is removed only if its class matches and both view IoUs exceed the NMS threshold. The kept pair indices stay aligned through RoI, mask, and disparity. Ties in anchor-free top-k and NMS prefer the original dense candidate order.

### Paired NMS on/off evaluation

Use the same checkpoint and KITTI validation split in both arms. The evaluator captures a dense candidate archive before score filtering/top-k, then replays NMS off and each IoU on the same pre-NMS pool at every score threshold:

```bash
python scripts/evaluate_paired_nms.py --openstereo-source OpenStereo \
  --dataset-root /path/to/KITTI --split-json /path/to/validation.json \
  --device cuda:0 --scores .05,.10,.15,.25,.30,.35,.40 \
  --nms-ious .5,.6,.7 --topk 500 \
  --save-candidates paired_candidates.json --output paired_nms_eval.json

python scripts/evaluate_paired_nms.py --candidates paired_candidates.json \
  --scores .25,.40 --nms-ious .5,.6,.7 --output paired_nms_replay.json
```

The report records checkpoint and split hashes, score distributions, pair counts, duplicates, kept dense IDs, one-to-one coverage/precision, and paired frame bootstrap intervals for on-minus-off differences. Matching is quality-first with both views required to hit the same GT. Use `--class-agnostic-matching` to compare with historical matcher semantics while keeping suppression class-aware. The NMS IoU and GT matching IoU are separate settings. Both evaluation and timing fail on missing assets referenced by the split's annotations so the cohort cannot silently shrink.

Choose an operating point on validation data subject to an agreed coverage loss, then confirm it on held-out data and measure full model latency. Keep the score, top-K, checkpoint and split consistent between the offline comparison and the benchmark. For example:

```bash
python scripts/benchmark_paired_nms.py --openstereo-source OpenStereo \
  --dataset-root /path/to/KITTI --split-json /path/to/validation.json \
  --device cuda:0 --score .40 --topk 500 --nms-iou .5 \
  --warmup 2 --repeats 3 --output paired_nms_timing.json
```

Timing runs both arms on each stereo input in alternating order, using batch-one forwards and CUDA synchronization. P50/P95 end-to-end latency excludes input loading, transfer, and output export. Stage timings include decode and RoI dispatch (including optional NMS) and the instance head (RoI, Mask, and iDisp). Empty outputs skip the instance head. Positive on-minus-off latency means the NMS arm took longer. Pair counts show RoI workload; Mask/iDisp accuracy requires separate evaluation.

## Evaluation

The 40-frame KITTI validation experiment uses the same frozen B2a pairs for both local disparity configurations. Prediction-driven EPE and D1 are measured on common GT foreground pixels. Accuracy below averages seeds 17, 2 and 5 at epoch 20. Timing includes all 2,323 retained pairs and the Mask branch, measured on an RTX 6000 Ada using seed 17.

| Metric | Native padding, 63 bins | Fixed 56×56, 33 bins |
|---|---:|---:|
| Pred-RoI EPE | 2.863 px | 2.637 px |
| Pred-RoI D1 | 22.63% | 21.07% |
| End-to-end P95 | 3,457 ms | 250 ms |
| Peak allocated GPU memory | 2,972 MiB | 1,319 MiB |

See [the full evaluation report](evaluation/summary.json). Training and evaluation entry points remain in `scripts/`. KITTI data is obtained separately.

[Real KITTI E20 paired-NMS validation](docs/paired_nms_e20_validation.md).

## Car foreground disparity

Train the Mask head with the published fixed-grid model and the frozen candidate archive from the paired NMS evaluation:

```bash
PYTHONPATH=src python scripts/train_mask_only.py \
  --dataset-root /path/to/KITTI --train-json /path/to/train.json --val-json /path/to/val.json \
  --openstereo-source OpenStereo --candidate-archive e20_candidates.json --replay e20_replay.json \
  --output-dir outputs/mask_only --device cpu
```

The entry point pins the 144/16 development split and the eight training frames. It recomputes the E20 Mask on archived pairs and starts the full training run after the eight-frame gate passes. See `baseline.json`, `overfit_gate.json`, and `formal_gate.json` for results. The crop-grid Mask IoU is a per-instance mean. Precision and recall pool valid pixels. The original-image Mask metrics use the existing left crop transform.

A checkpoint that passes the formal Car gate is exported as `foreground_ready.pt`. Given that checkpoint, `infer.py` returns `foreground_disparity` and `foreground_validity` on the fixed disparity grid. The row's `pair_ids` field holds the frame-local detector candidate ID. Background and unsupported categories have invalid foreground disparity values. Existing E20 bundles have no validated foreground classes. The CLI warns and writes foreground_supported=false for such bundles. The original `disparity` array still provides the rectangular result for diagnostics.

The optional `--mask-head context` trains a compact encoder-decoder on the same left RoI, preserving the `28×28` Mask output. It starts that Mask head fresh and uses class-weighted BCE. The rest of the published E20 network stays frozen. The training stage rechecks the eight-frame gate for the selected head before formal fitting.

## Experimental foreground evaluation checkpoint

`models/foreground_evaluation.pt` is the exact Car-only checkpoint used for the 40-frame foreground evaluation. It uses the epoch-25 context Mask from the per-instance BCE+Dice run with its development-selected Mask threshold 0.40 folded into the final logit bias. The detector, shared encoder and local iDisp weights come from E20. The inference entry point defaults to the original E20 checkpoint. The legacy E25 bundle uses an effective cutoff of 0.50 on its stored logits. New exports store unfurled logits and an explicit mask_threshold. The --mask-threshold override uses the original probability scale for both formats.

**Development gate override.** The checkpoint was evaluated after explicitly bypassing the development-set GT-RoI Mask IoU requirement of **75%**. It measured **73.83%** GT-RoI Mask IoU and **76.77%** Pred-RoI Mask IoU on the 16-frame development split. Those frames were held out from Mask fitting and had been seen by the frozen E20 network. Use the separate 40 frames for held-out results.

To replay the E23 to E25 continuation, first run the context Mask command above with --mask-head context and preserve its last.pt and neighboring overfit_gate.json. The first phase uses weighted BCE. Run the same data arguments with a fresh output directory and these additional options:

```bash
--stage formal --mask-head context --resume-mask outputs/mask_only/last.pt \
--formal-epochs 2 --formal-lr 1e-4 --formal-order-seed 731 --dice-weight 1 --mask-threshold .40 --dev-gate-iou .75 --allow-mask-gate-override
```

The continuation restores the Mask optimizer and uses unweighted per-instance BCE plus Dice. The published E25 was selected at 0.40 from the original development sweep. The replay pins that selection cutoff. Starting from the saved E23 checkpoint, it reproduced the development and held-out IoUs. All 625 tensors matched the published checkpoint exactly after applying its original bias fold. An override records the failed development gate and exports foreground_evaluation.pt. A ready export requires both gates to pass.

The original KITTI 40-frame acceptance criteria use the 28×28 crop grid as the primary coordinate system. Original-image metrics are reported separately.

| Cohort | Mask IoU | Foreground precision | Foreground recall |
| --- | ---: | ---: | ---: |
| GT-RoI, 257 cars | 69.42% | 82.86% | 89.83% |
| Fixed Pred-RoI, 149 cars | 75.47% | 86.29% | 89.81% |

The fixed Pred-RoI metrics passed. The GT-RoI result was below the 75% acceptance threshold. This artifact is designated for evaluation. With score 0.40, top-K 500 and class-aware paired NMS at IoU 0.70, the 149 matched cars had 92.65% valid GT foreground coverage and 10.49% background among exported foreground pixels. On the retained GT foreground pixels, EPE was 2.25 px and D1 was 19.02%.

Verify the artifact with `sha256sum -c models/SHA256SUMS`. Use it explicitly for inference:

```bash
PYTHONPATH=src python scripts/infer.py \
  --weights models/foreground_evaluation.pt --openstereo-source OpenStereo \
  --left /path/to/left.png --right /path/to/right.png \
  --score .40 --paired-nms --paired-nms-iou .70 --output foreground_eval.npz
```


## Near-car evaluation

The near-car validation cohort contains 23 annotated cars with calibrated GT foreground median depth below 10 m. Use score 0.40, top-K 500 and class-aware paired matching at bilateral IoU 0.50. Training validation defaults to score 0.40 and class-aware paired NMS IoU 0.70. Record threshold changes with --score-threshold and --validation-paired-nms-iou when comparing runs.

| Paired NMS IoU | Near cars matched | Extra qualifying pairs assigned to a near car | Near cars with extra pairs |
| --- | ---: | ---: | ---: |
| Disabled | 19 / 23 | 280 | 19 |
| 0.70 | 19 / 23 | 29 | 14 |
| 0.60 | 19 / 23 | 15 | 12 |
| 0.50 | 19 / 23 | 4 | 4 |

Each qualifying pair is assigned to its highest bilateral-IoU Car GT before counting extra pairs. All four settings missed the same four near cars. At 0.70, the E25 Mask yielded nonempty foreground disparity for 19 near cars. Seventeen of the original 23 also covered at least 85% of their valid GT foreground pixels. Changing the NMS threshold changed some winning pair IDs. A second run generated foreground disparity for the 19 matched cars at each setting, using the same E25 weights.

| Paired NMS IoU | At least 85% foreground coverage, out of 23 | EPE on common foreground (px) | D1 on common foreground |
| --- | ---: | ---: | ---: |
| 0.70 | 17 / 23 | 3.61 | 35.96% |
| 0.60 | 15 / 23 | 3.64 | 35.40% |
| 0.50 | 16 / 23 | 3.44 | 32.41% |

The common foreground contains 252,150 valid GT foreground pixels covered by all three settings. These are near-car validation measurements, conditional on detection for EPE and D1. The <10 m cohort is selected by GT for measurement. The inference model still processes all retained pairs.

See the [matrix NMS benchmark](docs/paired_nms_e20_validation.md#batched-nms-retest--same-e20-checkpoint) for updated latency.
