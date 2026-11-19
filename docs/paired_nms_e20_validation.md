# PR #1 · Paired NMS validation on KITTI

The optional inference path passes the live index checks. All tested hard-NMS IoUs lost ground-truth matches at the locked score threshold. The inference default remains off.

## Frozen evaluation protocol

- Tested implementation [e3040f1](https://github.com/Hbn040223/YOLO26-/commit/e3040f1a05c8a083f3bc8101a040165ca668deb4). All 165 CPU tests passed, including the overlap counterexample that distinguishes duplicate ownership from one-to-one TP assignment.
- Released seed 17 E20 `fixed_56` checkpoint, 33 local disparity bins, frozen B2a one-to-one detector and `legacy_clamp`. Weight SHA-256 `8e1caf6577dee2c92f64cf0bea62ec0e08b8d15832977b17830192f9fe1ffeb5`.
- KITTI `pilot_random_seed17/val.json`, 40 frames, 287 valid paired GT instances. Split SHA-256 `4e5a06c19d036c9234ed11c68e9ff8dd224dc610005e47aa5828d01975753651`. The available E30 checkpoint belongs to an earlier model without the one-to-one head or fixed grid. This report evaluates E20.
- Capture starts before score filtering and top-K. The replay applies score filtering, pre-NMS top-K 500, decoded validity filtering, optional class-aware paired NMS and RoI dispatch. No extra final cap was applied. All kept pairs count in the precision denominator. GT eligibility comes from valid paired annotations returned by the data loader. The evaluator applies no additional prediction-ignore exemption.
- Bilateral matching requires the same class and same GT in both views with joint IoU ≥0.50. Matching is quality-first, then score and fixed row order. Single-view recall uses the existing class-agnostic definition. Duplicate counts assign each qualifying prediction to its best eligible GT before counting repeats. The overlap subset uses same-class GTs with positive left-view IoU.
- Score selection preserves matched GT count and pre-NMS left hits versus score 0.40 with NMS off. Among eligible scores, the preregistered order is highest precision, then fewer pairs, then the higher score. The retention gate requires zero lost GT identities and no net reduction in matches. All three IoUs failed retention. The least-loss arm at 0.70 was used for CUDA benchmarking on this validation split.

## Score sweep with NMS off

| Score | Paired matches / GT | Precision | Pairs entering RoI | Pre-NMS left hits |
|---:|---:|---:|---:|---:|
| 0.05 | 191 / 287 | 1.14% | 16,823 | 222 |
| 0.10 | 190 / 287 | 1.62% | 11,742 | 220 |
| 0.15 | 185 / 287 | 2.25% | 8,240 | 211 |
| 0.25 | 172 / 287 | 3.70% | 4,655 | 193 |
| 0.30 | 166 / 287 | 4.51% | 3,680 | 187 |
| 0.35 | 163 / 287 | 5.54% | 2,941 | 185 |
| **0.40** | **156 / 287** | **6.72%** | **2,323** | **173** |

There were 386,739 valid candidate rows before score filtering or top-K. The histogram separates 4,235 rows eligible for at least one same-class GT pair and 382,504 other rows. Score bins span [0,1] in steps of 0.1. Eligible counts are 1,099 / 695 / 529 / 476 / 489 / 412 / 239 / 188 / 95 / 13. Other counts are 373,547 / 5,309 / 1,880 / 881 / 517 / 224 / 92 / 47 / 7 / 0.

## Same-weight comparison at score 0.40

| NMS IoU | Paired matches / GT | Coverage | Precision | RoI pairs | Duplicates | Overlap GT duplicates | Lost GT | Cars <10 m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Off | 156 / 287 | 54.36% | 6.72% | 2,323 | 1,282 | 597 | 0 | 19 / 23 |
| 0.50 | 142 / 287 | 49.48% | 26.79% | 530 | 15 | 11 | 14 | 19 / 23 |
| 0.60 | 146 / 287 | 50.87% | 22.29% | 655 | 52 | 30 | 10 | 19 / 23 |
| 0.70 | 151 / 287 | 52.61% | 17.50% | 863 | 129 | 73 | 5 | 19 / 23 |

Pre-NMS was identical at 2,323 pairs, 173 left GT hits and 163 right GT hits. At IoU 0.70, 10,000 paired frame bootstrap resamples give a coverage delta of −1.74 percentage points (95% interval −3.08 to −0.43) and a precision delta of +10.78 points (95% interval +9.03 to +13.13).

All five lost GTs had zero qualifying survivors. The higher-score pair that suppressed the previously matched pair had joint IoU below 0.50 against that GT every time. The five suppressor-to-GT IoUs were 0.492, 0.476, 0.498, 0.441 and 0.497.

## Live CUDA bench · off versus IoU 0.70

All 40 frames matched offline replay for the pre-NMS candidate and final RoI indexes. Pre-NMS boxes, scores and classes were identical for the two live arms. The off/on arms invoked NMS 0/40 times. On frame `000018_10`, instance input changes from 37 to 11 pairs.

| Latency | Off P50 | On P50 | Off P95 | On P95 |
|---|---:|---:|---:|---:|
| Pure paired NMS | 0.00 ms | 11.59 ms | 0.00 ms | 42.53 ms |
| Decode and RoI dispatch | 1.45 ms | 12.63 ms | 7.02 ms | 45.53 ms |
| Instance head | 18.63 ms | 11.27 ms | 77.38 ms | 25.95 ms |
| End-to-end | 31.60 ms | 35.38 ms | 88.73 ms | 86.31 ms |

RTX 6000 Ada, FP32, batch 1, two CPU threads, two warmups per arm and three alternating timed repeats per frame. Latency excludes input loading, transfer and output export. The paired end-to-end on-minus-off median is +2.10 ms. The measured end-to-end P95 falls slightly while its median increases.

Of the 151 GTs matched by both arms, 96 use a different winning dense ID. Common foreground contains 604,763 pixels. Local disparity EPE is 2.443 px off and 2.371 px on. D1 is 19.66% off and 19.48% on. The frozen Mask's mean instance IoU is 0.3052 off and 0.3065 on. These numbers use common matches. The five lost GTs are accounted for in coverage.

## Reproduction

Use the published checkpoint and OpenStereo revision in the [README](../README.md).

```bash
python scripts/evaluate_paired_nms.py --openstereo-source OpenStereo \
  --dataset-root /path/to/KITTI --split-json /path/to/val.json \
  --device cuda:0 --scores .05,.10,.15,.25,.30,.35,.40 \
  --nms-ious .5,.6,.7 --gt-iou .50 --topk 500 --bootstrap 10000 \
  --save-candidates e20_candidates.json --output e20_replay.json

python scripts/benchmark_paired_nms.py --openstereo-source OpenStereo \
  --dataset-root /path/to/KITTI --split-json /path/to/val.json \
  --device cuda:0 --score .40 --topk 500 --nms-iou .7 \
  --warmup 2 --repeats 3 --replay e20_replay.json --output e20_benchmark.json
```
