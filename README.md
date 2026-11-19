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

## Evaluation

The 40-frame KITTI validation experiment uses the same frozen B2a pairs for both local disparity configurations. Prediction-driven EPE and D1 are measured on common GT foreground pixels. Accuracy below averages seeds 17, 2 and 5 at epoch 20. Timing includes all 2,323 retained pairs and the Mask branch, measured on an RTX 6000 Ada using seed 17.

| Metric | Native padding, 63 bins | Fixed 56×56, 33 bins |
|---|---:|---:|
| Pred-RoI EPE | 2.863 px | 2.637 px |
| Pred-RoI D1 | 22.63% | 21.07% |
| End-to-end P95 | 3,457 ms | 250 ms |
| Peak allocated GPU memory | 2,972 MiB | 1,319 MiB |

See [the full evaluation report](evaluation/summary.json). Training and evaluation entry points remain in `scripts/`. KITTI data is obtained separately.
