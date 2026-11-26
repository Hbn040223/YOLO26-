import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import torch

from benchmark_edge_stream import infer_frame, run_stream
from infer import load_published_model

root = Path('outputs/v2-050')
out = root / 'idisp-fpn/jetson/20261126-replay-fp32'
out.mkdir(parents=True, exist_ok=True)
device = torch.device('cuda:0')
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.deterministic = True
model, bundle = load_published_model(root/'formal/best_r2.pt', Path('/tmp/yolo26-openstereo'), device,
    score=.48, apply_paired_nms=True, near10_mode='range', near10_mask_iou=.50, near10_review_box_iou=.45)
frames=[]
for frame in ('000000_10','000004_10','000005_10','000058_10'):
    if frame == '000000_10':
        meta_file = root/'nms_kitti2015_000000_10_range_048_warm/timing.json'
        paths = [root/'nms_kitti2015_000000_10_gt/official/testing'/f'image_{i}'/f'{frame}.png' for i in (2,3)]
    else:
        meta_file = root/'kitti2015_mixed_range_048'/frame/'timing.json'
        paths = [Path('/tmp/yolo26-mixed-frames')/f'{frame}_image_{i}.png' for i in (2,3)]
    meta = json.loads(meta_file.read_text())
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert('RGB'))
    frames.append((frame,(*images,(meta['bf_px_m'],meta['cx_delta_px']))))
def infer(frame):
    return infer_frame(model,frame,bundle['normalization'],device,'fp32')
with torch.inference_mode():
    for _ in range(3):
        for _,frame in frames:
            infer(frame)
    with (out/'tegrastats.log').open('w') as log, subprocess.Popen(['tegrastats','--interval','1000'],stdout=log) as proc:
        try:
            report = run_stream(frames,infer,rate_fps=5,duration_s=120)
        finally:
            proc.terminate()
    buckets = {}
    for row in report['rows']:
        buckets.setdefault(row['details']['idisp_pairs'], []).append(row['latency_ms'])
    report.update(commit='308064e', precision='fp32', backend='loop', tf32=False, power_mode='25W',
                  sequence='repeating four KITTI diagnostic frames',
                  per_load={n:dict(count=len(vals),p50_ms=float(np.percentile(vals,50)),
                                   p95_ms=float(np.percentile(vals,95))) for n,vals in buckets.items()})
    (out/'stream-range-5hz.json').write_text(json.dumps(report,indent=2))
    print({k:v for k,v in report.items() if k!='rows'},flush=True)
