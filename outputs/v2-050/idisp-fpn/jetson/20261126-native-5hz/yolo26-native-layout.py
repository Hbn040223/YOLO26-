import hashlib
import json
from pathlib import Path
import subprocess
from time import perf_counter
from PIL import Image
import numpy as np
import torch
from benchmark_edge_stream import infer_frame, run_stream
from infer import load_published_model

device = torch.device('cuda:0')
root = Path('outputs/v2-050')
out = root / 'idisp-fpn/jetson/20261126-native-5hz'
out.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
weights = root/'formal/best_r2.pt'
source = Path('/tmp/yolo26-openstereo')
model, bundle = load_published_model(weights,source,device,score=.48,apply_paired_nms=True,
    near10_mode='range',near10_mask_iou=.50,near10_review_box_iou=.45)
normalization=bundle['normalization']
sync=torch.cuda.synchronize
frames=[]
for frame in ('000000_10','000004_10','000005_10','000058_10'):
    if frame == '000000_10':
        meta_file = root/'nms_kitti2015_000000_10_range_048_warm/timing.json'
    else:
        meta_file = root/'kitti2015_mixed_range_048'/frame/'timing.json'
    paths = [Path('/tmp/yolo26-mixed-frames')/f'{frame}_image_{i}.png' for i in (2,3)]
    meta = json.loads(meta_file.read_text())
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert('RGB'))
    frames.append((frame,(*images,(meta['bf_px_m'],meta['cx_delta_px']))))


from contextlib import nullcontext
from unittest.mock import patch
import benchmark_edge_stream as streaming
from lightbackbone_stereo_yolo.stereo_input import StereoInputGeometry

layout_model, _ = load_published_model(weights,source,device,score=.48,apply_paired_nms=True,
    near10_mode='range',near10_mask_iou=.50,near10_review_box_iou=.45,idisp_cost_backend='bmm')
layout_model.encoder.to(memory_format=torch.channels_last)
layout_model.detector.to(memory_format=torch.channels_last)
fused_model, _ = load_published_model(weights,source,device,score=.48,apply_paired_nms=True,
    near10_mode='range',near10_mask_iou=.50,near10_review_box_iou=.45,idisp_cost_backend='bmm')
from torch import nn
from torch.nn.utils.fusion import fuse_conv_bn_eval
from timm.models.efficientnet_blocks import ConvBnAct, DepthwiseSeparableConv, InvertedResidual

def fuse(module, pairs):
    count=0
    for convolution,normalization in pairs:
        conv,bn=getattr(module,convolution),getattr(module,normalization)
        setattr(module,convolution,fuse_conv_bn_eval(conv,bn,transpose=isinstance(conv,nn.ConvTranspose2d)))
        setattr(module,normalization,nn.Identity())
        count+=1
    return count

backbone=fused_model.encoder.backbone
fused_count=fuse(backbone,(('conv_stem','bn1'),))
for module in list(backbone.modules()):
    if isinstance(module,InvertedResidual):
        fused_count+=fuse(module,(('conv_pw','bn1'),('conv_dw','bn2'),('conv_pwl','bn3')))
    elif isinstance(module,DepthwiseSeparableConv):
        fused_count+=fuse(module,(('conv_dw','bn1'),('conv_pw','bn2')))
    elif isinstance(module,ConvBnAct):
        fused_count+=fuse(module,(('conv','bn1'),))
    elif (isinstance(module,nn.Sequential) and len(module)>=2 and
          isinstance(module[0],(nn.Conv2d,nn.ConvTranspose2d)) and isinstance(module[1],nn.BatchNorm2d)):
        fused_count+=fuse(module,(('0','1'),))
fused_model.encoder.to(memory_format=torch.channels_last)
fused_model.detector.to(memory_format=torch.channels_last)
model.instance_head.idisp.cost_backend='bmm'
stream=torch.cuda.Stream(device=device)
mean=torch.tensor(normalization['mean'],device=device)
std=torch.tensor(normalization['std'],device=device)
stream.wait_stream(torch.cuda.current_stream(device))

def prepare_gpu(left,right,normalization,*,timings):
    if left.size!=right.size or left.mode!='RGB' or right.mode!='RGB':
        raise ValueError('GPU normalization probe expects decoded rectified RGB images')
    start=perf_counter()
    pair=torch.from_numpy(np.stack((np.asarray(left),np.asarray(right)))).to(device)
    pair=pair.to(torch.float32).div_(255.).sub_(mean).div_(std).permute(0,3,1,2)
    timings['gpu_prepare_enqueue_ms']=(perf_counter()-start)*1000
    return pair[:1],pair[1:],StereoInputGeometry((left.height,left.width),0)

def run(name,frame,trace=False):
    active=fused_model if 'fused-bn' in name else (layout_model if 'channels-last' in name else model)
    precision='encoder-fp16' if 'encoder-fp16' in name else 'amp-fp16'
    active.instance_head.idisp.amp_aggregation=precision=='amp-fp16'
    torch.backends.cudnn.allow_tf32=True
    preprocess=patch.object(streaming,'prepare_rectified_images',prepare_gpu) if 'gpu-input' in name else nullcontext()
    with torch.cuda.stream(stream),preprocess:
        return streaming.infer_frame(active,frame,normalization,device,precision,trace=trace)

modes=('native-amp','native-amp-channels-last','native-amp-gpu-input','native-amp-channels-last-gpu-input',
       'native-encoder-fp16','native-encoder-fp16-channels-last-gpu-input','native-amp-fused-bn-channels-last-gpu-input')
report=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),fused_batch_norm_count=fused_count,results={})
for frame_id,frame in frames:
    expected,expected_route,_=run(modes[0],frame,True)
    rows={}
    for name in modes:
        actual,route,_=run(name,frame,True)
        equal_ids=torch.equal(expected.pair_ids,actual.pair_ids)
        rows[name]=dict(samples=[],final_ids_equal=equal_ids,entry_ids_equal=torch.equal(expected_route['idisp'].indices,route['idisp'].indices),
                        max_distance_delta_m=(actual.near10_distance_m-expected.near10_distance_m).abs().max().item()
                        if equal_ids and expected.pair_ids.numel() else None)
        for _ in range(3):
            run(name,frame)
    for i in range(10):
        for name in modes[i%len(modes):]+modes[:i%len(modes)]:
            start=perf_counter()
            run(name,frame)
            rows[name]['samples'].append((perf_counter()-start)*1000)
    for name,row in rows.items():
        row['p50_p95_ms']=np.percentile(row['samples'],[50,95]).tolist()
    report['results'][frame_id]=rows
    (out/'native-layout-comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(frame_id,{k:[v['p50_p95_ms'],v['final_ids_equal']] for k,v in rows.items()},flush=True)

eligible=[mode for mode in modes if all(rows[mode]['final_ids_equal'] and rows[mode]['p50_p95_ms'][1]<=200
                                        for rows in report['results'].values())]
report['replay_eligible_modes']=eligible
(out/'native-layout-comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
if eligible:
    selected=min(eligible,key=lambda mode:max(rows[mode]['p50_p95_ms'][1] for rows in report['results'].values()))
    replay=run_stream(frames,lambda frame: run(selected,frame),rate_fps=5,duration_s=120)
    replay['mode']=selected
    replay['passed']=replay['completed']==600 and replay['latency_p95_ms']<=200 and replay['in_flight_at_end']==replay['dropped']==0
    (out/'native-layout-replay-5hz.json').write_text(json.dumps(replay,indent=2,allow_nan=False)+'\n')
