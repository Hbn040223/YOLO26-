from contextlib import nullcontext
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import patch
import warnings

import numpy as np
import torch
from torch.profiler import profile, ProfilerActivity, record_function

from infer import load_published_model, load_rectified_pair
from lightbackbone_stereo_yolo.edge_deployment import FEATURE_OUTPUTS

root=Path('outputs/v2-050')
out=root/'idisp-fpn/jetson/20261126-precision'
out.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
torch.backends.cudnn.deterministic=True
m,bundle=load_published_model(root/'formal/best_r2.pt',Path('/tmp/yolo26-openstereo'),'cuda:0',
    score=.48,apply_paired_nms=True,near10_mode='range')
frame='000004_10'
base=root/'kitti2015_mixed_range_048'/frame
meta=json.loads((base/'timing.json').read_text())
l,r,g=load_rectified_pair(*(Path('/tmp/yolo26-mixed-frames')/f'{frame}_image_{i}.png' for i in (2,3)),bundle['normalization'],0)
l,r=l.cuda(),r.cuda()
route={}
sync=torch.cuda.synchronize
head=m.instance_head
fields=('objectness','class_logits','box6')
variants=('fp32','encoder-fp16-experimental','amp-fp16')
result={v:{} for v in variants}

def cast(value,name,boundaries):
    with record_function(f'fp32_boundary.{name}'):
        converted=value.float()
    if boundaries is not None:
        boundaries[name]=dict(source_dtype=str(value.dtype),target_dtype=str(converted.dtype),
                              device=str(value.device),shape=list(value.shape),
                              storage_changed=converted.data_ptr()!=value.data_ptr())
    return converted

def apply(mode,n,boxes,boundaries=None):
    first=mode!='fp32'
    amp=mode=='amp-fp16'
    with torch.autocast('cuda',dtype=torch.float16) if first else nullcontext():
        with record_function('audit.encoder'):
            pyramid=m.encoder(l,r)
        if amp:
            with record_function('audit.detector'):
                dense=m.detector(pyramid)
    if first:
        names=FEATURE_OUTPUTS if amp else (*FEATURE_OUTPUTS,'p3_right','p4_right','c5_right')
        pyramid=SimpleNamespace(**{name:cast(getattr(pyramid,name),name,boundaries) for name in names},
                                metadata=pyramid.metadata)
    if not amp:
        with record_function('audit.detector'):
            dense=m.detector(pyramid)
    else:
        dense=SimpleNamespace(**{name:cast(getattr(dense,name),name,boundaries) for name in fields})
    head.idisp.amp_aggregation=amp
    if n is None:
        return m.decode_and_refine(l,pyramid,dense,input_geometry=g,
                                   calibration=(meta['bf_px_m'],meta['cx_delta_px']))
    expanded=SimpleNamespace(**{name:getattr(pyramid,name).expand(n,-1,-1,-1) for name in
                                ('p2_left','p2_right','p3_left','p4_left')})
    return head.forward_disparity(expanded,boxes[:n],*g.original_hw)[0]

with torch.inference_mode():
    m.forward_prepared(l,r,g,route=route,calibration=(meta['bf_px_m'],meta['cx_delta_px']))
    pairs=route['idisp']
    idx=sorted(range(len(pairs.indices)),key=lambda i:(-pairs.scores[i].item(),pairs.indices[i].item()))
    assert len(idx)>=9
    idx=idx[:9]
    boxes=pairs.boxes6[idx]
    for n in (6,9,None):
        ref=apply('fp32',n,boxes)
        for mode in variants:
            value=apply(mode,n,boxes)
            if n:
                common=ref.valid & value.valid
                result[mode][n]=dict(common_pixels=int(common.sum()),valid_equal=torch.equal(ref.valid,value.valid),
                    max_abs_disparity_delta_px=(value.absolute[common]-ref.absolute[common]).abs().max().item())
            else:
                result[mode]['automatic']=dict(output_ids=value.pair_ids.tolist(), reference_ids=ref.pair_ids.tolist(),
                    distances_m=value.near10_distance_m.tolist())
        if n is None:
            continue
        for _ in range(5):
            for mode in variants:
                apply(mode,n,boxes)
                sync()
        timings={v:[] for v in variants}
        for i in range(12):
            for mode in variants[i%3:]+variants[:i%3]:
                start=perf_counter()
                apply(mode,n,boxes)
                sync()
                timings[mode].append((perf_counter()-start)*1000)
        for mode in variants:
            boundaries={}
            result[mode][n].update(wall_timings_ms=timings[mode],wall_p50_p95_ms=np.percentile(timings[mode],[50,95]).tolist(),
                                   tensor_boundaries=boundaries)
            with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],record_shapes=True) as prof:
                apply(mode,n,boxes,boundaries)
                sync()
            selected={}
            for e in prof.events():
                if not any(part in e.name for part in ('copy_','_to_copy','Synchronize','contiguous','_local_scalar_dense')):
                    continue
                parent=e.cpu_parent
                names=[]
                while parent:
                    names.append(parent.name)
                    parent=parent.cpu_parent
                stage=next((name for name in names if name.startswith(('stereo.','audit.','fp32_boundary.'))),'outer')
                key=f'{stage}/{e.name}'
                item=selected.setdefault(key,dict(calls=0,self_cpu_us=0.,device_us=0.))
                item['calls']+=1
                item['self_cpu_us']+=e.self_cpu_time_total
                item['device_us']+=e.device_time_total
            result[mode][n]['profiler_selected']=selected
            print(mode,n,result[mode][n]['wall_p50_p95_ms'],flush=True)
    f=torch.ones(2,device='cuda',dtype=torch.float32)
    h=f.half()
    with profile(activities=[ProfilerActivity.CPU]) as cast_prof:
        f2=f.float()
        converted=h.float()
        twice=converted.float()
    no_op_test=dict(fp32_same_tensor=f2 is f, fp16_to_fp32_changes_storage=h.data_ptr()!=converted.data_ptr(),
                    second_float_same_tensor=twice is converted,
                    to_copy_calls=sum(e.name=='aten::_to_copy' for e in cast_prof.events()))
(out/'precision.json').write_text(json.dumps(dict(commit='308064e',frame_id=frame,pair_ids_9=pairs.indices[idx].tolist(),
    timing_samples_per_load=12,same_dtype_test=no_op_test,variants=result),indent=2,allow_nan=False))
