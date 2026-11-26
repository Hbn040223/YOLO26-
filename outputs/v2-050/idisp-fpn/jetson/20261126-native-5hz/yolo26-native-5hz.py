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


modes = {
    'fp32-loop': ('fp32','loop',False),
    'tf32-bmm': ('fp32','bmm',True),
    'amp-bmm': ('amp-fp16','bmm',False),
    'amp-tf32-bmm': ('amp-fp16','bmm',True),
}
head = model.instance_head.idisp
stream = torch.cuda.Stream(device=device)

def run(mode, frame, *, trace=False, timing=False):
    precision,backend,tf32 = modes[mode]
    torch.backends.cudnn.allow_tf32 = tf32
    head.cost_backend = backend
    head.amp_aggregation = precision == 'amp-fp16'
    with torch.cuda.stream(stream):
        return infer_frame(model,frame,normalization,device,precision,trace=trace,stage_timing=timing)

def percentiles(vals):
    return np.percentile(vals,[50,95]).tolist()

report = dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
              torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(),
              weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
              gpu_max_hz=int(Path('/sys/class/devfreq/17000000.gpu/max_freq').read_text()),
              mode_configs=modes, score=.48, records={})
for frame_id,frame in frames:
    ref,ref_route,_=run('fp32-loop',frame,trace=True)
    records={}
    for mode in modes:
        actual,route,_=run(mode,frame,trace=True)
        same=torch.equal(actual.pair_ids,ref.pair_ids)
        records[mode]=dict(samples_ms=[],entry_ids=route['idisp'].indices.tolist(),output_ids=actual.pair_ids.tolist(),
                           entry_ids_equal=torch.equal(route['idisp'].indices,ref_route['idisp'].indices),
                           final_ids_equal=same,distances_m=actual.near10_distance_m.tolist(),
                           max_distance_delta_m=(actual.near10_distance_m-ref.near10_distance_m).abs().max().item()
                           if same and actual.pair_ids.numel() else None)
        for _ in range(4):
            run(mode,frame)
    for i in range(12):
        order=list(modes)
        for mode in order[i%len(order):]+order[:i%len(order)]:
            stamp=perf_counter()
            run(mode,frame)
            records[mode]['samples_ms'].append((perf_counter()-stamp)*1000)
    for mode,record in records.items():
        record['p50_p95_ms']=percentiles(record['samples_ms'])
        record['stage_samples']=[run(mode,frame,timing=True) for _ in range(3)]
    report['records'][frame_id]=records
    (out/'native-comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(frame_id,{mode:[row['p50_p95_ms'],row['final_ids_equal']] for mode,row in records.items()},flush=True)

eligible = [mode for mode in modes if all(r[mode]['final_ids_equal'] for r in report['records'].values())]
selected = min(eligible, key=lambda m:max(r[m]['p50_p95_ms'][1] for r in report['records'].values()))
report['selected_mode'] = selected
(out/'native-comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print('120-second replay:',selected,flush=True)
for i in range(10):
    run(selected,frames[i%len(frames)][1])
with (out/'tegrastats.log').open('w') as log, subprocess.Popen(['tegrastats','--interval','1000'],stdout=log) as proc:
    try:
        replay=run_stream(frames,lambda frame: run(selected,frame),rate_fps=5,duration_s=120)
    finally:
        proc.terminate()
replay['mode'] = selected
replay['passed']=replay['completed']==600 and replay['latency_p95_ms']<=200 and replay['in_flight_at_end']==replay['dropped']==0
replay['per_load']={n:dict(service_p50_p95_ms=percentiles([row['service_ms'] for row in replay['rows'] if row['details']['idisp_pairs']==n]))
                    for n in sorted({row['details']['idisp_pairs'] for row in replay['rows']})}
(out/'native-replay-5hz.json').write_text(json.dumps(replay,indent=2,allow_nan=False)+'\n')
print({k:v for k,v in replay.items() if k!='rows'},flush=True)
