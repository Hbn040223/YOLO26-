import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from time import perf_counter
from unittest.mock import patch
from contextlib import contextmanager, ExitStack
import importlib.util

import numpy as np
import torch

from infer import load_published_model, load_rectified_pair
from lightbackbone_stereo_yolo import model as model_module, local_disparity as ld
from lightbackbone_stereo_yolo.roi_selection import subset_crop
from lightbackbone_stereo_yolo.local_aggregation import AttentionPyramid

out = Path('outputs/v2-050/idisp-fpn/jetson/20261126-fixed-pairs')
out.mkdir(parents=True, exist_ok=True)
historical = Path('outputs/v2-050/kitti2015_mixed_range_048')
weights = Path('outputs/v2-050/formal/best_r2.pt')
source = Path('/tmp/yolo26-openstereo')
spec = importlib.util.spec_from_file_location('cost_prototype', '/tmp/yolo26-cost-bmm.py')
prototype = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prototype)
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.deterministic = True
model, bundle = load_published_model(weights, source, 'cuda:0', score=.48, apply_paired_nms=True,
                                    near10_mode='range', near10_mask_iou=.50, near10_review_box_iou=.45)
head = model.instance_head
sync = torch.cuda.synchronize
loop = ld.build_local_cost_volume
bmm = prototype.batched_cost_volume
warmup, repeats = 20, 60

def digest(path):
    return hashlib.file_digest(path.open('rb'), 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else hashlib.sha256(path.read_bytes()).hexdigest()

def stats(values):
    array = np.asarray(values)
    q = [50, 95]
    rng = np.random.default_rng(17)
    draws = array[rng.integers(len(array), size=(10000, len(array)))]
    points = np.percentile(array, q)
    intervals = np.percentile(np.percentile(draws, q, axis=1), [2.5, 97.5], axis=1).T
    return {f'p{p}': dict(ms=float(x), ci95_ms=ci.tolist()) for p, x, ci in zip(q, points, intervals)}

def timed(fn):
    sync()
    torch.cuda.reset_peak_memory_stats()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    stamp = perf_counter()
    start.record()
    fn()
    end.record()
    end.synchronize()
    return dict(wall_ms=(perf_counter()-stamp)*1000, cuda_event_ms=start.elapsed_time(end),
                peak_allocated_bytes=torch.cuda.max_memory_allocated())

@contextmanager
def stages(rows):
    targets = {'roi_sample', 'cost_volume', 'local_aggregation', 'disparity_regression'}
    @contextmanager
    def probe(name):
        key = name.removeprefix('stereo.')
        if key not in targets:
            yield
            return
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        yield
        end.record()
        rows[key] = (start, end)
    with patch.object(ld, 'record_function', probe), patch.object(model_module, 'record_function', probe):
        yield

def test_volume():
    checks = []
    torch.manual_seed(17)
    for n in (1, 2, 6, 9):
        for w in (5, 56):
            l, r = (torch.randn(n,96,7,w,device='cuda') for _ in range(2))
            mask = torch.rand(n,1,7,w,device='cuda') > .2
            mask[:,:,0] = False
            args = l,r,head.idisp.search,mask,head.idisp.temperature
            a,b = loop(*args),bmm(*args)
            assert torch.equal(a.valid_bins,b.valid_bins) and torch.equal(a.spatial_valid,b.spatial_valid)
            torch.testing.assert_close(a.cost,b.cost,atol=1e-5,rtol=1e-5)
            checks.append(dict(N=n,width=w,max_abs_error=(a.cost-b.cost).abs().max().item()))
    return checks

@torch.inference_mode()
def run():
    result = dict(commit='308064ebe3d252f338b1b007ac13f6f05cee65fc', weights_sha256=digest(weights),
                  prototype_sha256=digest(Path(spec.origin)), torch=torch.__version__, cuda=torch.version.cuda,
                  device=torch.cuda.get_device_name(), tf32=False, power_mode='25W', warmup=warmup, repeats=repeats,
                  volume_checks=test_volume(), frames={})
    for frame in ('000004_10','000005_10','000058_10','000000_10'):
        training=frame != '000000_10'
        evidence=historical / frame if training else Path('outputs/v2-050/nms_kitti2015_000000_10_range_048_warm')
        meta=json.loads((evidence/'timing.json').read_text())
        images=([Path('/tmp/yolo26-mixed-frames') / f'{frame}_image_{i}.png' for i in (2,3)] if training else
                [Path('text data') / side / f'kitti2015_{frame}.png' for side in ('left','right')])
        hashes=list(map(digest,images))
        assert hashes == ([meta['left_sha256'],meta['right_sha256']] if training else [v['official_testing'] for v in json.loads((evidence.parent/'nms_kitti2015_000000_10_gt'/'test_frame_verification.json').read_text()).values()]), (frame,hashes)
        l,r,geometry=load_rectified_pair(*images,bundle['normalization'],0)
        l,r=l.cuda(),r.cuda()
        route, captured={}, {}
        def capture(module,args,output):
            captured['pyramid']=output
        def count(module,args):
            captured['calls']=captured.get('calls',0)+1
        with model.encoder.register_forward_hook(capture), head.idisp.register_forward_pre_hook(count):
            reference=model.forward_prepared(l,r,geometry,route=route,
                                             calibration=(meta['bf_px_m'],meta['cx_delta_px']))
        pairs=route['idisp']
        ids=pairs.indices.tolist()
        order=sorted(range(len(ids)),key=lambda i:(-pairs.scores[i].item(),ids[i]))
        assert len(ids)==len(set(ids))
        loads=[n for n in (0,2,6,9) if n<=len(ids)]
        ids6=[ids[i] for i in order[:6]] if 6 in loads else []
        ids9=[ids[i] for i in order[:9]] if 9 in loads else []
        assert set(ids6).issubset(ids9)
        if not ids:
            assert captured.get('calls',0)==0
        boxes=pairs.boxes6[order[:max(loads)]]
        pyr=captured['pyramid']
        def sample(n):
            pyramid=SimpleNamespace(**{k:getattr(pyr,k).expand(n,-1,-1,-1) for k in
                                      ('p2_left','p2_right','p3_left','p4_left')})
            return head.forward_disparity(pyramid,boxes[:n],*geometry.original_hw)
        cached={}
        if len(boxes):
            with patch.object(head.idisp,'forward',side_effect=head.idisp.forward) as tracker:
                sample(len(boxes))
            a,b,att,crop,valid=tracker.call_args.args
            for n in loads[1:]:
                idx=torch.arange(n,device=a.device)
                cached[n]=(a[:n],b[:n],AttentionPyramid(att.p2[:n],att.p3[:n],att.p4[:n]),
                           subset_crop(crop,idx),valid[:n])
        frame_data=dict(split='training' if training else 'testing', sha256=hashes,
                        calibration=[meta['bf_px_m'],meta['cx_delta_px']], available=len(ids),
                        actual_idisp_calls=captured.get('calls',0), pair_ids_6=ids6,pair_ids_9=ids9,
                        boxes6=boxes.tolist(),missing_loads=sorted(set((0,2,6,9))-set(loads)),
                        roi_shapes={n:list(args[0].shape) for n,args in cached.items()}, backends={})
        calibration=(meta['bf_px_m'],meta['cx_delta_px'])
        dense=model.detector(pyr)
        result['frames'][frame]=frame_data
        for name,backend in [('loop',loop),('bmm_prototype',bmm)]:
            rows, stage_rows=[], []
            with patch.object(ld,'build_local_cost_volume',backend):
                candidate=model.decode_and_refine(l,pyr,dense,input_geometry=geometry,
                                                  calibration=calibration,route=(candidate_route:={}))
                assert torch.equal(pairs.indices,candidate_route['idisp'].indices)
                assert torch.equal(reference.pair_ids,candidate.pair_ids)
                assert torch.equal(reference.foreground_validity,candidate.foreground_validity)
                valid_output=reference.validity
                assert torch.equal(valid_output,candidate.validity)
                frame_data['backends'][name]=dict(rows=rows, stage_rows=stage_rows,
                    final_ids=candidate.pair_ids.tolist(),
                    max_disparity_difference_px=(reference.disparity[valid_output]-candidate.disparity[valid_output]).abs().max().item() if valid_output.any() else None,
                    max_distance_difference_m=(reference.near10_distance_m-candidate.near10_distance_m).abs().max().item() if len(candidate.pair_ids) else None)
                for mode in ('roi_to_idisp','cached_roi'):
                    def call(n):
                        return sample(n) if mode=='roi_to_idisp' else head.idisp(*cached[n])
                    for n in loads[1:]:
                        for _ in range(warmup):
                            call(n)
                        sync()
                    for i in range(repeats):
                        other=[n for n in loads if n not in (6,9)]
                        paired=[n for n in (6,9) if n in loads]
                        order_run=other+(paired[::-1] if i%2 else paired)
                        one={}
                        for n in order_run:
                            one[n]=timed(lambda:call(n)) if n else dict(wall_ms=0.,cuda_event_ms=0.,peak_allocated_bytes=None)
                        delta={k:one[9][k]-one[6][k] for k in ('wall_ms','cuda_event_ms')} if 6 in one and 9 in one else None
                        rows.extend(dict(frame_id=frame,round_id=i,N=n,pair_ids_6=ids6,pair_ids_9=ids9,
                                         order=order_run,paired_delta_ms=delta,mode=mode,**one[n]) for n in order_run)
                for n in loads[1:]:
                    for i in range(repeats):
                        events={}
                        with stages(events):
                            sample(n)
                        sync()
                        stage_rows.append(dict(N=n,round_id=i, **{k:a.elapsed_time(b) for k,(a,b) in events.items()}))
            summary={}
            for mode in ('roi_to_idisp','cached_roi'):
                summary[mode]={n:{k:stats([row[k] for row in rows if row['N']==n and row['mode']==mode])
                                  for k in ('wall_ms','cuda_event_ms')} for n in loads}
                if 9 in loads:
                    summary[mode]['delta_9_minus_6']={k:stats([row['paired_delta_ms'][k] for row in rows if row['N']==9 and row['mode']==mode])
                                                        for k in ('wall_ms','cuda_event_ms')}
            if stage_rows:
                summary['stages_cuda_ms']={n:{key:stats([row[key] for row in stage_rows if row['N']==n]) for key in
                    ('roi_sample','cost_volume','local_aggregation','disparity_regression')} for n in loads[1:]}
            frame_data['backends'][name]['summary']=summary
            print(frame,name,len(ids),summary['cached_roi'].get('delta_9_minus_6'),flush=True)
        (out/'fixed_pairs.json').write_text(json.dumps(result,indent=2,allow_nan=False))
run()
