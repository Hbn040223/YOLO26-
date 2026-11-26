import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch
import numpy as np
import torch
from infer import load_published_model, load_rectified_pair
from lightbackbone_stereo_yolo.local_aggregation import AttentionPyramid
from lightbackbone_stereo_yolo.roi_selection import subset_crop

out = Path('outputs/v2-050/idisp-fpn/jetson/20261126-aggregation-graph')
out.mkdir(parents=True, exist_ok=True)
root = Path('outputs/v2-050')
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
torch.backends.cudnn.deterministic=True
model,bundle=load_published_model(root/'formal/best_r2.pt',Path('/tmp/yolo26-openstereo'),'cuda:0',
    score=.48,apply_paired_nms=True,near10_mode='range')
head=model.instance_head.idisp
frame='000004_10'
meta=json.loads((root/'kitti2015_mixed_range_048'/frame/'timing.json').read_text())
l,r,g=load_rectified_pair(*(Path('/tmp/yolo26-mixed-frames')/f'{frame}_image_{i}.png' for i in (2,3)),
    bundle['normalization'],0)
calibration=meta['bf_px_m'],meta['cx_delta_px']
sync=torch.cuda.synchronize
original=head.aggregation.forward
data={}
with torch.inference_mode():
    route={}
    with patch.object(head,'forward',side_effect=head.forward) as tracker:
        model.forward_prepared(l.cuda(),r.cuda(),g,route=route,calibration=calibration)
    a,b,attention,crop,mask=tracker.call_args.args
    pairs=route['idisp']
    indices=sorted(range(len(pairs.indices)),key=lambda i:(-pairs.scores[i].item(),pairs.indices[i].item()))
    for n in (2,6,9):
        graph=None
        try:
            idx=torch.tensor(indices[:n],device='cuda')
            args=(a[idx],b[idx],AttentionPyramid(a[idx],attention.p3[idx],attention.p4[idx]),subset_crop(crop,idx),mask[idx])
            with patch.object(head.aggregation,'forward',side_effect=original) as track_agg:
                reference=head(*args)
            cost, att, validity=track_agg.call_args.args
            incoming=(cost,att.p2,att.p3,att.p4,validity)
            static=tuple(t.clone() for t in incoming)
            def compute(tensors):
                return original(tensors[0],AttentionPyramid(*tensors[1:4]),tensors[4])
            stream=torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(20):
                    compute(static)
            torch.cuda.current_stream().wait_stream(stream)
            sync()
            torch.cuda.reset_peak_memory_stats()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_output=compute(static)
            peak=torch.cuda.max_memory_allocated()
            def replay(tensors):
                for dst,src in zip(static,tensors):
                    dst.copy_(src)
                graph.replay()
                return static_output
            expected=compute(incoming)
            torch.testing.assert_close(replay(incoming),expected,rtol=1e-5,atol=1e-5)
            changed=(cost*.9,*incoming[1:])
            torch.testing.assert_close(replay(changed),compute(changed),rtol=1e-5,atol=1e-5)
            with patch.object(head.aggregation,'forward',side_effect=lambda c,a,v: replay((c,a.p2,a.p3,a.p4,v))):
                candidate=head(*args)
            assert torch.equal(reference.valid,candidate.valid)
            error=(reference.absolute-candidate.absolute).abs().max().item()
            torch.testing.assert_close(reference.absolute,candidate.absolute,rtol=1e-5,atol=1e-5)
            rows={'ordinary':[],'graph_with_input_copies':[]}
            modes={'ordinary':compute,'graph_with_input_copies':replay}
            for i in range(80):
                for key in tuple(modes)[::(-1 if i%2 else 1)]:
                    start=perf_counter()
                    modes[key](incoming)
                    sync()
                    if i>=20:
                        rows[key].append((perf_counter()-start)*1000)
            data[n]=dict(rows=rows,wall_p50_p95_ms={key:np.percentile(values,[50,95]).tolist() for key,values in rows.items()},
                         graph_peak_allocated_bytes=peak,max_abs_disparity_error_px=error)
        except (RuntimeError,AssertionError) as exc:
            data[n]=dict(error=str(exc))
        print(n,{k:v for k,v in data[n].items() if k!='rows'},flush=True)
        del graph
(out/'aggregation-graph.json').write_text(json.dumps(dict(frame_id=frame,commit='308064e',precision='fp32',by_load=data),indent=2))
