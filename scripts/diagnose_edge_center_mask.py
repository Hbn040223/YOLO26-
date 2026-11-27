"""Server A/B for left-edge geometry and single-instance Mask ownership."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from edge_run_evidence import digest
from evaluate_fixed_pair_features import box_iou, image_array
from evaluate_paired_nms import write_json
from infer import load_published_model, load_rectified_pair
from lightbackbone_stereo_yolo.idisp_ab import project_disparity, summarize_error, error_totals
from lightbackbone_stereo_yolo.mask import foreground_disparity_validity, mask_logit_cutoff
from lightbackbone_stereo_yolo.roi_selection import near10_measure, subset_crop


def features(pyramid, n):
    return SimpleNamespace(**{k:getattr(pyramid,k).expand(n,-1,-1,-1)
        for k in ("p2_left","p2_right","p3_left","p4_left")})


def edge_run(head,pyramid,boxes,hw,truth,mask,calib):
    results={}
    yy,xx=torch.where(mask & (truth>0))
    target=truth[yy,xx]
    physical=(xx-target>=0)&(xx-target<hw[1])
    for label,enabled in (("baseline",False),("recenter",True)):
        head.recenter_left_edge=enabled
        out=head(features(pyramid,1),boxes,*hw)
        crop=out.crop
        c_l,c_r,scale=[float(v[0]) for v in (crop.c_l,crop.c_r,crop.c/crop.w_roi)]
        d0=c_l-c_r
        offset=((target-d0)/scale).round().long()
        idx=((xx-c_l)/scale).floor().long()-offset
        in_range=((target-d0).abs()<=16*scale)
        inside_right=(idx>=0)&(idx<56)
        right_x=c_r+(idx+.5)*scale
        valid_center=(right_x>=0)&(right_x<hw[1])
        disp,valid=project_disparity(SimpleNamespace(absolute=out.disparity,valid=out.validity),crop,hw)
        operational=foreground_disparity_validity(out.mask_logits,out.mask_validity,out.disparity,out.validity,
              out.validity.new_ones(1),mask_threshold=head.measurement_threshold)
        distance=near10_measure(out.disparity,operational,calib)[0]
        results[label]=dict(c_l=c_l,c_r=c_r,d0=d0,overflow=int((~in_range).sum()),
            target_usable=int((physical&in_range&inside_right&valid_center).sum()),
            right_padding_columns=int(((c_r+(torch.arange(56)+.5)*scale)<0).sum()),
            predicted_valid_on_truth=int(valid[0,yy,xx].sum()),mask_distance_m=float(distance) if distance.isfinite() else None)
        results[label]["_projection"]=(disp[0],valid[0])
    shared=mask&(truth>0)&results["baseline"]["_projection"][1]&results["recenter"]["_projection"][1]
    for row in results.values():
        disp,_=row.pop("_projection")
        row["shared_epe_d1"]=summarize_error(error_totals(disp,truth,shared))
    head.recenter_left_edge=False
    return dict(truth_pixels=len(target),physical_right_outside=int((~physical).sum()),arms=results)


def mask_run(model,pyramid,pairs,hw,labels,calib):
    head=model.instance_head
    k=int(torch.where(pairs.indices==9486)[0].item())
    out=head(features(pyramid,len(pairs.indices)),pairs.boxes6,*hw)
    adjusted=out.mask_logits-mask_logit_cutoff(model.mask_threshold)
    prob=F.interpolate(adjusted.sigmoid(),size=(56,56),mode="bilinear",align_corners=False)[:,0]
    mask=foreground_disparity_validity(out.mask_logits,out.mask_validity,out.disparity,out.validity,
          pairs.classes==model.foreground_class_indices[0],mask_threshold=model.mask_threshold)
    crop=out.crop
    cols=(crop.c_l[k]+(torch.arange(56,device=prob.device)+.5)*crop.c[k]/56).floor().long().clamp(0,hw[1]-1)
    rows=(crop.y0[k]+(torch.arange(56,device=prob.device)+.5)*crop.h_crop[k]/56).floor().long().clamp(0,hw[0]-1)
    ids=labels[rows[:,None],cols]
    target=ids==6669
    target_fg=mask[k]
    projected,valid=project_disparity(SimpleNamespace(absolute=prob,valid=out.validity),crop,hw)
    samples=projected[:,rows[:,None],cols]
    supports=valid[:,rows[:,None],cols]
    left,right=pairs.boxes6[:,:4],pairs.boxes6[:,[4,1,5,3]]
    duplicate=(box_iou(left,left[k].tolist())>=.5)&(box_iou(right,right[k].tolist())>=.5)
    eligible=(pairs.classes==pairs.classes[k])&~duplicate
    rivals=torch.where(eligible)[0]
    scores=samples[rivals].masked_fill(~supports[rivals],0)
    strongest,winner=(scores.max(0) if len(rivals) else
                     (torch.zeros_like(prob[k]),torch.zeros_like(ids)))
    def measure(m):
        d=near10_measure(out.disparity[k:k+1],m[None],calib)[0]
        return dict(pixels=int(m.sum()),target=int((m&target).sum()),
                    other_car=int((m&~target&(ids//256==26)).sum()),
                    distance_m=float(d) if d.isfinite() else None)
    trials={"baseline":measure(target_fg),
            "oracle_target_intersection":measure(target_fg&target),
            "confidence_065":measure(target_fg&(prob[k]>.65))}
    for margin in (0.,.10):
        # Only yield to candidates with confident mask support and stronger pixel ownership.
        remove=(strongest>.5)&(strongest>prob[k]+margin)
        trials["rival_margin_"+str(margin)]=measure(target_fg&~remove)
    owners={int(pairs.indices[r]):int((target_fg&(strongest>.5)&(winner==j)).sum())
            for j,r in enumerate(rivals)}
    return dict(trials=trials,rival_claims=owners)

@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("manifest","weights","openstereo","output"):
        parser.add_argument(name,type=Path)
    a=parser.parse_args()
    device=torch.device("cuda:0")
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    model,bundle=load_published_model(a.weights,a.openstereo,device,score=.48,apply_paired_nms=True,
                                     near10_mode="range",idisp_cost_backend="bmm")
    head=model.instance_head
    head.idisp.force_fp32_aggregation=True
    head.measurement_threshold=model.mask_threshold
    assert head.idisp.search.dmax_roi==16 and head.idisp.aggregation.stem[0].out_channels==32
    report={}
    for f in json.loads(a.manifest.read_text()):
        frame=f["frame_id"]
        if frame not in ("000004_10","000005_10"):
            continue
        l,r,g=load_rectified_pair(Path(f["left"]),Path(f["right"]),bundle["normalization"])
        l,r=l.to(device),r.to(device)
        hw=g.original_hw
        labels=image_array(f["instance_map"],np.int64).to(device)
        truth=image_array(f["disparity"],np.float32).to(device)/256
        calib=f["bf_px_m"],f["cx_delta_px"]
        item=next(t for t in f["instances"] if t["id"]==6674 and t["right_box_verified"])
        lb,rb=item["left_box"],item["right_box"]
        box=l.new_tensor([lb+[rb[0],rb[2]]])
        pyr=model.encoder(l,r)
        report[frame]={"edge":edge_run(head,pyr,box,hw,truth,labels==6674,calib)}
        route={}
        model.decode_and_refine(l,pyr,model.detector(pyr),input_geometry=g,route=route,calibration=calib)
        pairs=route["idisp"]
        if len(pairs.indices):
            iou1=box_iou(pairs.boxes6[:,:4],lb)
            iou2=box_iou(pairs.boxes6[:,[4,1,5,3]],rb)
            k=int(torch.minimum(iou1,iou2).argmax())
            report[frame]["edge"]["best_automatic"]={"pair_id":int(pairs.indices[k]),
               "left_iou":float(iou1[k]),"right_iou":float(iou2[k])}
        if frame=="000004_10":
            report[frame]["9486"]=mask_run(model,pyr,pairs,hw,labels,calib)
        print(frame,flush=True)
    write_json(a.output,{"weights_sha256":digest(a.weights),"manifest_sha256":digest(a.manifest),
        "settings":{"dmax_roi":16,"aggregation_width":32,"cost":"bmm","aggregation":"fp32"},"frames":report})

if __name__=="__main__":
    main()
