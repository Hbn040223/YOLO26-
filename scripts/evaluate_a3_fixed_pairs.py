"""A3 fixed-pair GT-RoI versus Pred-RoI local iDisp evaluation."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))

def quantiles(values):
    if not values: return {'count':0,'p10':None,'p50':None,'p90':None}
    x=torch.tensor(values,dtype=torch.float64); return {'count':len(values),'p10':float(torch.quantile(x,.1)),'p50':float(torch.quantile(x,.5)),'p90':float(torch.quantile(x,.9))}

def main():
 p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--pairs-cache',required=True); p.add_argument('--output',required=True); p.add_argument('--margins',default='16,32'); p.add_argument('--device',default='cuda:0'); a=p.parse_args()
 from lightbackbone_stereo_yolo.data.frame_dataset import KittiFrameDataset, build_frame_records
 from lightbackbone_stereo_yolo.lightstereo_encoder import LightStereoFeatureEncoder
 from lightbackbone_stereo_yolo.stereo_yolo26 import StereoYOLO26PanHead
 from lightbackbone_stereo_yolo.training import StereoYoloTrainingModel
 from lightbackbone_stereo_yolo.model import StereoLocalInstanceHead
 from lightbackbone_stereo_yolo.training_metrics import evaluate_fixed_roi_pairs
 device=torch.device(a.device); payload=torch.load(a.checkpoint,map_location='cpu',weights_only=False); cfg=payload['config']
 cache=json.load(open(a.pairs_cache)); cached={x['frame_id']:x for x in cache['frames']}; margins=[float(x) for x in a.margins.split(',')]
 dataset=KittiFrameDataset(build_frame_records(cfg['dataset_root'],cfg['val_json']),payload['class_mapping'],unknown_class='skip')
 encoder=LightStereoFeatureEncoder.from_checkpoint(cfg['lightstereo_source'],cfg['checkpoint'],adapter_channels=cfg['adapter_channels'])
 model=StereoYoloTrainingModel(encoder,StereoYOLO26PanHead(cfg['adapter_channels'],len(payload['class_mapping']),one_to_one=cfg.get('one_to_one',False)),StereoLocalInstanceHead(cfg['adapter_channels'],payload['dmax_roi'],roi_sampling=cfg.get('roi_sampling','native_pad'))).to(device); model.load_state_dict(payload['model']); model.eval()
 result={'pairs_cache':a.pairs_cache,'cached_matches':cache['matched_count'],'roi_sampling':model.instance_head.roi_sampling,'margins':{}}
 with torch.no_grad():
  for margin in margins:
   total={'common':0,'gt_epe_sum':0.,'pred_epe_sum':0.,'gt_d1_hits':0.,'pred_d1_hits':0.,'empty_instances':0,'coverage':[],'q_gt':[],'q_pred':[]}
   for sample in dataset:
    entry=cached.get(sample['frame_id'])
    if entry is None: continue
    left=sample['left'].unsqueeze(0).to(device); right=sample['right'].unsqueeze(0).to(device)
    pyramid=model.encoder(left,right)
    gt_boxes=sample['boxes6'].to(device)[torch.tensor(entry['target_indices'],device=device)]
    pred_boxes=torch.tensor(entry['boxes6'],device=device,dtype=gt_boxes.dtype)
    indices=torch.tensor(entry['target_indices'],device=device,dtype=torch.long)
    metrics,gt_out,pred_out=evaluate_fixed_roi_pairs(pyramid,model.instance_head,left,sample,gt_boxes,pred_boxes,indices,payload['dmax_roi'],margin)
    n=metrics['common_pixel_count']; total['common']+=n; total['gt_epe_sum']+=metrics['gt_epe']*n; total['pred_epe_sum']+=metrics['pred_epe']*n; total['gt_d1_hits']+=metrics['gt_d1']*n; total['pred_d1_hits']+=metrics['pred_d1']*n; total['empty_instances']+=metrics['empty_instances']; total['coverage']+=metrics['coverage_per_instance']; total['q_gt']+=(gt_out.crop.c/gt_out.crop.w_roi.float()).cpu().tolist(); total['q_pred']+=(pred_out.crop.c/pred_out.crop.w_roi.float()).cpu().tolist()
   c=max(total['common'],1); ge=total['gt_epe_sum']/c; pe=total['pred_epe_sum']/c
   result['margins'][str(int(margin))]={'common_pixel_count':total['common'],'empty_instances':total['empty_instances'],'gt_epe':ge,'pred_epe':pe,'delta_epe':pe-ge,'relative_epe_increase':(pe-ge)/max(ge,1e-8),'gt_d1':total['gt_d1_hits']/c,'pred_d1':total['pred_d1_hits']/c,'delta_d1':(total['pred_d1_hits']-total['gt_d1_hits'])/c,'coverage':quantiles(total['coverage']),'q_gt':quantiles(total['q_gt']),'q_pred':quantiles(total['q_pred'])}
 Path(a.output).write_text(json.dumps(result,indent=2)); print(json.dumps(result['margins'],indent=2))
if __name__=='__main__': main()
