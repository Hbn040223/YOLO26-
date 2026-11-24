"""Opt-in Car RoI budget and foreground duplicate selection before local disparity."""
from dataclasses import dataclass, fields, replace
from types import SimpleNamespace
import math

import numpy as np
import torch

from .detection import PairDetections, _iou, anchorfree_candidate_indices, anchorfree_prediction_scores
from .geometry import decode_anchorfree_boxes6, pair_boxes_to_lr
from .idisp_ab import project_disparity


@dataclass(frozen=True)
class RoiPairSelection:
    car_class: int
    bf_px_m: float
    cx_delta_px: float
    max_distance_m: float = 15.
    reserve_distance_m: float = 12.
    reserve_score: float = .05

    def __post_init__(self):
        if self.car_class < 0 or not math.isfinite(self.bf_px_m) or self.bf_px_m <= 0:
            raise ValueError("near selection requires a Car class and positive finite fB")
        if not math.isfinite(self.cx_delta_px):
            raise ValueError("principal-point difference must be finite")
        if not 0 < self.reserve_distance_m <= self.max_distance_m < float("inf"):
            raise ValueError("reserve range must fit within a positive finite near range")
        if not 0 < self.reserve_score < .40:
            raise ValueError("reserve score must be positive and below .40")


@dataclass(frozen=True)
class LeftOnlyDetections:
    """Left-view candidates without usable stereo geometry."""

    boxes: torch.Tensor
    scores: torch.Tensor
    ids: torch.Tensor
    classes: torch.Tensor
    distance_m: torch.Tensor


@dataclass(frozen=True)
class Near10Policy:
    mode: str
    car_class: int
    prefilter_m: float = 15.
    mask_iou: float = .60

    def __post_init__(self):
        if self.mode not in ("baseline", "range", "range-dedup"):
            raise ValueError("invalid 10 m policy")
        if not 10 <= self.prefilter_m < math.inf or not 0 < self.mask_iou <= 1:
            raise ValueError("invalid 10 m prefilter distance or Mask IoU")


@dataclass(frozen=True)
class Near10Geometry:
    disparity: torch.Tensor
    admitted: torch.Tensor
    far: torch.Tensor
    left_only: torch.Tensor
    invalid: torch.Tensor


def near10_geometry(pairs, image_hw, calibration, policy):
    bf, cx_delta = calibration
    if not math.isfinite(bf) or bf <= 0 or not math.isfinite(cx_delta):
        raise ValueError("10 m policy requires positive fB and finite cx difference")
    left, right = pair_boxes_to_lr(pairs.boxes6.float())
    height, width = image_hw
    bounds = left.new_tensor((width, height))

    def visible(box):
        return (torch.isfinite(box).all(1) & (box[:, 2:] > box[:, :2]).all(1)
                & (box[:, :2] < bounds).all(1) & (box[:, 2:] > 0).all(1))

    valid_left = visible(left)
    disparity = .5 * (left[:, 0] + left[:, 2] - right[:, 0] - right[:, 2]) - cx_delta
    stereo = valid_left & visible(right) & torch.isfinite(disparity) & (disparity > 0)
    # Border contact is conservative: the predicted center can shift on truncation.
    interior = ((left[:, :2] > 0) & (right[:, :2] > 0)
                & (left[:, 2:] < bounds) & (right[:, 2:] < bounds)).all(1)
    far = stereo & interior & (disparity < bf / policy.prefilter_m)
    return Near10Geometry(disparity, stereo & ~far, far, valid_left & ~stereo, ~valid_left)


def near10_conflicts(pairs, disparity, policy):
    left, right = pair_boxes_to_lr(pairs.boxes6.float())

    def overlaps(box):
        lo = torch.maximum(box[:, None, :2], box[None, :, :2])
        hi = torch.minimum(box[:, None, 2:], box[None, :, 2:])
        inter = (hi - lo).clamp_min(0).prod(2)
        area = (box[:, 2:] - box[:, :2]).prod(1)
        return inter / (area[:, None] + area - inter), inter / area

    left_iou, left_cover = overlaps(left)
    right_iou, right_cover = overlaps(right)
    contained = (left_cover >= .90) & (right_cover >= .90)
    # Row candidate may suppress a lower-scored column candidate.
    priority = ((pairs.scores[:, None] > pairs.scores)
                | ((pairs.scores[:, None] == pairs.scores) & (pairs.indices[:, None] < pairs.indices)))
    geometry = ((left_iou >= .35) & (right_iou >= .35)) | contained
    tolerance = (.10 * torch.minimum(disparity[:, None], disparity)).clamp_min(2.)
    close = (disparity[:, None] - disparity).abs() <= tolerance
    car = pairs.classes == policy.car_class
    return priority & geometry & close & car[:, None] & car


def near10_keep(pairs, conflicts, masks, policy):
    flat = masks.flatten(1).float()
    area = flat.sum(1)
    intersection = flat @ flat.T
    iou = intersection / (area[:, None] + area - intersection).clamp_min(1)
    conflicts = (conflicts & (iou >= policy.mask_iou)).cpu().numpy()
    order = np.lexsort((pairs.indices.cpu().numpy(), -pairs.scores.detach().cpu().numpy()))
    blocked = np.zeros(len(order), dtype=bool)
    for index in order:
        if not blocked[index]:
            blocked |= conflicts[index]
    return pairs.indices.new_tensor(np.flatnonzero(~blocked))


def near10_measure(disparity, valid, calibration):
    bf, cx_delta = calibration
    if len(disparity) == 0:
        return disparity.new_empty((0,), dtype=torch.float32)
    corrected = disparity.float() - cx_delta
    valid = valid & torch.isfinite(corrected) & (corrected > 0)
    median = corrected.masked_fill(~valid, float("nan")).flatten(1).nanmedian(dim=1).values
    return bf / median


def subset_pairs(pairs, selected):
    return PairDetections(pairs.boxes6[selected],pairs.scores[selected],pairs.indices[selected],
                          pairs.classes[selected])


def subset_crop(crop, indices):
    return replace(crop, **{f.name:getattr(crop,f.name)[indices] for f in fields(crop)
                           if isinstance(getattr(crop,f.name),torch.Tensor)})


def _depth(boxes, config, image_hw):
    disparity=.5*(boxes[:,0]+boxes[:,2]-boxes[:,4]-boxes[:,5])-config.cx_delta_px
    height, width = image_hw
    valid=(torch.isfinite(boxes).all(1)&torch.isfinite(disparity)&(disparity>0)&(boxes[:,2]>boxes[:,0])
           &(boxes[:,3]>boxes[:,1])&(boxes[:,5]>boxes[:,4])
           &(boxes[:,5]>0)&(boxes[:,4]<width)&(boxes[:,3]>0)&(boxes[:,1]<height))
    depth=torch.where(valid,config.bf_px_m/disparity,torch.full_like(disparity,float("inf")))
    return disparity,depth


def _greedy(conflicts, priority, scores):
    order=np.lexsort((np.arange(len(scores)),-scores,-priority))
    blocked=np.zeros(len(scores),dtype=bool)
    chosen=[]
    for i in order:
        if not blocked[i]:
            chosen.append(i)
            blocked|=conflicts[i]
    return chosen


def _select_left_only(boxes, depth, scores, classes, supported_boxes, image_hw, car_class):
    _, chosen = anchorfree_candidate_indices(scores, .40, 500)
    ids = chosen[(classes[chosen] == car_class) & ~torch.isfinite(depth[chosen])]
    raw_left = boxes[ids, :4]
    height, width = image_hw
    limit = raw_left.new_tensor((width, height, width, height))
    clipped = raw_left.clamp_min(0).minimum(limit)
    usable = torch.isfinite(raw_left).all(1) & (clipped[:, 2:] > clipped[:, :2]).all(1)
    ids, clipped = ids[usable], clipped[usable]
    if len(ids) and len(supported_boxes):
        covered = (_iou(clipped, supported_boxes.clamp_min(0).minimum(limit)) >= .50).any(1)
        ids, clipped = ids[~covered], clipped[~covered]
    if len(ids) > 1:
        overlap = _iou(clipped, clipped).cpu().numpy() >= .50
        order = _greedy(overlap, scores[ids].cpu().numpy(), scores[ids].cpu().numpy())
        ids, clipped = ids[order], clipped[order]
    return LeftOnlyDetections(clipped, scores[ids], ids, classes[ids],
                              scores.new_full((len(ids),), float("nan")))


def select_roi_pairs(pairs, predictions, locations, image_hw, config, right_offset_mode):
    """Return near stereo pairs and separate left observations with unknown distance."""
    d,z=_depth(pairs.boxes6,config,image_hw)
    supported_boxes = pairs.boxes6[(pairs.classes == config.car_class) & torch.isfinite(z), :4]
    admitted=(pairs.classes==config.car_class)&(z<=config.max_distance_m)
    pairs=subset_pairs(pairs,admitted)
    d=d[admitted]
    if len(d)>1:
        left,right=pair_boxes_to_lr(pairs.boxes6)
        joint=torch.minimum(_iou(left,left),_iou(right,right))
        conflict=(joint>.40)&((d[:,None]-d).abs()<=torch.maximum(
            torch.full_like(joint,2.),.10*torch.minimum(d[:,None],d)))
        chosen=_greedy(conflict.cpu().numpy(),np.arange(len(d),0,-1),pairs.scores.cpu().numpy())
        pairs=subset_pairs(pairs,torch.tensor(chosen,device=d.device,dtype=torch.long))
        d=d[chosen]

    scores,classes=anchorfree_prediction_scores(predictions)
    boxes=decode_anchorfree_boxes6(predictions["box6"].reshape(-1,6),locations,
                                  right_offset_mode=right_offset_mode)
    disp,depth=_depth(boxes,config,image_hw)
    left,right=pair_boxes_to_lr(boxes)
    height,width=image_hw
    limit=boxes.new_tensor([width-1,height-1])
    def area(view):
        return (view[:,2:].clamp_min(0).minimum(limit)-view[:,:2].clamp_min(0).minimum(limit)).clamp_min(0).prod(1)
    visible_area=area(left)
    edge=((left[:,0]<=0)&(right[:,0]<=0))|((left[:,2]>=width-1)&(right[:,2]>=width-1))
    eligible=((scores>=config.reserve_score)&(scores<.40)&(classes==config.car_class)&edge
              &(depth<=config.reserve_distance_m)&(visible_area>=1)&(area(right)>=1))
    if len(pairs.indices):
        bl,br=pair_boxes_to_lr(pairs.boxes6)
        joint=torch.minimum(_iou(left,bl),_iou(right,br))
        same_depth=(disp[:,None]-d).abs()<=torch.maximum(
            torch.full_like(joint,2.),.10*torch.minimum(disp[:,None],d))
        eligible&=~((joint>.35)&same_depth).any(1)
    if not len(scores):
        return pairs, _select_left_only(boxes, depth, scores, classes, supported_boxes, image_hw, config.car_class)
    area_rank=visible_area.masked_fill(~eligible,-1)
    biggest=eligible&(area_rank==area_rank.max())
    winner=scores.masked_fill(~biggest,-1).argmax().reshape(1)
    accepted=biggest[winner]
    if len(pairs.indices):
        accepted&=~(joint[winner]>=.65).any(1)
    index=torch.where(accepted,winner,-1).item()
    if index >= 0:
        ix = slice(index, index + 1)
        pairs = PairDetections(
            torch.cat((pairs.boxes6, boxes[ix])), torch.cat((pairs.scores, scores[ix])),
            torch.cat((pairs.indices, pairs.indices.new_tensor([index]))), torch.cat((pairs.classes, classes[ix])),
        )
        supported_boxes = torch.cat((supported_boxes, boxes[ix, :4]))
    left_only = _select_left_only(boxes, depth, scores, classes, supported_boxes, image_hw, config.car_class)
    return pairs, left_only


def project_foreground(logits, validity, crop, image_hw, mask_threshold):
    probability=logits.sigmoid()[:,0]
    height,width=probability.shape[-2:]
    quarter=replace(crop,c_l=crop.c_l/4,y0=crop.y0/4,c=crop.c/4,h_crop=crop.h_crop/4,
                    h_roi=torch.full_like(crop.h_roi,height),w_roi=torch.full_like(crop.w_roi,width))
    grid=tuple((n+3)//4 for n in image_hw)
    p,valid=project_disparity(SimpleNamespace(absolute=probability,valid=validity[:,0]),quarter,grid)
    return valid&(p>mask_threshold)


def keep_foreground_pairs(pairs, masks):
    """Suppress direct same-class overlaps, ordered by area times square-root score."""
    count=len(pairs.indices)
    if not count:
        return pairs.indices.new_empty((0,))
    flat=masks.flatten(1).float()
    area=flat.sum(1)
    intersection=flat@flat.T
    overlap=intersection/(area[:,None]+area-intersection).clamp_min(1)
    conflict=(_iou(pairs.boxes6[:,:4],pairs.boxes6[:,:4])>=.35)&(overlap>=.45)
    conflict&=pairs.classes[:,None]==pairs.classes
    priority=area*pairs.scores.clamp_min(0).sqrt()
    selected=_greedy(conflict.cpu().numpy(),priority.cpu().numpy(),pairs.scores.cpu().numpy())
    return pairs.indices.new_tensor(sorted(selected))
