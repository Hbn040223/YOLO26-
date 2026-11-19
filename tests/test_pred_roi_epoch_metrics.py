"""Train-loop contract for prediction-driven RoI metrics."""
from __future__ import annotations
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("train_script",ROOT/"scripts"/"train.py")
train_script=importlib.util.module_from_spec(spec); sys.modules["train_script"]=train_script; spec.loader.exec_module(train_script)


class PredRoiEpochMetricTest(unittest.TestCase):
    def test_validation_epoch_accumulates_pred_roi_metrics(self):
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder=nn.Identity(); self.detector=nn.Identity(); self.instance_head=nn.Identity()
        model=Model()
        batch={"left":torch.zeros(1,3,4,4),"right":torch.zeros(1,3,4,4),"targets":[{"boxes6":torch.empty(0,6)}]}
        args=SimpleNamespace(pred_roi_metrics=True,detector_variant="stereo_yolo26_pan",dmax_roi=2,roi_margin=16.,score_threshold=.05,pre_nms_topk=500,log_interval=0,device="cpu")
        output={"loss_obj":torch.tensor(1.),"loss_cls":torch.tensor(1.),"loss_box6":torch.tensor(1.),"loss_mask":torch.tensor(1.),"loss_disp":torch.tensor(1.),"loss_total":torch.tensor(1.),"metrics":{"iou_sum":0.,"iou_count":0,"epe_sum":0.,"epe_count":0,"d1_hit":0,"d1_count":0}}
        pred={"matched_count":1,"target_count":2,"coverage":.5,"iou_sum":.4,"iou_count":1,"epe_sum":3.,"epe_count":2,"d1_hit":1,"d1_count":2}
        with patch.object(train_script,"forward_batch",return_value=output), patch.object(train_script,"evaluate_pred_roi_frame",return_value=pred):
            result=train_script.run_epoch(model,[batch],args,None,0,True,lambda _:None)
        self.assertEqual(result["pred_coverage"],.5)
        self.assertEqual(result["pred_mask_iou"],.4)
        self.assertEqual(result["pred_epe"],1.5)
        self.assertEqual(result["pred_d1"],.5)


if __name__=="__main__":
    unittest.main()

