"""Checkpoint selection contracts for the training entry point."""
from __future__ import annotations
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("train_checkpoint_script",ROOT/"scripts"/"train.py")
train_script=importlib.util.module_from_spec(spec); sys.modules["train_checkpoint_script"]=train_script; spec.loader.exec_module(train_script)


class CheckpointPayloadTest(unittest.TestCase):
    def test_half_width_ranking_protects_original_near_targets(self):
        from lightbackbone_stereo_yolo.training import near10_selection_rank
        result = {"tp": 18, "gt": 23, "exactly_one_pair_targets": 17,
                  "new_missed_gt": 1, "foreground_coverage": .8,
                  "epe": 2., "d1": .1}
        rank, qualified = near10_selection_rank(result)
        self.assertFalse(qualified)
        fixed, qualified = near10_selection_rank({**result, "new_missed_gt": 0})
        self.assertTrue(qualified)
        self.assertGreater(fixed, rank)

    def test_checkpoint_persists_best_gt_roi_epe_for_resume(self):
        model=torch.nn.Linear(2,1)
        optimizer=torch.optim.SGD(model.parameters(),lr=.1)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"last.pt"
            train_script.save_checkpoint(path,model,optimizer,3,2.0,1.5,SimpleNamespace(dmax_roi=28),{24:0})
            payload=torch.load(path,map_location="cpu")
        self.assertEqual(payload["best_val_loss"],2.0)
        self.assertEqual(payload["best_gt_roi_epe"],1.5)


if __name__=="__main__":
    unittest.main()
