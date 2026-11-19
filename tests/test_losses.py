"""Joint detector/mask/instance-disparity loss contract."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
import torch
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
try:
    from lightbackbone_stereo_yolo.losses import assemble_losses
except ImportError:
    assemble_losses = None


class JointLossContractTest(unittest.TestCase):
    def test_joint_loss_exposes_normalized_component_terms(self) -> None:
        self.assertIsNotNone(assemble_losses)
        detection = {"loss_obj": torch.tensor(1., requires_grad=True), "loss_cls": torch.tensor(2., requires_grad=True), "loss_box6": torch.tensor(3., requires_grad=True)}
        mask_logits = torch.zeros(1, 1, 1, 1, requires_grad=True)
        result = assemble_losses(detection, mask_logits, torch.ones_like(mask_logits), torch.ones_like(mask_logits), torch.tensor([[[12.]]], requires_grad=True), torch.tensor([[[10.]]]), torch.tensor([[[True]]]))
        self.assertEqual(set(result), {"loss_det", "loss_mask", "loss_disp", "loss_total"})
        self.assertAlmostEqual(result["loss_det"].item(), 6.)
        result["loss_total"].backward()


if __name__ == "__main__":
    unittest.main()
