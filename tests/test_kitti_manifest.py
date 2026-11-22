"""KITTI paired-box label geometry."""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
import torch
import json
import tempfile
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
try:
    from lightbackbone_stereo_yolo.data.kitti import collate_instances, build_instance_target, build_kitti_manifest, project_right_box
except ImportError:
    collate_instances = build_instance_target = build_kitti_manifest = project_right_box = None


class KittiLabelContractTest(unittest.TestCase):
    def test_collate_stacks_images_but_keeps_instance_targets_per_sample(self) -> None:
        self.assertIsNotNone(collate_instances)
        sample = {"left": torch.zeros(3, 4, 5), "right": torch.zeros(3, 4, 5), "boxes6": torch.zeros(6), "mask": torch.zeros(4, 5, dtype=torch.bool)}
        batch = collate_instances([sample, sample])
        self.assertEqual(batch["left"].shape, (2, 3, 4, 5))
        self.assertEqual(len(batch["targets"]), 2)
        self.assertTrue(torch.equal(batch["targets"][0]["boxes6"], torch.zeros(6)))
    def test_instance_target_binds_mask_valid_disparity_and_paired_box(self) -> None:
        self.assertIsNotNone(build_instance_target)
        instances = torch.tensor([[0, 9, 9], [0, 9, 0]])
        disparity = torch.tensor([[0., 2., 3.], [0., 1., 0.]])
        target = build_instance_target(instances, disparity, instance_value=9, category_id=26)
        self.assertTrue(torch.equal(target.boxes6, torch.tensor([1., 0., 2., 1., -1., 0.])))
        self.assertTrue(torch.equal(target.mask, instances == 9))
        self.assertTrue(torch.equal(target.valid_disparity, torch.tensor([[False, True, True], [False, True, False]])))
    def test_manifest_binds_existing_stereo_mask_disparity_and_ignore_files(self) -> None:
        self.assertIsNotNone(build_kitti_manifest)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in ("training/image_2/000001_10.png", "training/image_3/000001_10.png", "training/instance/000001_10.png", "training/disp_occ_0/000001_10.png", "rpn_annotations_v1/ignore_masks/000001_10.png"):
                path = root / relative; path.parent.mkdir(parents=True, exist_ok=True); path.touch()
            annotation = {"images": [{"id": 1, "file_name": "training/image_2/000001_10.png", "width": 10, "height": 8, "ignore_mask_file": "rpn_annotations_v1/ignore_masks/000001_10.png"}], "annotations": [{"id": 1, "image_id": 1, "category_id": 26, "bbox": [1, 2, 3, 4], "source_instance_value": 9001}]}
            source = root / "split.json"; source.write_text(json.dumps(annotation))
            record = build_kitti_manifest(root, source)[0]
            self.assertEqual(record.left_box, (1., 2., 4., 6.))
            self.assertTrue(record.right_image.exists())
            self.assertEqual(record.instance_value, 9001)
            record.right_image.unlink()
            self.assertFalse(build_kitti_manifest(root, source))
            with self.assertRaisesRegex(FileNotFoundError, "image_3"):
                build_kitti_manifest(root, source, require_files=True)

    def test_a1_visible_projection_controls_right_box_without_support_gate(self):
        from lightbackbone_stereo_yolo.data.kitti import assess_right_projection

        mask = torch.ones((4, 20), dtype=torch.bool)
        disparity = torch.zeros(mask.shape)
        disparity[1, [0, 2, 10, 19]] = torch.tensor([2., 1., 1., float("inf")])
        valid = torch.ones_like(mask)
        projection = assess_right_projection(mask, disparity, valid, (4, 20))
        self.assertEqual(projection.raw_box.tolist(), [-2., 1., 9., 1.])
        self.assertEqual(projection.visible_box.tolist(), [1., 1., 9., 1.])
        self.assertEqual(projection.visible_width, 8.)
        self.assertEqual(projection.valid_pixels, 3)
        self.assertEqual(projection.visible_pixels, 2)
        self.assertEqual(projection.mask_pixels, 80)
        self.assertTrue(projection.partly_outside)
        self.assertTrue(projection.pair_evaluable)
        self.assertTrue(projection.low_support)
        target = build_instance_target(mask.long(), disparity, 1, 26, right_hw=(4, 20))
        self.assertEqual(target.boxes6[4:].tolist(), [1., 9.])

        valid[1, 10] = False
        self.assertFalse(assess_right_projection(mask, disparity, valid, (4, 20)).pair_evaluable)
        empty = assess_right_projection(mask & False, disparity, valid, (4, 20))
        self.assertIsNone(empty.raw_box)
        self.assertIsNone(empty.visible_box)
        self.assertFalse(empty.pair_evaluable)
        # The only surviving positive disparity projects outside the image.
        valid[1, 2] = False
        outside = assess_right_projection(mask, disparity, valid, (4, 20))
        self.assertTrue(outside.fully_outside)
        self.assertFalse(outside.partly_outside)
        self.assertIsNone(outside.visible_box)

    def test_right_box_uses_valid_left_instance_pixels_projected_by_disparity(self) -> None:
        self.assertIsNotNone(project_right_box)
        mask = torch.tensor([[False, True, True], [False, True, False]])
        disparity = torch.tensor([[0., 2., 3.], [0., 1., 0.]])
        self.assertTrue(torch.equal(project_right_box(mask, disparity), torch.tensor([-1., 0., 0., 1.])))


if __name__ == "__main__":
    unittest.main()
