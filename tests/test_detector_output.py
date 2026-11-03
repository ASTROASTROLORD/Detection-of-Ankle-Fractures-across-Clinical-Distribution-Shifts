import unittest

from algorithm.detector.detection_output import parse_detections
from algorithm.detector.preprocessing import make_resize_plan
from algorithm.detector.faster_rcnn_r50_fpn import model, test_pipeline


class DetectorOutputTests(unittest.TestCase):
    def test_manuscript_resize_and_coordinate_round_trip(self):
        plan = make_resize_plan(2000, 1500)
        self.assertEqual((plan.target_width, plan.target_height), (1067, 800))
        self.assertEqual((plan.pad_left, plan.pad_top), (0, 0))
        self.assertEqual((plan.padded_width, plan.padded_height), (1088, 800))
        self.assertEqual(plan.map_box_to_source((0, 0, 1067, 800)), (0, 0, 2000, 1500))
        self.assertEqual(make_resize_plan(500, 500).target_width, 800)
        wide = make_resize_plan(4000, 1000)
        self.assertEqual((wide.target_width, wide.target_height), (1333, 333))
        portrait = make_resize_plan(1500, 2000)
        self.assertEqual((portrait.target_width, portrait.target_height), (800, 1067))
        with self.assertRaises(ValueError):
            make_resize_plan(0, 1000)

    def test_public_inference_config_matches_manuscript(self):
        resize = test_pipeline[1]
        self.assertEqual(resize["scale"], (1333, 800))
        self.assertTrue(resize["keep_ratio"])
        self.assertEqual(model["data_preprocessor"]["pad_size_divisor"], 32)
        self.assertTrue(model["data_preprocessor"]["bgr_to_rgb"])
        self.assertEqual(model["test_cfg"]["rcnn"]["nms"]["iou_threshold"], 0.5)

    def test_valid_output_is_normalized(self):
        detections = parse_detections("image-1", {"bboxes": [[0, 0, 10, 10]], "scores": [0.8], "labels": [2]})
        self.assertEqual(detections[0].class_name, "posterior_malleolus_fracture")

    def test_invalid_label_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_detections("image-1", {"bboxes": [[0, 0, 10, 10]], "scores": [0.8], "labels": [4]})


if __name__ == "__main__":
    unittest.main()
