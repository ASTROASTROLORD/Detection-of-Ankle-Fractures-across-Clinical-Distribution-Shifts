import math
import unittest

from algorithm.detector.detection_output import Detection
from algorithm.patient_level.layer_2_lesion_localization import evaluate_lesion_detection
from algorithm.diagnostic_analysis.category_specific_froc import (
    fp_at_sensitivity,
    match_detections,
    operating_curve,
    sensitivity_at_fp,
)


class FrocTests(unittest.TestCase):
    def test_level_two_pools_negative_images_and_groups_classes(self):
        images = [
            {"image_id": "positive", "ground_truth": [{"label": 0, "box": [0, 0, 10, 10]}],
             "detections": [{"label": 1, "score": 0.9, "box": [0, 0, 10, 10]},
                            {"label": 0, "score": 0.8, "box": [0, 0, 10, 10]}]},
            {"image_id": "negative", "ground_truth": [], "detections": []},
        ]
        result = evaluate_lesion_detection(images)
        self.assertEqual(result["detected_lesions"], 1)
        self.assertEqual(result["false_positive_boxes"], 1)
        self.assertEqual(result["sensitivity_at_fp_per_radiograph"]["0.25"], 0)
        self.assertEqual(result["sensitivity_at_fp_per_radiograph"]["0.5"], 1)

    def test_level_two_score_floor_top_100_and_empty_references(self):
        predictions = [{"label": 1, "score": 0.9, "box": [20, 20, 30, 30]}] * 100
        predictions.append({"label": 0, "score": 0.8, "box": [0, 0, 10, 10]})
        image = {"image_id": "i", "detections": predictions,
                 "ground_truth": [{"label": 0, "box": [0, 0, 10, 10]}]}
        result = evaluate_lesion_detection([image])
        self.assertEqual(result["detected_lesions"], 0)
        self.assertEqual(result["false_positive_boxes"], 100)
        image["detections"] = [{"label": 0, "score": 0.049, "box": [0, 0, 10, 10]}]
        self.assertEqual(evaluate_lesion_detection([image])["detected_lesions"], 0)
        image["detections"][0]["score"] = 0.05
        self.assertEqual(evaluate_lesion_detection([image])["detected_lesions"], 1)
        image["ground_truth"] = []
        result = evaluate_lesion_detection([image])
        self.assertIsNone(result["detection_sensitivity"])
        self.assertIsNone(result["sensitivity_at_fp_per_radiograph"]["1.0"])
        with self.assertRaises(ValueError):
            evaluate_lesion_detection([image, image])

    def test_score_ordered_one_to_one_matching(self):
        detections = [Detection("i", 0, 0.9, (0, 0, 10, 10)), Detection("i", 0, 0.8, (0, 0, 10, 10))]
        events = match_detections(detections, [{"box": [0, 0, 10, 10]}])
        self.assertEqual(events, [(0.9, 1, 0), (0.8, 0, 1)])

    def test_operating_point_has_no_extrapolation(self):
        curve = operating_curve([(0.9, 1, 0)], 2, 1)
        self.assertEqual(sensitivity_at_fp(curve, 0.0), 0.5)
        self.assertTrue(math.isnan(fp_at_sensitivity(curve, 0.8)))


if __name__ == "__main__":
    unittest.main()
