import unittest

from algorithm.detector.detection_output import Detection
from algorithm.diagnostic_analysis.localization_confusion_matrix import localization_confusion_matrix


class ConfusionMatrixTests(unittest.TestCase):
    def test_competing_predictions_match_remaining_reference(self):
        result = localization_confusion_matrix(
            [Detection("i", 0, 0.9, (0, 0, 10, 10)),
             Detection("i", 0, 0.8, (0, 0, 10, 10))],
            {0: [{"box": [0, 0, 10, 10]}, {"box": [1, 0, 11, 10]}]},
        )
        self.assertEqual(result["counts"]["lateral_malleolus_fracture"]["lateral_malleolus_fracture"], 2)
        self.assertEqual(result["counts"]["lateral_malleolus_fracture"]["missed"], 0)

    def test_floor_and_custom_iou(self):
        reference = {0: [{"box": [0, 0, 10, 10]}]}
        prediction = Detection("i", 0, 0.04, (0, 0, 10, 10))
        result = localization_confusion_matrix([prediction], reference)
        self.assertEqual(result["counts"]["lateral_malleolus_fracture"]["missed"], 1)
        self.assertEqual(result["counts"]["background"]["lateral_malleolus_fracture"], 0)
        prediction = Detection("i", 0, 0.9, (4, 0, 14, 10))
        strict = localization_confusion_matrix([prediction], reference)
        relaxed = localization_confusion_matrix([prediction], reference, 0.25)
        self.assertEqual(strict["counts"]["lateral_malleolus_fracture"]["missed"], 1)
        self.assertEqual(relaxed["counts"]["lateral_malleolus_fracture"]["lateral_malleolus_fracture"], 1)

    def test_multiple_images_are_rejected(self):
        with self.assertRaises(ValueError):
            localization_confusion_matrix(
                [Detection("a", 0, 0.9, (0, 0, 10, 10)),
                 Detection("b", 0, 0.8, (0, 0, 10, 10))], {},
            )

    def test_wrong_site_is_counted_as_confusion(self):
        result = localization_confusion_matrix(
            [Detection("i", 1, 0.9, (0, 0, 10, 10))],
            {0: [{"box": [0, 0, 10, 10]}]},
        )
        self.assertEqual(result["counts"]["lateral_malleolus_fracture"]["medial_malleolus_fracture"], 1)


if __name__ == "__main__":
    unittest.main()
