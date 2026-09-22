import unittest

from algorithm.detector.detection_output import Detection
from algorithm.diagnostic_analysis.localization_confusion_matrix import localization_confusion_matrix


class ConfusionMatrixTests(unittest.TestCase):
    def test_wrong_site_is_counted_as_confusion(self):
        result = localization_confusion_matrix(
            [Detection("i", 1, 0.9, (0, 0, 10, 10))],
            {0: [{"box": [0, 0, 10, 10]}]},
        )
        self.assertEqual(result["counts"]["lateral_malleolus_fracture"]["medial_malleolus_fracture"], 1)


if __name__ == "__main__":
    unittest.main()
