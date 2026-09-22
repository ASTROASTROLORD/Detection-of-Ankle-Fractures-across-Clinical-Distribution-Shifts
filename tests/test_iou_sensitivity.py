import unittest

from algorithm.detector.detection_output import Detection
from algorithm.diagnostic_analysis.iou_sensitivity import evaluate_iou_sensitivity


class IouSensitivityTests(unittest.TestCase):
    def test_relaxed_iou_does_not_reduce_match_count(self):
        detections = [Detection("i", 0, 0.9, (0, 0, 9, 9))]
        result = evaluate_iou_sensitivity(detections, [{"box": [0, 0, 10, 10]}], (0.25, 0.5))
        self.assertGreaterEqual(result["0.25"]["true_positives"], result["0.5"]["true_positives"])


if __name__ == "__main__":
    unittest.main()
