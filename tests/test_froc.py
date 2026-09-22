import math
import unittest

from algorithm.detector.detection_output import Detection
from algorithm.diagnostic_analysis.category_specific_froc import (
    fp_at_sensitivity,
    match_detections,
    operating_curve,
    sensitivity_at_fp,
)


class FrocTests(unittest.TestCase):
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
