import unittest
import numpy as np

from algorithm.detector.detection_output import Detection
from algorithm.patient_level.patient_aggregation import aggregate_patient
from algorithm.patient_level.threshold_policy import select_thresholds


class PatientAggregationTests(unittest.TestCase):
    def test_manuscript_threshold_grid_and_tie_breaking(self):
        targets = np.array([[0, 0, 0], [1, 1, 1]])
        scores = np.array([[0.62, 0.94, 0.04], [0.64, 0.96, 0.05]])
        self.assertEqual(select_thresholds(targets, scores), [0.63, 0.95, 0.05])
        self.assertEqual(select_thresholds([[0], [1]], [[0.1], [0.9]],
                                           grid=[0.25, 0.75]), [0.75])

    def test_union_targets_and_maximum_patient_score(self):
        record = aggregate_patient(
            "p1",
            ["i1", "i2"],
            {"i1": [0], "i2": [1]},
            [
                Detection("i1", 0, 0.4, (0, 0, 10, 10)),
                Detection("i2", 0, 0.8, (1, 1, 11, 11)),
                Detection("i2", 1, 0.7, (0, 0, 10, 10)),
            ],
        )
        self.assertEqual(record.targets.tolist(), [1, 1, 0, 0])
        self.assertEqual(record.scores.tolist(), [0.8, 0.7, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
