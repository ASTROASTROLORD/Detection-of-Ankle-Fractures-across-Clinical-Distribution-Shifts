import unittest

from algorithm.detector.detection_output import Detection
from algorithm.patient_level.patient_aggregation import aggregate_patient


class PatientAggregationTests(unittest.TestCase):
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
