import unittest

from algorithm.detector.detection_output import Detection
from algorithm.patient_level.patient_aggregation import aggregate_patient
from algorithm.patient_level.pipeline import run_four_layer_diagnosis


class FourLayerTests(unittest.TestCase):
    def test_pipeline_exposes_four_layers(self):
        record = aggregate_patient("p", ["i"], {"i": [0]}, [Detection("i", 0, 0.9, (0, 0, 10, 10))])
        result = run_four_layer_diagnosis(record, [0.5, 0.5, 0.5, 0.5], {0: [{"box": [0, 0, 10, 10]}]})
        self.assertIn("layer_1_patient_fracture_recognition", result)
        self.assertIn("layer_2_lesion_localization", result)
        self.assertIn("layer_3_post_localization_typing", result)
        self.assertIn("layer_4_fixed_threshold_detection", result)


if __name__ == "__main__":
    unittest.main()
