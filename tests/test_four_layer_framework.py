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
        self.assertEqual(result["five_lesion_analysis"]["lesions"][0]["outcome"],
                         "correct_frozen_threshold_met")

    def test_original_class_references_and_image_references_use_same_analysis(self):
        record = aggregate_patient("p", ["i"], {"i": [0]},
                                   [Detection("i", 1, 0.9, (0, 0, 10, 10))])
        old = run_four_layer_diagnosis(record, [0.5] * 4,
                                       ground_truth_by_class={0: [{"gt_id": 1, "box": [0, 0, 10, 10]}]})
        new = run_four_layer_diagnosis(record, [0.5] * 4,
                                       ground_truth_by_image={"i": [{"gt_id": 1, "label": 0, "box": [0, 0, 10, 10]}]})
        self.assertEqual(old, new)
        self.assertEqual(old["five_lesion_analysis"]["summary"]["all"]["counts"]["wrong_class"], 1)
        self.assertEqual(old["layer_2_lesion_localization"]["detected_lesions"], 0)
        self.assertEqual(old["layer_2_lesion_localization"]["false_positive_boxes"], 1)
        self.assertEqual(old["layer_3_post_localization_typing"]["localized_lesions"], 1)

    def test_image_annotations_share_the_five_state_matching_result(self):
        record = aggregate_patient("p", ["a", "b"], {"a": [0]},
                                   [Detection("b", 0, 0.9, (0, 0, 10, 10))])
        references = {"a": [{"gt_id": 1, "label": 0, "box": [0, 0, 10, 10]}], "b": []}
        result = run_four_layer_diagnosis(record, [0.63, 0.76, 0.79, 0.68],
                                           ground_truth_by_image=references)
        self.assertEqual(result["patient_site_summary"]["state"], "lateral_malleolus_fracture")
        self.assertEqual(result["five_lesion_analysis"]["lesions"][0]["outcome"], "no_effective_candidate")
        self.assertEqual(result["layer_2_lesion_localization"]["localized_lesions"], 0)
        self.assertEqual(result["layer_3_post_localization_typing"]["localized_lesions"], 0)
        self.assertEqual(result["layer_4_fixed_threshold_detection"]["reference_lesion_outcomes"]["no_effective_candidate"], 1)
        self.assertEqual(result["class_predictions"]["lateral_malleolus_fracture"], 1)
        self.assertEqual(result["layers"]["layer_1_patient_fracture_recognition"],
                         result["layer_1_patient_fracture_recognition"])

    def test_incomplete_or_pooled_multi_image_references_are_rejected(self):
        record = aggregate_patient("p", ["a", "b"], {}, [])
        with self.assertRaisesRegex(ValueError, "every patient image"):
            run_four_layer_diagnosis(record, [0.5] * 4, ground_truth_by_image={"a": []})
        with self.assertRaisesRegex(ValueError, "multiple images"):
            run_four_layer_diagnosis(record, [0.5] * 4, ground_truth_by_class={0: []})


if __name__ == "__main__":
    unittest.main()
