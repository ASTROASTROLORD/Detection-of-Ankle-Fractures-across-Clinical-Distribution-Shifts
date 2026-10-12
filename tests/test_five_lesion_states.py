import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from algorithm.detector.detection_output import Detection
from algorithm.diagnostic_analysis.category_specific_froc import match_detections
from algorithm.diagnostic_analysis.five_lesion_states import (
    OUTCOMES,
    assign_lesion_state,
    summarize_diagnosis,
    analyze_lesion_states,
    decompose_image,
    summarize_outcomes,
)


class FiveLesionStateTests(unittest.TestCase):
    def test_priority_produces_one_state(self):
        scores = {"lateral_malleolus_fracture": 0.9, "distal_tibial_fracture": 0.8}
        thresholds = {name: 0.5 for name in scores}
        self.assertEqual(assign_lesion_state(scores, thresholds), "distal_tibial_fracture")

    def test_no_positive_is_normal_state(self):
        self.assertEqual(assign_lesion_state({}, {}), "no_core_fracture")


class LesionOutcomeTests(unittest.TestCase):
    def reference(self, gt_id=1, label=0, box=(0, 0, 10, 10)):
        return {"gt_id": gt_id, "label": label, "box": list(box)}

    def prediction(self, score=0.9, label=0, box=(0, 0, 10, 10), image_id="i"):
        return Detection(image_id, label, score, box)

    def test_all_five_outcomes_and_denominator(self):
        cases = [
            [],
            [self.prediction(box=(0, 0, 4, 10))],
            [self.prediction(label=1)],
            [self.prediction(score=0.62)],
            [self.prediction(score=0.63)],
        ]
        rows = [decompose_image("i", predictions, [self.reference()])[0] for predictions in cases]
        self.assertEqual([row["outcome"] for row in rows], list(OUTCOMES))
        summary = summarize_outcomes(rows)["all"]
        self.assertEqual(summary["reference_lesions"], 5)
        self.assertEqual(list(summary["counts"].values()), [1] * 5)
        self.assertEqual(list(summary["proportions"].values()), [0.2] * 5)

    def test_candidate_and_localization_iou_boundaries(self):
        for width, expected in ((2.49, OUTCOMES[0]), (2.5, OUTCOMES[1]),
                                (4.99, OUTCOMES[1]), (5.0, OUTCOMES[4])):
            with self.subTest(width=width):
                row = decompose_image("i", [self.prediction(box=(0, 0, width, 10))],
                                      [self.reference()])[0]
                self.assertEqual(row["outcome"], expected)

    def test_score_floor_boundary_and_declared_thresholds(self):
        low = decompose_image("i", [self.prediction(score=0.049)], [self.reference()])[0]
        floor = decompose_image("i", [self.prediction(score=0.05)], [self.reference()])[0]
        self.assertEqual(low["outcome"], OUTCOMES[0])
        self.assertEqual(floor["outcome"], OUTCOMES[3])
        for label, threshold in enumerate((0.63, 0.76, 0.79, 0.68)):
            row = decompose_image("i", [self.prediction(score=threshold, label=label)],
                                  [self.reference(label=label)])[0]
            self.assertEqual(row["outcome"], OUTCOMES[4])

    def test_matching_precedes_typing_and_thresholding(self):
        # The high-score wrong-class box consumes the reference, despite a
        # lower-score, correctly typed box that also passes the threshold.
        predictions = [self.prediction(score=0.9, label=1), self.prediction(score=0.8)]
        row = decompose_image("i", predictions, [self.reference()])[0]
        self.assertEqual(row["outcome"], OUTCOMES[2])
        self.assertEqual(row["prediction_index"], 0)
        low_wrong = decompose_image("i", [self.prediction(score=0.05, label=1)],
                                    [self.reference()])[0]
        self.assertEqual(low_wrong["outcome"], OUTCOMES[2])

    def test_one_prediction_cannot_match_two_references(self):
        rows = decompose_image("i", [self.prediction()], [self.reference(1), self.reference(2)])
        self.assertEqual([row["outcome"] for row in rows], [OUTCOMES[4], OUTCOMES[1]])
        self.assertEqual(rows[1]["best_any_iou"], 1.0)

    def test_next_prediction_chooses_best_available_reference(self):
        references = [self.reference(1), self.reference(2, box=(0, 0, 11, 10))]
        predictions = [self.prediction(score=0.9), self.prediction(score=0.8)]
        rows = decompose_image("i", predictions, references)
        self.assertTrue(all(row["matched"] for row in rows))
        self.assertAlmostEqual(rows[1]["matched_iou"], 10 / 11)
        events = match_detections(predictions, references)
        self.assertEqual(sum(row["matched"] for row in rows), sum(event[1] for event in events))

    def test_equal_scores_use_class_grouped_stable_order(self):
        predictions = [self.prediction(label=1), self.prediction(label=0)]
        row = decompose_image("i", predictions, [self.reference()])[0]
        self.assertEqual(row["prediction_index"], 1)
        self.assertEqual(row["outcome"], OUTCOMES[4])

    def test_candidate_search_respects_top_100_limit(self):
        predictions = [self.prediction(box=(100, 100, 110, 110)) for _ in range(100)]
        predictions.append(self.prediction(score=0.8))
        row = decompose_image("i", predictions, [self.reference()])[0]
        self.assertEqual(row["outcome"], OUTCOMES[0])

    def test_images_are_matched_separately_and_negative_scenes_are_retained(self):
        images = [
            {"image_id": "a", "scene": "positive", "ground_truth": [self.reference()], "detections": []},
            {"image_id": "b", "scene": "negative", "ground_truth": [],
             "detections": [{"label": 0, "score": 0.9, "box": [0, 0, 10, 10]}]},
        ]
        result = analyze_lesion_states(images)
        self.assertEqual(result["lesions"][0]["outcome"], OUTCOMES[0])
        self.assertEqual(result["summary"]["all"]["reference_lesions"], 1)
        self.assertEqual(result["summary"]["by_scene"]["negative"]["reference_lesions"], 0)
        self.assertTrue(all(value is None for value in result["summary"]["by_scene"]["negative"]["proportions"].values()))
        json.dumps(result, allow_nan=False)

    def test_invalid_inputs_are_rejected(self):
        for thresholds in ((0.5,), (0.5, 0.5, 0.5, float("nan"))):
            with self.assertRaises(ValueError):
                decompose_image("i", [], [], thresholds)
        for reference in (self.reference(label=4), self.reference(label=0.5),
                          self.reference(box=(0, 0, 0, 10))):
            with self.assertRaises(ValueError):
                decompose_image("i", [], [reference])
        with self.assertRaisesRegex(ValueError, "duplicate gt_id"):
            decompose_image("i", [], [self.reference(1), self.reference("1")])
        with self.assertRaisesRegex(ValueError, "this image"):
            decompose_image("i", [self.prediction(image_id="other")], [])
        image = {"image_id": "i", "ground_truth": [], "detections": []}
        with self.assertRaisesRegex(ValueError, "duplicate image_id"):
            analyze_lesion_states([image, image])

    def test_command_line_exports_synthetic_example(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out.json"
            subprocess.run([sys.executable, "-m", "interface.summarize_patient_diagnosis",
                            str(root / "examples/five_lesion_states_input.json"), str(output), "--threshold", "0.63", "0.76", "0.79", "0.68"],
                           cwd=root, check=True, capture_output=True, text=True)
            result = json.loads(output.read_text())
            self.assertEqual(list(result["summary"]["all"]["counts"].values()), [1] * 5)
            self.assertEqual(result["method"]["class_thresholds"]["lateral_malleolus_fracture"], 0.63)


class UnifiedDiagnosisTests(unittest.TestCase):
    def scores(self):
        return dict(zip(("lateral_malleolus_fracture", "medial_malleolus_fracture",
                         "posterior_malleolus_fracture", "distal_tibial_fracture"), (0.9, 0.1, 0.1, 0.1)))

    def test_existing_patient_input_keeps_its_original_fields(self):
        payload = {"patient_id": "p", "class_scores": self.scores(), "extra": "preserved"}
        result = summarize_diagnosis(payload)
        self.assertEqual(result["lesion_state"], "lateral_malleolus_fracture")
        self.assertEqual(result["extra"], "preserved")
        self.assertEqual(result["patient_site_summary"]["unit"], "patient")
        self.assertNotIn("lesion_state", payload)

    def test_patient_summary_and_lesion_analysis_share_one_entry(self):
        payload = {"patient_id": "p", "class_scores": self.scores(), "images": [
            {"image_id": "i", "ground_truth": [{"gt_id": 1, "label": 0, "box": [0, 0, 10, 10]}],
             "detections": []}]}
        result = summarize_diagnosis(payload)
        self.assertEqual(result["patient_site_summary"]["state"], "lateral_malleolus_fracture")
        self.assertEqual(result["lesions"][0]["outcome"], "no_effective_candidate")
        self.assertEqual(result["patient_id"], "p")

    def test_original_nested_layer_input_is_supported(self):
        result = summarize_diagnosis({"layers": {"layer_1_patient_fracture_recognition":
                                    {"scores": [0.9, 0.1, 0.1, 0.1]}}})
        self.assertEqual(result["lesion_state"], "lateral_malleolus_fracture")

    def test_missing_or_incomplete_patient_scores_are_rejected(self):
        for payload in ({}, {"class_scores": {}}, {"class_scores": {"lateral_malleolus_fracture": 0.9}}):
            with self.assertRaises(ValueError):
                summarize_diagnosis(payload)


if __name__ == "__main__":
    unittest.main()
