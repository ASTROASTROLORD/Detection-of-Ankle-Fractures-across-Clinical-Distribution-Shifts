"""Composition of the four diagnostic layers."""

from __future__ import annotations

from .layer_1_patient_fracture_recognition import recognize_patient
from .layer_2_lesion_localization import summarize_localization
from .layer_3_post_localization_typing import type_after_localization
from .layer_4_fixed_threshold_detection import fixed_threshold_detection
from algorithm.detector.detection_output import CLASS_NAMES


def run_four_layer_diagnosis(record, thresholds, ground_truth_by_class=None):
    layer_1 = recognize_patient(record, thresholds)
    result = {
        "patient_id": record.patient_id,
        "class_scores": {
            name: float(score) for name, score in zip(CLASS_NAMES, record.scores)
        },
        "layer_1_patient_fracture_recognition": layer_1,
        "layer_4_fixed_threshold_detection": fixed_threshold_detection(record, thresholds),
    }
    if ground_truth_by_class is not None:
        result["layer_2_lesion_localization"] = summarize_localization(record, ground_truth_by_class)
        result["layer_3_post_localization_typing"] = type_after_localization(record, ground_truth_by_class)
    return result
