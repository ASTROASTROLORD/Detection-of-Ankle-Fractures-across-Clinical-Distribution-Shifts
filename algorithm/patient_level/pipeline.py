"""Composition of the four diagnostic layers."""

from __future__ import annotations

from .layer_1_patient_fracture_recognition import recognize_patient
from .layer_4_fixed_threshold_detection import fixed_threshold_detection
from algorithm.detector.detection_output import CLASS_NAMES
from algorithm.diagnostic_analysis.five_lesion_states import summarize_diagnosis


def run_four_layer_diagnosis(record, thresholds, ground_truth_by_class=None, ground_truth_by_image=None):
    if ground_truth_by_class is not None and ground_truth_by_image is not None:
        raise ValueError("provide ground_truth_by_image or ground_truth_by_class, not both")
    if ground_truth_by_class is not None and len(record.image_ids) != 1:
        raise ValueError("multiple images require ground_truth_by_image to prevent cross-image matching")
    if ground_truth_by_class is not None:
        ground_truth_by_image = {record.image_ids[0]: [
            {"gt_id": item.get("gt_id", item.get("annotation_id", f"class-{label}-reference-{index}")),
             "label": label, "box": item["box"]}
            for label, items in sorted(ground_truth_by_class.items())
            for index, item in enumerate(items, start=1)
        ]}
    layer_1 = recognize_patient(record, thresholds)
    result = {
        "patient_id": record.patient_id,
        "class_scores": {
            name: float(score) for name, score in zip(CLASS_NAMES, record.scores)
        },
        "layer_1_patient_fracture_recognition": layer_1,
        "layer_4_fixed_threshold_detection": fixed_threshold_detection(record, thresholds),
    }
    if ground_truth_by_image is not None:
        if set(ground_truth_by_image) != set(record.image_ids):
            raise ValueError("ground_truth_by_image must declare every patient image, including empty references")
        images = [{
            "image_id": image_id,
            "ground_truth": ground_truth_by_image[image_id],
            "detections": [{"label": item.label, "score": item.score, "box": list(item.box)}
                           for item in record.detections if item.image_id == image_id],
        } for image_id in record.image_ids]
        analysis = summarize_diagnosis({"images": images}, thresholds)
        counts = analysis["summary"]["all"]["counts"]
        total = len(analysis["lesions"])
        localized = sum(row["matched"] for row in analysis["lesions"])
        result["five_lesion_analysis"] = analysis
        result["layer_2_lesion_localization"] = {
            "iou_threshold": analysis["method"]["localization_iou"],
            "matching": analysis["method"]["matching"],
            "reference_lesions": total,
            "localized_lesions": localized,
            "localization_sensitivity": localized / total if total else None,
        }
        matrix = {name: {predicted: 0 for predicted in CLASS_NAMES} for name in CLASS_NAMES}
        for row in analysis["lesions"]:
            if row["matched"]:
                matrix[row["gt_class"]][row["prediction_class"]] += 1
        result["layer_3_post_localization_typing"] = {
            "iou_threshold": analysis["method"]["localization_iou"],
            "labels": list(CLASS_NAMES),
            "counts": matrix,
            "localized_lesions": localized,
            "wrong_class_lesions": counts["wrong_class"],
        }
        result["layer_4_fixed_threshold_detection"]["reference_lesion_outcomes"] = counts
    result["class_predictions"] = dict(zip(CLASS_NAMES, layer_1["predictions"]))
    result["layers"] = {key: value for key, value in result.items() if key.startswith("layer_")}
    return summarize_diagnosis(result, thresholds)
