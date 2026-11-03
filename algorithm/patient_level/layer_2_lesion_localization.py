"""Layer 2: same-class lesion detection and pooled image-level FROC."""

from __future__ import annotations

from collections import Counter

from algorithm.detector.detection_output import CLASS_NAMES, parse_detections
from algorithm.diagnostic_analysis.category_specific_froc import (
    match_detections, operating_curve, sensitivity_at_fp,
)
from algorithm.diagnostic_analysis.five_lesion_states import (
    SCORE_FLOOR, MAX_PREDICTIONS_PER_IMAGE,
)


def evaluate_lesion_detection(images, iou_threshold: float = 0.50) -> dict:
    """Pool per-image, same-class matches across the supplied patient or cohort.

    Inputs use the annotated ``images`` format accepted by the patient pipeline.
    Detections must already have undergone class-wise NMS. The score floor and
    cross-class top-100 limit are applied here before matching. Negative images
    contribute to the FP/radiograph denominator. Cohort FROC requires passing
    all cohort images together, rather than averaging patient operating points.
    """
    events = []
    reference_count = 0
    image_ids = set()
    for image in images:
        image_id = image["image_id"]
        if not isinstance(image_id, str) or not image_id.strip() or image_id in image_ids:
            raise ValueError("each image must have a unique non-empty image_id")
        image_ids.add(image_id)
        predictions = parse_detections(image_id, {
            "bboxes": [item["box"] for item in image["detections"]],
            "scores": [item["score"] for item in image["detections"]],
            "labels": [item["label"] for item in image["detections"]],
        })
        predictions = sorted(predictions, key=lambda item: -item.score)
        predictions = [item for item in predictions[:MAX_PREDICTIONS_PER_IMAGE]
                       if item.score >= SCORE_FLOOR]
        references = parse_detections(image_id, {
            "bboxes": [item["box"] for item in image["ground_truth"]],
            "scores": [1.0] * len(image["ground_truth"]),
            "labels": [item["label"] for item in image["ground_truth"]],
        })
        reference_count += len(references)
        for label in range(len(CLASS_NAMES)):
            events.extend(match_detections(
                [item for item in predictions if item.label == label],
                [{"box": item.box} for item in references if item.label == label],
                iou_threshold,
            ))
    if not image_ids:
        raise ValueError("at least one image is required")
    curve = operating_curve(events, reference_count, len(image_ids))
    detected = sum(event[1] for event in events)
    sensitivity = detected / reference_count if reference_count else None
    return {
        "iou_threshold": iou_threshold,
        "matching": "per-image, same-class, descending-score, one-to-one",
        "score_floor": SCORE_FLOOR,
        "max_predictions_per_image": MAX_PREDICTIONS_PER_IMAGE,
        "images": len(image_ids),
        "reference_lesions": reference_count,
        "detected_lesions": detected,
        "false_positive_boxes": sum(event[2] for event in events),
        "detection_sensitivity": sensitivity,
        # Compatibility fields now follow the same-class Level 2 definition.
        "localized_lesions": detected,
        "localization_sensitivity": sensitivity,
        "froc_curve": [
            {"threshold": score, "sensitivity": recall if reference_count else None,
             "fp_per_radiograph": fp}
            for score, recall, fp in curve
        ],
        "sensitivity_at_fp_per_radiograph": {
            str(budget): sensitivity_at_fp(curve, budget) if reference_count else None
            for budget in (0.125, 0.25, 0.5, 1.0, 2.0)
        },
    }


def summarize_localization(record, ground_truth_by_class: dict[int, list[dict]], iou_threshold: float = 0.5) -> dict:
    if len(record.image_ids) != 1:
        raise ValueError("multiple images require evaluate_lesion_detection with per-image references")
    matched = Counter()
    false_positive = Counter()
    retained = sorted(record.detections, key=lambda item: -item.score)[:MAX_PREDICTIONS_PER_IMAGE]
    for label in range(len(CLASS_NAMES)):
        predictions = [item for item in retained if item.label == label and item.score >= SCORE_FLOOR]
        events = match_detections(predictions, ground_truth_by_class.get(label, []), iou_threshold)
        matched[label] += sum(event[1] for event in events)
        false_positive[label] += sum(event[2] for event in events)
    return {
        "iou_threshold": iou_threshold,
        "true_positives": {CLASS_NAMES[key]: value for key, value in matched.items()},
        "false_positives": {CLASS_NAMES[key]: value for key, value in false_positive.items()},
    }
