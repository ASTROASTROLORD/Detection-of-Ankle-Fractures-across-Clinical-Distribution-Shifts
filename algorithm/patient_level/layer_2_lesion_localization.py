"""Layer 2: lesion localization summaries from image-level detections."""

from __future__ import annotations

from collections import Counter

from algorithm.detector.detection_output import CLASS_NAMES
from algorithm.diagnostic_analysis.category_specific_froc import match_detections


def summarize_localization(record, ground_truth_by_class: dict[int, list[dict]], iou_threshold: float = 0.5) -> dict:
    matched = Counter()
    false_positive = Counter()
    for label in range(len(CLASS_NAMES)):
        predictions = [item for item in record.detections if item.label == label]
        events = match_detections(predictions, ground_truth_by_class.get(label, []), iou_threshold)
        matched[label] += sum(event[1] for event in events)
        false_positive[label] += sum(event[2] for event in events)
    return {
        "iou_threshold": iou_threshold,
        "true_positives": {CLASS_NAMES[key]: value for key, value in matched.items()},
        "false_positives": {CLASS_NAMES[key]: value for key, value in false_positive.items()},
    }
