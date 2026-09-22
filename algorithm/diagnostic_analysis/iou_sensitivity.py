"""Sensitivity of lesion detection to the IoU acceptance criterion."""

from __future__ import annotations

from .category_specific_froc import match_detections


def evaluate_iou_sensitivity(predictions, ground_truth, thresholds=(0.25, 0.5, 0.75)) -> dict[str, dict[str, float]]:
    result = {}
    for iou_threshold in thresholds:
        events = match_detections(predictions, ground_truth, float(iou_threshold))
        tp = sum(row[1] for row in events)
        fp = sum(row[2] for row in events)
        result[str(iou_threshold)] = {
            "true_positives": tp,
            "false_positives": fp,
            "sensitivity": tp / len(ground_truth) if ground_truth else 0.0,
        }
    return result
