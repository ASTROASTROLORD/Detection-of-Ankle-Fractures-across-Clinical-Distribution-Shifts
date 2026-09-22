"""Layer 3: post-localization fracture-site typing."""

from __future__ import annotations

from algorithm.diagnostic_analysis.localization_confusion_matrix import localization_confusion_matrix


def type_after_localization(record, ground_truth, iou_threshold: float = 0.5) -> dict:
    return localization_confusion_matrix(record.detections, ground_truth, iou_threshold)
