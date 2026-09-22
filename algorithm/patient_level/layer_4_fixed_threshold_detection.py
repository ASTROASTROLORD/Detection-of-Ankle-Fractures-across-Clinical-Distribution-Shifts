"""Layer 4: fixed-threshold detection operating point."""

from __future__ import annotations

from algorithm.detector.detection_output import CLASS_NAMES


def fixed_threshold_detection(record, thresholds: list[float]) -> dict:
    if len(thresholds) != len(CLASS_NAMES):
        raise ValueError("one fixed threshold is required per detector class")
    kept = [item for item in record.detections if item.score >= thresholds[item.label]]
    return {
        "thresholds": dict(zip(CLASS_NAMES, thresholds)),
        "detections": [
            {"image_id": item.image_id, "class": item.class_name, "score": item.score, "box": list(item.box)}
            for item in kept
        ],
    }
