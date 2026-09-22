"""One-to-one lesion matching and category-specific FROC primitives."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def box_iou(box: Iterable[float], boxes: np.ndarray) -> np.ndarray:
    box = np.asarray(tuple(box), dtype=float)
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    if not len(boxes):
        return np.empty(0, dtype=float)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    area_b = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    union = area_a + area_b - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def match_detections(predictions, ground_truth, iou_threshold: float = 0.5):
    """Return score-ordered ``(score, tp, fp)`` events."""
    if not 0 <= iou_threshold <= 1:
        raise ValueError("IoU threshold must be between 0 and 1")
    gt_boxes = np.asarray([row["box"] for row in ground_truth], dtype=float).reshape(-1, 4)
    available = np.ones(len(gt_boxes), dtype=bool)
    events = []
    for prediction in sorted(predictions, key=lambda item: -item.score)[:100]:
        overlaps = box_iou(prediction.box, gt_boxes)
        candidates = np.flatnonzero(available)
        matched = False
        if len(candidates):
            index = int(candidates[int(np.argmax(overlaps[candidates]))])
            if overlaps[index] >= iou_threshold:
                available[index] = False
                matched = True
        events.append((float(prediction.score), int(matched), int(not matched)))
    return events


def operating_curve(events, ground_truth_count: int, image_count: int):
    if ground_truth_count < 0 or image_count <= 0:
        raise ValueError("ground truth must be non-negative and image count positive")
    curve = [(1.0, 0.0, 0.0)]
    tp = fp = 0
    ordered = sorted(events, key=lambda item: -item[0])
    index = 0
    while index < len(ordered):
        score = ordered[index][0]
        while index < len(ordered) and ordered[index][0] == score:
            tp += ordered[index][1]
            fp += ordered[index][2]
            index += 1
        curve.append((score, tp / ground_truth_count if ground_truth_count else math.nan, fp / image_count))
    return curve


def sensitivity_at_fp(curve, fp_budget: float) -> float:
    values = [row[1] for row in curve if row[2] <= fp_budget]
    return max(values) if values else 0.0


def fp_at_sensitivity(curve, target: float) -> float:
    values = [row[2] for row in curve if row[1] >= target]
    return min(values) if values else math.nan
