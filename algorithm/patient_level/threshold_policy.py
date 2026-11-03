"""Frozen patient-level threshold and binary metric definitions."""

from __future__ import annotations

import numpy as np


def binary_stats(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float | int]:
    targets = np.asarray(targets, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if targets.shape != predictions.shape:
        raise ValueError("targets and predictions must have the same shape")
    tp = int(((targets == 1) & (predictions == 1)).sum())
    tn = int(((targets == 0) & (predictions == 0)).sum())
    fp = int(((targets == 0) & (predictions == 1)).sum())
    fn = int(((targets == 1) & (predictions == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "precision": precision,
            "recall": recall, "specificity": specificity, "f1": f1}


def select_thresholds(targets: np.ndarray, scores: np.ndarray, grid=None) -> list[float]:
    targets = np.asarray(targets, dtype=np.int64)
    scores = np.asarray(scores, dtype=float)
    if targets.shape != scores.shape or targets.ndim != 2:
        raise ValueError("targets and scores must be equally shaped matrices")
    grid = np.asarray(grid if grid is not None else np.arange(5, 96) / 100.0)
    selected = []
    for column in range(targets.shape[1]):
        candidates = [(binary_stats(targets[:, column], scores[:, column] >= threshold)["f1"],
                      -abs(float(threshold) - 0.5), float(threshold)) for threshold in grid]
        selected.append(max(candidates)[2])
    return selected


def patient_scores(records) -> tuple[np.ndarray, np.ndarray]:
    if not records:
        raise ValueError("at least one patient record is required")
    return (np.stack([row.targets for row in records]), np.stack([row.scores for row in records]))
