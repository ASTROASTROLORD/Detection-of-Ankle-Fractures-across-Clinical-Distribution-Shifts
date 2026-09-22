"""Layer 1: patient-level fracture recognition."""

from __future__ import annotations

import numpy as np

from .patient_aggregation import PatientRecord
from .threshold_policy import binary_stats


def recognize_patient(record: PatientRecord, thresholds: list[float]) -> dict:
    if len(thresholds) != len(record.scores):
        raise ValueError("one threshold is required for each fracture class")
    predictions = record.scores >= np.asarray(thresholds)
    per_class = [
        binary_stats(np.asarray([record.targets[index]]), np.asarray([predictions[index]]))
        for index in range(len(predictions))
    ]
    return {
        "patient_id": record.patient_id,
        "scores": record.scores.tolist(),
        "predictions": predictions.astype(int).tolist(),
        "per_class": per_class,
    }
