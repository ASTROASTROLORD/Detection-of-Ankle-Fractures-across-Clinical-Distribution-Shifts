"""Patient-level aggregation of image annotations and detector outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

from algorithm.detector.detection_output import CLASS_NAMES, Detection


@dataclass
class PatientRecord:
    patient_id: str
    image_ids: tuple[str, ...]
    targets: np.ndarray
    scores: np.ndarray
    detections: tuple[Detection, ...]


def aggregate_patient(
    patient_id: str,
    image_ids: Iterable[str],
    target_labels_by_image: Mapping[str, Iterable[int]],
    detections: Iterable[Detection],
) -> PatientRecord:
    image_ids = tuple(str(value) for value in image_ids)
    image_set = set(image_ids)
    if not patient_id or not image_ids:
        raise ValueError("patient_id and at least one image are required")
    detections = tuple(detections)
    if any(item.image_id not in image_set for item in detections):
        raise ValueError("a detection references an image outside the patient")
    targets = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    for image_id in image_ids:
        for label in target_labels_by_image.get(image_id, ()):
            if not 0 <= int(label) < len(CLASS_NAMES):
                raise ValueError(f"target label out of range: {label}")
            targets[int(label)] = 1
    scores = np.zeros(len(CLASS_NAMES), dtype=float)
    for detection in detections:
        scores[detection.label] = max(scores[detection.label], detection.score)
    return PatientRecord(patient_id, tuple(sorted(image_set)), targets, scores, detections)


def aggregate_patients(rows: Iterable[dict]) -> list[PatientRecord]:
    records = []
    for row in rows:
        records.append(
            aggregate_patient(
                row["patient_id"],
                row["image_ids"],
                row.get("target_labels_by_image", {}),
                row.get("detections", ()),
            )
        )
    return records
