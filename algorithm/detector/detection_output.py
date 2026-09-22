"""Validated detector-output records used by the diagnostic layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

CLASS_NAMES = (
    "lateral_malleolus_fracture",
    "medial_malleolus_fracture",
    "posterior_malleolus_fracture",
    "distal_tibial_fracture",
)


@dataclass(frozen=True)
class Detection:
    image_id: str
    label: int
    score: float
    box: tuple[float, float, float, float]

    @property
    def class_name(self) -> str:
        return CLASS_NAMES[self.label]


def _array(value: Any, dtype: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=dtype)


def parse_detections(image_id: str, output: dict[str, Any]) -> list[Detection]:
    """Convert an MMDetection-style output dictionary into validated records."""
    boxes = _array(output.get("bboxes", []), float).reshape(-1, 4)
    scores = _array(output.get("scores", []), float).reshape(-1)
    labels = _array(output.get("labels", []), int).reshape(-1)
    if not (len(boxes) == len(scores) == len(labels)):
        raise ValueError("bboxes, scores, and labels must have equal lengths")
    detections = []
    for box, score, label in zip(boxes, scores, labels):
        if not 0 <= int(label) < len(CLASS_NAMES):
            raise ValueError(f"detector label out of range: {label}")
        if not np.isfinite(score) or not 0 <= float(score) <= 1:
            raise ValueError(f"detector score must be in [0, 1]: {score}")
        if not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError(f"invalid detection box: {box.tolist()}")
        detections.append(
            Detection(str(image_id), int(label), float(score), tuple(map(float, box)))
        )
    return detections


def group_by_image(detections: Iterable[Detection]) -> dict[str, list[Detection]]:
    grouped: dict[str, list[Detection]] = {}
    for detection in detections:
        grouped.setdefault(detection.image_id, []).append(detection)
    return grouped
