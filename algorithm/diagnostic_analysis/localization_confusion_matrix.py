"""Post-localization predicted-versus-reference fracture-site confusion."""

from __future__ import annotations

from collections import Counter

import numpy as np

from algorithm.detector.detection_output import CLASS_NAMES, Detection
from .category_specific_froc import box_iou


def localization_confusion_matrix(predictions: list[Detection], ground_truth: dict[int, list[dict]], iou_threshold: float = 0.5) -> dict:
    rows = Counter()
    references = [(label, item) for label, items in ground_truth.items() for item in items]
    used: set[int] = set()
    for prediction in sorted(predictions, key=lambda item: -item.score):
        if not references:
            rows[("background", prediction.class_name)] += 1
            continue
        overlaps = [float(box_iou(prediction.box, np.asarray([item[1]["box"]]))[0]) for item in references]
        index = int(np.argmax(overlaps))
        if overlaps[index] < iou_threshold or index in used:
            rows[("background", prediction.class_name)] += 1
            continue
        used.add(index)
        rows[(CLASS_NAMES[references[index][0]], prediction.class_name)] += 1
    for index, (label, _) in enumerate(references):
        if index not in used:
            rows[(CLASS_NAMES[label], "missed")] += 1
    labels = list(CLASS_NAMES) + ["background", "missed"]
    return {
        "iou_threshold": iou_threshold,
        "labels": labels,
        "counts": {
            reference: {prediction: rows[(reference, prediction)] for prediction in labels}
            for reference in list(CLASS_NAMES) + ["background"]
        },
    }
