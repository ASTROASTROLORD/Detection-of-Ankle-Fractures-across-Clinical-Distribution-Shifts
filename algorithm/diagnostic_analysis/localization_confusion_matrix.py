"""Post-localization predicted-versus-reference fracture-site confusion."""

from __future__ import annotations

from collections import Counter

from algorithm.detector.detection_output import CLASS_NAMES, Detection
from .five_lesion_states import decompose_image, SCORE_FLOOR, MAX_PREDICTIONS_PER_IMAGE


def localization_confusion_matrix(predictions: list[Detection], ground_truth: dict[int, list[dict]], iou_threshold: float = 0.5) -> dict:
    """Summarize one image using the canonical class-agnostic lesion matches."""
    predictions = list(predictions)
    image_ids = {prediction.image_id for prediction in predictions}
    if len(image_ids) > 1:
        raise ValueError("confusion matrix inputs must belong to one image")
    image_id = next(iter(image_ids), "reference-only-image")
    references = [
        {"gt_id": f"{label}-{index}", "label": label, "box": item["box"]}
        for label, items in ground_truth.items()
        for index, item in enumerate(items)
    ]
    lesions = decompose_image(image_id, predictions, references,
                              localization_iou=iou_threshold)
    rows = Counter()
    used = set()
    for lesion in lesions:
        if lesion["matched"]:
            used.add(lesion["prediction_index"])
            rows[(lesion["gt_class"], lesion["prediction_class"])] += 1
        else:
            rows[(lesion["gt_class"], "missed")] += 1
    ordered = sorted(enumerate(predictions), key=lambda pair: pair[1].label)
    ordered.sort(key=lambda pair: -pair[1].score)
    for index, prediction in ordered[:MAX_PREDICTIONS_PER_IMAGE]:
        if prediction.score >= SCORE_FLOOR and index not in used:
            rows[("background", prediction.class_name)] += 1
    labels = list(CLASS_NAMES) + ["background", "missed"]
    return {
        "iou_threshold": iou_threshold,
        "labels": labels,
        "counts": {
            reference: {prediction: rows[(reference, prediction)] for prediction in labels}
            for reference in list(CLASS_NAMES) + ["background"]
        },
    }
