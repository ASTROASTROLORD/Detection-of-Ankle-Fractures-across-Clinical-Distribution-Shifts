"""Unified patient site summaries and five reference-lesion analysis outcomes."""

from __future__ import annotations

from collections import Counter
from numbers import Integral
from typing import Iterable, Sequence

import numpy as np

from algorithm.detector.detection_output import CLASS_NAMES, Detection, parse_detections
from .category_specific_froc import box_iou


OUTCOMES = (
    "no_effective_candidate",
    "localization_insufficient",
    "wrong_class",
    "correct_below_frozen_threshold",
    "correct_frozen_threshold_met",
)
FROZEN_CLASS_THRESHOLDS = (0.63, 0.76, 0.79, 0.68)
SCORE_FLOOR = 0.05
CANDIDATE_IOU = 0.25
LOCALIZATION_IOU = 0.50
MAX_PREDICTIONS_PER_IMAGE = 100


def _label(value) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= value < len(CLASS_NAMES):
        raise ValueError("label must be an integer between 0 and 3")
    return int(value)


def _thresholds(values: Sequence[float]) -> tuple[float, ...]:
    if len(values) != len(CLASS_NAMES):
        raise ValueError("exactly four class thresholds are required")
    values = tuple(float(value) for value in values)
    if any(not np.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise ValueError("class thresholds must be finite and in [0, 1]")
    return values


def decompose_image(
    image_id: str,
    predictions: Iterable[Detection],
    ground_truth: Iterable[dict],
    thresholds: Sequence[float] = FROZEN_CLASS_THRESHOLDS,
) -> list[dict]:
    """Assign one outcome to each ``{gt_id, label, box}`` reference lesion.

    Match score-ordered candidates to the best available reference, irrespective
    of class, at IoU >= 0.50. Class-grouped stable sorting resolves score ties.
    Thresholds are applied only after matching and correct fracture-site typing.
    """
    if not isinstance(image_id, str) or not image_id.strip():
        raise ValueError("image_id must be a non-empty string")
    thresholds = _thresholds(thresholds)
    predictions = list(predictions)
    if any(prediction.image_id != image_id for prediction in predictions):
        raise ValueError("all predictions must belong to this image")
    labels = [_label(prediction.label) for prediction in predictions]
    validated = parse_detections(image_id, {
        "bboxes": [prediction.box for prediction in predictions],
        "scores": [prediction.score for prediction in predictions],
        "labels": labels,
    })
    references = []
    seen = set()
    for reference in ground_truth:
        gt_id = reference["gt_id"]
        if isinstance(gt_id, bool) or not isinstance(gt_id, (str, int)) or not str(gt_id).strip():
            raise ValueError("gt_id must be a non-empty string or integer")
        if str(gt_id) in seen:
            raise ValueError("duplicate gt_id within image")
        seen.add(str(gt_id))
        label = _label(reference["label"])
        validated_gt = parse_detections(image_id, {
            "bboxes": [reference["box"]], "scores": [1.0], "labels": [label],
        })[0]
        references.append({"gt_id": gt_id, "label": label, "box": validated_gt.box})
    references.sort(key=lambda reference: reference["label"])
    # Final FROC groups by class before stable score sorting. Preserve the
    # original input index so a matching decision remains traceable.
    ordered = sorted(enumerate(validated), key=lambda pair: pair[1].label)
    ordered.sort(key=lambda pair: -pair[1].score)
    ordered = [(index, pred) for index, pred in ordered[:MAX_PREDICTIONS_PER_IMAGE]
               if pred.score >= SCORE_FLOOR]
    gt_boxes = np.asarray([ref["box"] for ref in references], dtype=float).reshape(-1, 4)
    available = np.ones(len(references), dtype=bool)
    best_iou = np.zeros(len(references), dtype=float)
    matches = {}
    for rank, (prediction_index, prediction) in enumerate(ordered, start=1):
        overlaps = box_iou(prediction.box, gt_boxes)
        best_iou = np.maximum(best_iou, overlaps)
        eligible = np.flatnonzero(available)
        if len(eligible):
            chosen = int(eligible[np.argmax(overlaps[eligible])])
            if overlaps[chosen] >= LOCALIZATION_IOU:
                available[chosen] = False
                matches[chosen] = (prediction_index, prediction, rank, float(overlaps[chosen]))
    rows = []
    for index, reference in enumerate(references):
        match = matches.get(index)
        if match is None:
            outcome = OUTCOMES[0] if best_iou[index] < CANDIDATE_IOU else OUTCOMES[1]
        elif match[1].label != reference["label"]:
            outcome = OUTCOMES[2]
        elif match[1].score < thresholds[reference["label"]]:
            outcome = OUTCOMES[3]
        else:
            outcome = OUTCOMES[4]
        rows.append({
            "image_id": image_id,
            "gt_id": reference["gt_id"],
            "gt_class": CLASS_NAMES[reference["label"]],
            "gt_box": list(reference["box"]),
            "matched": match is not None,
            "prediction_index": match[0] if match else None,
            "prediction_class": CLASS_NAMES[match[1].label] if match else None,
            "prediction_score": match[1].score if match else None,
            "prediction_rank": match[2] if match else None,
            "prediction_box": list(match[1].box) if match else None,
            "matched_iou": match[3] if match else None,
            "best_any_iou": float(best_iou[index]),
            "outcome": outcome,
        })
    return rows


def summarize_outcomes(rows: Iterable[dict]) -> dict:
    """Count outcomes using all reference lesions as the denominator."""
    rows = list(rows)

    def count(selected):
        counts = Counter(row["outcome"] for row in selected)
        if any(outcome not in OUTCOMES for outcome in counts):
            raise ValueError("unknown lesion outcome")
        total = len(selected)
        return {
            "reference_lesions": total,
            "counts": {outcome: counts[outcome] for outcome in OUTCOMES},
            "proportions": {outcome: counts[outcome] / total if total else None for outcome in OUTCOMES},
        }

    scenes = sorted({row.get("scene", "unspecified") for row in rows})
    return {"all": count(rows), "by_scene": {
        scene: count([row for row in rows if row.get("scene", "unspecified") == scene]) for scene in scenes
    }}


def analyze_lesion_states(
    images: Iterable[dict], thresholds: Sequence[float] = FROZEN_CLASS_THRESHOLDS,
) -> dict:
    """Analyze JSON-compatible per-image records; never match across images."""
    thresholds = _thresholds(thresholds)
    rows = []
    seen = set()
    scenes = set()
    for image in images:
        image_id = image["image_id"]
        if not isinstance(image_id, str) or not image_id.strip():
            raise ValueError("image_id must be a non-empty string")
        if image_id in seen:
            raise ValueError("duplicate image_id")
        seen.add(image_id)
        scene = image.get("scene", "unspecified")
        if not isinstance(scene, str) or not scene.strip():
            raise ValueError("scene must be a non-empty string")
        scenes.add(scene)
        detections = [Detection(image_id, _label(item["label"]), item["score"], tuple(item["box"]))
                      for item in image["detections"]]
        image_rows = decompose_image(image_id, detections, image["ground_truth"], thresholds)
        rows.extend({"scene": scene, **row} for row in image_rows)
    summary = summarize_outcomes(rows)
    summary["by_scene"] = {
        scene: summary["by_scene"].get(scene, summarize_outcomes([])["all"])
        for scene in sorted(scenes)
    }
    return {
        "method": {
            "unit": "reference lesion",
            "denominator": "all reference lesions in the selected images",
            "matching": "per-image, class-agnostic, descending-score, stable, one-to-one",
            "score_floor": SCORE_FLOOR,
            "candidate_iou": CANDIDATE_IOU,
            "localization_iou": LOCALIZATION_IOU,
            "max_predictions_per_image": MAX_PREDICTIONS_PER_IMAGE,
            "class_thresholds": dict(zip(CLASS_NAMES, thresholds)),
            "images": len(seen),
        },
        "summary": summary,
        "lesions": rows,
    }


PATIENT_STATE_PRIORITY = (
    "distal_tibial_fracture",
    "posterior_malleolus_fracture",
    "medial_malleolus_fracture",
    "lateral_malleolus_fracture",
)
# Compatibility name used by the original public site-summary API.
STATE_PRIORITY = PATIENT_STATE_PRIORITY


def assign_patient_state(class_scores: dict[str, float], thresholds: dict[str, float]) -> str:
    """Assign one patient site state; this does not classify lesion errors."""
    for values in (class_scores, thresholds):
        if any(name not in CLASS_NAMES for name in values):
            raise ValueError("unknown fracture-site name")
        if any(not np.isfinite(float(value)) or not 0 <= float(value) <= 1 for value in values.values()):
            raise ValueError("site scores and thresholds must be finite and in [0, 1]")
    positives = [name for name in PATIENT_STATE_PRIORITY
                 if name in thresholds and float(class_scores.get(name, 0.0)) >= float(thresholds[name])]
    return positives[0] if positives else "no_core_fracture"


def assign_lesion_state(class_scores: dict[str, float], thresholds: dict[str, float]) -> str:
    """Compatibility alias for the original patient site-summary function."""
    return assign_patient_state(class_scores, thresholds)


def summarize_states(records, thresholds: dict[str, float]) -> dict[str, int]:
    """Summarize patient site states using the original public API."""
    counts = {name: 0 for name in (*PATIENT_STATE_PRIORITY, "no_core_fracture")}
    for record in records:
        scores = dict(zip(CLASS_NAMES, record.scores)) if hasattr(record, "scores") else record
        counts[assign_patient_state(scores, thresholds)] += 1
    return counts


def summarize_diagnosis(payload: dict, thresholds: Sequence[float] = FROZEN_CLASS_THRESHOLDS) -> dict:
    """Unified entry point for reference-lesion analysis and patient site summaries.

    An images input produces lesion analysis. Patient scores, when explicitly
    supplied, add the original site summary to that same result. A patient-only
    input preserves the existing patient output and its lesion_state field.
    """
    thresholds = _thresholds(thresholds)
    score_map = payload.get("class_scores")
    if score_map is None and "layers" in payload:
        values = payload["layers"]["layer_1_patient_fracture_recognition"]["scores"]
        if len(values) != len(CLASS_NAMES):
            raise ValueError("exactly four patient scores are required")
        score_map = dict(zip(CLASS_NAMES, values))
    if "images" in payload:
        result = analyze_lesion_states(payload["images"], thresholds)
    elif score_map is not None:
        result = dict(payload)
    else:
        raise ValueError("input must contain images or patient class_scores/layers")
    if score_map is not None:
        if set(score_map) != set(CLASS_NAMES):
            raise ValueError("patient class_scores must contain exactly the four fracture sites")
        state = assign_patient_state(score_map, dict(zip(CLASS_NAMES, thresholds)))
        result["patient_site_summary"] = {
            "unit": "patient",
            "state": state,
            "class_scores": {name: float(score_map[name]) for name in CLASS_NAMES},
            "class_thresholds": dict(zip(CLASS_NAMES, thresholds)),
            "priority_order": list(PATIENT_STATE_PRIORITY),
        }
        if "images" not in payload:
            result["lesion_state"] = state
        if "patient_id" in payload:
            result["patient_id"] = payload["patient_id"]
    return result
