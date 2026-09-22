"""Five-state diagnostic summary; this is not a detector output head."""

from __future__ import annotations

from algorithm.detector.detection_output import CLASS_NAMES

STATE_PRIORITY = (
    "distal_tibial_fracture",
    "posterior_malleolus_fracture",
    "medial_malleolus_fracture",
    "lateral_malleolus_fracture",
)


def assign_lesion_state(class_scores: dict[str, float], thresholds: dict[str, float]) -> str:
    positives = [name for name in STATE_PRIORITY if class_scores.get(name, 0.0) >= thresholds.get(name, 1.0)]
    return positives[0] if positives else "no_core_fracture"


def summarize_states(records, thresholds: dict[str, float]) -> dict[str, int]:
    counts = {name: 0 for name in (*STATE_PRIORITY, "no_core_fracture")}
    for record in records:
        scores = dict(zip(CLASS_NAMES, record.scores)) if hasattr(record, "scores") else record
        state = assign_lesion_state(scores, thresholds)
        counts[state] += 1
    return counts
