"""Run the four-layer patient-level algorithm over a JSON patient record."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from algorithm.detector.detection_output import Detection
from algorithm.patient_level.patient_aggregation import aggregate_patient
from algorithm.patient_level.pipeline import run_four_layer_diagnosis


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--threshold", type=float, nargs=4, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    detections = [
        Detection(str(item["image_id"]), int(item["label"]), float(item["score"]), tuple(item["box"]))
        for item in payload.get("detections", [])
    ]
    record = aggregate_patient(
        payload["patient_id"],
        payload["image_ids"],
        payload.get("target_labels_by_image", {}),
        detections,
    )
    ground_truth = payload.get("ground_truth_by_class")
    if ground_truth is not None:
        ground_truth = {int(label): value for label, value in ground_truth.items()}
    result = run_four_layer_diagnosis(record, args.threshold, ground_truth)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
