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
    if args.input.resolve() == args.output.resolve():
        parser.error("input and output must be different files")
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    image_ids = payload.get("image_ids")
    detection_items = payload.get("detections", [])
    ground_truth_by_image = payload.get("ground_truth_by_image")
    if "images" in payload:
        if image_ids is not None or "detections" in payload:
            raise ValueError("provide images or image_ids/detections, not both")
        images = payload["images"]
        image_ids = [image["image_id"] for image in images]
        if len(set(image_ids)) != len(image_ids):
            raise ValueError("duplicate image_id")
        detection_items = [{**item, "image_id": image["image_id"]}
                           for image in images for item in image["detections"]]
        if any("ground_truth" in image for image in images):
            if ground_truth_by_image is not None:
                raise ValueError("declare references inside images or ground_truth_by_image, not both")
            if not all("ground_truth" in image for image in images):
                raise ValueError("every image must declare ground_truth, including empty references")
            ground_truth_by_image = {image["image_id"]: image["ground_truth"] for image in images}
    detections = [
        Detection(str(item["image_id"]), item["label"], float(item["score"]), tuple(item["box"]))
        for item in detection_items
    ]
    targets = payload.get("target_labels_by_image")
    if targets is None:
        targets = {image_id: sorted({item["label"] for item in references})
                   for image_id, references in (ground_truth_by_image or {}).items()}
    record = aggregate_patient(
        payload["patient_id"],
        image_ids,
        targets,
        detections,
    )
    ground_truth = payload.get("ground_truth_by_class")
    if ground_truth is not None:
        ground_truth = {int(label): value for label, value in ground_truth.items()}
    result = run_four_layer_diagnosis(record, args.threshold, ground_truth,
                                      ground_truth_by_image)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
