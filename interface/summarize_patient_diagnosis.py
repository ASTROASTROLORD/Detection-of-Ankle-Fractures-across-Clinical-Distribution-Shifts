"""Assign the configured five-state diagnostic summary to patient output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from algorithm.diagnostic_analysis.five_lesion_states import assign_lesion_state
from algorithm.detector.detection_output import CLASS_NAMES


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--threshold", type=float, nargs=4, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    scores = payload.get("class_scores") or {
        name: value for name, value in zip(CLASS_NAMES, payload["layers"]["layer_1_patient_fracture_recognition"]["scores"])
    }
    thresholds = dict(zip(CLASS_NAMES, args.threshold))
    payload["lesion_state"] = assign_lesion_state(scores, thresholds)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
