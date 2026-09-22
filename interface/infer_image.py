"""Normalize one detector output into the public image-level schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from algorithm.detector.detection_output import parse_detections


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON with image_id and detector arrays")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    detections = parse_detections(payload["image_id"], payload["detections"])
    result = {
        "image_id": payload["image_id"],
        "detections": [
            {"label": item.label, "class": item.class_name, "score": item.score, "box": list(item.box)}
            for item in detections
        ],
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
