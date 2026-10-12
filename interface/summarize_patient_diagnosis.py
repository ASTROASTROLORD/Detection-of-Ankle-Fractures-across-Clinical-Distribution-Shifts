"""Summarize patient fracture sites and annotated five-outcome lesion analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from algorithm.diagnostic_analysis.five_lesion_states import summarize_diagnosis


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="patient output or JSON containing annotated images")
    parser.add_argument("output", type=Path, help="JSON diagnosis summary and optional lesion analysis")
    parser.add_argument("--threshold", type=float, nargs=4, required=True)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("input and output must be different files")
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    result = summarize_diagnosis(payload, args.threshold)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
