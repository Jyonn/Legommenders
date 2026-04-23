#!/usr/bin/env python3
"""Export global Operator/Predictor metadata from source code.

The exporter is intentionally AST-based: it does not import torch, construct
models, or execute project modules.  It infers the boring fields automatically,
then merges semantic docstring annotations and optional human overrides.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.operator_metadata import export_metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "operator_metadata.json",
        help="JSON file to write.",
    )
    parser.add_argument(
        "--overrides",
        type=Path,
        default=ROOT / "tools" / "operator_metadata_overrides.json",
        help="Optional human supplement file for constraints/descriptions.",
    )
    args = parser.parse_args()

    payload = export_metadata(ROOT, overrides_path=args.overrides)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Exported {len(payload['operators'])} operators and {len(payload['predictors'])} predictors to {args.output}")


if __name__ == "__main__":
    main()
