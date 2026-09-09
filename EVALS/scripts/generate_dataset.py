"""
Regenerate the synthetic evaluation dataset (examples.json) and its simulated human labels
(human_labels.json).

This is a thin CLI wrapper around src.dataset.generate_dataset() / generate_human_labels() —
the README documents `python scripts/generate_dataset.py` as the setup step, so this script
needs to exist for that instruction to actually work.

NOTE: this only ever produces SYNTHETIC data. It has nothing to do with the real human
calibration labels (Phase 2 of the PDF spec) — those come only from a person using the
Streamlit labeling app (dashboards/label_app.py) against the real Postgres `human_labels`
table. Re-running this script does not create or replace any real human label.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dataset import generate_dataset, generate_human_labels

_DATASET_PATH = Path(__file__).resolve().parent.parent / "datasets" / "examples.json"
_LABELS_PATH = Path(__file__).resolve().parent.parent / "datasets" / "human_labels.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=200, help="Number of synthetic examples to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (for reproducibility)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing dataset files")
    args = parser.parse_args()

    if _DATASET_PATH.exists() and not args.force:
        print(f"[skip] {_DATASET_PATH} already exists. Pass --force to regenerate.")
        return

    print(f"Generating {args.n} synthetic examples (seed={args.seed})...")
    examples = generate_dataset(n=args.n, seed=args.seed)
    _DATASET_PATH.write_text(json.dumps(examples, indent=2))
    print(f"Wrote {len(examples)} examples to {_DATASET_PATH}")

    print("Generating simulated human labels (SYNTHETIC — not real human judgments)...")
    labels = generate_human_labels(examples, seed=args.seed)
    _LABELS_PATH.write_text(json.dumps(labels, indent=2))
    print(f"Wrote {len(labels)} label rows to {_LABELS_PATH}")
    print(
        "\nReminder: these labels are simulated (expected_scores +/- noise), not real people. "
        "For real calibration data, use dashboards/label_app.py against the live Postgres instance."
    )


if __name__ == "__main__":
    main()
