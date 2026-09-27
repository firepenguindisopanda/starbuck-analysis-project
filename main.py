"""
Starbucks Customer Segmentation & Offer Recommendation - Pipeline Entry Point.

Runs the full end-to-end pipeline: data validation, EDA, feature engineering,
segmentation, causal inference and recommendation, then the send-time offer
response model. Expects the raw Udacity files in data/raw/ (see README, Data).
"""

import os
import sys
import subprocess
from pathlib import Path

PHASES = [
    ("Phase 1: Data Ingestion & Validation", "src/data/load_data.py"),
    ("Phase 2: Exploratory Data Analysis", "src/data/eda.py"),
    ("Phase 3: Feature Engineering", "src/data/feature_engineering.py"),
    ("Phase 4: Customer Segmentation", "src/models/clustering.py"),
    ("Phase 5: Causal Inference & Recommendation", "src/models/recommendation.py"),
    ("Phase 6: Send-Time Offer Response Model", "src/models/offer_response.py"),
]


def main():
    root = Path(__file__).parent
    if not (root / "data" / "raw" / "transcript.json").exists():
        print("Raw data not found in data/raw/. See README, Data, for where to get it.")
        sys.exit(1)

    print("=" * 60)
    print("  STARBUCKS OFFER OPTIMIZATION - FULL PIPELINE")
    print("=" * 60)

    for phase_name, script in PHASES:
        script_path = root / script
        if not script_path.exists():
            print(f"\n  Skipping {phase_name} - {script} not found")
            continue

        print(f"\n{'-' * 60}")
        print(f"  {phase_name}")
        print(f"{'-' * 60}")

        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=root,
            # The phases print symbols such as the greater-than-or-equal sign;
            # without this, a redirected Windows console falls back to cp1252.
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )

        if result.returncode != 0:
            print(f"\n  {phase_name} FAILED (exit code {result.returncode})")
            sys.exit(result.returncode)

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE - all phases passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
