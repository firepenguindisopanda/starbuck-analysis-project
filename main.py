"""
Starbucks Customer Segmentation & Offer Recommendation - Pipeline Entry Point.

Runs the full end-to-end pipeline: data ingestion to EDA to feature engineering
to clustering to predictive modeling to recommendation to reports.
"""

import sys
import subprocess
from pathlib import Path

PHASES = [
    ("Phase 1: Data Ingestion & Validation", "src/data/load_data.py"),
    ("Phase 2: Exploratory Data Analysis", "src/data/eda.py"),
    ("Phase 3: Feature Engineering", "src/data/feature_engineering.py"),
    ("Phase 4: Customer Segmentation", "src/models/clustering.py"),
    ("Phase 5: Predictive Modeling", "src/models/predictive_modeling.py"),
    ("Phase 6: Causal Inference & Recommendation", "src/models/recommendation.py"),
    ("Phase 7: Reports & Power BI Export", "src/reporting/generate_report.py"),
    ("Phase 8: Power BI Export", "src/reporting/export_powerbi.py"),
]


def main():
    root = Path(__file__).parent

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
        )

        if result.returncode != 0:
            print(f"\n  {phase_name} FAILED (exit code {result.returncode})")
            sys.exit(result.returncode)

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE - all phases passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
