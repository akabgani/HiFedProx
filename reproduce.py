# Author: Alireza Kabgani <a.kabgani@gmail.com>
# SPDX-License-Identifier: MIT
"""Run the HiFedProx checks or reproduce the complete paper experiment matrix."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="run all 105 HiFedProx exponent-study experiments and generate supported tables and figures",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume an interrupted full run without rerunning completed records",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "generated_output",
        help="directory created for newly generated run records, tables, figures, and reports",
    )
    args = parser.parse_args()

    python = sys.executable
    run([python, "test_code.py"])

    if not args.full:
        run([python, "smoke_test.py"])
        print("\nQuick validation completed. Use `python reproduce.py --full` for all 105 paper runs.")
        return

    # Never recursively delete a user-selected output directory.
    # The paper command uses --fresh to recreate only paper_runs.csv.
    args.out.mkdir(parents=True, exist_ok=True)

    run_csv = args.out / "runs" / "paper_runs.csv"
    analysis_dir = args.out / "analysis"
    validation_json = args.out / "validation_report.json"

    command = [python, "run_experiments.py", "paper", "--out", str(run_csv)]
    if not args.resume:
        command.append("--fresh")
    run(command)
    run([python, "validate_generated.py", "--runs", str(run_csv), "--out", str(validation_json)])
    run([python, "analyze_results.py", "--runs", str(run_csv), "--out", str(analysis_dir)])

    print("\nFull reproduction completed.")
    print(f"Run matrix: {run_csv}")
    print(f"Tables and figures: {analysis_dir}")
    print(f"Validation report: {validation_json}")


if __name__ == "__main__":
    main()
