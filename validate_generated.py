# Author: Alireza Kabgani <a.kabgani@gmail.com>
# SPDX-License-Identifier: MIT
"""Validate a generated HiFedProx paper run matrix and protocol fields."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_results import load_runs, validate_paper_matrix
from hifedprox import method_name

NUMERIC_METRICS = (
    "test_loss",
    "test_accuracy",
    "writer_accuracy_q10",
    "tail_ratio_orderstat_mean",
    "armijo_trial_evals",
    "grad_evals",
    "failed_linesearch",
    "zero_step_fraction",
    "update_norm_unmodified_mean",
)


def validate_protocol(runs: pd.DataFrame) -> dict[str, object]:
    """Check the paper matrix, protocol fields, and numerical sanity."""
    validate_paper_matrix(runs)

    required_protocol = {
        "rounds_completed",
        "unstable",
        "participation",
        "base_local_steps",
        "initial_step_size",
        "batch_size",
        "mu",
        "reference_radius",
        "backtracking_factor",
        "armijo_c",
        "max_backtracks",
    }
    missing = required_protocol.difference(runs.columns)
    if missing:
        raise ValueError(f"generated run file is missing protocol columns: {sorted(missing)}")

    expected_methods = runs["power"].map(method_name)
    if not expected_methods.equals(runs["method"].astype(str)):
        raise ValueError("method names do not agree with the recorded powers")

    exact_checks = {
        "rounds_completed": 100,
        "unstable": 0,
        "batch_size": 16,
        "max_backtracks": 12,
    }
    for column, expected in exact_checks.items():
        values = set(runs[column].astype(int).tolist())
        if values != {expected}:
            raise ValueError(f"{column} differs from the paper protocol: found {sorted(values)}")

    float_checks = {
        "participation": 0.3,
        "base_local_steps": 3.0,
        "initial_step_size": 0.5,
        "mu": 0.03,
        "reference_radius": 1.0,
        "backtracking_factor": 0.5,
        "armijo_c": 1e-4,
    }
    for column, expected in float_checks.items():
        values = runs[column].to_numpy(dtype=float)
        if not np.allclose(values, expected, rtol=0.0, atol=1e-12):
            raise ValueError(f"{column} differs from the paper protocol")

    for metric in NUMERIC_METRICS:
        values = runs[metric].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite values found in {metric}")

    stress_values = runs.loc[runs["condition"] != "clean", "update_norm_stress_test_mean"].to_numpy(dtype=float)
    if not np.isfinite(stress_values).all():
        raise ValueError("non-finite stressed-client displacement means found")

    return {
        "rows": int(len(runs)),
        "complete_matrix": True,
        "powers": sorted(int(value) for value in runs["power"].unique()),
        "conditions": sorted(runs["condition"].unique().tolist()),
        "seeds": sorted(int(value) for value in runs["seed"].unique()),
        "failed_line_searches": int(runs["failed_linesearch"].sum()),
        "maximum_zero_step_fraction": float(runs["zero_step_fraction"].max()),
        "all_metrics_finite": True,
        "paper_protocol_fields_match": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    if not args.runs.is_file():
        parser.error(f"run file not found: {args.runs}")

    runs = load_runs(args.runs)
    report = validate_protocol(runs)
    print(json.dumps(report, indent=2))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"validation report written to {args.out}")


if __name__ == "__main__":
    main()
