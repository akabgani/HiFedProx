# Author: Alireza Kabgani <a.kabgani@gmail.com>
# SPDX-License-Identifier: MIT
"""Run a short execution check on the packaged data and implementation."""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import pandas as pd

from hifedprox import load_cache
from run_experiments import make_config, run_one

ROOT = Path(__file__).resolve().parent


def main() -> None:
    data = load_cache(str(ROOT / "data" / "femnist_evaluation.npz"))
    with tempfile.TemporaryDirectory(prefix="hifedprox_smoke_") as directory:
        output = Path(directory) / "smoke_runs.csv"
        p2 = run_one(data, output, make_config(2.0, "clean", 3001, rounds=1))
        p8 = run_one(data, output, make_config(8.0, "clean", 3001, rounds=1))
        fractional = run_one(data, output, make_config(2.5, "moderate_stress", 3001, rounds=1))

        frame = pd.read_csv(output)
        if len(frame) != 3:
            raise AssertionError(f"expected three smoke-test rows, found {len(frame)}")
        for record in (p2, p8, fractional):
            for metric in ("test_loss", "test_accuracy", "tail_ratio_orderstat_mean"):
                if not math.isfinite(float(record[metric])):
                    raise AssertionError(f"non-finite smoke-test metric: {metric}")
        if int(p2["grad_evals"]) != int(p8["grad_evals"]):
            raise AssertionError("paired powers used different gradient budgets")

    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
