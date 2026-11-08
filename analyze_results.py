
# Author: Alireza Kabgani <a.kabgani@gmail.com>
# SPDX-License-Identifier: MIT
"""Reconstruct the HiFedProx paper tables, figures, and headline comparisons."""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CONDITIONS = ("clean", "moderate_stress", "severe_stress")
CONDITION_LABELS = {
    "clean": "Clean",
    "moderate_stress": "Moderate stress",
    "severe_stress": "Severe stress",
}
PAPER_POWERS = tuple(range(2, 9))
PAPER_SEEDS = (3001, 3002, 3003, 3004, 3005)
REQUIRED_COLUMNS = {
    "method",
    "condition",
    "seed",
    "power",
    "test_loss",
    "test_accuracy",
    "writer_accuracy_q10",
    "tail_ratio_orderstat_mean",
    "armijo_trial_evals",
    "grad_evals",
    "failed_linesearch",
    "zero_step_fraction",
    "update_norm_unmodified_mean",
    "update_norm_stress_test_mean",
}


def load_runs(path: Path) -> pd.DataFrame:
    runs = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(runs.columns)
    if missing:
        raise ValueError(f"run file is missing columns: {sorted(missing)}")
    runs = runs.copy()
    runs["power"] = pd.to_numeric(runs["power"], errors="coerce")
    inferred_power = runs["method"].astype(str).str.extract(r"^p([0-9.]+)_bt$")[0].astype(float)
    runs["power"] = runs["power"].fillna(inferred_power)
    if runs["power"].isna().any():
        raise ValueError("could not infer the power for every run")
    runs["seed"] = runs["seed"].astype(int)
    if runs.duplicated(["power", "condition", "seed"]).any():
        raise ValueError("run file contains duplicate power-condition-seed rows")
    return runs


def validate_paper_matrix(runs: pd.DataFrame) -> None:
    expected = {
        (float(power), condition, seed)
        for power in PAPER_POWERS
        for condition in CONDITIONS
        for seed in PAPER_SEEDS
    }
    actual = set(zip(runs["power"], runs["condition"], runs["seed"]))
    missing = expected.difference(actual)
    extra = actual.difference(expected)
    if missing or extra:
        raise ValueError(f"incomplete paper matrix; missing={sorted(missing)}, extra={sorted(extra)}")
    if len(runs) != 105:
        raise ValueError(f"expected 105 runs, found {len(runs)}")
    for (condition, seed), block in runs.groupby(["condition", "seed"]):
        if block["grad_evals"].nunique() != 1:
            raise ValueError(f"gradient budgets differ across powers for {condition}, seed={seed}")

def exact_bootstrap_interval(values: np.ndarray) -> tuple[float, float]:
    """Exhaustive ordered bootstrap interval used in the paper (5^5 resamples)."""
    values = np.asarray(values, dtype=float)
    if values.shape != (5,):
        raise ValueError("the exhaustive paper bootstrap expects exactly five paired values")
    means = np.fromiter(
        (values[list(indices)].mean() for indices in itertools.product(range(5), repeat=5)),
        dtype=float,
        count=5**5,
    )
    return (
        float(np.quantile(means, 0.025, method="linear")),
        float(np.quantile(means, 0.975, method="linear")),
    )


def summarize_main(runs: pd.DataFrame) -> pd.DataFrame:
    grouped = runs.groupby(["power", "condition"])
    rows: list[dict[str, float]] = []
    for power in PAPER_POWERS:
        row: dict[str, float] = {"power": power}
        for condition in CONDITIONS:
            block = grouped.get_group((float(power), condition))
            prefix = condition.replace("_stress", "")
            row[f"{prefix}_loss_mean"] = float(block["test_loss"].mean())
            row[f"{prefix}_loss_sd"] = float(block["test_loss"].std(ddof=1))
            row[f"{prefix}_accuracy_mean_percent"] = 100.0 * float(block["test_accuracy"].mean())
            row[f"{prefix}_accuracy_sd_percent"] = 100.0 * float(block["test_accuracy"].std(ddof=1))
            row[f"{prefix}_writer_q10_mean_percent"] = 100.0 * float(block["writer_accuracy_q10"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_tail_cost(runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    baseline_trials = float(
        runs[(runs["power"] == 2.0) & (runs["condition"] == "severe_stress")]["armijo_trial_evals"].mean()
    )
    for power in PAPER_POWERS:
        row: dict[str, float] = {"power": power}
        for condition in CONDITIONS:
            block = runs[(runs["power"] == power) & (runs["condition"] == condition)]
            row[f"{condition}_tail_mean"] = float(block["tail_ratio_orderstat_mean"].mean())
            row[f"{condition}_tail_sd"] = float(block["tail_ratio_orderstat_mean"].std(ddof=1))
        severe = runs[(runs["power"] == power) & (runs["condition"] == "severe_stress")]
        trials = float(severe["armijo_trial_evals"].mean())
        row["severe_trial_evals_mean"] = trials
        row["trial_increase_percent_vs_p2"] = 100.0 * (trials - baseline_trials) / baseline_trials
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_groups(runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for condition in ("moderate_stress", "severe_stress"):
        for power in PAPER_POWERS:
            block = runs[(runs["power"] == power) & (runs["condition"] == condition)]
            for group, column in (
                ("Unmodified clients", "update_norm_unmodified_mean"),
                ("Stress-test clients", "update_norm_stress_test_mean"),
            ):
                rows.append(
                    {
                        "condition": condition,
                        "power": power,
                        "group": group,
                        "mean_displacement": float(block[column].mean()),
                        "sd_displacement": float(block[column].std(ddof=1)),
                    }
                )
    return pd.DataFrame(rows)


def paired_comparisons(runs: pd.DataFrame) -> pd.DataFrame:
    comparisons = [
        ("moderate_stress", 2, 7),
        ("severe_stress", 2, 6),
        ("severe_stress", 5, 6),
        ("moderate_stress", 5, 7),
        ("moderate_stress", 6, 7),
        ("severe_stress", 6, 7),
        ("severe_stress", 6, 8),
    ]
    rows = []
    for condition, reference_power, new_power in comparisons:
        block = runs[runs["condition"] == condition]
        pivot = block.pivot(index="seed", columns="power", values="test_loss")
        differences = (pivot[float(reference_power)] - pivot[float(new_power)]).to_numpy(dtype=float)
        low, high = exact_bootstrap_interval(differences)
        rows.append(
            {
                "condition": condition,
                "reference_power": reference_power,
                "new_power": new_power,
                "mean_loss_improvement": float(differences.mean()),
                "bootstrap_low": low,
                "bootstrap_high": high,
                "wins_new": int(np.sum(differences > 0.0)),
                "pairs": len(differences),
                "minimum_difference": float(differences.min()),
                "maximum_difference": float(differences.max()),
            }
        )
    return pd.DataFrame(rows)


def plot_scale_matched(output: Path) -> None:
    ratio = np.linspace(0.0, 1.4, 300)
    fig, axis = plt.subplots(figsize=(5.6, 3.7))
    for power in (2, 3, 5, 8):
        axis.plot(ratio, ratio ** (power - 1), marker="o", markevery=45, label=rf"$p={power}$")
    axis.scatter([1.0], [1.0], zorder=4)
    axis.annotate(r"Common calibration at $\|d\|=R$", xy=(1.0, 1.0), xytext=(0.38, 1.25), arrowprops={"arrowstyle": "->"})
    axis.set_xlabel(r"normalized displacement $\|d\|/R$")
    axis.set_ylabel(r"normalized gradient magnitude $\|\nabla\phi_{p,R,\mu}(d)\|/(\mu R)$")
    axis.set_xlim(0.0, 1.4)
    axis.set_ylim(0.0, 4.1)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_loss(runs: pd.DataFrame, output: Path) -> None:
    fig, axis = plt.subplots(figsize=(5.4, 3.6))
    for condition in CONDITIONS:
        block = runs[runs["condition"] == condition].groupby("power")["test_loss"]
        means = block.mean().reindex(PAPER_POWERS)
        stds = block.std(ddof=1).reindex(PAPER_POWERS)
        axis.errorbar(PAPER_POWERS, means, yerr=stds, marker="o", capsize=3, label=CONDITION_LABELS[condition])
    axis.set_xlabel(r"Power $p$")
    axis.set_ylabel("Final pooled loss")
    axis.set_xticks(PAPER_POWERS)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_tail(runs: pd.DataFrame, output: Path) -> None:
    fig, axis = plt.subplots(figsize=(5.4, 3.6))
    for condition in CONDITIONS:
        block = runs[runs["condition"] == condition].groupby("power")["tail_ratio_orderstat_mean"]
        means = block.mean().reindex(PAPER_POWERS)
        stds = block.std(ddof=1).reindex(PAPER_POWERS)
        axis.errorbar(PAPER_POWERS, means, yerr=stds, marker="o", capsize=3, label=CONDITION_LABELS[condition])
    axis.set_xlabel(r"Power $p$")
    axis.set_ylabel("Upper-tail displacement ratio")
    axis.set_xticks(PAPER_POWERS)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_groups(groups: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4), sharey=False)
    for axis, condition in zip(axes, ("moderate_stress", "severe_stress")):
        block = groups[groups["condition"] == condition]
        for group in ("Unmodified clients", "Stress-test clients"):
            group_block = block[block["group"] == group].sort_values("power")
            axis.errorbar(
                group_block["power"],
                group_block["mean_displacement"],
                yerr=group_block["sd_displacement"],
                marker="o",
                capsize=3,
                label=group,
            )
        axis.set_title(CONDITION_LABELS[condition])
        axis.set_xlabel(r"Power $p$")
        axis.set_xticks(PAPER_POWERS)
        axis.set_ylabel("Mean displacement norm")
        axis.legend()
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def write_report(
    runs: pd.DataFrame,
    main: pd.DataFrame,
    tail: pd.DataFrame,
    comparisons: pd.DataFrame,
    output: Path,
) -> None:
    moderate_p2 = main.loc[main["power"] == 2, "moderate_loss_mean"].item()
    severe_p2 = main.loc[main["power"] == 2, "severe_loss_mean"].item()
    moderate_best_row = main.loc[main["moderate_loss_mean"].idxmin()]
    severe_best_row = main.loc[main["severe_loss_mean"].idxmin()]
    moderate_best_power = int(moderate_best_row["power"])
    severe_best_power = int(severe_best_row["power"])
    moderate_best_loss = float(moderate_best_row["moderate_loss_mean"])
    severe_best_loss = float(severe_best_row["severe_loss_mean"])
    severe_p8_tail = tail.loc[tail["power"] == 8, "severe_stress_tail_mean"].item()
    moderate_improvement = 100.0 * (moderate_p2 - moderate_best_loss) / moderate_p2
    severe_improvement = 100.0 * (severe_p2 - severe_best_loss) / severe_p2
    p5_p6 = comparisons[
        (comparisons["condition"] == "severe_stress")
        & (comparisons["reference_power"] == 5)
        & (comparisons["new_power"] == 6)
    ].iloc[0]
    text = f"""# Generated HiFedProx paper results

- Run matrix: {len(runs)} runs = 7 powers x 3 conditions x 5 paired seeds.
- Moderate-stress minimum loss: p={moderate_best_power}, {moderate_best_loss:.6f}; improvement over p=2: {moderate_improvement:.4f}%.
- Severe-stress minimum loss: p={severe_best_power}, {severe_best_loss:.6f}; improvement over p=2: {severe_improvement:.4f}%.
- Severe-stress p=5 minus p=6 paired loss difference: {p5_p6['mean_loss_improvement']:.6f}; exhaustive 95% bootstrap interval [{p5_p6['bootstrap_low']:.6f}, {p5_p6['bootstrap_high']:.6f}].
- Severe-stress p=8 tail ratio: {severe_p8_tail:.6f}.
- Complete line-search failures across generated runs: {int(runs['failed_linesearch'].sum())}.

The runs were generated on CPU with deterministic PyTorch algorithms and one CPU thread. Bitwise identity is not guaranteed across PyTorch releases, operating systems, compiler libraries, or hardware; use the pinned environment for the closest reproduction.
"""
    output.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True, help="CSV generated by run_experiments.py paper")
    parser.add_argument("--out", type=Path, default=ROOT / "reproduced_outputs")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    runs = load_runs(args.runs)
    validate_paper_matrix(runs)
    main_table = summarize_main(runs)
    tail_table = summarize_tail_cost(runs)
    groups = summarize_groups(runs)
    comparisons = paired_comparisons(runs)

    main_table.to_csv(args.out / "table_main_results.csv", index=False)
    tail_table.to_csv(args.out / "table_tail_and_cost.csv", index=False)
    groups.to_csv(args.out / "table_group_displacements.csv", index=False)
    comparisons.to_csv(args.out / "paired_comparisons.csv", index=False)
    plot_scale_matched(args.out / "scale_matched_gradient_comparison.pdf")
    plot_loss(runs, args.out / "hifedprox_loss_vs_power.pdf")
    plot_tail(runs, args.out / "hifedprox_tail_vs_power.pdf")
    plot_groups(groups, args.out / "hifedprox_group_displacements.pdf")
    write_report(runs, main_table, tail_table, comparisons, args.out / "RESULTS.md")

    metadata = {
        "runs_file": str(args.runs),
        "rows": len(runs),
        "powers": list(PAPER_POWERS),
        "conditions": list(CONDITIONS),
        "seeds": list(PAPER_SEEDS),
    }
    (args.out / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nMAIN TABLE")
    print(main_table.to_string(index=False))
    print("\nTAIL AND COST TABLE")
    print(tail_table.to_string(index=False))
    print("\nPAIRED COMPARISONS")
    print(comparisons.to_string(index=False))
    print(f"\noutputs written to {args.out}")


if __name__ == "__main__":
    main()
