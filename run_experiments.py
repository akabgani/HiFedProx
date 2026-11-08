# Author: Alireza Kabgani <a.kabgani@gmail.com>
# SPDX-License-Identifier: MIT
"""Run the HiFedProx FEMNIST experiments reported in the FLTA paper."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from hifedprox import CONDITIONS, PAPER_POWERS, RunConfig, load_cache, method_name, run_federated

ROOT = Path(__file__).resolve().parent
DEFAULT_EVAL_DATA = ROOT / "data" / "femnist_evaluation.npz"
DEFAULT_CAL_DATA = ROOT / "data" / "femnist_calibration.npz"
PAPER_SEEDS = (3001, 3002, 3003, 3004, 3005)
PAPER_CONDITIONS = ("clean", "moderate_stress", "severe_stress")

# Calibration grid used to select the common (mu, R) pair reported in the paper.
CALIBRATION_GRID = {
    2.0: ((0.03, 1.0), (0.10, 1.0), (0.30, 1.0), (1.00, 1.0)),
    3.0: ((0.03, 1.0), (0.10, 1.0), (0.03, 0.5), (0.10, 0.5)),
}

KEY_COLUMNS = ["method", "condition", "seed", "mu", "reference_radius", "rounds_completed"]


def write_record(path: Path, record: dict[str, object]) -> None:
    """Append or replace one run atomically, making interrupted grids resumable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([record])
    if path.exists():
        old = pd.read_csv(path)
        combined = pd.concat([old, new], ignore_index=True, sort=False)
        combined = combined.drop_duplicates(subset=KEY_COLUMNS, keep="last")
    else:
        combined = new
    temporary = path.with_suffix(path.suffix + ".tmp")
    combined.to_csv(temporary, index=False)
    temporary.replace(path)


def make_config(
    power: float,
    condition: str,
    seed: int,
    *,
    rounds: int = 100,
    participation: float = 0.3,
    local_steps: int = 3,
    initial_step_size: float = 0.5,
    batch_size: int = 16,
    mu: float = 0.03,
    reference_radius: float = 1.0,
    rho: float = 0.5,
    armijo_c: float = 1e-4,
    max_backtracks: int = 12,
) -> RunConfig:
    checkpoints = tuple(sorted({min(60, rounds), rounds}))
    return RunConfig(
        method=method_name(power),
        condition=condition,
        seed=seed,
        rounds=rounds,
        participation=participation,
        local_steps=local_steps,
        initial_step_size=initial_step_size,
        batch_size=batch_size,
        mu=mu,
        reference_radius=reference_radius,
        backtracking_factor=rho,
        armijo_c=armijo_c,
        max_backtracks=max_backtracks,
        checkpoints=checkpoints,
    )


def run_one(data: dict, output: Path, config: RunConfig) -> dict[str, object]:
    print(
        f"running power={config.method[1:-3]}, condition={config.condition}, "
        f"seed={config.seed}, rounds={config.rounds}, mu={config.mu}, R={config.reference_radius}",
        flush=True,
    )
    record = run_federated(data, config)
    write_record(output, record)
    print(
        f"loss={record['test_loss']:.9f}, accuracy={100.0 * record['test_accuracy']:.4f}%, "
        f"tail={record['tail_ratio_orderstat_mean']:.6f}, trials={record['armijo_trial_evals']}",
        flush=True,
    )
    return record



def completed_keys(path: Path) -> set[tuple[str, str, int, float, float, int]]:
    if not path.is_file():
        return set()
    frame = pd.read_csv(path)
    required = {"method", "condition", "seed", "mu", "reference_radius", "rounds_completed"}
    if not required.issubset(frame.columns):
        return set()
    return {
        (str(row.method), str(row.condition), int(row.seed), float(row.mu), float(row.reference_radius), int(row.rounds_completed))
        for row in frame.itertuples(index=False)
    }


def config_key(config: RunConfig) -> tuple[str, str, int, float, float, int]:
    return (config.method, config.condition, config.seed, config.mu, config.reference_radius, config.rounds)


def run_paper(args: argparse.Namespace) -> None:
    data = load_cache(str(args.data))
    if args.fresh and args.out.exists():
        args.out.unlink()
    done = completed_keys(args.out)
    for power in args.powers:
        for condition in args.conditions:
            for seed in args.seeds:
                config = make_config(
                    power,
                    condition,
                    seed,
                    rounds=args.rounds,
                    mu=args.mu,
                    reference_radius=args.reference_radius,
                )
                if not args.force and config_key(config) in done:
                    print(f"skipping completed power={power:g}, condition={condition}, seed={seed}", flush=True)
                    continue
                run_one(data, args.out, config)
    print(f"\nrun matrix written to {args.out}")


def run_single(args: argparse.Namespace) -> None:
    data = load_cache(str(args.data))
    config = make_config(
        args.power,
        args.condition,
        args.seed,
        rounds=args.rounds,
        participation=args.participation,
        local_steps=args.local_steps,
        initial_step_size=args.step_size,
        batch_size=args.batch_size,
        mu=args.mu,
        reference_radius=args.reference_radius,
        rho=args.rho,
        armijo_c=args.armijo_c,
        max_backtracks=args.max_backtracks,
    )
    run_one(data, args.out, config)


def run_calibration(args: argparse.Namespace) -> None:
    data = load_cache(str(args.data))
    for power, grid in CALIBRATION_GRID.items():
        for mu, radius in grid:
            for condition in ("clean", "moderate_stress"):
                run_one(
                    data,
                    args.out,
                    make_config(
                        power,
                        condition,
                        1501,
                        rounds=60,
                        mu=mu,
                        reference_radius=radius,
                    ),
                )
    raw = pd.read_csv(args.out)
    raw["power"] = raw["method"].str.extract(r"^p([0-9.]+)_bt$")[0].astype(float)
    summary = (
        raw.groupby(["power", "mu", "reference_radius"], as_index=False)["test_loss"]
        .mean()
        .rename(columns={"test_loss": "mean_clean_moderate_loss"})
        .sort_values(["power", "mu", "reference_radius"])
    )
    summary_path = args.out.with_name(args.out.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False)
    selected = summary.loc[summary.groupby("power")["mean_clean_moderate_loss"].idxmin()]
    print("\nselected configurations")
    print(selected.to_string(index=False))
    print(f"raw calibration records: {args.out}")
    print(f"calibration summary: {summary_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    paper = subparsers.add_parser("paper", help="run the 105 paper experiments")
    paper.add_argument("--data", type=Path, default=DEFAULT_EVAL_DATA)
    paper.add_argument("--out", type=Path, default=ROOT / "results" / "paper_runs.csv")
    paper.add_argument("--powers", type=float, nargs="+", default=list(PAPER_POWERS))
    paper.add_argument("--conditions", nargs="+", choices=PAPER_CONDITIONS, default=list(PAPER_CONDITIONS))
    paper.add_argument("--seeds", type=int, nargs="+", default=list(PAPER_SEEDS))
    paper.add_argument("--rounds", type=int, default=100)
    paper.add_argument("--mu", type=float, default=0.03)
    paper.add_argument("--reference-radius", type=float, default=1.0)
    paper.add_argument("--force", action="store_true", help="rerun rows already present in the output CSV")
    paper.add_argument(
        "--fresh",
        action="store_true",
        help="delete the output CSV before starting, so every experiment is rerun from scratch",
    )
    paper.set_defaults(function=run_paper)

    single = subparsers.add_parser("single", help="run one power-condition-seed combination")
    single.add_argument("--data", type=Path, default=DEFAULT_EVAL_DATA)
    single.add_argument("--out", type=Path, default=ROOT / "results" / "single_run.csv")
    single.add_argument("--power", type=float, required=True)
    single.add_argument("--condition", choices=PAPER_CONDITIONS, required=True)
    single.add_argument("--seed", type=int, required=True)
    single.add_argument("--rounds", type=int, default=100)
    single.add_argument("--participation", type=float, default=0.3)
    single.add_argument("--local-steps", type=int, default=3)
    single.add_argument("--step-size", type=float, default=0.5)
    single.add_argument("--batch-size", type=int, default=16)
    single.add_argument("--mu", type=float, default=0.03)
    single.add_argument("--reference-radius", type=float, default=1.0)
    single.add_argument("--rho", type=float, default=0.5)
    single.add_argument("--armijo-c", type=float, default=1e-4)
    single.add_argument("--max-backtracks", type=int, default=12)
    single.set_defaults(function=run_single)

    calibration = subparsers.add_parser("calibration", help="rerun the original 16 p=2/p=3 calibration jobs")
    calibration.add_argument("--data", type=Path, default=DEFAULT_CAL_DATA)
    calibration.add_argument("--out", type=Path, default=ROOT / "results" / "calibration_runs.csv")
    calibration.set_defaults(function=run_calibration)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error(f"data file not found: {args.data}")
    args.function(args)


if __name__ == "__main__":
    main()
