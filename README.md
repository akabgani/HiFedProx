# HiFedProx: High-Order Federated Proximal Regularization

This package accompanies the conference paper

**Beyond Conventional Federated Learning via High-Order Regularization**

**Code author:** Alireza Kabgani  
**Email:** a.kabgani@gmail.com

This repository contains the implementation and reproducibility scripts for the **HiFedProx exponent-sensitivity experiments** in the accompanying paper. It includes the frozen 60-writer FEMNIST-derived caches, the client optimizer, tests, and scripts for Tables II and IV and the underlying numerical data for Figure 4. The plotting scripts produce related figures, but do not reproduce the exact published layouts. No generated experiment records are included in the source archive.

**Scope limitation:** The paper also compares with FedAvg, conventional FedProx, and SCAFFOLD (Table III). Those baseline implementations, their tuning histories, and the Table III reproduction script are **not in this release**. Thus `--full` means the full **105-run HiFedProx exponent matrix**, not full reproduction of every table in the paper.

Paper: [arXiv:2609.09904](https://arxiv.org/abs/2609.09904).

## Experimental matrix

The main experiment contains:

- powers `p = 2, 3, 4, 5, 6, 7, 8`;
- conditions `clean`, `moderate_stress`, and `severe_stress`;
- paired seeds `3001, 3002, 3003, 3004, 3005`;
- 100 communication rounds;
- 60 writers and 18 sampled clients per round;
- common parameters `mu = 0.03` and `R = 1`.

For a fixed condition and seed, all powers use the same initial network, persistent stress-test clients, client selections, local minibatch streams, local-step budgets, and communication budget. Only the exponent of the scale-matched regularizer changes.

## Files

- `hifedprox.py`: model, power regularizer, same-minibatch Armijo update, federated training loop, and metrics.
- `run_experiments.py`: single runs, the 105-run HiFedProx exponent matrix, and the p=2/p=3 calibration study.
- `analyze_results.py`: exponent-study tables, related plots, paired bootstrap intervals, and a numerical summary (not Table III).
- `validate_generated.py`: protocol and numerical-sanity checks for a generated run matrix.
- `reproduce.py`: main exponent-study reproduction entry point.
- `smoke_test.py`: short execution check.
- `test_code.py`: focused implementation tests.
- `data/`: frozen FEMNIST-derived caches used by the experiments.
- `DATA_NOTICE.md`: data provenance, reuse considerations, and limitations.
- `CITATION.cff`: GitHub citation metadata, pointing readers to the paper.
- `.gitignore`: generated output and local working files excluded from version control.

## Tested environment

- Python 3.13.5
- PyTorch 2.10.0 CPU
- NumPy 2.3.5
- pandas 2.2.3
- Matplotlib 3.10.8

The implementation uses one CPU thread and requests deterministic PyTorch algorithms. PyTorch does not guarantee bitwise identity across releases, platforms, or CPU implementations, so the pinned environment is recommended for the closest reproduction.

## Installation

### Python virtual environment

Linux or macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Conda

```bash
conda env create -f environment.yml
conda activate hifedprox-flta2026
```

After the dependencies are installed, the experiments do not require internet access.

## Quick check

```bash
python reproduce.py
```

This command runs the implementation tests and three short training jobs, including one fractional exponent. It checks the code and packaged data but does not reproduce the full paper matrix.

## Full paper reproduction

```bash
python reproduce.py --full
```

The command:

1. runs the implementation tests;
2. trains all `7 x 3 x 5 = 105` configurations;
3. writes every completed run to `generated_output/runs/paper_runs.csv`;
4. validates the generated matrix and protocol;
5. creates the **HiFedProx exponent-study** tables, related figures, bootstrap intervals, and summary under `generated_output/analysis/`. It does not create the conventional-baseline Table III.

The full matrix is serial and may require several hours, depending on the processor. The run CSV is updated after each completed configuration. To continue an interrupted run, use:

```bash
python reproduce.py --full --resume
```

## Individual commands

Run one reported configuration:

```bash
python run_experiments.py single \
  --power 6 \
  --condition severe_stress \
  --seed 3001 \
  --out generated_output/single_run.csv
```

Run a selected subset:

```bash
python run_experiments.py paper \
  --powers 5 6 7 \
  --conditions moderate_stress severe_stress \
  --seeds 3001 3002 \
  --out generated_output/subset.csv
```

The implementation accepts any real exponent `p >= 2`, for example:

```bash
python run_experiments.py single \
  --power 2.5 \
  --condition moderate_stress \
  --seed 3001 \
  --out generated_output/fractional_run.csv
```

Rerun the calibration used for the common `(mu, R)` pair:

```bash
python run_experiments.py calibration \
  --out generated_output/calibration_runs.csv
```

## Output files

A full run creates:

```text
generated_output/
├── runs/
│   └── paper_runs.csv
├── analysis/
│   ├── table_main_results.csv
│   ├── table_tail_and_cost.csv
│   ├── table_group_displacements.csv
│   ├── paired_comparisons.csv
│   ├── scale_matched_gradient_comparison.pdf
│   ├── hifedprox_loss_vs_power.pdf
│   ├── hifedprox_tail_vs_power.pdf
│   ├── hifedprox_group_displacements.pdf
│   ├── analysis_metadata.json
│   └── RESULTS.md
└── validation_report.json
```

## Numerical definitions

For the 18 displacement norms in a communication round, sorted as `r_(1), ..., r_(18)`, the reported upper-tail diagnostic is

```text
r_(17) / max(r_(9), 1e-12).
```

The statistic is averaged over 100 rounds. Writer-level tenth-percentile accuracy uses linear interpolation across the 60 writer accuracies. Figure error bars are one sample standard deviation over the five paired seeds. The paired intervals enumerate all `5^5 = 3125` ordered bootstrap resamples and use the 2.5% and 97.5% linear quantiles.

## Data

The included caches are a frozen 60-writer subset derived from FEMNIST/LEAF and are sufficient to reproduce the reported experiments. They do not reconstruct the earlier raw-data acquisition, writer-selection, or resizing pipeline. See `DATA_NOTICE.md`.

## License and citation

The **software code** is released under the MIT License. This does **not** claim to relicense the FEMNIST-derived data. See `DATA_NOTICE.md` and the source dataset terms.

The preferred citation is Alireza Kabgani and Masoud Ahookhosh, *Beyond Conventional Federated Learning via High-Order Regularization*, 2026, [arXiv:2609.09904](https://arxiv.org/abs/2609.09904). A GitHub-readable `CITATION.cff` is included.
