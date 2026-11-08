"""Deterministic CPU implementation of the HiFedProx experiments.

Author: Alireza Kabgani <a.kabgani@gmail.com>
License: MIT
"""
from __future__ import annotations

import copy
import json
import math
import random
import time
from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

CONDITIONS: dict[str, dict[str, float]] = {
    "clean": {"stress_fraction": 0.0, "lr_multiplier": 1.0, "step_multiplier": 1.0},
    "moderate_stress": {"stress_fraction": 0.2, "lr_multiplier": 2.0, "step_multiplier": 3.0},
    "severe_stress": {"stress_fraction": 0.2, "lr_multiplier": 4.0, "step_multiplier": 3.0},
}
PAPER_POWERS = tuple(range(2, 9))


def method_name(power: float) -> str:
    """Return the stable CSV method label for a real power p >= 2."""
    if not math.isfinite(power) or power < 2.0:
        raise ValueError("power must be finite and at least 2")
    token = str(float(power)).removesuffix(".0")
    return f"p{token}_bt"


def power_from_method(method: str) -> float:
    """Parse labels such as p2_bt, p7_bt, or p2.5_bt."""
    if not (method.startswith("p") and method.endswith("_bt")) or method == "p0_bt":
        raise ValueError(f"method does not encode a positive power: {method!r}")
    try:
        power = float(method[1:-3])
    except ValueError as exc:
        raise ValueError(f"cannot parse power from method {method!r}") from exc
    if not math.isfinite(power) or power < 2.0:
        raise ValueError("encoded power must be finite and at least 2")
    return power


@dataclass(frozen=True)
class RunConfig:
    method: str
    condition: str
    seed: int
    rounds: int = 100
    participation: float = 0.3
    local_steps: int = 3
    initial_step_size: float = 0.5
    batch_size: int = 16
    mu: float = 0.03
    reference_radius: float = 1.0
    backtracking_factor: float = 0.5
    armijo_c: float = 1e-4
    max_backtracks: int = 12
    checkpoints: tuple[int, ...] = (60, 100)

    def validate(self, client_count: int | None = None) -> None:
        config = self
        power_from_method(config.method)
        if config.condition not in CONDITIONS:
            raise ValueError(f"unknown condition: {config.condition!r}")
        if not isinstance(config.seed, int):
            raise ValueError("seed must be an integer")
        if config.rounds < 1 or config.local_steps < 1 or config.batch_size < 1:
            raise ValueError("rounds, local_steps, and batch_size must be positive")
        if not 0.0 < config.participation <= 1.0:
            raise ValueError("participation must lie in (0, 1]")
        if config.initial_step_size <= 0.0:
            raise ValueError("initial_step_size must be positive")
        if config.mu <= 0.0:
            raise ValueError("mu must be positive")
        if config.reference_radius <= 0.0:
            raise ValueError("reference_radius must be positive")
        if not 0.0 < config.backtracking_factor < 1.0:
            raise ValueError("backtracking_factor must lie in (0, 1)")
        if not 0.0 < config.armijo_c < 1.0:
            raise ValueError("armijo_c must lie in (0, 1)")
        if config.max_backtracks < 0:
            raise ValueError("max_backtracks must be nonnegative")
        if len(set(config.checkpoints)) != len(config.checkpoints):
            raise ValueError("checkpoints must be unique")
        if any(c < 1 or c > config.rounds for c in config.checkpoints):
            raise ValueError("checkpoints must lie between 1 and rounds")
        if client_count is not None:
            selected = max(3, int(round(config.participation * client_count)))
            if selected > client_count:
                raise ValueError("participation selects more clients than exist")


class SmallCNN(nn.Module):
    """CNN used in the paper; it has exactly 27,326 trainable parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(32 * 3 * 3, 64)
        self.fc2 = nn.Linear(64, 62)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def flatten_parameters(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def set_parameters(model: nn.Module, vector: torch.Tensor) -> None:
    if vector.ndim != 1:
        raise ValueError("parameter vector must be one-dimensional")
    expected = sum(p.numel() for p in model.parameters())
    if vector.numel() != expected:
        raise ValueError(f"parameter vector has {vector.numel()} entries; expected {expected}")
    offset = 0
    with torch.no_grad():
        for parameter in model.parameters():
            count = parameter.numel()
            parameter.copy_(vector[offset : offset + count].view_as(parameter))
            offset += count


def flatten_gradients(model: nn.Module) -> torch.Tensor:
    return torch.cat(
        [
            torch.zeros_like(p).reshape(-1) if p.grad is None else p.grad.reshape(-1)
            for p in model.parameters()
        ]
    )


def power_regularizer(
    model: nn.Module,
    anchors: Iterable[torch.Tensor],
    power: float,
    mu: float,
    reference_radius: float,
) -> torch.Tensor:
    """Return mu*||theta-anchor||^p/(p*R^(p-2))."""
    if power < 2.0:
        raise ValueError("power must be at least 2")
    if mu < 0.0 or reference_radius <= 0.0:
        raise ValueError("mu must be nonnegative and reference_radius positive")
    anchor_list = list(anchors)
    parameters = list(model.parameters())
    if len(anchor_list) != len(parameters):
        raise ValueError("anchor and model parameter lists have different lengths")
    squared_norm = torch.zeros((), dtype=parameters[0].dtype, device=parameters[0].device)
    for parameter, anchor in zip(parameters, anchor_list):
        if parameter.shape != anchor.shape:
            raise ValueError("anchor and model parameter shapes differ")
        squared_norm = squared_norm + (parameter - anchor).square().sum()
    return mu * squared_norm.pow(power / 2.0) / (
        power * reference_radius ** (power - 2.0)
    )


def regularized_objective(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    anchors: Iterable[torch.Tensor],
    power: float,
    mu: float,
    reference_radius: float,
) -> torch.Tensor:
    return F.cross_entropy(model(x), y) + power_regularizer(
        model, anchors, power, mu, reference_radius
    )


def sample_minibatch(
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(y) == 0:
        raise ValueError("client has no training examples")
    indices = torch.randperm(len(y), generator=generator)[: min(batch_size, len(y))]
    return x[indices], y[indices]


def local_power_update(
    global_model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    power: float,
    mu: float,
    reference_radius: float,
    initial_step_size: float,
    steps: int,
    batch_size: int,
    seed: int,
    backtracking_factor: float,
    armijo_c: float,
    max_backtracks: int,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Run same-mini-batch Armijo updates around the fixed server anchor."""
    model = copy.deepcopy(global_model)
    model.train()
    anchors = [p.detach().clone() for p in global_model.parameters()]
    start = flatten_parameters(global_model)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    grad_evals = objective_evals = trial_evals = 0
    accepted_steps: list[float] = []
    backtracks: list[int] = []
    failures: list[int] = []
    accepted_on_last_trial: list[int] = []

    for _ in range(steps):
        xb, yb = sample_minibatch(x, y, batch_size, generator)
        model.zero_grad(set_to_none=True)
        loss = regularized_objective(
            model, xb, yb, anchors, power, mu, reference_radius
        )
        loss.backward()
        gradient = flatten_gradients(model).detach()
        current = flatten_parameters(model)
        current_value = float(loss.detach())
        gradient_norm_sq = float(torch.dot(gradient, gradient))
        grad_evals += 1
        objective_evals += 1  # current forward pass is shared with backward

        step_size = initial_step_size
        accepted = False
        for trial_index in range(max_backtracks + 1):
            candidate = current - step_size * gradient
            if torch.isfinite(candidate).all():
                set_parameters(model, candidate)
                with torch.no_grad():
                    trial_value = float(
                        regularized_objective(
                            model, xb, yb, anchors, power, mu, reference_radius
                        )
                    )
            else:
                trial_value = math.inf
            objective_evals += 1
            trial_evals += 1
            if math.isfinite(trial_value) and trial_value <= (
                current_value - armijo_c * step_size * gradient_norm_sq
            ):
                accepted = True
                accepted_steps.append(step_size)
                backtracks.append(trial_index)
                failures.append(0)
                accepted_on_last_trial.append(int(trial_index == max_backtracks))
                break
            if trial_index < max_backtracks:
                step_size *= backtracking_factor

        if not accepted:
            set_parameters(model, current)
            accepted_steps.append(0.0)
            backtracks.append(max_backtracks)
            failures.append(1)
            accepted_on_last_trial.append(0)

    return flatten_parameters(model) - start, {
        "grad_evals": grad_evals,
        "objective_evals": objective_evals,
        "trial_evals": trial_evals,
        "accepted_steps": accepted_steps,
        "backtracks": backtracks,
        "failures": failures,
        "accepted_on_last_trial": accepted_on_last_trial,
    }


def order_quantile(values: Iterable[float], q: float) -> float:
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must lie in [0,1]")
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return math.nan
    if not np.isfinite(array).all():
        raise ValueError("quantile values must be finite")
    return float(np.quantile(array, q, method="inverted_cdf"))


def linear_quantile(values: Iterable[float], q: float) -> float:
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must lie in [0,1]")
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return math.nan
    if not np.isfinite(array).all():
        raise ValueError("quantile values must be finite")
    return float(np.quantile(array, q, method="linear"))


def summarize(values: Iterable[float], prefix: str) -> dict[str, float]:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return {f"{prefix}_{key}": math.nan for key in ("mean", "q10", "q50", "q90")}
    if not np.isfinite(array).all():
        raise ValueError(f"{prefix} values must be finite")
    return {
        f"{prefix}_mean": float(array.mean()),
        f"{prefix}_q10": linear_quantile(array, 0.1),
        f"{prefix}_q50": linear_quantile(array, 0.5),
        f"{prefix}_q90": linear_quantile(array, 0.9),
    }


def evaluate(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    writers: torch.Tensor,
) -> tuple[float, float, float]:
    previous_mode = model.training
    model.eval()
    try:
        total_loss = 0.0
        predictions: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(y), 512):
                logits = model(x[start : start + 512])
                total_loss += float(
                    F.cross_entropy(logits, y[start : start + 512], reduction="sum")
                )
                predictions.append(logits.argmax(dim=1))
        prediction = torch.cat(predictions)
        accuracy = float((prediction == y).float().mean())
        writer_accuracies = [
            float((prediction[writers == writer] == y[writers == writer]).float().mean())
            for writer in torch.unique(writers)
        ]
    finally:
        model.train(previous_mode)
    return total_loss / len(y), accuracy, linear_quantile(writer_accuracies, 0.1)


def validate_cache(data: Mapping[str, np.ndarray]) -> int:
    required = {"x_train", "y_train", "writer_train", "x_test", "y_test", "writer_test"}
    missing = required.difference(data)
    if missing:
        raise ValueError(f"cache is missing arrays: {sorted(missing)}")
    x_train = np.asarray(data["x_train"])
    y_train = np.asarray(data["y_train"])
    writer_train = np.asarray(data["writer_train"])
    x_test = np.asarray(data["x_test"])
    y_test = np.asarray(data["y_test"])
    writer_test = np.asarray(data["writer_test"])
    if len(x_train) != len(y_train) or len(y_train) != len(writer_train):
        raise ValueError("training arrays have inconsistent lengths")
    if len(x_test) != len(y_test) or len(y_test) != len(writer_test):
        raise ValueError("evaluation arrays have inconsistent lengths")
    if x_train.shape[1:] not in {(196,), (1, 14, 14)} or x_test.shape[1:] not in {(196,), (1, 14, 14)}:
        raise ValueError("images must be flattened or image-shaped 14x14 arrays")
    if x_train.dtype != np.float32 or x_test.dtype != np.float32:
        raise ValueError("images must have float32 dtype")
    if not np.isfinite(x_train).all() or not np.isfinite(x_test).all():
        raise ValueError("images must be finite")
    if min(float(x_train.min()), float(x_test.min())) < 0.0 or max(float(x_train.max()), float(x_test.max())) > 1.0:
        raise ValueError("images must lie in [0,1]")
    if not np.issubdtype(y_train.dtype, np.integer) or not np.issubdtype(y_test.dtype, np.integer):
        raise ValueError("labels must be integers")
    if not np.issubdtype(writer_train.dtype, np.integer) or not np.issubdtype(writer_test.dtype, np.integer):
        raise ValueError("writer IDs must be integers")
    if y_train.min() < 0 or y_train.max() >= 62 or y_test.min() < 0 or y_test.max() >= 62:
        raise ValueError("labels must lie in {0,...,61}")
    writer_ids = np.unique(writer_train)
    if not np.array_equal(writer_ids, np.arange(len(writer_ids))):
        raise ValueError("training writer IDs must be contiguous and start at zero")
    if not np.isin(np.unique(writer_test), writer_ids).all():
        raise ValueError("all evaluation writers must appear in training")
    return len(writer_ids)


def load_cache(path: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as cache:
        return {key: cache[key] for key in cache.files}


def run_federated(data: Mapping[str, np.ndarray], config: RunConfig) -> dict[str, object]:
    client_count = validate_cache(data)
    config.validate(client_count)
    seed_everything(config.seed)

    x_train = torch.as_tensor(data["x_train"].reshape(-1, 1, 14, 14), dtype=torch.float32)
    y_train = torch.as_tensor(data["y_train"], dtype=torch.long)
    writer_train = torch.as_tensor(data["writer_train"], dtype=torch.long)
    x_test = torch.as_tensor(data["x_test"].reshape(-1, 1, 14, 14), dtype=torch.float32)
    y_test = torch.as_tensor(data["y_test"], dtype=torch.long)
    writer_test = torch.as_tensor(data["writer_test"], dtype=torch.long)

    client_indices = [torch.where(writer_train == client_id)[0] for client_id in range(client_count)]
    condition = CONDITIONS[config.condition]
    stress_rng = np.random.default_rng(config.seed + 77)
    stressed = np.zeros(client_count, dtype=bool)
    stressed_count = int(round(condition["stress_fraction"] * client_count))
    if stressed_count:
        stressed[stress_rng.choice(client_count, stressed_count, replace=False)] = True

    client_labels: list[torch.Tensor] = []
    for client_id, indices in enumerate(client_indices):
        labels = y_train[indices].clone()
        if stressed[client_id]:
            labels = (labels + 1) % 62
        client_labels.append(labels)

    model = SmallCNN()
    selection_rng = np.random.default_rng(config.seed + 991)
    selected_count = max(3, int(round(config.participation * client_count)))

    grad_evals = objective_evals = trial_evals = 0
    all_steps: list[float] = []
    all_backtracks: list[int] = []
    all_failures: list[int] = []
    all_last_trial: list[int] = []
    norms_all: list[float] = []
    norms_unmodified: list[float] = []
    norms_stress_test: list[float] = []
    round_tail_ratios: list[float] = []
    checkpoints: list[dict[str, float | int]] = []
    unstable = 0
    started = time.perf_counter()

    for round_index in range(1, config.rounds + 1):
        selected = np.sort(selection_rng.choice(client_count, selected_count, replace=False))
        displacements: list[torch.Tensor] = []
        for client_id_raw in selected:
            client_id = int(client_id_raw)
            indices = client_indices[client_id]
            local_steps = config.local_steps * (
                int(condition["step_multiplier"]) if stressed[client_id] else 1
            )
            local_step_size = config.initial_step_size * (
                condition["lr_multiplier"] if stressed[client_id] else 1.0
            )
            local_seed = config.seed * 100000 + round_index * 1000 + client_id

            power = power_from_method(config.method)
            displacement, local = local_power_update(
                model, x_train[indices], client_labels[client_id],
                power, config.mu, config.reference_radius,
                local_step_size, local_steps, config.batch_size, local_seed,
                config.backtracking_factor, config.armijo_c, config.max_backtracks,
            )
            displacements.append(displacement)
            grad_evals += int(local["grad_evals"])
            objective_evals += int(local["objective_evals"])
            trial_evals += int(local["trial_evals"])
            all_steps.extend(local["accepted_steps"])
            all_backtracks.extend(local["backtracks"])
            all_failures.extend(local["failures"])
            all_last_trial.extend(local["accepted_on_last_trial"])

        updates = torch.stack(displacements)
        norms = torch.linalg.vector_norm(updates, dim=1).cpu().numpy()
        norms_all.extend(norms.tolist())
        unmodified_mask = ~stressed[selected]
        stress_mask = stressed[selected]
        norms_unmodified.extend(norms[unmodified_mask].tolist())
        norms_stress_test.extend(norms[stress_mask].tolist())
        q50 = order_quantile(norms, 0.5)
        q90 = order_quantile(norms, 0.9)
        round_tail_ratios.append(q90 / max(q50, 1e-12))

        set_parameters(model, flatten_parameters(model) + updates.mean(dim=0))
        if not torch.isfinite(flatten_parameters(model)).all():
            unstable = 1
            break
        if round_index in config.checkpoints:
            loss, accuracy, writer_q10 = evaluate(model, x_test, y_test, writer_test)
            checkpoints.append({
                "round": round_index,
                "test_loss": loss,
                "test_accuracy": accuracy,
                "writer_accuracy_q10": writer_q10,
            })

    if unstable:
        loss = accuracy = writer_q10 = math.nan
    else:
        loss, accuracy, writer_q10 = evaluate(model, x_test, y_test, writer_test)

    record: dict[str, object] = {
        "method": config.method,
        "power": power_from_method(config.method),
        "condition": config.condition,
        "seed": config.seed,
        "rounds_completed": round_index,
        "unstable": unstable,
        "test_loss": loss,
        "test_accuracy": accuracy,
        "writer_accuracy_q10": writer_q10,
        "grad_evals": grad_evals,
        "objective_evals": objective_evals,
        "armijo_trial_evals": trial_evals,
        "failed_linesearch": int(sum(all_failures)),
        "local_steps_attempted": len(all_failures),
        "zero_step_fraction": float(np.mean(np.asarray(all_steps) == 0.0)) if all_steps else 0.0,
        "accepted_on_last_trial_fraction": float(np.mean(all_last_trial)) if all_last_trial else 0.0,
        "tail_ratio_orderstat_mean": float(np.mean(round_tail_ratios)),
        "stressed_writer_ids_json": json.dumps(np.flatnonzero(stressed).tolist()),
        "participation": config.participation,
        "base_local_steps": config.local_steps,
        "initial_step_size": config.initial_step_size,
        "batch_size": config.batch_size,
        "mu": config.mu,
        "reference_radius": config.reference_radius,
        "backtracking_factor": config.backtracking_factor,
        "armijo_c": config.armijo_c,
        "max_backtracks": config.max_backtracks,
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint_json": json.dumps(checkpoints),
    }
    record.update(summarize(all_steps, "accepted_step"))
    record.update(summarize(all_backtracks, "backtracks"))
    record.update(summarize(norms_all, "update_norm"))
    record.update(summarize(norms_unmodified, "update_norm_unmodified"))
    record.update(summarize(norms_stress_test, "update_norm_stress_test"))
    return record
