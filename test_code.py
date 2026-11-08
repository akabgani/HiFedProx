# Author: Alireza Kabgani <a.kabgani@gmail.com>
# SPDX-License-Identifier: MIT
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from analyze_results import exact_bootstrap_interval
from hifedprox import (
    RunConfig,
    SmallCNN,
    flatten_parameters,
    local_power_update,
    method_name,
    order_quantile,
    power_from_method,
    power_regularizer,
    run_federated,
    set_parameters,
    validate_cache,
)

ROOT = Path(__file__).resolve().parent


class TinyLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(2, 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class HiFedProxTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(10)
        torch.set_num_threads(1)

    def test_model_parameter_count(self) -> None:
        self.assertEqual(sum(parameter.numel() for parameter in SmallCNN().parameters()), 27326)

    def test_method_names_support_integer_and_fractional_powers(self) -> None:
        for power in (2.0, 2.0000001, 2.5, 2.500001, 7.0, 8.0):
            self.assertEqual(power_from_method(method_name(power)), power)
        with self.assertRaises(ValueError):
            method_name(1.9)

    def test_regularizer_gradient_matches_formula(self) -> None:
        model = TinyLinear()
        anchors = [parameter.detach().clone() for parameter in model.parameters()]
        displacement = torch.tensor([0.2, -0.3, 0.5, -0.1])
        set_parameters(model, flatten_parameters(model) + displacement)
        for power in (2.0, 2.5, 3.0, 8.0):
            model.zero_grad(set_to_none=True)
            value = power_regularizer(model, anchors, power, 0.7, 1.3)
            value.backward()
            actual = torch.cat([parameter.grad.reshape(-1) for parameter in model.parameters()])
            expected = (
                0.7
                / (1.3 ** (power - 2.0))
                * torch.linalg.vector_norm(displacement).pow(power - 2.0)
                * displacement
            )
            torch.testing.assert_close(actual, expected)

    def test_scale_matching_at_reference_radius(self) -> None:
        direction = torch.tensor([3.0, 4.0])
        direction = direction / torch.linalg.vector_norm(direction)
        radius = 1.7
        mu = 0.4
        displacement = radius * direction
        for power in (2.0, 3.0, 5.0, 8.0):
            gradient = mu * radius ** (2.0 - power) * torch.linalg.vector_norm(displacement).pow(power - 2.0) * displacement
            self.assertAlmostEqual(float(torch.linalg.vector_norm(gradient)), mu * radius, places=6)

    def test_local_update_is_deterministic(self) -> None:
        model = TinyLinear()
        x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        y = torch.tensor([0, 1, 0])
        args = (model, x, y, 7.0, 0.03, 1.0, 0.2, 3, 2, 555, 0.5, 1e-4, 12)
        first, first_info = local_power_update(*args)
        second, second_info = local_power_update(*args)
        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
        self.assertEqual(first_info, second_info)

    def test_armijo_cost_identity(self) -> None:
        model = TinyLinear()
        x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        y = torch.tensor([0, 1])
        _, info = local_power_update(model, x, y, 6.0, 0.1, 1.0, 0.5, 2, 2, 77, 0.5, 1e-4, 12)
        self.assertEqual(info["grad_evals"], 2)
        self.assertEqual(info["trial_evals"], info["objective_evals"] - info["grad_evals"])
        self.assertEqual(sum(info["failures"]), 0)

    def test_failed_search_restores_iterate(self) -> None:
        model = TinyLinear()
        x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        y = torch.tensor([0, 1])
        displacement, info = local_power_update(model, x, y, 3.0, 0.03, 1.0, 32.0, 1, 2, 77, 0.5, 1e-4, 1)
        torch.testing.assert_close(displacement, torch.zeros_like(displacement), rtol=0.0, atol=0.0)
        self.assertEqual(info["failures"], [1])
        self.assertEqual(info["accepted_steps"], [0.0])

    def test_order_statistic_quantile(self) -> None:
        self.assertEqual(order_quantile([1.0, 2.0, 3.0, 4.0], 0.5), 2.0)
        self.assertEqual(order_quantile([1.0, 2.0, 3.0, 4.0], 0.9), 4.0)

    def test_cache_validation(self) -> None:
        data = {
            "x_train": np.zeros((6, 196), dtype=np.float32),
            "y_train": np.array([0, 1, 0, 1, 0, 1]),
            "writer_train": np.array([0, 0, 1, 1, 2, 2]),
            "x_test": np.zeros((3, 196), dtype=np.float32),
            "y_test": np.array([0, 1, 0]),
            "writer_test": np.array([0, 1, 2]),
        }
        self.assertEqual(validate_cache(data), 3)

    def test_one_round_paired_gradient_budgets(self) -> None:
        data = {
            "x_train": np.zeros((6, 196), dtype=np.float32),
            "y_train": np.array([0, 1, 0, 1, 0, 1]),
            "writer_train": np.array([0, 0, 1, 1, 2, 2]),
            "x_test": np.zeros((3, 196), dtype=np.float32),
            "y_test": np.array([0, 1, 0]),
            "writer_test": np.array([0, 1, 2]),
        }
        common = dict(condition="clean", seed=4, rounds=1, participation=1.0, local_steps=1, batch_size=2, checkpoints=(1,), mu=0.03)
        p2 = run_federated(data, RunConfig(method="p2_bt", **common))
        p8 = run_federated(data, RunConfig(method="p8_bt", **common))
        self.assertEqual(p2["grad_evals"], p8["grad_evals"])
        self.assertEqual(p2["grad_evals"], 3)
        self.assertEqual(p2["failed_linesearch"], 0)
        self.assertEqual(p8["failed_linesearch"], 0)

    def test_exhaustive_bootstrap_interval_for_constant_values(self) -> None:
        low, high = exact_bootstrap_interval(np.ones(5))
        self.assertEqual(low, 1.0)
        self.assertEqual(high, 1.0)

    def test_packaged_calibration_evaluation_disjointness(self) -> None:
        """The frozen train split is shared; held-out splits are disjoint and exhaustive."""
        from collections import Counter

        with np.load(ROOT / "data" / "femnist_calibration.npz", allow_pickle=False) as src:
            calibration = {key: src[key] for key in src.files}
        with np.load(ROOT / "data" / "femnist_evaluation.npz", allow_pickle=False) as src:
            evaluation = {key: src[key] for key in src.files}
        with np.load(ROOT / "data" / "femnist_original_heldout.npz", allow_pickle=False) as src:
            original = {key: src[key] for key in src.files}

        for key in ("x_train", "y_train", "writer_train"):
            np.testing.assert_array_equal(calibration[key], evaluation[key])

        def samples(cache: dict, split: str) -> Counter:
            return Counter(
                (image.tobytes(), int(label), int(writer))
                for image, label, writer in zip(
                    cache[f"x_{split}"], cache[f"y_{split}"], cache[f"writer_{split}"]
                )
            )

        train = samples(evaluation, "train")
        cal = samples(calibration, "test")
        test = samples(evaluation, "test")
        heldout = samples(original, "test")
        self.assertFalse(train.keys() & cal.keys())
        self.assertFalse(train.keys() & test.keys())
        self.assertFalse(cal.keys() & test.keys())
        self.assertEqual(cal + test, heldout)

    def test_packaged_femnist_cache(self) -> None:
        path = ROOT / "data" / "femnist_evaluation.npz"
        with np.load(path, allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
        self.assertEqual(validate_cache(data), 60)
        self.assertEqual(data["x_train"].shape, (8064, 196))
        self.assertEqual(data["x_test"].shape, (1021, 196))



if __name__ == "__main__":
    unittest.main(verbosity=2)
