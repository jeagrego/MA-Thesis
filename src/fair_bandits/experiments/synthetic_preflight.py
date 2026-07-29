from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..data import (
    make_imbalanced_synthetic_cmab_dataset,
    make_preprocessing_weights,
    make_synthetic_cmab_dataset,
    make_synthetic_expert_advice,
    make_synthetic_potential_rewards,
)
from ..policies import SyntheticPolicyParams
from .synthetic_runner import replay_one_synthetic_trajectory


def _display_or_print(
    value: Any,
    *,
    display_fn: Callable[[Any], None] | None = None,
) -> None:
    """
    Display an object in a notebook when display_fn is provided.
    Otherwise, print a readable text representation.
    """
    if display_fn is not None:
        display_fn(value)
        return

    if isinstance(value, pd.DataFrame):
        print(value.to_string(index=False))
    else:
        print(value)


def run_synthetic_preflight(
    *,
    regimes: Sequence[str],
    policies_by_regime: Mapping[str, Sequence[str]],
    params: SyntheticPolicyParams,
    d: int,
    n_experts: int,
    t: int = 250,
    seed: int = 42,
    preprocessing: str = "reweigh_group_label",
    display_fn: Callable[[Any], None] | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Run preflight checks for the balanced synthetic CMAB experiments.

    For each regime, this function:
    - generates a short synthetic dataset;
    - computes the Reweighting support table;
    - creates expert advice when required by EXP4;
    - replays one fairness-aware trajectory;
    - prints and returns the final temporal metrics.
    """
    results: dict[str, dict[str, Any]] = {}

    for regime in regimes:
        print("Preflight regime:", regime)

        dataset = make_synthetic_cmab_dataset(
            T=t,
            d=d,
            regime=regime,
            seed=seed,
        )

        groups = np.asarray(dataset["group"]).astype(str)
        oracle_actions = np.asarray(dataset["y_opt"], dtype=int)

        print("Groups:", np.unique(groups, return_counts=True))

        _, support = make_preprocessing_weights(
            preprocessing=preprocessing,
            groups=groups,
            oracle_actions=oracle_actions,
        )

        print("\nReweighting support:")
        _display_or_print(
            support,
            display_fn=display_fn,
        )

        expert_advice = None
        expert_names = None

        if regime == "adversarial_switching":
            expert_advice, expert_names = make_synthetic_expert_advice(
                dataset,
                n_experts=n_experts,
                seed=seed,
            )

            print("Expert advice shape:", expert_advice.shape)
            print("Expert names:", expert_names)

        if regime not in policies_by_regime:
            raise KeyError(f"No policies defined for regime: {regime}")

        if len(policies_by_regime[regime]) < 2:
            raise ValueError(
                f"Expected at least two policies for regime {regime}, "
                "with the fairness-aware policy in second position."
            )

        policy_name = str(policies_by_regime[regime][1])

        potential_rewards = make_synthetic_potential_rewards(
            dataset,
            environment_seed=seed,
            regime=regime,
        )

        logs_df, _ = replay_one_synthetic_trajectory(
            dataset=dataset,
            potential_rewards=potential_rewards,
            policy_name=policy_name,
            preprocessing=preprocessing,
            seed=seed,
            params=params,
            expert_advice=expert_advice,
        )

        final_metrics = logs_df[
            [
                "avg_reward",
                "cumulative_prediction_error",
                "DP_gap_over_time",
                "EO_gap_over_time",
                "UtilityGap_over_time",
            ]
        ].tail(1)

        print("Replay rows:", len(logs_df))
        print("Final metrics:")

        _display_or_print(
            final_metrics,
            display_fn=display_fn,
        )

        results[regime] = {
            "dataset": dataset,
            "support": support,
            "expert_advice": expert_advice,
            "expert_names": expert_names,
            "logs": logs_df,
            "final_metrics": final_metrics,
        }

    print("\nPreflight completed successfully.")

    return results


def run_imbalanced_synthetic_preflight(
    *,
    regimes: Sequence[str],
    policies_by_regime: Mapping[str, Sequence[str]],
    params: SyntheticPolicyParams,
    d: int,
    n_experts: int,
    minority_fraction: float,
    t: int = 250,
    seed: int = 42,
    preprocessing: str = "reweigh_group_label",
    display_fn: Callable[[Any], None] | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Run preflight checks for the imbalanced synthetic CMAB sensitivity analysis.

    For each regime, this function:
    - generates a short imbalanced synthetic dataset;
    - computes the Reweighting support table;
    - creates expert advice when required by EXP4;
    - replays one fairness-aware trajectory;
    - prints and returns the final temporal metrics.
    """
    results: dict[str, dict[str, Any]] = {}

    for regime in regimes:
        print("Imbalanced preflight regime:", regime)

        dataset = make_imbalanced_synthetic_cmab_dataset(
            T=t,
            d=d,
            regime=regime,
            seed=seed,
            minority_fraction=minority_fraction,
        )

        groups = np.asarray(dataset["group"]).astype(str)
        oracle_actions = np.asarray(dataset["y_opt"], dtype=int)

        print("Groups:", np.unique(groups, return_counts=True))

        _, support = make_preprocessing_weights(
            preprocessing=preprocessing,
            groups=groups,
            oracle_actions=oracle_actions,
        )

        print("\nReweighting support:")
        _display_or_print(
            support,
            display_fn=display_fn,
        )

        expert_advice = None
        expert_names = None

        if regime == "adversarial_switching":
            expert_advice, expert_names = make_synthetic_expert_advice(
                dataset,
                n_experts=n_experts,
                seed=seed,
            )

            print("Expert advice shape:", expert_advice.shape)
            print("Expert names:", expert_names)

        if regime not in policies_by_regime:
            raise KeyError(f"No policies defined for regime: {regime}")

        if len(policies_by_regime[regime]) < 2:
            raise ValueError(
                f"Expected at least two policies for regime {regime}, "
                "with the fairness-aware policy in second position."
            )

        policy_name = str(policies_by_regime[regime][1])

        potential_rewards = make_synthetic_potential_rewards(
            dataset,
            environment_seed=seed,
            regime=regime,
        )

        logs_df, _ = replay_one_synthetic_trajectory(
            dataset=dataset,
            potential_rewards=potential_rewards,
            policy_name=policy_name,
            preprocessing=preprocessing,
            seed=seed,
            params=params,
            expert_advice=expert_advice,
        )

        final_metrics = logs_df[
            [
                "avg_reward",
                "cumulative_prediction_error",
                "DP_gap_over_time",
                "EO_gap_over_time",
                "UtilityGap_over_time",
            ]
        ].tail(1)

        print("Replay rows:", len(logs_df))
        print("Final metrics:")

        _display_or_print(
            final_metrics,
            display_fn=display_fn,
        )

        results[regime] = {
            "dataset": dataset,
            "support": support,
            "expert_advice": expert_advice,
            "expert_names": expert_names,
            "logs": logs_df,
            "final_metrics": final_metrics,
        }

    print("\nImbalanced synthetic preflight completed successfully.")

    return results