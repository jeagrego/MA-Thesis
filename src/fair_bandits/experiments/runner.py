from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..metrics.summary import summarize_metrics
from ..metrics.temporal import (
    add_temporal_columns_single_run,
    aggregate_temporal_over_seeds,
)
from ..policies.base import BaseContextualPolicy


PolicyFactory = Callable[[int, np.ndarray, int], BaseContextualPolicy]


@dataclass
class ExperimentBundle:
    logs: pd.DataFrame
    seed_summary: pd.DataFrame
    temporal_summary: pd.DataFrame
    metadata: dict[str, Any]


def _add_constant_columns(df: pd.DataFrame, constants: dict[str, Any] | None) -> pd.DataFrame:
    out = df.copy()
    if constants:
        for k, v in constants.items():
            out[k] = v
    return out


def sample_idx_seq(
    *,
    n_obs: int,
    horizon: int,
    rng: np.random.Generator,
    probs: np.ndarray | None = None,
    replace: bool = True,
) -> np.ndarray:
    """
    Sample a sequence of row indices for offline replay.

    If probs is provided, it must sum to 1.
    """
    if n_obs <= 0:
        raise ValueError("n_obs must be positive.")
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if not replace and horizon > n_obs:
        raise ValueError("Cannot sample without replacement when horizon > n_obs.")

    if probs is not None:
        p = np.asarray(probs, dtype=float)
        if p.shape[0] != n_obs:
            raise ValueError("Length of probs must match n_obs.")
        if np.any(p < 0):
            raise ValueError("Sampling probabilities must be non-negative.")
        s = p.sum()
        if s <= 0:
            raise ValueError("Sampling probabilities must sum to a positive value.")
        p = p / s
    else:
        p = None

    return rng.choice(n_obs, size=horizon, replace=replace, p=p)


def run_bandit_on_index_sequence(
    *,
    X: np.ndarray,
    y: np.ndarray,
    group: np.ndarray,
    idx_seq: np.ndarray,
    policy: BaseContextualPolicy,
    policy_name: str | None = None,
    seed: int | None = None,
    rolling_window: int = 50,
    constants: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Run one policy on one fixed index sequence.
    The policy is updated online as it goes through the sequence.
    """
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=int)
    g_arr = np.asarray(group).astype(str)
    idx = np.asarray(idx_seq, dtype=int)

    rows: list[dict[str, Any]] = []

    for t, i in enumerate(idx, start=1):
        x_t = X_arr[i]
        y_t = int(y_arr[i])
        g_t = str(g_arr[i])

        action = int(policy.select(x_t, g_t))
        reward = float(action == y_t)

        policy.update(x_t, action, reward, g_t)

        rows.append(
            {
                "t": t,
                "original_index": int(i),
                "group": g_t,
                "y_true": y_t,
                "action": action,
                "reward": reward,
            }
        )

    logs = pd.DataFrame(rows)
    logs["policy"] = policy_name if policy_name is not None else policy.__class__.__name__
    if seed is not None:
        logs["seed"] = int(seed)

    logs = _add_constant_columns(logs, constants)
    logs = add_temporal_columns_single_run(logs, window=rolling_window)
    return logs


def summarize_seed_runs(
    logs_df: pd.DataFrame,
    *,
    extra_group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build one summary row per (seed, policy, condition) combination, aggregating over time.
    The final row of each seed's logs is used to get the final cumulative reward, regret, and fairness gap.
    """
    if logs_df.empty:
        return pd.DataFrame()

    group_cols = []
    if extra_group_cols:
        group_cols.extend(extra_group_cols)

    for col in ["policy", "seed"]:
        if col in logs_df.columns:
            group_cols.append(col)

    rows: list[dict[str, Any]] = []

    for keys, gdf in logs_df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = {col: val for col, val in zip(group_cols, keys)}
        gdf = gdf.sort_values("t").reset_index(drop=True)

        metrics = summarize_metrics(
            y_true=gdf["y_true"].to_numpy(),
            y_hat=gdf["action"].to_numpy(),
            group=gdf["group"].to_numpy(),
            reward=gdf["reward"].to_numpy(),
            oracle_reward=1.0,
        )
        row.update(metrics)

        final = gdf.iloc[-1]
        row["n_rounds"] = int(len(gdf))
        row["final_cum_reward"] = float(final["cum_reward"])
        row["final_avg_reward"] = float(final["avg_reward"])
        row["final_rolling_reward"] = float(final["rolling_reward"])
        row["final_dp_gap"] = float(final["dp_gap"])
        row["final_cum_regret"] = float(final["cum_regret"])

        rows.append(row)

    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def summarize_over_seeds(
    seed_summary: pd.DataFrame,
    *,
    metric_cols: list[str] | None = None,
    group_cols: list[str] | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Aggregate seed-level summaries across seeds.
    For each metric column, compute the mean and confidence interval across seeds.
    By default, group by all non-metric columns (except seed).
    """
    if seed_summary.empty:
        return pd.DataFrame()

    df = seed_summary.copy()

    if group_cols is None:
        group_cols = [c for c in df.columns if c in {"policy", "dataset", "sensitive", "preprocessing"}]

    if metric_cols is None:
        exclude = set(group_cols + ["seed"])
        metric_cols = [
            c for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
        ]

    rows: list[dict[str, Any]] = []

    for keys, gdf in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = {col: val for col, val in zip(group_cols, keys)}
        row["n_seeds"] = int(gdf["seed"].nunique()) if "seed" in gdf.columns else len(gdf)

        for metric in metric_cols:
            values = gdf[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_low"] = float(np.quantile(values, alpha / 2))
            row[f"{metric}_high"] = float(np.quantile(values, 1 - alpha / 2))

        rows.append(row)

    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def run_multi_seed_bandit_experiment(
    *,
    X: np.ndarray,
    y: np.ndarray,
    group: np.ndarray,
    policy_factories: dict[str, PolicyFactory],
    horizon: int,
    n_seeds: int = 10,
    seed0: int = 42,
    sampling_probs: np.ndarray | None = None,
    replace: bool = True,
    rolling_window: int = 50,
    condition_cols: dict[str, Any] | None = None,
) -> ExperimentBundle:
    """
    Run several policies over several seeds.

    Important design choice:
    for a given seed, all policies see the exact same index sequence.
    """
    if not policy_factories:
        raise ValueError("policy_factories cannot be empty.")

    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=int)
    g_arr = np.asarray(group).astype(str)

    all_logs: list[pd.DataFrame] = []

    for seed_offset in range(n_seeds):
        seed = int(seed0 + seed_offset)
        rng = np.random.default_rng(seed)

        idx_seq = sample_idx_seq(
            n_obs=X_arr.shape[0],
            horizon=horizon,
            rng=rng,
            probs=sampling_probs,
            replace=replace,
        )

        for policy_name, factory in policy_factories.items():
            policy = factory(X_arr.shape[1], g_arr, seed)

            logs = run_bandit_on_index_sequence(
                X=X_arr,
                y=y_arr,
                group=g_arr,
                idx_seq=idx_seq,
                policy=policy,
                policy_name=policy_name,
                seed=seed,
                rolling_window=rolling_window,
                constants=condition_cols,
            )
            all_logs.append(logs)

    logs_df = pd.concat(all_logs, ignore_index=True)

    extra_group_cols = list(condition_cols.keys()) if condition_cols else []
    seed_summary = summarize_seed_runs(logs_df, extra_group_cols=extra_group_cols)

    temporal_group_cols = [*extra_group_cols, "policy"] if "policy" in logs_df.columns else extra_group_cols
    temporal_summary = aggregate_temporal_over_seeds(
        logs_df,
        group_cols=temporal_group_cols,
        value_cols=["avg_reward", "rolling_reward", "dp_gap", "cum_regret"],
    )

    metadata = {
        "horizon": int(horizon),
        "n_seeds": int(n_seeds),
        "seed0": int(seed0),
        "replace": bool(replace),
        "rolling_window": int(rolling_window),
        "n_samples": int(X_arr.shape[0]),
        "n_features": int(X_arr.shape[1]),
        "policies": list(policy_factories.keys()),
    }
    if condition_cols:
        metadata.update(condition_cols)

    return ExperimentBundle(
        logs=logs_df,
        seed_summary=seed_summary,
        temporal_summary=temporal_summary,
        metadata=metadata,
    )
