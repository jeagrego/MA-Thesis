from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def _quantile_ci(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float, float]:
    """
    Simple nonparametric CI from empirical quantiles across seeds.
    """
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return 0.0, 0.0, 0.0
    return (
        float(v.mean()),
        float(np.quantile(v, alpha / 2)),
        float(np.quantile(v, 1 - alpha / 2)),
    )


def add_temporal_columns_single_run(
    logs_df: pd.DataFrame,
    *,
    window: int = 50,
    t_col: str = "t",
    reward_col: str = "reward",
    action_col: str = "action",
    group_col: str = "group",
    oracle_reward_col: str | None = None,
    positive_action: int = 1,
) -> pd.DataFrame:
    """
    Enrich a single-run log with temporal columns:
    - cum_reward
    - avg_reward
    - rolling_reward
    - cum_regret
    - dp_gap

    Assumes logs_df contains a single policy and a single seed.
    """
    if logs_df.empty:
        return logs_df.copy()

    df = logs_df.sort_values(t_col).reset_index(drop=True).copy()

    reward = df[reward_col].to_numpy(dtype=float)
    df["cum_reward"] = np.cumsum(reward)
    df["avg_reward"] = df["cum_reward"] / np.arange(1, len(df) + 1)

    df["rolling_reward"] = (
        pd.Series(reward, dtype=float)
        .rolling(window=window, min_periods=1)
        .mean()
        .to_numpy()
    ) # compute rolling average reward with specified window size
      # the min_periods=1 argument ensures that we get an average even for the first few time steps where the window is not full

    if oracle_reward_col is None:
        oracle = np.ones(len(df), dtype=float)
    else:
        oracle = df[oracle_reward_col].to_numpy(dtype=float)

    df["cum_regret"] = np.cumsum(oracle - reward)
    actions = df[action_col].to_numpy(dtype=int)
    groups = df[group_col].astype(str).to_numpy()
    uniq_groups = np.unique(groups)
    group_total = {g: 0 for g in uniq_groups}
    group_pos = {g: 0 for g in uniq_groups}
    dp_gap_values: list[float] = []
    for a, g in zip(actions, groups):
        group_total[g] += 1
        if int(a) == positive_action:
            group_pos[g] += 1

        rates = []
        for gg in uniq_groups:
            den = group_total[gg]
            rate = (group_pos[gg] / den) if den > 0 else 0.0
            rates.append(rate)

        dp_gap_values.append(float(max(rates) - min(rates)))

    df["dp_gap"] = dp_gap_values
    return df


def aggregate_temporal_over_seeds(
    logs_df: pd.DataFrame,
    *,
    value_cols: Iterable[str] = ("avg_reward", "rolling_reward", "dp_gap", "cum_regret"),
    group_cols: list[str] | None = None,
    t_col: str = "t",
    seed_col: str = "seed",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Aggregate temporal trajectories across seeds.

    Output columns for each value_col:
    - <value>_mean
    - <value>_low
    - <value>_high
    """
    if logs_df.empty:
        return pd.DataFrame()

    if group_cols is None:
        group_cols = ["policy"] if "policy" in logs_df.columns else []

    group_keys = [*group_cols, t_col]
    rows: list[dict] = []

    for keys, gdf in logs_df.groupby(group_keys, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = {col: val for col, val in zip(group_keys, keys)}
        row["n_seeds"] = int(gdf[seed_col].nunique()) if seed_col in gdf.columns else len(gdf)

        for value_col in value_cols:
            if value_col not in gdf.columns:
                continue
            mean_, low_, high_ = _quantile_ci(gdf[value_col].to_numpy(dtype=float), alpha=alpha)
            row[f"{value_col}_mean"] = mean_
            row[f"{value_col}_low"] = low_
            row[f"{value_col}_high"] = high_

        rows.append(row)

    out = pd.DataFrame(rows).sort_values(group_keys).reset_index(drop=True)
    return out
