from __future__ import annotations

from typing import Iterable

from .naming import normalize_metric_columns

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
    
def _finite_range_or_nan(
    values: list[float],
    *,
    expected_count: int,
) -> float:
    """
    Return max - min when all expected group-specific values are available.
    Otherwise return NaN.
    """
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]

    if len(array) < expected_count:
        return np.nan

    return float(array.max() - array.min())


def add_synthetic_temporal_metrics(
    logs_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add cumulative reward, cumulative prediction error, DP, EO, and UtilityGap
    over time for one synthetic CMAB trajectory.
    """
    if logs_df.empty:
        return logs_df.copy()

    out = normalize_metric_columns(logs_df)

    required_columns = [
        "reward",
        "prediction_error_increment",
        "group",
        "action",
        "oracle_action",
    ]

    missing_columns = [
        column for column in required_columns if column not in out.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Cannot compute synthetic temporal metrics. Missing columns: "
            f"{missing_columns}"
        )

    out = out.sort_values("t").reset_index(drop=True).copy()

    out["cum_reward"] = out["reward"].cumsum()
    out["avg_reward"] = out["cum_reward"] / np.arange(1, len(out) + 1)

    out["cumulative_prediction_error"] = out[
        "prediction_error_increment"
    ].cumsum()

    # Backward-compatible alias only.
    out["cumulative_regret"] = out["cumulative_prediction_error"]

    groups = sorted(out["group"].astype(str).unique().tolist())
    n_groups = len(groups)

    count = {group: 0 for group in groups}
    positive_action_count = {group: 0 for group in groups}
    reward_sum = {group: 0.0 for group in groups}

    true_positive_denominator = {group: 0 for group in groups}
    true_positive_numerator = {group: 0 for group in groups}
    false_positive_denominator = {group: 0 for group in groups}
    false_positive_numerator = {group: 0 for group in groups}

    dp_gap_values = []
    tpr_gap_values = []
    fpr_gap_values = []
    eo_gap_values = []
    utility_gap_values = []

    for row in out.itertuples():
        group = str(row.group)
        action = int(row.action)
        oracle_action = int(row.oracle_action)
        reward = float(row.reward)

        count[group] += 1
        positive_action_count[group] += int(action == 1)
        reward_sum[group] += reward

        if oracle_action == 1:
            true_positive_denominator[group] += 1
            true_positive_numerator[group] += int(action == 1)
        else:
            false_positive_denominator[group] += 1
            false_positive_numerator[group] += int(action == 1)

        dp_rates = [
            positive_action_count[g] / count[g] if count[g] > 0 else np.nan
            for g in groups
        ]

        utility_rates = [
            reward_sum[g] / count[g] if count[g] > 0 else np.nan
            for g in groups
        ]

        tpr_rates = [
            true_positive_numerator[g] / true_positive_denominator[g]
            if true_positive_denominator[g] > 0
            else np.nan
            for g in groups
        ]

        fpr_rates = [
            false_positive_numerator[g] / false_positive_denominator[g]
            if false_positive_denominator[g] > 0
            else np.nan
            for g in groups
        ]

        dp_gap = _finite_range_or_nan(dp_rates, expected_count=n_groups)
        utility_gap = _finite_range_or_nan(
            utility_rates,
            expected_count=n_groups,
        )
        tpr_gap = _finite_range_or_nan(tpr_rates, expected_count=n_groups)
        fpr_gap = _finite_range_or_nan(fpr_rates, expected_count=n_groups)

        eo_gap = (
            max(tpr_gap, fpr_gap)
            if np.isfinite(tpr_gap) and np.isfinite(fpr_gap)
            else np.nan
        )

        dp_gap_values.append(dp_gap)
        utility_gap_values.append(utility_gap)
        tpr_gap_values.append(tpr_gap)
        fpr_gap_values.append(fpr_gap)
        eo_gap_values.append(eo_gap)

    out["DP_gap_over_time"] = dp_gap_values
    out["TPR_gap_over_time"] = tpr_gap_values
    out["FPR_gap_over_time"] = fpr_gap_values
    out["EO_gap_over_time"] = eo_gap_values
    out["UtilityGap_over_time"] = utility_gap_values

    return out

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
    value_cols: Iterable[str] = ("avg_reward", "rolling_reward", "dp_gap", "cumulative_prediction_error",),
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
