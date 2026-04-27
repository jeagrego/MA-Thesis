from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _apply_filters(df: pd.DataFrame, filters: dict[str, Any] | None) -> pd.DataFrame:
    """
    Applies filters to the DataFrame, returning a new filtered DataFrame.
    """
    out = df.copy()
    if filters:
        for col, val in filters.items():
            out = out[out[col] == val]
    return out


def plot_benchmark_bars(
    seed_summary: pd.DataFrame,
    *,
    metric: str,
    filters: dict[str, Any] | None = None,
    policy_col: str = "policy",
    seed_col: str = "seed",
    policy_order: list[str] | None = None,
    alpha: float = 0.05,
    ax=None,
    title: str | None = None,
    rotate_xticks: int = 0,
):
    """
    Plots a bar chart comparing the specified metric across policies, with error bars representing 
    confidence intervals across seeds.

    Parameters:
    - seed_summary: DataFrame containing per-seed metrics for each policy.
    - metric: The column name of the metric to plot.
    - filters: Optional dict of column-value pairs to filter the DataFrame before plotting.
    - policy_col: The column name identifying the policy in seed_summary.
    - seed_col: The column name identifying the seed in seed_summary.
    - policy_order: Optional list specifying the order of policies on the x-axis. 
      If None, uses the order they appear in the DataFrame.
    - alpha: Significance level for confidence intervals (default 0.05 for 95% CI).
    - ax: Optional matplotlib Axes to plot on. If None, creates a new figure and axes.
    - title: Optional title for the plot.
    - rotate_xticks: Degrees to rotate x-axis tick labels (default 0).

    Returns:
    - ax: The matplotlib Axes object containing the plot.
    """
    df = _apply_filters(seed_summary, filters)
    if df.empty:
        raise ValueError("No rows left after applying filters.")
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found in seed_summary.")

    policies = policy_order if policy_order is not None else df[policy_col].drop_duplicates().tolist()

    means = []
    low_err = []
    high_err = []

    for policy in policies:
        values = df.loc[df[policy_col] == policy, metric].to_numpy(dtype=float)
        if values.size == 0:
            means.append(np.nan)
            low_err.append(0.0)
            high_err.append(0.0)
            continue

        mean_ = float(values.mean())
        low_ = float(np.quantile(values, alpha / 2))
        high_ = float(np.quantile(values, 1 - alpha / 2))

        means.append(mean_)
        low_err.append(mean_ - low_)
        high_err.append(high_ - mean_)

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    x = np.arange(len(policies))
    ax.bar(
        x,
        means,
        yerr=np.vstack([low_err, high_err]),
        capsize=4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(policies, rotation=rotate_xticks)
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} across policies")
    ax.grid(True, axis="y", alpha=0.3)
    return ax
