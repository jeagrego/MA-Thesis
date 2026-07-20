from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from ..metrics.temporal import aggregate_temporal_over_seeds


def _apply_filters(df: pd.DataFrame, filters: dict[str, Any] | None) -> pd.DataFrame:
    """
    Applies filters to the DataFrame, returning a new filtered DataFrame.
    """
    out = df.copy()
    if filters:
        for col, val in filters.items():
            out = out[out[col] == val]
    return out


def _plot_temporal_metric(
    logs_df: pd.DataFrame,
    *,
    metric: str,
    ylabel: str,
    title: str | None = None,
    filters: dict[str, Any] | None = None,
    policy_col: str = "policy",
    t_col: str = "t",
    seed_col: str = "seed",
    policy_order: list[str] | None = None,
    alpha: float = 0.05,
    ax=None,
):
    """
    Plots a temporal metric over time with confidence intervals across seeds.
    """
    df = _apply_filters(logs_df, filters)
    if df.empty:
        raise ValueError("No rows left after applying filters.")

    agg = aggregate_temporal_over_seeds(
        df,
        group_cols=[policy_col],
        t_col=t_col,
        seed_col=seed_col,
        value_cols=[metric],
        alpha=alpha,
    )    

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    policies = policy_order if policy_order is not None else agg[policy_col].drop_duplicates().tolist()

    for policy in policies:
        sub = agg[agg[policy_col] == policy].sort_values(t_col)
        if sub.empty:
            continue

        ax.plot(sub[t_col], sub[f"{metric}_mean"], label=policy)
        ax.fill_between(
            sub[t_col],
            sub[f"{metric}_low"],
            sub[f"{metric}_high"],
            alpha=0.2,
        )

    ax.set_xlabel("Round")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_average_reward_over_time(
    logs_df: pd.DataFrame,
    *,
    filters: dict[str, Any] | None = None,
    policy_order: list[str] | None = None,
    alpha: float = 0.05,
    ax=None,
    title: str | None = None,
):
    """
    Plots the average reward over time for each policy, with confidence intervals across seeds.
    """
    return _plot_temporal_metric(
        logs_df,
        metric="avg_reward",
        ylabel="Average reward",
        title=title or "Average reward over time",
        filters=filters,
        policy_order=policy_order,
        alpha=alpha,
        ax=ax,
    )



def plot_rolling_reward(
    logs_df: pd.DataFrame,
    *,
    filters: dict[str, Any] | None = None,
    policy_order: list[str] | None = None,
    alpha: float = 0.05,
    ax=None,
    title: str | None = None,
):
    """
    Plots the rolling reward over time for each policy, with confidence intervals across seeds.
    """
    return _plot_temporal_metric(
        logs_df,
        metric="rolling_reward",
        ylabel="Rolling reward",
        title=title or "Rolling reward",
        filters=filters,
        policy_order=policy_order,
        alpha=alpha,
        ax=ax,
    )


def plot_dp_gap_over_time(
    logs_df: pd.DataFrame,
    *,
    filters: dict[str, Any] | None = None,
    policy_order: list[str] | None = None,
    alpha: float = 0.05,
    ax=None,
    title: str | None = None,
):
    """
    Plots the demographic parity gap over time for each policy, with confidence intervals across seeds.
    """
    return _plot_temporal_metric(
        logs_df,
        metric="dp_gap",
        ylabel="DP-gap",
        title=title or "Demographic parity gap over time",
        filters=filters,
        policy_order=policy_order,
        alpha=alpha,
        ax=ax,
    )

