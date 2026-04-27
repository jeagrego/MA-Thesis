from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_synth_temporal_metric(
    temporal_df: pd.DataFrame,
    regime: str,
    T: int,
    metric: str,
    policies: list[str] | None = None,
    title: str | None = None,
    ax=None,
):
    """Plot a temporal metric for the synthetic benchmark."""
    sub = temporal_df[
        (temporal_df["regime"] == regime)
        & (temporal_df["T"] == T)
        & (temporal_df["metric"] == metric)
    ].copy()

    if policies is not None:
        sub = sub[sub["policy"].isin(policies)].copy()

    if sub.empty:
        raise ValueError(f"No data for regime={regime}, T={T}, metric={metric}")

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    for policy, grp in sub.groupby("policy"):
        grp = grp.sort_values("t")
        x = grp["t"].to_numpy()
        y = grp["mean"].to_numpy()
        ylow = grp["ci_low"].to_numpy()
        yhigh = grp["ci_high"].to_numpy()

        ax.plot(x, y, label=policy)
        ax.fill_between(x, ylow, yhigh, alpha=0.2)

    ax.set_xlabel("Round")
    ax.set_ylabel(metric)
    ax.set_title(title if title else f"{metric} over time | {regime} | T={T}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax