from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_POLICY_LABELS: dict[str, str] = {
    "LinUCB": "LinUCB",
    "FairLinUCB_DP": "FairLinUCB-DP",
    "LinTS": "LinTS",
    "FairLinTS_DP": "FairLinTS-DP",
    "EXP4": "EXP4",
    "FairEXP4_DP": "FairEXP4-DP",
}


DEFAULT_PREPROCESSING_LABELS: dict[str, str] = {
    "uniform": "Uniform",
    "reweigh_group_label": "Reweighting",
}

def friendly_label(
    value: str,
    mapping: dict[str, str] | None = None,
) -> str:
    """
    Return a human-readable label while keeping unknown values unchanged.
    """
    if mapping is None:
        mapping = {}

    return mapping.get(str(value), str(value))

def curve_style(
    *,
    policy: str,
    preprocessing: str,
) -> dict[str, str]:
    """
    Use a fixed visual convention for synthetic CMAB curves.

    Baseline policies:
        LinUCB, LinTS, EXP4 -> blue

    Fairness-aware policies:
        FairLinUCB, FairLinTS, FairEXP4 -> red

    Uniform preprocessing:
        solid line

    Reweighting preprocessing:
        dotted line
    """
    policy_text = str(policy)
    preprocessing_text = str(preprocessing)

    color = "red" if policy_text.startswith("Fair") else "blue"
    linestyle = ":" if preprocessing_text == "reweigh_group_label" else "-"

    return {
        "color": color,
        "linestyle": linestyle,
    }

def aggregate_temporal_mean_sd(
    dataframe: pd.DataFrame,
    *,
    metric: str,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Aggregate a temporal synthetic metric across seeds.

    Expected input:
        one row per regime x preprocessing x policy x seed x t.

    Output:
        one row per regime x preprocessing x policy x t, with mean and SD.
    """
    if group_cols is None:
        group_cols = [
            "regime",
            "preprocessing",
            "policy",
            "t",
        ]

    if dataframe.empty:
        return pd.DataFrame()

    if metric not in dataframe.columns:
        raise KeyError(
            f"Metric '{metric}' not found. "
            f"Available columns are: {dataframe.columns.tolist()}"
        )

    missing_group_cols = [
        column for column in group_cols if column not in dataframe.columns
    ]

    if missing_group_cols:
        raise KeyError(
            f"Missing grouping columns: {missing_group_cols}. "
            f"Available columns are: {dataframe.columns.tolist()}"
        )

    summary = (
        dataframe
        .groupby(group_cols, as_index=False)[metric]
        .agg(
            mean="mean",
            sd="std",
            n="count",
        )
    )

    summary["sd"] = summary["sd"].fillna(0.0)
    summary["n"] = summary["n"].astype(int)
    summary["se"] = summary["sd"] / np.sqrt(summary["n"].clip(lower=1))

    try:
        from scipy.stats import t as student_t

        summary["t_critical"] = summary["n"].apply(
            lambda n: float(student_t.ppf(0.975, df=int(n) - 1)) if int(n) > 1 else 0.0
        )
    except Exception:
        summary["t_critical"] = 1.96
        summary.loc[summary["n"] <= 1, "t_critical"] = 0.0

    summary["ci_half_width"] = summary["t_critical"] * summary["se"]
    summary.loc[summary["n"] <= 1, "ci_half_width"] = 0.0

    summary["low"] = summary["mean"] - summary["ci_half_width"]
    summary["high"] = summary["mean"] + summary["ci_half_width"]

    return summary

def plot_curve_with_sd(
    ax,
    curve: pd.DataFrame,
    *,
    x_col: str = "t",
    label: str,
    color: str | None = None,
    linestyle: str = "-",
    linewidth: float = 2.0,
    alpha_line: float = 1.0,
    alpha_band: float = 0.12,
) -> None:
    """
    Draw temporal mean ± SD curve.
    """
    x = curve[x_col].to_numpy(dtype=float)
    mean = curve["mean"].to_numpy(dtype=float)
    low = curve["low"].to_numpy(dtype=float)
    high = curve["high"].to_numpy(dtype=float)

    ax.plot(
        x,
        mean,
        label=label,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        alpha=alpha_line,
    )

    ax.fill_between(
        x,
        low,
        high,
        color=color,
        alpha=alpha_band,
    )

def plot_synthetic_preprocessing_fairness(
    temporal_df: pd.DataFrame,
    *,
    regime: str,
    output_dir: str | Path,
    policies: list[str],
    preprocessings: list[str],
    regime_label: str,
    regime_file_tag: str,
    policy_labels: dict[str, str] | None = None,
    preprocessing_labels: dict[str, str] | None = None,
    zoom_start: int = 250,
    show: bool = True,
) -> Path:
    """
    Plot preprocessing comparison for DP and Equalized Odds Gaps in one two-panel figure.
    """
    policy_labels = DEFAULT_POLICY_LABELS | (policy_labels or {})
    preprocessing_labels = DEFAULT_PREPROCESSING_LABELS | (preprocessing_labels or {})

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subset = temporal_df[temporal_df["regime"] == regime].copy()

    if subset.empty:
        raise ValueError(f"No temporal data available for regime={regime}.")

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    fig.suptitle(
        f"Synthetic CMAB | {regime_label} | Preprocessing comparison",
        fontweight="bold",
        y=1.02,
    )

    for ax, metric_col, ylabel in [
        (axes[0], "DP_gap_over_time", "Demographic Parity Gap"),
        (axes[1], "EO_gap_over_time", "Equalized Odds Gap"),
    ]:
        summary = aggregate_temporal_mean_sd(
            subset,
            metric=metric_col,
            group_cols=["policy", "preprocessing", "t"],
        )

        summary = summary[summary["t"] >= int(zoom_start)].copy()

        for policy in policies:
            for preprocessing in preprocessings:
                curve = summary[
                    (summary["policy"] == policy)
                    & (summary["preprocessing"] == preprocessing)
                ].sort_values("t")

                if curve.empty:
                    continue

                label = (
                    f"{friendly_label(policy, policy_labels)} | "
                    f"{friendly_label(preprocessing, preprocessing_labels)}"
                )

                style = curve_style(policy=policy, preprocessing=preprocessing,)
                plot_curve_with_sd(ax,curve,label=label,**style,)

        ax.set_xlabel("Rounds")
        ax.set_ylabel(ylabel)
        ax.set_title(
            f"Preprocessing | {ylabel} over time",
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()

    path = output_dir / f"synthetic_{regime_file_tag}_preprocessing_fairness.png"

    fig.savefig(path, dpi=300, bbox_inches="tight")
    print("Saved:", path)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return path


def plot_synthetic_inprocessing_fairness(
    temporal_df: pd.DataFrame,
    *,
    regime: str,
    output_dir: str | Path,
    policies: list[str],
    regime_label: str,
    regime_file_tag: str,
    policy_labels: dict[str, str] | None = None,
    zoom_start: int = 250,
    show: bool = True,
) -> Path:
    """
    Plot Demographic Parity Gap and Equalized Odds Gap over time for one synthetic regime, 
    comparing multiple in-processing policies under uniform preprocessing.
    """
    policy_labels = DEFAULT_POLICY_LABELS | (policy_labels or {})

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subset = temporal_df[
        (temporal_df["regime"] == regime)
        & (temporal_df["preprocessing"] == "uniform")
    ].copy()

    if subset.empty:
        raise ValueError(
            f"No temporal data available for regime={regime} "
            "and preprocessing='uniform'."
        )

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    fig.suptitle(
        f"Synthetic CMAB | {regime_label} | "
        "In-processing comparison under uniform preprocessing",
        fontweight="bold",
        y=1.02,
    )

    for ax, metric_col, ylabel in [
        (axes[0], "DP_gap_over_time", "Demographic Parity Gap"),
        (axes[1], "EO_gap_over_time", "Equalized Odds Gap"),
    ]:
        summary = aggregate_temporal_mean_sd(
            subset,
            metric=metric_col,
            group_cols=["policy", "t"],
        )

        summary = summary[summary["t"] >= int(zoom_start)].copy()

        for policy in policies:
            curve = summary[summary["policy"] == policy].sort_values("t")

            if curve.empty:
                continue

            style = curve_style(policy=policy,preprocessing="uniform",)
            plot_curve_with_sd(ax,curve,label=friendly_label(policy, policy_labels),**style,)

        ax.set_xlabel("Rounds")
        ax.set_ylabel(ylabel)
        ax.set_title(
            f"In-processing | {ylabel} over time",
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()

    path = output_dir / f"synthetic_{regime_file_tag}_inprocessing_fairness.png"

    fig.savefig(path, dpi=300, bbox_inches="tight")
    print("Saved:", path)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return path


def plot_synthetic_performance_metric(
    temporal_df: pd.DataFrame,
    *,
    regime: str,
    metric_col: str,
    ylabel: str,
    filename_suffix: str,
    output_dir: str | Path,
    policies: list[str],
    preprocessings: list[str],
    regime_label: str,
    regime_file_tag: str,
    policy_labels: dict[str, str] | None = None,
    preprocessing_labels: dict[str, str] | None = None,
    zoom_start: int = 250,
    show: bool = True,
) -> Path:
    """
    Plot one performance metric (average reward or cumulative prediction error)
    for one synthetic regime, comparing multiple policies and preprocessings.
    """
    policy_labels = DEFAULT_POLICY_LABELS | (policy_labels or {})
    preprocessing_labels = DEFAULT_PREPROCESSING_LABELS | (preprocessing_labels or {})

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subset = temporal_df[temporal_df["regime"] == regime].copy()

    if subset.empty:
        raise ValueError(f"No temporal data available for regime={regime}.")

    if metric_col not in subset.columns:
        raise KeyError(
            f"Metric '{metric_col}' not found. "
            f"Available columns are: {subset.columns.tolist()}"
        )

    summary = aggregate_temporal_mean_sd(
        subset,
        metric=metric_col,
        group_cols=["policy", "preprocessing", "t"],
    )

    summary = summary[summary["t"] >= int(zoom_start)].copy()

    fig, ax = plt.subplots(figsize=(10, 5))

    for policy in policies:
        for preprocessing in preprocessings:
            curve = summary[
                (summary["policy"] == policy)
                & (summary["preprocessing"] == preprocessing)
            ].sort_values("t")

            if curve.empty:
                continue

            label = (
                f"{friendly_label(policy, policy_labels)} | "
                f"{friendly_label(preprocessing, preprocessing_labels)}"
            )

            style = curve_style(policy=policy,preprocessing=preprocessing,)
            plot_curve_with_sd(ax,curve,label=label,**style,)

    ax.set_title(f"Synthetic CMAB | {regime_label} | {ylabel} over time",fontweight="bold",)

    ax.set_xlabel("Rounds")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = output_dir / f"synthetic_{regime_file_tag}_{filename_suffix}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print("Saved:", path)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return path