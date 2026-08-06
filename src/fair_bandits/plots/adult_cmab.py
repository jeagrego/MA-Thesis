from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_ADULT_POLICY_LABELS = {
    "LinUCB": "LinUCB",
    "FairLinUCB": "FairLinUCB",
    "FairLinUCB+PP": "FairLinUCB+PP",
    "LinTS": "LinTS",
    "FairLinTS": "FairLinTS",
    "FairLinTS+PP": "FairLinTS+PP",
    "EXP4": "EXP4",
    "FairEXP4": "FairEXP4",
    "FairEXP4+PP": "FairEXP4+PP",
}


DEFAULT_ADULT_PREPROCESSING_LABELS = {
    "uniform": "Uniform",
    "reweigh_group_label": "Reweighting",
}


DEFAULT_ADULT_FAMILY_LABELS = {
    "linucb": "LinUCB",
    "linear_ts": "Linear Thompson Sampling",
    "exp4": "EXP4",
}


def adult_friendly_label(value: str, labels: dict[str, str] | None = None) -> str:
    """
    Return a friendly label for an Adult policy or preprocessing value.
    """
    labels = labels or {}
    return labels.get(str(value), str(value))


def adult_curve_style(
    *,
    policy: str,
    preprocessing: str = "uniform",
) -> dict[str, str]:
    """
    Return a style dictionary for an Adult policy and preprocessing combination.
    """
    
    policy_text = str(policy)

    if policy_text in {"LinUCB", "LinTS", "EXP4"}:
        color = "blue"
    elif policy_text in {"FairLinUCB", "FairLinTS", "FairEXP4"}:
        color = "red"
    elif policy_text in {"FairLinUCB+PP", "FairLinTS+PP", "FairEXP4+PP"}:
        color = "green"
    else:
        color = "black"

    linestyle = ":" if str(preprocessing) == "reweigh_group_label" else "-"

    return {
        "color": color,
        "linestyle": linestyle,
    }

def _t_critical(
    n: int,
    confidence: float = 0.95,
) -> float:
    """
    Return the two-sided Student-t critical value.

    If scipy is unavailable, fall back to the normal approximation.
    """
    n = int(n)

    if n <= 1:
        return 0.0

    try:
        from scipy.stats import t as student_t

        alpha = 1.0 - float(confidence)
        return float(student_t.ppf(1.0 - alpha / 2.0, df=n - 1))

    except Exception:
        return 1.96


def add_mean_ci_columns(
    summary: pd.DataFrame,
    *,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """
    Add standard error and pointwise confidence interval columns.

    Expected columns:
    - mean
    - sd
    - n

    Output columns:
    - se
    - ci_half_width
    - low
    - high
    """
    out = summary.copy()

    out["sd"] = out["sd"].fillna(0.0)
    out["n"] = out["n"].astype(int)

    out["se"] = out["sd"] / np.sqrt(out["n"].clip(lower=1))

    out["t_critical"] = out["n"].apply(
        lambda n: _t_critical(
            n,
            confidence=confidence,
        )
    )

    out["ci_half_width"] = out["t_critical"] * out["se"]

    out.loc[out["n"] <= 1, "ci_half_width"] = 0.0

    out["low"] = out["mean"] - out["ci_half_width"]
    out["high"] = out["mean"] + out["ci_half_width"]

    return out

def aggregate_adult_curve(
    dataframe: pd.DataFrame,
    metric: str,
    group_cols: list[str],
    *,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """
    Aggregate an Adult temporal curve across seeds.

    The plotted band is the pointwise confidence interval of the mean:
        mean ± t_(0.975, n-1) × SD / sqrt(n)

    The unit of replication is the full experimental run/seed.
    """
    summary = (
        dataframe.groupby(group_cols + ["t"], as_index=False)[metric]
        .agg(
            mean="mean",
            sd="std",
            n="count",
        )
    )

    return add_mean_ci_columns(
        summary,
        confidence=confidence,
    )


def draw_adult_curve(
    ax,
    curve: pd.DataFrame,
    *,
    label: str,
    color: str | None = None,
    linestyle: str = "-",
    linewidth: float = 2.0,
    alpha_band: float = 0.12,
) -> None:
    """
    Draw an Adult temporal mean curve with pointwise 95% CI bands.
    """
    if curve.empty:
        return

    ax.plot(
        curve["t"],
        curve["mean"],
        label=label,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        alpha=0.95,
    )

    ax.fill_between(
        curve["t"].to_numpy(dtype=float),
        curve["low"].to_numpy(dtype=float),
        curve["high"].to_numpy(dtype=float),
        color=color,
        alpha=alpha_band,
    )


def plot_adult_preprocessing_fairness(
    temporal_df: pd.DataFrame,
    *,
    family: str,
    family_label: str,
    policies: list[str],
    preprocessings: list[str],
    fig_dir: str | Path,
    policy_labels: dict[str, str] | None = None,
    preprocessing_labels: dict[str, str] | None = None,
    show: bool = True,
) -> Path:
    """
    Plot fairness metrics for Adult policies with different preprocessing methods.
    """
    df = temporal_df[temporal_df["family"] == family].copy() if "family" in temporal_df.columns else temporal_df.copy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    fig.suptitle(
        f"ADULT | sensitive=sex | {family_label} | Preprocessing comparison",
        fontweight="bold",
        y=1.02,
    )

    for ax, metric, ylabel in [
        (axes[0], "DP_gap", "Demographic Parity Gap"),
        (axes[1], "EO_gap", "Equalized Odds Gap"),
    ]:
        summary = aggregate_adult_curve(df, metric, ["policy", "preprocessing"])

        for policy in policies:
            for preprocessing in preprocessings:
                curve = summary[
                    (summary["policy"] == policy)
                    & (summary["preprocessing"] == preprocessing)
                ].sort_values("t")

                label = (
                    f"{adult_friendly_label(policy, policy_labels)} | "
                    f"{adult_friendly_label(preprocessing, preprocessing_labels)}"
                )

                style = adult_curve_style(policy=policy, preprocessing=preprocessing)
                draw_adult_curve(ax, curve, label=label, **style)

        ax.set_title(f"{ylabel} over time", fontweight="bold")
        ax.set_xlabel("Rounds")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    output_path = fig_dir / f"adult_sex_{family}_preprocessing_fairness.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print("Saved:", output_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return output_path


def plot_adult_inprocessing_fairness(
    temporal_df: pd.DataFrame,
    *,
    family: str,
    family_label: str,
    policies: list[str],
    fig_dir: str | Path,
    policy_labels: dict[str, str] | None = None,
    show: bool = True,
) -> Path:
    """
    Plot fairness metrics for Adult policies with different in-processing methods.
    """
    df = temporal_df[temporal_df["family"] == family].copy() if "family" in temporal_df.columns else temporal_df.copy()
    df = df[df["preprocessing"] == "uniform"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    fig.suptitle(
        f"ADULT | sensitive=sex | {family_label} | In-processing comparison",
        fontweight="bold",
        y=1.02,
    )

    for ax, metric, ylabel in [
        (axes[0], "DP_gap", "Demographic Parity Gap"),
        (axes[1], "EO_gap", "Equalized Odds Gap"),
    ]:
        summary = aggregate_adult_curve(df, metric, ["policy"])

        for policy in policies:
            curve = summary[summary["policy"] == policy].sort_values("t")
            label = adult_friendly_label(policy, policy_labels)
            style = adult_curve_style(policy=policy, preprocessing="uniform")
            draw_adult_curve(ax, curve, label=label, **style)

        ax.set_title(f"{ylabel} over time", fontweight="bold")
        ax.set_xlabel("Rounds")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    output_path = fig_dir / f"adult_sex_{family}_inprocessing_fairness.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print("Saved:", output_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return output_path


def plot_adult_performance_metric(
    temporal_df: pd.DataFrame,
    *,
    family: str,
    family_label: str,
    metric_col: str,
    ylabel: str,
    filename_suffix: str,
    policies: list[str],
    preprocessings: list[str],
    fig_dir: str | Path,
    policy_labels: dict[str, str] | None = None,
    preprocessing_labels: dict[str, str] | None = None,
    show: bool = True,
) -> Path:
    """
    Plot a performance metric for Adult policies with different preprocessing methods.
    """
    df = temporal_df[temporal_df["family"] == family].copy() if "family" in temporal_df.columns else temporal_df.copy()
    summary = aggregate_adult_curve(df, metric_col, ["policy", "preprocessing"])

    fig, ax = plt.subplots(figsize=(10, 5))

    for policy in policies:
        for preprocessing in preprocessings:
            curve = summary[
                (summary["policy"] == policy)
                & (summary["preprocessing"] == preprocessing)
            ].sort_values("t")

            label = (
                f"{adult_friendly_label(policy, policy_labels)} | "
                f"{adult_friendly_label(preprocessing, preprocessing_labels)}"
            )

            style = adult_curve_style(policy=policy, preprocessing=preprocessing)
            draw_adult_curve(ax, curve, label=label, **style)

    ax.set_title(
        f"ADULT | sensitive=sex | {family_label} | {ylabel} over time",
        fontweight="bold",
    )
    ax.set_xlabel("Rounds")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    output_path = fig_dir / f"adult_sex_{family}_{filename_suffix}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print("Saved:", output_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return output_path


def plot_adult_family_figure_set(
    *,
    temporal_df: pd.DataFrame,
    family: str,
    family_label: str,
    policies: list[str],
    preprocessings: list[str],
    fig_dir: str | Path,
    policy_labels: dict[str, str] | None = None,
    preprocessing_labels: dict[str, str] | None = None,
    show: bool = True,
) -> list[Path]:
    """
    Generate the five standard online Adult figures for one policy family.
    """
    paths = []

    paths.append(
        plot_adult_preprocessing_fairness(
            temporal_df,
            family=family,
            family_label=family_label,
            policies=policies,
            preprocessings=preprocessings,
            fig_dir=fig_dir,
            policy_labels=policy_labels,
            preprocessing_labels=preprocessing_labels,
            show=show,
        )
    )

    paths.append(
        plot_adult_inprocessing_fairness(
            temporal_df,
            family=family,
            family_label=family_label,
            policies=policies,
            fig_dir=fig_dir,
            policy_labels=policy_labels,
            show=show,
        )
    )

    paths.append(
        plot_adult_performance_metric(
            temporal_df,
            family=family,
            family_label=family_label,
            metric_col="average_reward",
            ylabel="Average reward",
            filename_suffix="average_reward",
            policies=policies,
            preprocessings=preprocessings,
            fig_dir=fig_dir,
            policy_labels=policy_labels,
            preprocessing_labels=preprocessing_labels,
            show=show,
        )
    )

    paths.append(
        plot_adult_performance_metric(
            temporal_df,
            family=family,
            family_label=family_label,
            metric_col="cumulative_prediction_error",
            ylabel="Cumulative prediction error",
            filename_suffix="cumulative_prediction_error",
            policies=policies,
            preprocessings=preprocessings,
            fig_dir=fig_dir,
            policy_labels=policy_labels,
            preprocessing_labels=preprocessing_labels,
            show=show,
        )
    )

    paths.append(
        plot_adult_performance_metric(
            temporal_df,
            family=family,
            family_label=family_label,
            metric_col="UtilityGap",
            ylabel="UtilityGap",
            filename_suffix="utility_gap",
            policies=policies,
            preprocessings=preprocessings,
            fig_dir=fig_dir,
            policy_labels=policy_labels,
            preprocessing_labels=preprocessing_labels,
            show=show,
        )
    )

    return paths


def plot_adult_postprocessing_metric(
    postproc_df: pd.DataFrame,
    *,
    family: str,
    family_label: str,
    metric: str,
    ylabel: str,
    filename_suffix: str,
    fig_dir: str | Path,
    policy_labels: dict[str, str] | None = None,
    preprocessing_labels: dict[str, str] | None = None,
    show: bool = True,
) -> Path:
    """
    Plot fairness metrics for Adult policies with different post-processing methods.
    """
    df = postproc_df[postproc_df["family"] == family].copy() if "family" in postproc_df.columns else postproc_df.copy()
    summary = (df.groupby(["policy", "preprocessing"], as_index=False)[metric].agg(mean="mean", sd="std", n="count",))
    summary = add_mean_ci_columns(summary, confidence=0.95)
    fig, ax = plt.subplots(figsize=(10, 5))

    for _, row in summary.iterrows():
        policy = str(row["policy"])
        preprocessing = str(row["preprocessing"])
        label = (
            f"{adult_friendly_label(policy, policy_labels)} | "
            f"{adult_friendly_label(preprocessing, preprocessing_labels)}"
        )
        style = adult_curve_style(policy=policy, preprocessing=preprocessing)

        ax.errorbar(
            [row["mean"]],
            [label],
            xerr=[row["ci_half_width"]],
            fmt="o",
            color=style["color"],
            capsize=4,
        )

    ax.set_title(
        f"ADULT | sensitive=sex | {family_label} | Post-processing {ylabel}",
        fontweight="bold",
    )
    ax.set_xlabel("Mean ± 95% CI")
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    output_path = fig_dir / f"adult_sex_{family}_postprocessing_{filename_suffix}.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print("Saved:", output_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return output_path


def plot_adult_postprocessing_summary(
    *,
    postproc_df: pd.DataFrame,
    family: str,
    family_label: str,
    fig_dir: str | Path,
    policy_labels: dict[str, str] | None = None,
    preprocessing_labels: dict[str, str] | None = None,
    show: bool = True,
) -> list[Path]:
    """
    Generate held-out post-processing summary figures for one family.
    """
    return [
        plot_adult_postprocessing_metric(
            postproc_df,
            family=family,
            family_label=family_label,
            metric="DP_gap",
            ylabel="Demographic Parity Gap",
            filename_suffix="dp_gap",
            fig_dir=fig_dir,
            policy_labels=policy_labels,
            preprocessing_labels=preprocessing_labels,
            show=show,
        ),
        plot_adult_postprocessing_metric(
            postproc_df,
            family=family,
            family_label=family_label,
            metric="EO_gap",
            ylabel="Equalized Odds Gap",
            filename_suffix="eo_gap",
            fig_dir=fig_dir,
            policy_labels=policy_labels,
            preprocessing_labels=preprocessing_labels,
            show=show,
        ),
        plot_adult_postprocessing_metric(
            postproc_df,
            family=family,
            family_label=family_label,
            metric="average_reward",
            ylabel="Average reward",
            filename_suffix="average_reward",
            fig_dir=fig_dir,
            policy_labels=policy_labels,
            preprocessing_labels=preprocessing_labels,
            show=show,
        ),
    ]
