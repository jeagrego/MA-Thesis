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
        f"ADULT | sensitive = sex | {family_label} | Preprocessing comparison",
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
        f"ADULT | sensitive = sex | {family_label} | In-processing comparison",
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
        f"ADULT | sensitive = sex | {family_label} | {ylabel} over time",
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

def plot_adult_linucb_inprocessing_average_reward(
    temporal_df: pd.DataFrame,
    *,
    fig_dir: str | Path,
    policy_labels: dict[str, str] | None = None,
    show: bool = True,
) -> Path:
    """
    Plot the Adult LinUCB in-processing comparison for average reward.

    The figure contains only two curves:
    - LinUCB under uniform preprocessing;
    - FairLinUCB under uniform preprocessing.
    """
    df = temporal_df.copy()

    if "family" in df.columns:
        df = df[df["family"] == "linucb"].copy()

    if "average_reward" not in df.columns:
        if "avg_reward" in df.columns:
            df["average_reward"] = df["avg_reward"]
        elif "accuracy" in df.columns:
            df["average_reward"] = df["accuracy"]
        else:
            raise KeyError(
                "Could not find an average reward column. "
                f"Available columns are: {df.columns.tolist()}"
            )

    policies = [
        "LinUCB",
        "FairLinUCB",
    ]

    df = df[
        (df["policy"].isin(policies))
        & (df["preprocessing"] == "uniform")
    ].copy()

    if df.empty:
        raise ValueError(
            "No Adult LinUCB/FairLinUCB uniform rows found in temporal_df."
        )

    summary = aggregate_adult_curve(
        df,
        "average_reward",
        ["policy"],
    )

    fig, ax = plt.subplots(
        figsize=(10, 5),
    )

    fig.suptitle(
        "ADULT | sensitive = sex | LinUCB | In-processing average reward",
        fontweight="bold",
        y=1.02,
    )

    for policy in policies:
        curve = (
            summary[
                summary["policy"] == policy
            ]
            .sort_values("t")
        )

        if curve.empty:
            print("Missing curve:", policy)
            continue

        label = adult_friendly_label(
            policy,
            policy_labels,
        )

        style = adult_curve_style(
            policy=policy,
            preprocessing="uniform",
        )

        draw_adult_curve(
            ax,
            curve,
            label=label,
            **style,
        )

    ax.set_xlabel("Rounds")
    ax.set_ylabel("Average reward")
    y_min = float(summary["low"].min())
    y_max = float(summary["high"].max())
    y_range = y_max - y_min

    y_padding = max(
        0.005,
        0.10 * y_range,
    )

    ax.set_ylim(
        max(0.0, y_min - y_padding),
        min(1.0, y_max + y_padding),
    )
    ax.grid(
        True,
        alpha=0.3,
    )
    ax.legend(
        fontsize=9,
        loc="best",
    )

    plt.tight_layout()

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        fig_dir
        / "adult_sex_linucb_inprocessing_average_reward_uniform.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    print("Saved:", output_path)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path

def plot_adult_linucb_postprocessing_average_reward(
    postproc_horizon_df: pd.DataFrame,
    *,
    fig_dir: str | Path,
    policy_labels: dict[str, str] | None = None,
    show: bool = True,
) -> Path:
    """
    Plot the Adult LinUCB post-processing comparison for average reward.

    The figure contains only two held-out curves:
    - FairLinUCB under uniform preprocessing;
    - FairLinUCB+PP under uniform preprocessing.

    This isolates the utility effect of post-processing after FairLinUCB.
    """
    df = postproc_horizon_df.copy()

    if df.empty:
        raise ValueError("postproc_horizon_df is empty.")

    if "horizon" not in df.columns:
        if "t" in df.columns:
            df["horizon"] = df["t"]
        else:
            raise KeyError(
                "postproc_horizon_df must contain either 'horizon' or 't'."
            )

    if "family" in df.columns:
        df = df[df["family"] == "linucb"].copy()

    if "average_reward" not in df.columns:
        if "avg_reward" in df.columns:
            df["average_reward"] = df["avg_reward"]
        elif "accuracy" in df.columns:
            df["average_reward"] = df["accuracy"]
        else:
            raise KeyError(
                "Could not find an average reward column. "
                f"Available columns are: {df.columns.tolist()}"
            )

    policies = [
        "FairLinUCB",
        "FairLinUCB+PP",
    ]

    df = df[
        (df["policy"].isin(policies))
        & (df["preprocessing"] == "uniform")
    ].copy()

    if df.empty:
        raise ValueError(
            "No Adult FairLinUCB/FairLinUCB+PP uniform rows found "
            "in postproc_horizon_df."
        )

    summary = (
        df.groupby(
            [
                "policy",
                "horizon",
            ],
            as_index=False,
        )["average_reward"]
        .agg(
            mean="mean",
            sd="std",
            n="count",
        )
    )

    summary = add_mean_ci_columns(
        summary,
        confidence=0.95,
    )

    fig, ax = plt.subplots(
        figsize=(10, 5),
    )

    fig.suptitle(
        "ADULT | sensitive = sex | LinUCB | Post-processing average reward",
        fontweight="bold",
        y=1.02,
    )

    for policy in policies:
        curve = (
            summary[
                summary["policy"] == policy
            ]
            .sort_values("horizon")
        )

        if curve.empty:
            print("Missing curve:", policy)
            continue

        label = adult_friendly_label(
            policy,
            policy_labels,
        )

        style = adult_curve_style(
            policy=policy,
            preprocessing="uniform",
        )

        ax.plot(
            curve["horizon"],
            curve["mean"],
            label=label,
            color=style["color"],
            linestyle=style["linestyle"],
            marker="o",
            linewidth=2.0,
            markersize=4,
            alpha=0.95,
        )

        ax.fill_between(
            curve["horizon"].to_numpy(dtype=float),
            curve["low"].to_numpy(dtype=float),
            curve["high"].to_numpy(dtype=float),
            color=style["color"],
            alpha=0.10,
        )

        ax.fill_between(
            curve["horizon"].to_numpy(dtype=float),
            curve["low"].to_numpy(dtype=float),
            curve["high"].to_numpy(dtype=float),
            color=style["color"],
            alpha=0.10,
        )

    ax.set_xlabel("Training horizon / rounds")
    ax.set_ylabel("Held-out average reward")

    y_min = float(summary["low"].min())
    y_max = float(summary["high"].max())
    y_range = y_max - y_min

    y_padding = max(
        0.005,
        0.10 * y_range,
    )

    ax.set_ylim(
        max(0.0, y_min - y_padding),
        min(1.0, y_max + y_padding),
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend(
        fontsize=9,
        loc="best",
    )

    plt.tight_layout()

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        fig_dir
        / "adult_sex_linucb_postprocessing_average_reward_uniform.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

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
        f"ADULT | sensitive = sex | {family_label} | Post-processing {ylabel}",
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

ADULT_FINAL_METRIC_DISPLAY_NAMES = {
    "average_reward": "Average reward",
    "cumulative_prediction_error": "Cumulative prediction error",
    "DP_gap": "Demographic Parity Gap",
    "EO_gap": "Equalized Odds Gap",
    "UtilityGap": "UtilityGap",
}


ADULT_LOWER_IS_BETTER = {
    "cumulative_prediction_error",
    "DP_gap",
    "EO_gap",
    "UtilityGap",
}


def _ensure_adult_plot_metric_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure compatibility with old and new metric column names.
    """
    df = dataframe.copy()

    if "average_reward" not in df.columns and "accuracy" in df.columns:
        df["average_reward"] = df["accuracy"]

    if (
        "cumulative_prediction_error" not in df.columns
        and "cumulative_regret" in df.columns
    ):
        df["cumulative_prediction_error"] = df["cumulative_regret"]

    return df


def _adult_postprocessed_policy_name(fair_policy: str) -> str:
    """
    Return the post-processed policy name from the fairness-aware policy name.
    """
    return f"{fair_policy}+PP"


def prepare_adult_final_plot_dataframe(
    *,
    endpoint_df: pd.DataFrame,
    postproc_df: pd.DataFrame,
    family: str,
    policies: list[str],
    preprocessings: list[str],
    policy_labels: dict[str, str] | None = None,
    preprocessing_labels: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Combine online endpoint rows and held-out post-processing rows for final dot plots.

    The plotted methods are:
    - baseline policy under both preprocessing settings;
    - fairness-aware policy under both preprocessing settings;
    - post-processed fairness-aware policy under both preprocessing settings.
    """
    policy_labels = DEFAULT_ADULT_POLICY_LABELS | (policy_labels or {})
    preprocessing_labels = DEFAULT_ADULT_PREPROCESSING_LABELS | (
        preprocessing_labels or {}
    )

    endpoint = _ensure_adult_plot_metric_columns(endpoint_df)
    postproc = _ensure_adult_plot_metric_columns(postproc_df)

    endpoint = (
        endpoint[
            (endpoint["family"] == family)
            & (endpoint["policy"].isin(policies))
            & (endpoint["preprocessing"].isin(preprocessings))
        ]
        .copy()
    )

    fair_policy = str(policies[1])
    postprocessed_policy = _adult_postprocessed_policy_name(fair_policy)

    postproc = (
        postproc[
            (postproc["family"] == family)
            & (postproc["policy"] == postprocessed_policy)
            & (postproc["preprocessing"].isin(preprocessings))
        ]
        .copy()
    )

    combined = pd.concat(
        [
            endpoint,
            postproc,
        ],
        ignore_index=True,
    )

    method_order = []

    for policy in [
        str(policies[0]),
        fair_policy,
        postprocessed_policy,
    ]:
        for preprocessing in preprocessings:
            method = (
                f"{adult_friendly_label(policy, policy_labels)} | "
                f"{adult_friendly_label(preprocessing, preprocessing_labels)}"
            )
            method_order.append(method)

    combined["Method"] = [
        (
            f"{adult_friendly_label(policy, policy_labels)} | "
            f"{adult_friendly_label(preprocessing, preprocessing_labels)}"
        )
        for policy, preprocessing in zip(
            combined["policy"],
            combined["preprocessing"],
        )
    ]

    return combined, method_order


def aggregate_adult_final_metric(
    dataframe: pd.DataFrame,
    *,
    metric: str,
) -> pd.DataFrame:
    """
    Aggregate a final metric across seeds and compute 95% CI for the mean.
    """
    if metric not in dataframe.columns:
        raise KeyError(
            f"Metric '{metric}' not found. "
            f"Available columns are: {dataframe.columns.tolist()}"
        )

    summary = (
        dataframe
        .groupby(
            [
                "policy",
                "preprocessing",
                "Method",
            ],
            as_index=False,
        )[metric]
        .agg(
            mean="mean",
            sd="std",
            n="count",
        )
    )

    return add_mean_ci_columns(summary, confidence=0.95)


def plot_adult_final_metric_dot_panel(
    *,
    endpoint_df: pd.DataFrame,
    postproc_df: pd.DataFrame,
    family: str,
    family_label: str,
    policies: list[str],
    preprocessings: list[str],
    metrics: list[str],
    title_suffix: str,
    filename_suffix: str,
    fig_dir: str | Path,
    policy_labels: dict[str, str] | None = None,
    preprocessing_labels: dict[str, str] | None = None,
    show: bool = True,
) -> Path:
    """
    Plot final dot plots with 95% CI error bars.

    Baseline policies are blue, fairness-aware policies are red, and
    post-processed policies are green.
    """
    combined, method_order = prepare_adult_final_plot_dataframe(
        endpoint_df=endpoint_df,
        postproc_df=postproc_df,
        family=family,
        policies=policies,
        preprocessings=preprocessings,
        policy_labels=policy_labels,
        preprocessing_labels=preprocessing_labels,
    )

    if combined.empty:
        raise ValueError(
            f"No final data available for family={family}."
        )

    y_positions = {
        method: index
        for index, method in enumerate(method_order)
    }

    fig, axes = plt.subplots(
        1,
        len(metrics),
        figsize=(5.2 * len(metrics), 5.5),
        sharey=True,
    )

    if len(metrics) == 1:
        axes = [axes]

    fig.suptitle(
        f"ADULT | sensitive = sex | {family_label} | {title_suffix}",
        fontweight="bold",
        y=1.02,
    )

    for ax, metric in zip(axes, metrics):
        summary = aggregate_adult_final_metric(
            combined,
            metric=metric,
        )

        for _, row in summary.iterrows():
            method = str(row["Method"])

            if method not in y_positions:
                continue

            style = adult_curve_style(
                policy=str(row["policy"]),
                preprocessing=str(row["preprocessing"]),
            )

            ax.errorbar(
                row["mean"],
                y_positions[method],
                xerr=row["ci_half_width"],
                fmt="o",
                color=style["color"],
                capsize=4,
                markersize=5,
            )

        display_metric = ADULT_FINAL_METRIC_DISPLAY_NAMES.get(
            metric,
            metric,
        )

        ax.set_title(
            display_metric,
            fontweight="bold",
        )

        ax.set_yticks(
            list(y_positions.values())
        )
        ax.set_yticklabels(
            list(y_positions.keys())
        )

        if metric in ADULT_LOWER_IS_BETTER:
            ax.set_xlabel(
                "Mean ± 95% CI\n(lower is better)"
            )
            ax.invert_xaxis()
        else:
            ax.set_xlabel(
                "Mean ± 95% CI\n(higher is better)"
            )

        ax.grid(
            True,
            axis="x",
            alpha=0.3,
        )

    plt.tight_layout()

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        fig_dir
        / f"adult_sex_{family}_{filename_suffix}.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    print("Saved:", output_path)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path


def plot_adult_final_dot_plots(
    *,
    endpoint_df: pd.DataFrame,
    postproc_df: pd.DataFrame,
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
    Generate final predictive and fairness dot plots for one Adult policy family.
    """
    return [
        plot_adult_final_metric_dot_panel(
            endpoint_df=endpoint_df,
            postproc_df=postproc_df,
            family=family,
            family_label=family_label,
            policies=policies,
            preprocessings=preprocessings,
            metrics=[
                "average_reward",
                "cumulative_prediction_error",
            ],
            title_suffix="Predictive performance",
            filename_suffix="final_predictive_performance",
            fig_dir=fig_dir,
            policy_labels=policy_labels,
            preprocessing_labels=preprocessing_labels,
            show=show,
        ),
        plot_adult_final_metric_dot_panel(
            endpoint_df=endpoint_df,
            postproc_df=postproc_df,
            family=family,
            family_label=family_label,
            policies=policies,
            preprocessings=preprocessings,
            metrics=[
                "DP_gap",
                "EO_gap",
                "UtilityGap",
            ],
            title_suffix="Fairness performance",
            filename_suffix="final_fairness_performance",
            fig_dir=fig_dir,
            policy_labels=policy_labels,
            preprocessing_labels=preprocessing_labels,
            show=show,
        ),
    ]


def _add_adult_online_baseline_to_postproc_horizon(
    *,
    postproc_horizon_df: pd.DataFrame,
    temporal_df: pd.DataFrame,
    family: str,
    baseline_policy: str,
    horizons: list[int],
    preprocessings: list[str],
) -> pd.DataFrame:
    """
    Add standard-policy reference rows at the requested horizons.

    Fairness-aware and post-processed rows are held-out evaluations. The
    standard-policy rows are reconstructed from the online temporal
    trajectories so that the three policy variants can be compared in the
    same figure.
    """
    postproc = postproc_horizon_df.copy()

    if "policy" not in postproc.columns:
        raise KeyError("postproc_horizon_df must contain a 'policy' column.")

    if (postproc["policy"] == baseline_policy).any():
        return postproc

    required = {
        "family",
        "seed",
        "policy",
        "preprocessing",
        "t",
    }
    missing = required - set(temporal_df.columns)

    if missing:
        raise KeyError(
            f"Cannot reconstruct {baseline_policy}. Missing temporal columns: "
            f"{sorted(missing)}"
        )

    metric_columns: dict[str, str] = {}

    for metric, legacy in [
        ("DP_gap", "DP_gap_over_time"),
        ("EO_gap", "EO_gap_over_time"),
    ]:
        if metric in temporal_df.columns:
            metric_columns[metric] = metric
        elif legacy in temporal_df.columns:
            metric_columns[metric] = legacy
        else:
            raise KeyError(f"Missing temporal metric: {metric}")

    baseline_temporal = temporal_df[
        (temporal_df["family"] == family)
        & (temporal_df["policy"] == baseline_policy)
        & (temporal_df["preprocessing"].isin(preprocessings))
    ].copy()

    rows: list[dict[str, object]] = []

    for horizon in horizons:
        available = baseline_temporal[
            baseline_temporal["t"] <= int(horizon)
        ].copy()

        if available.empty:
            continue

        last_rows = (
            available
            .sort_values("t")
            .groupby(
                ["seed", "preprocessing"],
                as_index=False,
            )
            .tail(1)
        )

        for _, row in last_rows.iterrows():
            rows.append(
                {
                    "family": family,
                    "seed": int(row["seed"]),
                    "policy": baseline_policy,
                    "preprocessing": str(row["preprocessing"]),
                    "horizon": int(horizon),
                    "t": int(horizon),
                    "DP_gap": float(row[metric_columns["DP_gap"]]),
                    "EO_gap": float(row[metric_columns["EO_gap"]]),
                }
            )

    if not rows:
        print(f"No online reference rows reconstructed for {family}.")
        return postproc

    return pd.concat(
        [postproc, pd.DataFrame(rows)],
        ignore_index=True,
    )


def plot_adult_postprocessing_over_horizon(
    postproc_horizon_df: pd.DataFrame,
    *,
    temporal_df: pd.DataFrame,
    family: str,
    family_label: str,
    baseline_policy: str,
    fair_policy: str,
    horizons: list[int],
    preprocessings: list[str],
    fig_dir: str | Path,
    policy_labels: dict[str, str] | None = None,
    preprocessing_labels: dict[str, str] | None = None,
    show: bool = True,
) -> Path:
    """
    Compare the baseline, fairness-aware, and post-processed policies.

    Expected input:
        one row per family x seed x preprocessing x policy x horizon.

    The policies shown are:
    - standard online baseline policy;
    - fairness-aware in-processing policy;
    - fairness-aware policy after post-processing.
    """
    if postproc_horizon_df.empty:
        raise ValueError(
            "postproc_horizon_df is empty."
        )

    df = _add_adult_online_baseline_to_postproc_horizon(
        postproc_horizon_df=postproc_horizon_df,
        temporal_df=temporal_df,
        family=family,
        baseline_policy=baseline_policy,
        horizons=horizons,
        preprocessings=preprocessings,
    )

    df = _ensure_adult_plot_metric_columns(df)

    if "horizon" not in df.columns:
        if "t" in df.columns:
            df = df.copy()
            df["horizon"] = df["t"]
        else:
            raise KeyError(
                "postproc_horizon_df must contain either 'horizon' or 't'."
            )

    postprocessed_policy = _adult_postprocessed_policy_name(fair_policy)
    policies = [
        baseline_policy,
        fair_policy,
        postprocessed_policy,
    ]

    df = (
        df[
            (df["family"] == family)
            & (df["policy"].isin(policies))
            & (df["preprocessing"].isin(preprocessings))
        ]
        .copy()
    )

    if df.empty:
        raise ValueError(
            f"No post-processing horizon data available for family={family}."
        )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 5),
    )

    fig.suptitle(
        f"ADULT | sensitive = sex | {family_label} | Post-processing comparison",
        fontweight="bold",
        y=1.02,
    )

    for ax, metric, ylabel in [
        (axes[0], "DP_gap", "Demographic Parity Gap"),
        (axes[1], "EO_gap", "Equalized Odds Gap"),
    ]:
        summary = (
            df
            .groupby(
                [
                    "policy",
                    "preprocessing",
                    "horizon",
                ],
                as_index=False,
            )[metric]
            .agg(
                mean="mean",
                sd="std",
                n="count",
            )
        )

        summary = add_mean_ci_columns(
            summary,
            confidence=0.95,
        )

        for policy in policies:
            for preprocessing in preprocessings:
                curve = (
                    summary[
                        (summary["policy"] == policy)
                        & (summary["preprocessing"] == preprocessing)
                    ]
                    .sort_values("horizon")
                )

                if curve.empty:
                    continue

                label = (
                    f"{adult_friendly_label(policy, policy_labels)} | "
                    f"{adult_friendly_label(preprocessing, preprocessing_labels)}"
                )

                style = adult_curve_style(
                    policy=policy,
                    preprocessing=preprocessing,
                )

                ax.plot(
                    curve["horizon"],
                    curve["mean"],
                    label=label,
                    color=style["color"],
                    linestyle=style["linestyle"],
                    marker="o",
                    linewidth=2.0,
                    markersize=4,
                    alpha=0.95,
                )

                ax.fill_between(
                    curve["horizon"].to_numpy(dtype=float),
                    curve["low"].to_numpy(dtype=float),
                    curve["high"].to_numpy(dtype=float),
                    color=style["color"],
                    alpha=0.10,
                )

        ax.set_title(
            f"{ylabel} over training horizon",
            fontweight="bold",
        )
        ax.set_xlabel("Training horizon / rounds")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(
            fontsize=8,
            loc="best",
            frameon=True,
        )

    plt.tight_layout()

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        fig_dir
        / f"adult_sex_{family}_postprocessing_over_horizon.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    print("Saved:", output_path)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path
