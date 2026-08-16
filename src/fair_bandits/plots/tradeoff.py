from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_TRADEOFF_POLICY_LABELS = {
    "LinUCB": "LinUCB",
    "FairLinUCB": "FairLinUCB",
    "FairLinUCB+PP": "FairLinUCB+PP",
    "FairLinUCB_DP": "FairLinUCB-DP",

    "LinTS": "LinTS",
    "FairLinTS": "FairLinTS",
    "FairLinTS+PP": "FairLinTS+PP",
    "FairLinTS_DP": "FairLinTS-DP",

    "EXP4": "EXP4",
    "FairEXP4": "FairEXP4",
    "FairEXP4+PP": "FairEXP4+PP",
    "FairEXP4_DP": "FairEXP4-DP",
}


DEFAULT_TRADEOFF_PREPROCESSING_LABELS = {
    "uniform": "Uniform",
    "reweigh_group_label": "Reweighting",
}


FAIRNESS_METRIC_LABELS = {
    "DP_gap": "Demographic Parity Gap",
    "EO_gap": "Equalized Odds Gap",
}


def _friendly_label(value: str, labels: dict[str, str] | None = None) -> str:
    labels = labels or {}
    return labels.get(str(value), str(value))


def _t_critical(n: int, confidence: float = 0.95) -> float:
    n = int(n)

    if n <= 1:
        return 0.0

    try:
        from scipy.stats import t as student_t

        alpha = 1.0 - float(confidence)
        return float(student_t.ppf(1.0 - alpha / 2.0, df=n - 1))

    except Exception:
        return 1.96


def _mean_ci(values: pd.Series, confidence: float = 0.95) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce").dropna()
    n = int(values.shape[0])

    if n == 0:
        return pd.Series(
            {
                "mean": np.nan,
                "sd": np.nan,
                "n": 0,
                "ci_half_width": np.nan,
                "low": np.nan,
                "high": np.nan,
            }
        )

    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if n > 1 else 0.0
    se = sd / np.sqrt(max(n, 1))
    ci_half_width = _t_critical(n, confidence=confidence) * se if n > 1 else 0.0

    return pd.Series(
        {
            "mean": mean,
            "sd": sd,
            "n": n,
            "ci_half_width": ci_half_width,
            "low": mean - ci_half_width,
            "high": mean + ci_half_width,
        }
    )


def _ensure_tradeoff_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize metric names used across Adult, COMPAS, and synthetic outputs.
    """
    df = dataframe.copy()

    if "average_reward" not in df.columns:
        if "avg_reward" in df.columns:
            df["average_reward"] = df["avg_reward"]
        elif "accuracy" in df.columns:
            df["average_reward"] = df["accuracy"]

    if "DP_gap" not in df.columns and "DP_gap_over_time" in df.columns:
        df["DP_gap"] = df["DP_gap_over_time"]

    if "EO_gap" not in df.columns and "EO_gap_over_time" in df.columns:
        df["EO_gap"] = df["EO_gap_over_time"]

    if "UtilityGap" not in df.columns and "UtilityGap_over_time" in df.columns:
        df["UtilityGap"] = df["UtilityGap_over_time"]

    if (
        "cumulative_prediction_error" not in df.columns
        and "cumulative_regret" in df.columns
    ):
        df["cumulative_prediction_error"] = df["cumulative_regret"]

    required = [
        "seed",
        "policy",
        "preprocessing",
        "average_reward",
        "DP_gap",
        "EO_gap",
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise KeyError(
            f"Missing columns for trade-off plotting: {missing}. "
            f"Available columns are: {df.columns.tolist()}"
        )

    return df


def _curve_style(policy: str, preprocessing: str) -> dict[str, object]:
    """
    Visual style for fairness-utility trade-off plots.

    Color encodes the intervention family:
    - blue  = baseline policy
    - red   = fairness-aware in-processing policy
    - green = post-processed policy

    Shade encodes preprocessing:
    - lighter shade = uniform
    - darker shade  = reweighting

    Marker encodes preprocessing:
    - triangle = uniform
    - square   = reweighting
    """
    policy_text = str(policy)
    preprocessing_text = str(preprocessing)

    is_uniform = preprocessing_text == "uniform"

    if "+PP" in policy_text:
        color = "#3BE13E" if is_uniform else "green"   # light/dark green
    elif policy_text.startswith("Fair"):
        color = "#F76262" if is_uniform else "red"   # light/dark red
    else:
        color = "#3CABF6" if is_uniform else "blue"   # light/dark blue

    marker = "^" if is_uniform else "s"

    return {
        "color": color,
        "marker": marker,
    }

def _policy_stage_order(policy: str) -> int:
    """
    Order policy types in legends and plots:
    baseline first, then fairness-aware, then post-processed.
    """
    policy_text = str(policy)

    if "+PP" in policy_text:
        return 2

    if policy_text.startswith("Fair"):
        return 1

    return 0


def _preprocessing_order(preprocessing: str) -> int:
    """
    Put Uniform before Reweighting in plots and legends.
    """
    preprocessing_text = str(preprocessing)

    if preprocessing_text == "uniform":
        return 0

    if preprocessing_text == "reweigh_group_label":
        return 1

    return 99

def _method_label(
    policy: str,
    preprocessing: str,
    *,
    policy_labels: dict[str, str] | None = None,
    preprocessing_labels: dict[str, str] | None = None,
) -> str:
    policy_labels = DEFAULT_TRADEOFF_POLICY_LABELS | (policy_labels or {})
    preprocessing_labels = DEFAULT_TRADEOFF_PREPROCESSING_LABELS | (
        preprocessing_labels or {}
    )

    return (
        f"{_friendly_label(policy, policy_labels)} | "
        f"{_friendly_label(preprocessing, preprocessing_labels)}"
    )


def _aggregate_tradeoff_points(
    dataframe: pd.DataFrame,
    *,
    fairness_metric: str,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """
    Aggregate average reward and one fairness metric across seeds.

    Each output row is one plotted method:
    policy × preprocessing.
    """
    df = _ensure_tradeoff_columns(dataframe)

    if fairness_metric not in df.columns:
        raise KeyError(
            f"Missing fairness metric: {fairness_metric}. "
            f"Available columns are: {df.columns.tolist()}"
        )

    group_cols = [
        "policy",
        "preprocessing",
        "Method",
    ]

    x_summary = (
        df.groupby(group_cols)["average_reward"]
        .apply(lambda s: _mean_ci(s, confidence=confidence))
        .unstack()
        .reset_index()
    )

    y_summary = (
        df.groupby(group_cols)[fairness_metric]
        .apply(lambda s: _mean_ci(s, confidence=confidence))
        .unstack()
        .reset_index()
    )

    x_summary = x_summary.rename(
        columns={
            "mean": "x_mean",
            "sd": "x_sd",
            "n": "x_n",
            "ci_half_width": "x_ci",
            "low": "x_low",
            "high": "x_high",
        }
    )

    y_summary = y_summary.rename(
        columns={
            "mean": "y_mean",
            "sd": "y_sd",
            "n": "y_n",
            "ci_half_width": "y_ci",
            "low": "y_low",
            "high": "y_high",
        }
    )

    summary = x_summary.merge(
        y_summary,
        on=group_cols,
        how="inner",
    )

    summary["_policy_order"] = summary["policy"].map(_policy_stage_order)
    summary["_preprocessing_order"] = summary["preprocessing"].map(_preprocessing_order)

    summary = (
        summary.sort_values(
            [
                "_policy_order",
                "_preprocessing_order",
                "policy",
                "preprocessing",
            ]
        )
        .drop(columns=["_policy_order", "_preprocessing_order"])
        .reset_index(drop=True)
    )

    return summary


def _axis_limits_from_summaries(
    summaries: list[pd.DataFrame],
    *,
    x_padding_fraction: float = 0.08,
    y_padding_fraction: float = 0.10,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Compute common x and y limits from one or several summary dataframes.
    """
    data = pd.concat(
        [summary for summary in summaries if not summary.empty],
        ignore_index=True,
    )

    if data.empty:
        return (0.0, 1.0), (0.0, 1.0)

    x_min = float(data["x_low"].min())
    x_max = float(data["x_high"].max())
    y_min = float(data["y_low"].min())
    y_max = float(data["y_high"].max())

    x_range = max(x_max - x_min, 1e-6)
    y_range = max(y_max - y_min, 1e-6)

    x_pad = x_padding_fraction * x_range
    y_pad = y_padding_fraction * y_range

    xlim = (
        max(0.0, x_min - x_pad),
        min(1.0, x_max + x_pad),
    )

    ylim = (
        max(0.0, y_min - y_pad),
        y_max + y_pad,
    )

    return xlim, ylim


def _plot_tradeoff_panel(
    dataframe: pd.DataFrame,
    *,
    dataset_label: str,
    sensitive_label: str,
    family_label: str,
    fairness_metric: str,
    output_path: str | Path,
    title_prefix: str | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    confidence: float = 0.95,
    show: bool = True,
) -> Path:
    """
    Draw one fairness-utility trade-off plot:
    x-axis = average reward;
    y-axis = fairness gap.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = _aggregate_tradeoff_points(
        dataframe,
        fairness_metric=fairness_metric,
        confidence=confidence,
    )

    if summary.empty:
        raise ValueError(
            f"No data available for trade-off plot: "
            f"{dataset_label}, {family_label}, {fairness_metric}"
        )

    fig, ax = plt.subplots(figsize=(8.5, 6.2))

    ylabel = FAIRNESS_METRIC_LABELS.get(fairness_metric, fairness_metric)

    if title_prefix is None:
        title_prefix = dataset_label

    ax.set_title(
        (
            f"{title_prefix} | sensitive = {sensitive_label} | "
            f"{family_label}\n"
            f"Average reward vs {ylabel}"
        ),
        fontweight="bold",
    )

    for _, row in summary.iterrows():
        policy = str(row["policy"])
        preprocessing = str(row["preprocessing"])
        method = str(row["Method"])

        style = _curve_style(
            policy=policy,
            preprocessing=preprocessing,
        )

        ax.errorbar(
            row["x_mean"],
            row["y_mean"],
            xerr=row["x_ci"],
            yerr=row["y_ci"],
            fmt=style["marker"],
            color=style["color"],
            ecolor=style["color"],
            markerfacecolor=style["color"],
            markeredgecolor="black",
            markeredgewidth=0.5,
            markersize=8,
            capsize=4,
            linewidth=1.4,
            label=method,
            alpha=0.95,
        )

    ax.set_xlabel(
        "Average reward\n(higher is better)"
    )
    ax.set_ylabel(
        f"{ylabel}\n(lower is better)"
    )

    if xlim is not None:
        ax.set_xlim(*xlim)

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.grid(True, alpha=0.3)

    ax.text(
            0.03,
            0.97,
            "Better region\n→ higher reward\n↓ lower disparity",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": "lightgray",
                "alpha": 0.9,
            },
        )

    ax.legend(
        fontsize=8,
        loc="upper right",
        frameon=True,
    )

    plt.tight_layout()
    fig.savefig(
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


def synthetic_final_tradeoff_source(
    *,
    temporal_df: pd.DataFrame,
    regime: str,
    policies: list[str],
    preprocessings: list[str],
    policy_labels: dict[str, str] | None = None,
    preprocessing_labels: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Build final online endpoint rows for synthetic trade-off plots.

    IMPORTANT:
    Synthetic plots intentionally exclude post-processing.
    """
    df = _ensure_tradeoff_columns(temporal_df)

    if "regime" not in df.columns:
        raise KeyError(
            "Synthetic temporal dataframe must contain a 'regime' column."
        )

    df = df[
        (df["regime"] == regime)
        & (df["policy"].isin(policies))
        & (df["preprocessing"].isin(preprocessings))
    ].copy()

    if df.empty:
        raise ValueError(f"No synthetic rows for regime={regime}.")

    df = (
        df.sort_values("t")
        .groupby(
            [
                "regime",
                "seed",
                "policy",
                "preprocessing",
            ],
            as_index=False,
        )
        .tail(1)
        .copy()
    )

    df["Method"] = [
        _method_label(
            policy,
            preprocessing,
            policy_labels=policy_labels,
            preprocessing_labels=preprocessing_labels,
        )
        for policy, preprocessing in zip(
            df["policy"],
            df["preprocessing"],
        )
    ]

    return df


def real_dataset_final_tradeoff_source(
    *,
    endpoint_df: pd.DataFrame,
    postproc_df: pd.DataFrame,
    family: str,
    policies: list[str],
    preprocessings: list[str],
    policy_labels: dict[str, str] | None = None,
    preprocessing_labels: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Build final rows for Adult/COMPAS trade-off plots.

    Online rows:
        baseline and fairness-aware policies at the final online checkpoint.

    Post-processing rows:
        fairness-aware + PP evaluated on held-out test data.
    """
    endpoint = _ensure_tradeoff_columns(endpoint_df)

    if "family" in endpoint.columns:
        endpoint = endpoint[endpoint["family"] == family].copy()

    endpoint = endpoint[
        endpoint["policy"].isin(policies)
        & endpoint["preprocessing"].isin(preprocessings)
    ].copy()

    if "t" in endpoint.columns:
        endpoint = (
            endpoint.sort_values("t")
            .groupby(
                [
                    "seed",
                    "policy",
                    "preprocessing",
                ],
                as_index=False,
            )
            .tail(1)
            .copy()
        )

    if postproc_df is None or postproc_df.empty:
        postproc = pd.DataFrame()
    else:
        postproc = _ensure_tradeoff_columns(postproc_df)

        if "family" in postproc.columns:
            postproc = postproc[postproc["family"] == family].copy()

        fair_policy = str(policies[1])
        postprocessed_policy = f"{fair_policy}+PP"

        postproc = postproc[
            (postproc["policy"] == postprocessed_policy)
            & (postproc["preprocessing"].isin(preprocessings))
        ].copy()

    combined = pd.concat(
        [
            endpoint,
            postproc,
        ],
        ignore_index=True,
    )

    if combined.empty:
        raise ValueError(f"No final rows available for family={family}.")

    combined["Method"] = [
        _method_label(
            policy,
            preprocessing,
            policy_labels=policy_labels,
            preprocessing_labels=preprocessing_labels,
        )
        for policy, preprocessing in zip(
            combined["policy"],
            combined["preprocessing"],
        )
    ]

    return combined


def plot_synthetic_tradeoff_set(
    *,
    temporal_df: pd.DataFrame,
    regimes: list[str],
    policies_by_regime: dict[str, list[str]],
    regime_labels: dict[str, str],
    regime_file_tags: dict[str, str],
    preprocessings: list[str],
    fig_dir: str | Path,
    policy_labels: dict[str, str] | None = None,
    preprocessing_labels: dict[str, str] | None = None,
    confidence: float = 0.95,
    show: bool = True,
) -> list[Path]:
    """
    Generate synthetic fairness-utility trade-off plots.

    Produces two plots per regime:
    - average reward vs DP gap;
    - average reward vs EO gap.

    No post-processing variants are included.
    """
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    output_paths: list[Path] = []

    for regime in regimes:
        source = synthetic_final_tradeoff_source(
            temporal_df=temporal_df,
            regime=regime,
            policies=policies_by_regime[regime],
            preprocessings=preprocessings,
            policy_labels=policy_labels,
            preprocessing_labels=preprocessing_labels,
        )

        summaries = [
            _aggregate_tradeoff_points(
                source,
                fairness_metric=metric,
                confidence=confidence,
            )
            for metric in ["DP_gap", "EO_gap"]
        ]

        xlim, _ = _axis_limits_from_summaries(summaries)

        for fairness_metric in ["DP_gap", "EO_gap"]:
            metric_tag = "dp_gap" if fairness_metric == "DP_gap" else "eo_gap"

            metric_summary = _aggregate_tradeoff_points(
                source,
                fairness_metric=fairness_metric,
                confidence=confidence,
            )
            _, ylim = _axis_limits_from_summaries([metric_summary])

            output_path = (
                fig_dir
                / f"synthetic_{regime_file_tags[regime]}_tradeoff_{metric_tag}.png"
            )

            output_paths.append(
                _plot_tradeoff_panel(
                    source,
                    dataset_label="Synthetic CMAB",
                    sensitive_label="simulated group",
                    family_label=regime_labels[regime],
                    fairness_metric=fairness_metric,
                    output_path=output_path,
                    title_prefix="Synthetic CMAB",
                    xlim=xlim,
                    ylim=ylim,
                    confidence=confidence,
                    show=show,
                )
            )

    return output_paths


def plot_real_dataset_tradeoff_set(
    *,
    dataset_label: str,
    sensitive_label: str,
    endpoint_df: pd.DataFrame,
    postproc_df: pd.DataFrame,
    policy_families: dict[str, dict],
    preprocessings: list[str],
    fig_dir: str | Path,
    file_prefix: str,
    policy_labels: dict[str, str] | None = None,
    preprocessing_labels: dict[str, str] | None = None,
    confidence: float = 0.95,
    show: bool = True,
) -> list[Path]:
    """
    Generate Adult or COMPAS fairness-utility trade-off plots.

    Produces two plots per family:
    - average reward vs DP gap;
    - average reward vs EO gap.

    Includes post-processed variants only for Adult/COMPAS.
    """
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    sources: dict[str, pd.DataFrame] = {}

    for family, config in policy_families.items():
        sources[family] = real_dataset_final_tradeoff_source(
            endpoint_df=endpoint_df,
            postproc_df=postproc_df,
            family=family,
            policies=config["policies"],
            preprocessings=preprocessings,
            policy_labels=policy_labels,
            preprocessing_labels=preprocessing_labels,
        )

    # Common axis scales within one dataset/sensitive-attribute setting.
    all_dp_summaries = []
    all_eo_summaries = []

    for source in sources.values():
        all_dp_summaries.append(
            _aggregate_tradeoff_points(
                source,
                fairness_metric="DP_gap",
                confidence=confidence,
            )
        )
        all_eo_summaries.append(
            _aggregate_tradeoff_points(
                source,
                fairness_metric="EO_gap",
                confidence=confidence,
            )
        )

    xlim_dp, ylim_dp = _axis_limits_from_summaries(all_dp_summaries)
    xlim_eo, ylim_eo = _axis_limits_from_summaries(all_eo_summaries)

    # Use the same x-axis for DP and EO plots.
    xlim = (
        min(xlim_dp[0], xlim_eo[0]),
        max(xlim_dp[1], xlim_eo[1]),
    )

    output_paths: list[Path] = []

    for family, config in policy_families.items():
        family_label = config["label"]
        source = sources[family]

        for fairness_metric, metric_tag, ylim in [
            ("DP_gap", "dp_gap", ylim_dp),
            ("EO_gap", "eo_gap", ylim_eo),
        ]:
            output_path = (
                fig_dir
                / f"{file_prefix}_{family}_tradeoff_{metric_tag}.png"
            )

            output_paths.append(
                _plot_tradeoff_panel(
                    source,
                    dataset_label=dataset_label,
                    sensitive_label=sensitive_label,
                    family_label=family_label,
                    fairness_metric=fairness_metric,
                    output_path=output_path,
                    title_prefix=dataset_label,
                    xlim=xlim,
                    ylim=ylim,
                    confidence=confidence,
                    show=show,
                )
            )

    return output_paths