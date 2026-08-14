from __future__ import annotations

import re

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PREPROCESSING_LABELS: dict[str, str] = {
    "uniform": "Uniform",
    "reweigh_group_label": "Reweighting",
}

METRIC_DISPLAY_NAMES: dict[str, str] = {
    "DP gap": "Demographic Parity Gap",
    "EO gap": "Equalized Odds Gap",
    "UtilityGap": "UtilityGap",
    "Accuracy": "Accuracy",
    "Average reward": "Average reward",
    "Cumulative prediction error": "Cumulative prediction error",
}

def friendly_preprocessing(preprocessing: str) -> str:
    return PREPROCESSING_LABELS.get(str(preprocessing), str(preprocessing))

def exp4_curve_style(
    *,
    policy: str,
    preprocessing: str,
) -> dict[str, str]:
    """
    Visual convention for COMPAS EXP4-family curves.

    EXP4:
        blue

    FairEXP4:
        red

    FairEXP4+PP:
        green

    Uniform:
        solid line

    Reweighting:
        dotted line
    """
    policy_text = str(policy)

    if policy_text == "EXP4":
        color = "blue"
    elif policy_text == "FairEXP4":
        color = "red"
    elif policy_text == "FairEXP4+PP":
        color = "green"
    else:
        color = "black"

    linestyle = ":" if preprocessing == "reweigh_group_label" else "-"

    return {"color": color,"linestyle": linestyle,}

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
        return float(
            student_t.ppf(
                1.0 - alpha / 2.0,
                df=n - 1,
            )
        )

    except Exception:
        return 1.96

def add_mean_ci_columns(
    summary: pd.DataFrame,
    *,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """
    Add pointwise confidence interval columns for the mean.

    Expected input columns:
    - mean
    - sd
    - n

    The plotted interval is:
        mean ± t_(0.975, n-1) × SD / sqrt(n)
    """
    out = summary.copy()

    out["sd"] = out["sd"].fillna(0.0)
    out["n"] = out["n"].astype(int)

    out["se"] = out["sd"] / np.sqrt(
        out["n"].clip(lower=1)
    )

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

def aggregate_temporal_metric(
    dataframe: pd.DataFrame,
    metric: str,
    *,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """
    Aggregate a temporal metric across seeds.

    The plotted band is the pointwise confidence interval of the mean:
        mean ± t_(0.975, n-1) × SD / sqrt(n)

    The unit of replication is the full experimental run/seed.
    """
    summary = (
        dataframe
        .groupby(
            [
                "policy",
                "preprocessing",
                "t",
            ],
            as_index=False,
        )[metric]
        .agg(
            mean="mean",
            sd="std",
            n="count",
        )
    )

    return add_mean_ci_columns(summary, confidence=confidence,)

def draw_curve(
    ax,
    curve: pd.DataFrame,
    label: str,
    *,
    color: str | None = None,
    linestyle: str = "-",
) -> None:
    """
    Draw a mean temporal curve with pointwise 95% confidence interval bands.
    """
    ax.plot(
        curve["t"],
        curve["mean"],
        label=label,
        color=color,
        linestyle=linestyle,
        marker=None,
        linewidth=2.0,
        alpha=0.95,
    )

    ax.fill_between(
        curve["t"],
        curve["low"],
        curve["high"],
        color=color,
        alpha=0.10,
    )


def plot_two_panel(
    dataframe: pd.DataFrame,
    *,
    fig_dir: Path,
    title: str,
    filename: str,
    policies: list[str],
    preprocessings: list[str],
    filter_uniform: bool = False,
) -> Path:
    """
    Plot two panels with temporal fairness metrics.
    Parameters:
    - dataframe: pd.DataFrame
        The input DataFrame containing the temporal data.
    - fig_dir: Path
        The directory where the figure will be saved.
    - title: str
        The title of the figure.
    - filename: str
        The filename for the saved figure.
    - policies: list[str]
        A list of policies to include in the plot.
    - preprocessings: list[str]
        A list of preprocessing methods to include in the plot.
    - filter_uniform: bool, optional (default=False)
        If True, only include the "reweigh_group_label" preprocessing method in the plot
    Returns:
    - Path
        The path to the saved figure.
    """
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5),
    )

    fig.suptitle(
        title,
        fontweight="bold",
        y=1.02,
    )

    metrics = [
        (
            "DP_gap",
            "Demographic Parity Gap",
        ),
        (
            "EO_gap",
            "Equalized Odds Gap",
        ),
    ]

    for ax, (metric, ylabel) in zip(axes, metrics):
        summary = aggregate_temporal_metric(
            dataframe,
            metric,
        )

        for policy in policies:
            for preprocessing in preprocessings:
                if filter_uniform and preprocessing != "reweigh_group_label":
                    continue

                curve = summary[
                    (summary["policy"] == policy)
                    & (summary["preprocessing"] == preprocessing)
                ].sort_values("t")

                if curve.empty:
                    continue

                label = f"{policy} | {friendly_preprocessing(preprocessing)}"
                style = exp4_curve_style(policy=policy,preprocessing=preprocessing,)
                draw_curve(ax,curve,label,**style,)

        ax.set_title(f"{ylabel} over time",fontweight="bold",)
        ax.set_xlabel("Rounds")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()

    fig_dir.mkdir(parents=True,exist_ok=True,)
    output_path = fig_dir / filename
    plt.savefig(output_path,dpi=300,bbox_inches="tight",)
    print("Saved:", output_path)
    plt.show()

    return output_path


def plot_exp4_temporal_fairness(
    temporal_df: pd.DataFrame,
    *,
    fig_dir: Path,
    policies: list[str],
    preprocessings: list[str],
) -> list[Path]:
    """
    Plot temporal fairness metrics for the COMPAS dataset.
    Parameters:
    - temporal_df: pd.DataFrame
        The input DataFrame containing the temporal data.
    - fig_dir: Path
        The directory where the figures will be saved.
    - policies: list[str]
        A list of policies to include in the plots.
    - preprocessings: list[str]
        A list of preprocessing methods to include in the plots.
    Returns:
    - list[Path]
        A list of paths to the saved figures.
    """
    if temporal_df.empty:
        print(
            "Temporal dataframe is empty."
            "Run the benchmark before generating temporal fairness plots."
        )
        return []

    output_paths = []

    output_paths.append(
        plot_two_panel(
            temporal_df,
            fig_dir=fig_dir,
            title="COMPAS | sensitive = race binary | EXP4 | Preprocessing comparison",
            filename="compas_race_binary_exp4_preprocessing_fairness.png",
            policies=policies,
            preprocessings=preprocessings,
        )
    )

    output_paths.append(
        plot_two_panel(
            temporal_df,
            fig_dir=fig_dir,
            title="COMPAS | sensitive = race binary | EXP4 | In-processing comparison",
            filename="compas_race_binary_exp4_inprocessing_fairness.png",
            policies=policies,
            preprocessings=preprocessings,
            filter_uniform=True,
        )
    )

    return output_paths

def plot_utility_gap_over_time(
    temporal_df: pd.DataFrame,
    *,
    fig_dir: Path,
) -> Path:
    """
    Plot UtilityGap over time for the COMPAS race-binary EXP4 experiment.

    UtilityGap measures the between-group difference in average reward.
    Lower values indicate smaller utility disparities between groups.
    """
    if temporal_df.empty:
        raise ValueError(
            "temporal_df is empty. "
            "Run the benchmark before plotting UtilityGap."
        )

    metric = "UtilityGap"

    if metric not in temporal_df.columns:
        raise KeyError(
            f"Column not found: {metric}. "
            f"Available columns are: {temporal_df.columns.tolist()}"
        )

    summary = aggregate_temporal_metric(
        temporal_df,
        metric,
    )

    curves = [
    ("EXP4", "uniform"),
    ("EXP4", "reweigh_group_label"),
    ("FairEXP4", "uniform"),
    ("FairEXP4", "reweigh_group_label"),
    ]

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(11, 6),
    )

    fig.suptitle(
        "COMPAS | sensitive = race binary | EXP4 | UtilityGap over time",
        fontweight="bold",
        y=1.02,
    )

    for policy, preprocessing in curves:
        curve = summary[
            (
                summary["policy"]
                == policy
            )
            & (
                summary["preprocessing"]
                == preprocessing
            )
        ].sort_values("t")

        if curve.empty:
            print(
                "Missing curve:",
                policy,
                preprocessing,
            )
            continue

        label = f"{policy} | {friendly_preprocessing(preprocessing)}"
        style = exp4_curve_style(policy=policy,preprocessing=preprocessing,)
        draw_curve(ax,curve,label,**style,)

    ax.set_xlabel("Rounds")
    ax.set_ylabel("UtilityGap")
    ax.grid(True,alpha=0.3,)
    ax.legend(fontsize=8,loc="upper right",)
    plt.tight_layout()
    fig_dir.mkdir(parents=True,exist_ok=True,)
    output_path = (fig_dir/ "compas_race_binary_exp4_utilitygap_over_time.png")
    plt.savefig(output_path,dpi=300,bbox_inches="tight",)
    print("Saved:",output_path,)
    plt.show()

    return output_path

def aggregate_postprocessing_curve(
    *,
    dataframe: pd.DataFrame,
    metric: str,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """
    Aggregate a post-processing metric across seeds.

    The plotted band is the pointwise confidence interval of the mean:
        mean ± t_(0.975, n-1) × SD / sqrt(n)
    """
    summary = (
        dataframe
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

    return add_mean_ci_columns(
        summary,
        confidence=confidence,
    )

def plot_exp4_postprocessing_over_horizon(
    postproc_long_df: pd.DataFrame,
    *,
    fig_dir: Path,
) -> Path:
    """
    Plot post-processing curves over the training horizon.

    Curves:
    - EXP4 | Uniform
    - EXP4 | Reweighting
    - FairEXP4 | Uniform
    - FairEXP4 | Reweighting
    - FairEXP4+PP | Uniform
    - FairEXP4+PP | Reweighting
    """
    if postproc_long_df.empty:
        raise ValueError(
            "postproc_long_df is empty. "
            "Run the longitudinal post-processing benchmark first."
        )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 5),
    )

    fig.suptitle(
        "COMPAS | sensitive = race binary | EXP4 | Post-processing comparison",
        fontweight="bold",
        y=1.02,
    )

    plot_settings = [
        {
            "policy": "EXP4",
            "preprocessing": "uniform",
            "label": "EXP4 | Uniform",
            "color": "blue",
            "linestyle": "-",
            "marker": "o",
        },
        {
            "policy": "EXP4",
            "preprocessing": "reweigh_group_label",
            "label": "EXP4 | Reweighting",
            "color": "blue",
            "linestyle": ":",
            "marker": "o",
        },
        {
            "policy": "FairEXP4",
            "preprocessing": "uniform",
            "label": "FairEXP4 | Uniform",
            "color": "red",
            "linestyle": "-",
            "marker": "o",
        },
        {
            "policy": "FairEXP4",
            "preprocessing": "reweigh_group_label",
            "label": "FairEXP4 | Reweighting",
            "color": "red",
            "linestyle": ":",
            "marker": "o",
        },
        {
            "policy": "FairEXP4+PP",
            "preprocessing": "uniform",
            "label": "FairEXP4+PP | Uniform",
            "color": "green",
            "linestyle": "-",
            "marker": "o",
        },
        {
            "policy": "FairEXP4+PP",
            "preprocessing": "reweigh_group_label",
            "label": "FairEXP4+PP | Reweighting",
            "color": "green",
            "linestyle": ":",
            "marker": "o",
        },
    ]

    metrics = [
        (
            axes[0],
            "DP_gap",
            "Demographic Parity Gap",
        ),
        (
            axes[1],
            "EO_gap",
            "Equalized Odds Gap",
        ),
    ]

    for ax, metric, ylabel in metrics:
        summary = aggregate_postprocessing_curve(
            dataframe=postproc_long_df,
            metric=metric,
        )

        for setting in plot_settings:
            policy = setting["policy"]
            preprocessing = setting["preprocessing"]

            curve = summary[
                (summary["policy"] == policy)
                & (summary["preprocessing"] == preprocessing)
            ].sort_values("horizon")

            if curve.empty:
                print(
                    "Missing curve:",
                    policy,
                    preprocessing,
                )
                continue

            ax.plot(
                curve["horizon"],
                curve["mean"],
                label=setting["label"],
                color=setting["color"],
                linestyle=setting["linestyle"],
                marker=setting["marker"],
                linewidth=2.0,
                markersize=5,
                alpha=0.95,
            )

            ax.fill_between(
                curve["horizon"],
                curve["low"],
                curve["high"],
                color=setting["color"],
                alpha=0.08,
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

    fig_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        fig_dir
        / "compas_race_binary_exp4_postprocessing_over_horizon.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    print("Saved:", output_path)

    plt.show()

    return output_path


def parse_mean_sd(value):
    """
    Parse the mean and standard deviation from a string.
    """
    if pd.isna(value):
        return np.nan, np.nan

    text = str(value).strip()
    text = text.replace("$", "")
    text = text.replace("\\pm", "±")
    text = text.replace("+/-", "±")

    numbers = re.findall(
        r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?",
        text,
    )

    if len(numbers) >= 2:
        return float(numbers[0]), float(numbers[1])

    if len(numbers) == 1:
        return float(numbers[0]), np.nan

    return np.nan, np.nan


def prepare_performance_plot_df(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare the DataFrame for performance plotting.
    """
    df = dataframe.copy()

    df.columns = [
        str(col).strip()
        for col in df.columns
    ]

    rename_map = {
        "policy": "Policy",
        "preprocessing": "Preprocessing",

        "accuracy": "Accuracy",
        "Accuracy": "Accuracy",

        "average_reward": "Average reward",
        "average_reward_mean": "Average reward_mean",
        "average_reward_std": "Average reward_sd",
        "average_reward_sd": "Average reward_sd",
        "Average Reward": "Average reward",
        "Average reward": "Average reward",

        "cumulative_prediction_error": "Cumulative prediction error",
        "cumulative_prediction_error_mean": "Cumulative prediction error_mean",
        "cumulative_prediction_error_std": "Cumulative prediction error_sd",
        "cumulative_prediction_error_sd": "Cumulative prediction error_sd",

        "cumulative_regret": "Cumulative prediction error",
        "cumulative_regret_mean": "Cumulative prediction error_mean",
        "cumulative_regret_std": "Cumulative prediction error_sd",
        "cumulative_regret_sd": "Cumulative prediction error_sd",
        "Cumulative Regret": "Cumulative prediction error",
        "Cumulative regret": "Cumulative prediction error",

        "DP_gap": "DP gap",
        "DP_gap_mean": "DP gap_mean",
        "DP_gap_std": "DP gap_sd",
        "DP_gap_sd": "DP gap_sd",
        "DP Gap": "DP gap",
        "DP gap": "DP gap",

        "EO_gap": "EO gap",
        "EO_gap_mean": "EO gap_mean",
        "EO_gap_std": "EO gap_sd",
        "EO_gap_sd": "EO gap_sd",
        "EO Gap": "EO gap",
        "EO gap": "EO gap",

        "UtilityGap": "UtilityGap",
        "UtilityGap_mean": "UtilityGap_mean",
        "UtilityGap_std": "UtilityGap_sd",
        "UtilityGap_sd": "UtilityGap_sd",
        "utility_gap": "UtilityGap",
        "utility_gap_mean": "UtilityGap_mean",
        "utility_gap_std": "UtilityGap_sd",
        "utility_gap_sd": "UtilityGap_sd",
    }

    df = df.rename(
        columns={
            old: new
            for old, new in rename_map.items()
            if old in df.columns
        }
    )

    required_columns = [
        "Policy",
        "Preprocessing",
    ]

    for col in required_columns:
        if col not in df.columns:
            raise KeyError(
                f"Missing required column: {col}. "
                f"Available columns are: {df.columns.tolist()}"
            )

    df["Preprocessing"] = df["Preprocessing"].replace(
        {
            "uniform": "Uniform",
            "reweigh_group_label": "Reweighting",
            "Reweighing": "Reweighting",
            "Reweighting": "Reweighting",
            "Uniform": "Uniform",
        }
    )

    df["Method"] = (
        df["Policy"].astype(str)
        + " | "
        + df["Preprocessing"].astype(str)
    )

    metrics = [
        "Accuracy",
        "Average reward",
        "Cumulative prediction error",
        "DP gap",
        "EO gap",
        "UtilityGap",
    ]

    for metric in metrics:
        mean_col = f"{metric}_mean"
        sd_col = f"{metric}_sd"

        if mean_col in df.columns:
            df[mean_col] = pd.to_numeric(
                df[mean_col],
                errors="coerce",
            )

            if sd_col in df.columns:
                df[sd_col] = pd.to_numeric(
                    df[sd_col],
                    errors="coerce",
                )
            else:
                df[sd_col] = 0.0

            continue

        if metric not in df.columns:
            print(f"Skipping missing metric: {metric}")
            continue

        means = []
        sds = []

        for value in df[metric]:
            mean, sd = parse_mean_sd(value)
            means.append(mean)
            sds.append(sd)

        df[mean_col] = means
        df[sd_col] = sds

    return df


def horizontal_metric_plot(
    df: pd.DataFrame,
    metrics: list[str],
    title: str,
    save_path: Path | None = None,
    method_order: list[str] | None = None,
    figsize: tuple[int, int] = (11, 6),
    invert_lower_is_better: bool = False,
    n_seeds: int = 50,
    confidence: float = 0.95,
) -> None:
    """
    Plot horizontal error bars for specified metrics.
    Parameters:
    - df: pd.DataFrame
        The input DataFrame containing the metrics data.
    - metrics: list[str]
        A list of metrics to plot (e.g., ["DP gap", "EO gap"]).
    - title: str
        The title of the plot.
    - save_path: Path | None, optional (default=None)
        The path to save the plot. If None, the plot will not be saved.
    - method_order: list[str] | None, optional (default=None)
        The order of methods to display on the y-axis. If None, the order will be determined by the DataFrame.
    - figsize: tuple[int, int], optional (default=(11, 6))
        The size of the figure.
    - invert_lower_is_better: bool, optional (default=False)
        If True, invert the x-axis for metrics where lower values are better.
    """
    data = df.copy()

    if method_order is not None:
        data["Method"] = pd.Categorical(
            data["Method"],
            categories=method_order,
            ordered=True,
        )

        data = (
            data
            .dropna(subset=["Method"])
            .sort_values("Method")
        )

    methods = data["Method"].astype(str).tolist()
    y = np.arange(len(methods))

    fig, axes = plt.subplots(
        1,
        len(metrics),
        figsize=figsize,
        sharey=True,
    )

    if len(metrics) == 1:
        axes = [axes]

    fig.suptitle(
        title,
        fontweight="bold",
        y=1.02,
    )

    for ax, metric in zip(axes, metrics):
        mean_col = f"{metric}_mean"
        sd_col = f"{metric}_sd"

        if mean_col not in data.columns:
            raise KeyError(
                f"Missing plotting column: {mean_col}. "
                f"Available columns are: {data.columns.tolist()}"
            )

        if sd_col not in data.columns:
            data[sd_col] = 0.0

        ci_half_width = (
            _t_critical(
                n_seeds,
                confidence=confidence,
            )
            * data[sd_col]
            / np.sqrt(float(n_seeds))
        )

        ax.errorbar(
            data[mean_col],
            y,
            xerr=ci_half_width,
            fmt="o",
            capsize=4,
        )

        display_metric = METRIC_DISPLAY_NAMES.get(metric, metric)

        ax.set_title(
            display_metric,
            fontweight="bold",
        )

        ax.grid(
            True,
            axis="x",
            alpha=0.3,
        )

        ax.set_yticks(y)
        ax.set_yticklabels(methods)

        if metric in [
            "DP gap",
            "EO gap",
            "UtilityGap",
            "Cumulative prediction error",
        ]:
            ax.set_xlabel(
                "Mean ± 95% CI\n(lower is better)"
            )

            if invert_lower_is_better:
                ax.invert_xaxis()

        else:
            ax.set_xlabel(
                "Mean ± 95% CI\n(higher is better)"
            )

    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

        print("Saved:", save_path)

    plt.show()


def plot_cumulative_prediction_error_scale_diagnostic(
    temporal_df: pd.DataFrame,
    *,
    fig_dir: Path,
) -> Path:
    """
    Plot cumulative prediction error under several scale settings.

    Visual convention:
    - EXP4: blue
    - FairEXP4: red
    - Uniform: solid line
    - Reweighting: dotted line
    """
    dataframe = temporal_df.copy()

    if (
        "cumulative_prediction_error" not in dataframe.columns
        and "cumulative_regret" in dataframe.columns
    ):
        dataframe["cumulative_prediction_error"] = dataframe["cumulative_regret"]

    metric = "cumulative_prediction_error"

    if metric not in dataframe.columns:
        raise KeyError(
            f"Column not found: {metric}. "
            f"Available columns are: {dataframe.columns.tolist()}"
        )

    summary = aggregate_temporal_metric(
        dataframe,
        metric,
    )

    curves = [
        ("EXP4", "uniform"),
        ("EXP4", "reweigh_group_label"),
        ("FairEXP4", "uniform"),
        ("FairEXP4", "reweigh_group_label"),
    ]

    scale_settings = [
        ("linear", "linear", "Linear scale"),
        ("linear", "symlog", "Log-like y-axis"),
        ("log", "linear", "Log x-axis"),
    ]

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(20, 5),
        sharey=False,
    )

    fig.suptitle(
        "COMPAS | sensitive = race binary | EXP4 | Cumulative prediction error scale diagnostic",
        fontweight="bold",
        y=1.03,
    )

    for ax, (xscale, yscale, panel_title) in zip(axes, scale_settings):
        for policy, preprocessing in curves:
            curve = summary[
                (summary["policy"] == policy)
                & (summary["preprocessing"] == preprocessing)
            ].sort_values("t")

            if curve.empty:
                print("Missing curve:", policy, preprocessing)
                continue

            if xscale == "log":
                curve = curve[curve["t"] > 0]

            label = f"{policy} | {friendly_preprocessing(preprocessing)}"

            style = exp4_curve_style(
                policy=policy,
                preprocessing=preprocessing,
            )

            ax.plot(
                curve["t"],
                curve["mean"],
                label=label,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2.0,
                alpha=0.95,
            )

            ax.fill_between(
                curve["t"],
                curve["low"],
                curve["high"],
                color=style["color"],
                alpha=0.08,
            )

        ax.set_title(panel_title, fontweight="bold")
        ax.set_xlabel("Rounds")
        ax.set_ylabel("Cumulative prediction error")
        ax.grid(True, alpha=0.3)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)

        if yscale == "symlog":
            ax.set_ylabel("Cumulative prediction error\n(symlog scale)")

        ax.legend(fontsize=8)

    plt.tight_layout()

    fig_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        fig_dir
        / "compas_race_binary_exp4_cumulative_prediction_error_scale_diagnostic.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    print("Saved:", output_path)
    plt.show()

    return output_path

def plot_average_reward_over_time(
    temporal_df: pd.DataFrame,
    *,
    fig_dir: Path,
) -> Path:
    """
    Plot average reward over time for the COMPAS race-binary EXP4 experiment.

    Average reward corresponds to accuracy in this offline classification-derived
    bandit setting because the reward is 1 when the selected action matches the
    observed label and 0 otherwise.
    """
    if temporal_df.empty:
        raise ValueError(
            "temporal_df is empty. "
            "Run the benchmark before plotting average reward."
        )

    metric = "average_reward"

    if metric not in temporal_df.columns:
        if "avg_reward" in temporal_df.columns:
            temporal_df = temporal_df.copy()
            temporal_df["average_reward"] = temporal_df["avg_reward"]
        else:
            raise KeyError(
                f"Column not found: {metric}. "
                f"Available columns are: {temporal_df.columns.tolist()}"
            )

    summary = aggregate_temporal_metric(
        temporal_df,
        metric,
    )

    curves = [
        ("EXP4", "uniform"),
        ("EXP4", "reweigh_group_label"),
        ("FairEXP4", "uniform"),
        ("FairEXP4", "reweigh_group_label"),
    ]

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(11, 6),
    )

    fig.suptitle(
        "COMPAS | sensitive = race binary | EXP4 | Average reward over time",
        fontweight="bold",
        y=1.02,
    )

    for policy, preprocessing in curves:
        curve = summary[
            (summary["policy"] == policy)
            & (summary["preprocessing"] == preprocessing)
        ].sort_values("t")

        if curve.empty:
            print(
                "Missing curve:",
                policy,
                preprocessing,
            )
            continue

        label = f"{policy} | {friendly_preprocessing(preprocessing)}"
        style = exp4_curve_style(
            policy=policy,
            preprocessing=preprocessing,
        )

        draw_curve(
            ax,
            curve,
            label=label,
            **style,
        )

    ax.set_xlabel("Rounds")
    ax.set_ylabel("Average reward")
    ax.set_ylim(0.0, 1.0)
    ax.grid(
        True,
        alpha=0.3,
    )
    ax.legend(
        fontsize=8,
        loc="lower right",
    )

    plt.tight_layout()

    fig_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        fig_dir
        / "compas_race_binary_exp4_average_reward_over_time.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    print("Saved:", output_path)
    plt.show()

    return output_path

def plot_differential_cumulative_prediction_error(
    temporal_df: pd.DataFrame,
    *,
    fig_dir: Path,
) -> Path:
    """
    Plot differential cumulative prediction error relative to EXP4 | Uniform.

    Visual convention:
    - EXP4: blue
    - FairEXP4: red
    - Uniform: solid line
    - Reweighting: dotted line
    """
    dataframe = temporal_df.copy()

    if (
        "cumulative_prediction_error" not in dataframe.columns
        and "cumulative_regret" in dataframe.columns
    ):
        dataframe["cumulative_prediction_error"] = dataframe["cumulative_regret"]

    metric = "cumulative_prediction_error"

    if metric not in dataframe.columns:
        raise KeyError(
            f"Column not found: {metric}. "
            f"Available columns are: {dataframe.columns.tolist()}"
        )

    summary = aggregate_temporal_metric(
        dataframe,
        metric,
    )

    baseline = (
        summary[
            (summary["policy"] == "EXP4")
            & (summary["preprocessing"] == "uniform")
        ][
            [
                "t",
                "mean",
            ]
        ]
        .rename(
            columns={
                "mean": "baseline_mean",
            }
        )
    )

    if baseline.empty:
        raise ValueError(
            "Baseline EXP4 | Uniform not found in temporal_df."
        )

    curves = [
        ("EXP4", "uniform"),
        ("EXP4", "reweigh_group_label"),
        ("FairEXP4", "uniform"),
        ("FairEXP4", "reweigh_group_label"),
    ]

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(11, 6),
    )

    for policy, preprocessing in curves:
        curve = (
            summary[
                (summary["policy"] == policy)
                & (summary["preprocessing"] == preprocessing)
            ]
            .merge(
                baseline,
                on="t",
                how="inner",
            )
            .sort_values("t")
        )

        if curve.empty:
            print(
                "Missing curve:",
                policy,
                preprocessing,
            )
            continue

        curve["diff_mean"] = curve["mean"] - curve["baseline_mean"]
        curve["diff_low"] = curve["low"] - curve["baseline_mean"]
        curve["diff_high"] = curve["high"] - curve["baseline_mean"]

        label = f"{policy} | {friendly_preprocessing(preprocessing)}"

        style = exp4_curve_style(
            policy=policy,
            preprocessing=preprocessing,
        )

        ax.plot(
            curve["t"],
            curve["diff_mean"],
            label=label,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.0,
            alpha=0.95,
        )

        ax.fill_between(
            curve["t"],
            curve["diff_low"],
            curve["diff_high"],
            color=style["color"],
            alpha=0.08,
        )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=1,
        color="black",
        alpha=0.7,
    )

    ax.set_title(
        "COMPAS | sensitive = race binary | EXP4 | Differential cumulative prediction error",
        fontweight="bold",
    )

    ax.set_xlabel("Rounds")
    ax.set_ylabel(
        "Difference vs EXP4 | Uniform\n(Cumulative prediction error)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    plt.tight_layout()

    fig_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        fig_dir
        / "compas_race_binary_exp4_differential_cumulative_prediction_error.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    print("Saved:", output_path)
    plt.show()

    return output_path