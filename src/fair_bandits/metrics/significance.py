from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def safe_wilcoxon(
    baseline_values: Sequence[float],
    intervention_values: Sequence[float],
) -> float:
    """
    Run a paired Wilcoxon signed-rank test safely.

    Returns NaN when there are too few paired values, and 1.0 when all paired
    differences are exactly zero.
    """
    baseline = np.asarray(baseline_values, dtype=float)
    intervention = np.asarray(intervention_values, dtype=float)

    if baseline.shape != intervention.shape:
        raise ValueError(
            "baseline_values and intervention_values must have the same shape."
        )

    finite_mask = np.isfinite(baseline) & np.isfinite(intervention)
    baseline = baseline[finite_mask]
    intervention = intervention[finite_mask]

    if baseline.size < 2:
        return np.nan

    differences = intervention - baseline

    if np.allclose(differences, 0.0):
        return 1.0

    try:
        result = wilcoxon(
            baseline,
            intervention,
            zero_method="wilcox",
            alternative="two-sided",
        )
        return float(result.pvalue)
    except ValueError:
        return np.nan


def holm_adjust(
    p_values: Sequence[float],
) -> list[float]:
    """
    Holm-adjust a list of p-values.

    NaN p-values are preserved as NaN.
    """
    p_array = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p_array, np.nan, dtype=float)

    finite_indices = np.flatnonzero(np.isfinite(p_array))

    if finite_indices.size == 0:
        return adjusted.tolist()

    finite_p = p_array[finite_indices]
    order = np.argsort(finite_p)

    sorted_indices = finite_indices[order]
    sorted_p = finite_p[order]

    n_tests = len(sorted_p)
    running_max = 0.0

    for rank, original_index in enumerate(sorted_indices):
        multiplier = n_tests - rank
        candidate = min(float(sorted_p[rank]) * multiplier, 1.0)
        running_max = max(running_max, candidate)
        adjusted[original_index] = running_max

    return adjusted.tolist()


def format_p_value(
    p_value: float,
    *,
    threshold: float = 0.001,
    digits: int = 3,
) -> str:
    """
    Format p-values for thesis tables.
    """
    if not np.isfinite(p_value):
        return "NA"

    if p_value < threshold:
        return f"<{threshold:.3f}"

    return f"{p_value:.{digits}f}"


def format_mean_sd(
    values: Sequence[float],
    *,
    digits: int = 3,
) -> str:
    """
    Format values as mean ± standard deviation.
    """
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]

    if array.size == 0:
        return "NA"

    if array.size == 1:
        return f"{array[0]:.{digits}f} ± 0.000"

    return f"{array.mean():.{digits}f} ± {array.std(ddof=1):.{digits}f}"


def method_label(
    *,
    policy: str,
    preprocessing: str,
    policy_labels: dict[str, str] | None = None,
    preprocessing_labels: dict[str, str] | None = None,
) -> str:
    """
    Build a display label for one policy-preprocessing combination.
    """
    policy_labels = policy_labels or {}
    preprocessing_labels = preprocessing_labels or {}

    policy_text = policy_labels.get(str(policy), str(policy))
    preprocessing_text = preprocessing_labels.get(
        str(preprocessing),
        str(preprocessing),
    )

    return f"{policy_text} | {preprocessing_text}"


def extract_paired_values(
    dataframe: pd.DataFrame,
    *,
    metric: str,
    baseline_policy: str,
    baseline_preprocessing: str,
    intervention_policy: str,
    intervention_preprocessing: str,
    seed_col: str = "seed",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract paired baseline and intervention values for one metric.

    Pairing is done by seed.
    """
    required_columns = {
        seed_col,
        "policy",
        "preprocessing",
        metric,
    }

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise KeyError(
            f"Missing columns for paired extraction: {sorted(missing_columns)}"
        )

    baseline = dataframe[
        (dataframe["policy"] == baseline_policy)
        & (dataframe["preprocessing"] == baseline_preprocessing)
    ][[seed_col, metric]].copy()

    intervention = dataframe[
        (dataframe["policy"] == intervention_policy)
        & (dataframe["preprocessing"] == intervention_preprocessing)
    ][[seed_col, metric]].copy()

    baseline = (
        baseline
        .groupby(seed_col, as_index=False)[metric]
        .mean()
        .rename(columns={metric: "baseline"})
    )

    intervention = (
        intervention
        .groupby(seed_col, as_index=False)[metric]
        .mean()
        .rename(columns={metric: "intervention"})
    )

    paired = baseline.merge(
        intervention,
        on=seed_col,
        how="inner",
    )

    return (
        paired["baseline"].to_numpy(dtype=float),
        paired["intervention"].to_numpy(dtype=float),
    )


def build_paired_significance_table(
    *,
    dataframe: pd.DataFrame,
    comparisons: list[dict],
    metrics: list[tuple[str, str]],
    policy_labels: dict[str, str] | None = None,
    preprocessing_labels: dict[str, str] | None = None,
    seed_col: str = "seed",
    alpha: float = 0.05,
    value_digits: int = 3,
) -> pd.DataFrame:
    """
    Build a paired significance table with Holm correction.
    
    Parameters:
    - dataframe: DataFrame containing the performance metrics.
    - comparisons: List of dictionaries specifying the baseline and intervention policies and preprocessings.
    - metrics: List of tuples specifying the metric column and its display label.
    - policy_labels: Optional dictionary mapping policy names to display labels.
    - preprocessing_labels: Optional dictionary mapping preprocessing names to display labels.
    - seed_col: Column name for the random seed used for pairing.
    - alpha: Significance level for determining significance.
    - value_digits: Number of decimal places to display for metric values.
    
    Returns:
    - A DataFrame containing the paired significance results with Holm-adjusted p-values and significance flags
    """
    rows = []
    raw_p_values = []

    for comparison in comparisons:
        for metric_col, metric_label in metrics:
            baseline_values, intervention_values = extract_paired_values(
                dataframe,
                metric=metric_col,
                baseline_policy=comparison["baseline_policy"],
                baseline_preprocessing=comparison["baseline_preprocessing"],
                intervention_policy=comparison["intervention_policy"],
                intervention_preprocessing=comparison["intervention_preprocessing"],
                seed_col=seed_col,
            )

            raw_p = safe_wilcoxon(
                baseline_values,
                intervention_values,
            )

            raw_p_values.append(raw_p)

            rows.append(
                {
                    "Comparison": comparison["Comparison"],
                    "Metric": metric_label,
                    "Baseline": method_label(
                        policy=comparison["baseline_policy"],
                        preprocessing=comparison["baseline_preprocessing"],
                        policy_labels=policy_labels,
                        preprocessing_labels=preprocessing_labels,
                    ),
                    "Intervention": method_label(
                        policy=comparison["intervention_policy"],
                        preprocessing=comparison["intervention_preprocessing"],
                        policy_labels=policy_labels,
                        preprocessing_labels=preprocessing_labels,
                    ),
                    "Baseline value": format_mean_sd(
                        baseline_values,
                        digits=value_digits,
                    ),
                    "Intervention value": format_mean_sd(
                        intervention_values,
                        digits=value_digits,
                    ),
                    "Raw p": raw_p,
                }
            )

    table = pd.DataFrame(rows)

    if table.empty:
        return pd.DataFrame(
            columns=[
                "Comparison",
                "Metric",
                "Baseline",
                "Intervention",
                "Baseline value",
                "Intervention value",
                "Holm p",
                "Sig.",
            ]
        )

    adjusted_p_values = holm_adjust(raw_p_values)

    table["Holm p"] = [
        format_p_value(p_value)
        for p_value in adjusted_p_values
    ]

    table["Sig."] = [
        "Yes" if np.isfinite(p_value) and p_value < alpha else "No"
        for p_value in adjusted_p_values
    ]

    table = table.drop(columns=["Raw p"])

    return table[
        [
            "Comparison",
            "Metric",
            "Baseline",
            "Intervention",
            "Baseline value",
            "Intervention value",
            "Holm p",
            "Sig.",
        ]
    ]