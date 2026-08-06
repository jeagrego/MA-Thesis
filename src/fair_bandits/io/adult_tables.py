from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..metrics import build_paired_significance_table, normalize_metric_columns
from .latex import export_latex_table, fmt_mean_sd
from .save import save_dataframe


ADULT_SUMMARY_METRICS = [
    ("average_reward", "Average reward"),
    ("cumulative_prediction_error", "Cumulative prediction error"),
    ("UtilityGap", "UtilityGap"),
    ("DP_gap", "Demographic Parity Gap"),
    ("EO_gap", "Equalized Odds Gap"),
]


ADULT_SIGNIFICANCE_METRICS = [
    ("DP_gap", "Demographic Parity Gap"),
    ("EO_gap", "Equalized Odds Gap"),
    ("UtilityGap", "UtilityGap"),
    ("average_reward", "Average reward"),
    ("cumulative_prediction_error", "Cumulative prediction error"),
]


def adult_friendly_label(value: str, labels: dict[str, str] | None = None) -> str:
    labels = labels or {}
    return labels.get(str(value), str(value))


def build_adult_combined_final_frame(
    *,
    endpoint_df: pd.DataFrame,
    postproc_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine final online endpoints with held-out post-processing rows.
    """
    endpoint_df = normalize_metric_columns(endpoint_df)
    postproc_df = normalize_metric_columns(postproc_df)

    if endpoint_df.empty:
        online_final = pd.DataFrame()
    else:
        group_cols = ["family", "seed", "policy", "preprocessing"]
        available = [column for column in group_cols if column in endpoint_df.columns]
        online_final = (
            endpoint_df.sort_values("t")
            .groupby(available, as_index=False)
            .tail(1)
        )

    return normalize_metric_columns(
        pd.concat([online_final, postproc_df], ignore_index=True)
    )


def export_adult_final_summary_tables(
    *,
    endpoint_df: pd.DataFrame,
    postproc_df: pd.DataFrame,
    table_dir: str | Path,
    families: dict[str, dict],
    preprocessings: list[str],
    policy_labels: dict[str, str] | None = None,
    preprocessing_labels: dict[str, str] | None = None,
    summary_metrics: list[tuple[str, str]] | None = None,
    digits: int = 4,
) -> dict[str, pd.DataFrame]:
    """
    Export final Adult summary tables, one per policy family.
    """
    table_dir = Path(table_dir)
    table_dir.mkdir(parents=True, exist_ok=True)

    summary_metrics = ADULT_SUMMARY_METRICS if summary_metrics is None else summary_metrics
    combined = build_adult_combined_final_frame(
        endpoint_df=endpoint_df,
        postproc_df=postproc_df,
    )

    tables: dict[str, pd.DataFrame] = {}

    for family, family_config in families.items():
        family_label = str(family_config.get("label", family))
        policies = list(family_config.get("policies", []))
        pp_policy = str(family_config.get("postprocessed_policy", ""))
        display_policies = policies + ([pp_policy] if pp_policy else [])

        subset = combined[combined["family"] == family].copy()
        rows = []

        for policy in display_policies:
            for preprocessing in preprocessings:
                group = subset[
                    (subset["policy"] == policy)
                    & (subset["preprocessing"] == preprocessing)
                ].copy()

                if group.empty:
                    continue

                row = {
                    "Policy": adult_friendly_label(policy, policy_labels),
                    "Preprocessing": adult_friendly_label(preprocessing, preprocessing_labels),
                    "Seeds": int(group["seed"].nunique()),
                }

                for source_col, display_col in summary_metrics:
                    if source_col not in group.columns:
                        raise KeyError(
                            f"Missing summary column {source_col}. "
                            f"Available columns: {group.columns.tolist()}"
                        )
                    row[display_col] = fmt_mean_sd(group[source_col], digits=digits)

                rows.append(row)

        table = pd.DataFrame(rows)
        tables[family] = table

        csv_path = table_dir / f"adult_sex_{family}_final_summary.csv"
        tex_path = table_dir / f"adult_sex_{family}_final_summary.tex"

        save_dataframe(table, csv_path)
        export_latex_table(
            dataframe=table,
            path=tex_path,
            caption=(
                "Final utility and fairness summary for the Adult dataset "
                f"using sex as sensitive attribute and {family_label}. Values are "
                "reported as mean $\\pm$ standard deviation across seeds."
            ),
            label=f"tab:adult_sex_{family}_final_summary",
        )

        print("Saved:", csv_path)
        print("Saved:", tex_path)
        print(table.to_string(index=False))
        print()

    return tables


def default_adult_significance_comparisons_by_family() -> dict[str, list[dict]]:
    """
    Default Adult comparisons for online preprocessing, in-processing and PP.
    """
    return {
        "linear_ts": [
            {
                "Comparison": "Preprocessing: LinTS uniform vs reweighting",
                "baseline_policy": "LinTS",
                "baseline_preprocessing": "uniform",
                "intervention_policy": "LinTS",
                "intervention_preprocessing": "reweigh_group_label",
            },
            {
                "Comparison": "Preprocessing: FairLinTS uniform vs reweighting",
                "baseline_policy": "FairLinTS",
                "baseline_preprocessing": "uniform",
                "intervention_policy": "FairLinTS",
                "intervention_preprocessing": "reweigh_group_label",
            },
            {
                "Comparison": "In-processing: LinTS vs FairLinTS under uniform",
                "baseline_policy": "LinTS",
                "baseline_preprocessing": "uniform",
                "intervention_policy": "FairLinTS",
                "intervention_preprocessing": "uniform",
            },
            {
                "Comparison": "Post-processing: FairLinTS vs FairLinTS+PP under uniform",
                "baseline_policy": "FairLinTS",
                "baseline_preprocessing": "uniform",
                "intervention_policy": "FairLinTS+PP",
                "intervention_preprocessing": "uniform",
            },
            {
                "Comparison": "Post-processing: FairLinTS vs FairLinTS+PP under reweighting",
                "baseline_policy": "FairLinTS",
                "baseline_preprocessing": "reweigh_group_label",
                "intervention_policy": "FairLinTS+PP",
                "intervention_preprocessing": "reweigh_group_label",
            },
        ],
        "exp4": [
            {
                "Comparison": "Preprocessing: EXP4 uniform vs reweighting",
                "baseline_policy": "EXP4",
                "baseline_preprocessing": "uniform",
                "intervention_policy": "EXP4",
                "intervention_preprocessing": "reweigh_group_label",
            },
            {
                "Comparison": "Preprocessing: FairEXP4 uniform vs reweighting",
                "baseline_policy": "FairEXP4",
                "baseline_preprocessing": "uniform",
                "intervention_policy": "FairEXP4",
                "intervention_preprocessing": "reweigh_group_label",
            },
            {
                "Comparison": "In-processing: EXP4 vs FairEXP4 under uniform",
                "baseline_policy": "EXP4",
                "baseline_preprocessing": "uniform",
                "intervention_policy": "FairEXP4",
                "intervention_preprocessing": "uniform",
            },
            {
                "Comparison": "Post-processing: FairEXP4 vs FairEXP4+PP under uniform",
                "baseline_policy": "FairEXP4",
                "baseline_preprocessing": "uniform",
                "intervention_policy": "FairEXP4+PP",
                "intervention_preprocessing": "uniform",
            },
            {
                "Comparison": "Post-processing: FairEXP4 vs FairEXP4+PP under reweighting",
                "baseline_policy": "FairEXP4",
                "baseline_preprocessing": "reweigh_group_label",
                "intervention_policy": "FairEXP4+PP",
                "intervention_preprocessing": "reweigh_group_label",
            },
        ],
    }


def export_adult_significance_tables(
    *,
    endpoint_df: pd.DataFrame,
    postproc_df: pd.DataFrame,
    table_dir: str | Path,
    families: dict[str, dict],
    comparisons_by_family: dict[str, list[dict]] | None = None,
    significance_metrics: list[tuple[str, str]] | None = None,
    policy_labels: dict[str, str] | None = None,
    preprocessing_labels: dict[str, str] | None = None,
    seed_col: str = "seed",
    alpha: float = 0.05,
    value_digits: int = 3,
) -> dict[str, pd.DataFrame]:
    """
    Export paired Wilcoxon/Holm significance tables, one per Adult family.
    """
    table_dir = Path(table_dir)
    table_dir.mkdir(parents=True, exist_ok=True)

    combined = build_adult_combined_final_frame(
        endpoint_df=endpoint_df,
        postproc_df=postproc_df,
    )

    comparisons_by_family = (
        default_adult_significance_comparisons_by_family()
        if comparisons_by_family is None
        else comparisons_by_family
    )

    significance_metrics = (
        ADULT_SIGNIFICANCE_METRICS
        if significance_metrics is None
        else significance_metrics
    )

    tables: dict[str, pd.DataFrame] = {}

    for family, family_config in families.items():
        family_label = str(family_config.get("label", family))
        subset = combined[combined["family"] == family].copy()

        table = build_paired_significance_table(
            dataframe=subset,
            comparisons=comparisons_by_family[family],
            metrics=significance_metrics,
            policy_labels=policy_labels,
            preprocessing_labels=preprocessing_labels,
            seed_col=seed_col,
            alpha=alpha,
            value_digits=value_digits,
        )

        tables[family] = table

        csv_path = table_dir / f"adult_sex_{family}_significance_tests.csv"
        tex_path = table_dir / f"adult_sex_{family}_significance_tests.tex"

        save_dataframe(table, csv_path)
        export_latex_table(
            dataframe=table,
            path=tex_path,
            caption=(
                "Paired Wilcoxon signed-rank tests with Holm correction for "
                "the Adult dataset using sex as sensitive attribute and "
                f"{family_label}."
            ),
            label=f"tab:adult_sex_{family}_significance_tests",
        )

        print("Saved:", csv_path)
        print("Saved:", tex_path)
        print(table.to_string(index=False))
        print()

    return tables
