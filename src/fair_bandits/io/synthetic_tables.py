from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..metrics import build_paired_significance_table
from .latex import export_latex_table, fmt_mean_sd
from .save import save_dataframe


def friendly_label(
    value: str,
    labels: dict[str, str] | None = None,
) -> str:
    """
    Return a display label while keeping unknown values unchanged.
    """
    if labels is None:
        labels = {}

    return labels.get(str(value), str(value))


def build_synthetic_final_summary_table(
    *,
    final_endpoint_df: pd.DataFrame,
    regime: str,
    policies_by_regime: dict[str, list[str]],
    preprocessings: list[str],
    summary_metrics: list[tuple[str, str]],
    policy_labels: dict[str, str] | None = None,
    preprocessing_labels: dict[str, str] | None = None,
    digits: int = 4,
) -> pd.DataFrame:
    """
    Build a final synthetic utility/fairness summary table for one regime.

    The input dataframe must already be restricted to the final checkpoint.
    """
    subset = final_endpoint_df[
        final_endpoint_df["regime"] == regime
    ].copy()

    rows = []

    for policy in policies_by_regime[regime]:
        for preprocessing in preprocessings:
            group = subset[
                (subset["policy"] == policy)
                & (subset["preprocessing"] == preprocessing)
            ].copy()

            if group.empty:
                continue

            row = {
                "Policy": friendly_label(policy, policy_labels),
                "Preprocessing": friendly_label(
                    preprocessing,
                    preprocessing_labels,
                ),
            }

            for source_col, display_col in summary_metrics:
                if source_col not in group.columns:
                    raise KeyError(
                        f"Missing summary metric column: {source_col}. "
                        f"Available columns are: {group.columns.tolist()}"
                    )

                row[display_col] = fmt_mean_sd(
                    group[source_col],
                    digits=digits,
                )

            rows.append(row)

    return pd.DataFrame(rows)


def export_synthetic_final_summary_table(
    *,
    final_endpoint_df: pd.DataFrame,
    regime: str,
    table_dir: str | Path,
    policies_by_regime: dict[str, list[str]],
    preprocessings: list[str],
    summary_metrics: list[tuple[str, str]],
    regime_labels: dict[str, str],
    regime_file_tags: dict[str, str],
    final_t: int,
    policy_labels: dict[str, str] | None = None,
    preprocessing_labels: dict[str, str] | None = None,
    caption_prefix: str = "Final utility and fairness summary for the synthetic",
    label_prefix: str = "tab:synthetic",
    file_prefix: str = "synthetic",
    digits: int = 4,
) -> pd.DataFrame:
    """
    Build and export one final synthetic summary table as CSV and LaTeX.
    """
    table_dir = Path(table_dir)
    table_dir.mkdir(parents=True, exist_ok=True)

    table = build_synthetic_final_summary_table(
        final_endpoint_df=final_endpoint_df,
        regime=regime,
        policies_by_regime=policies_by_regime,
        preprocessings=preprocessings,
        summary_metrics=summary_metrics,
        policy_labels=policy_labels,
        preprocessing_labels=preprocessing_labels,
        digits=digits,
    )

    file_tag = regime_file_tags[regime]

    csv_path = table_dir / f"{file_prefix}_{file_tag}_final_summary.csv"
    tex_path = table_dir / f"{file_prefix}_{file_tag}_final_summary.tex"

    save_dataframe(table, csv_path)

    export_latex_table(
        dataframe=table,
        path=tex_path,
        caption=(
            f"{caption_prefix} {regime_labels[regime].lower()} regime. "
            f"Values are reported as mean $\\pm$ standard deviation across "
            f"seeds at round {final_t}."
        ),
        label=f"{label_prefix}_{file_tag}_final_summary",
    )

    print("Saved:", csv_path)
    print("Saved:", tex_path)
    print()
    print(table.to_string(index=False))

    return table


def export_synthetic_final_summary_tables(
    *,
    final_endpoint_df: pd.DataFrame,
    regimes: list[str],
    table_dir: str | Path,
    policies_by_regime: dict[str, list[str]],
    preprocessings: list[str],
    summary_metrics: list[tuple[str, str]],
    regime_labels: dict[str, str],
    regime_file_tags: dict[str, str],
    final_t: int,
    policy_labels: dict[str, str] | None = None,
    preprocessing_labels: dict[str, str] | None = None,
    caption_prefix: str = "Final utility and fairness summary for the synthetic",
    label_prefix: str = "tab:synthetic",
    file_prefix: str = "synthetic",
    digits: int = 4,
) -> dict[str, pd.DataFrame]:
    """
    Export final synthetic summary tables for several regimes.
    """
    tables: dict[str, pd.DataFrame] = {}

    for regime in regimes:
        print("Final utility and fairness summary:", regime)

        tables[regime] = export_synthetic_final_summary_table(
            final_endpoint_df=final_endpoint_df,
            regime=regime,
            table_dir=table_dir,
            policies_by_regime=policies_by_regime,
            preprocessings=preprocessings,
            summary_metrics=summary_metrics,
            regime_labels=regime_labels,
            regime_file_tags=regime_file_tags,
            final_t=final_t,
            policy_labels=policy_labels,
            preprocessing_labels=preprocessing_labels,
            caption_prefix=caption_prefix,
            label_prefix=label_prefix,
            file_prefix=file_prefix,
            digits=digits,
        )

    return tables


def export_synthetic_significance_table(
    *,
    final_endpoint_df: pd.DataFrame,
    regime: str,
    table_dir: str | Path,
    comparisons_by_regime: dict[str, list[dict]],
    significance_metrics: list[tuple[str, str]],
    regime_labels: dict[str, str],
    regime_file_tags: dict[str, str],
    final_t: int,
    policy_labels: dict[str, str] | None = None,
    preprocessing_labels: dict[str, str] | None = None,
    caption_prefix: str = "Paired Wilcoxon tests with Holm correction for the synthetic",
    label_prefix: str = "tab:synthetic",
    file_prefix: str = "synthetic",
    seed_col: str = "seed",
    alpha: float = 0.05,
    value_digits: int = 3,
) -> pd.DataFrame:
    """
    Build and export one paired significance table as CSV and LaTeX.
    """
    table_dir = Path(table_dir)
    table_dir.mkdir(parents=True, exist_ok=True)

    subset = final_endpoint_df[
        final_endpoint_df["regime"] == regime
    ].copy()

    comparisons = comparisons_by_regime[regime]

    table = build_paired_significance_table(
        dataframe=subset,
        comparisons=comparisons,
        metrics=significance_metrics,
        policy_labels=policy_labels,
        preprocessing_labels=preprocessing_labels,
        seed_col=seed_col,
        alpha=alpha,
        value_digits=value_digits,
    )

    file_tag = regime_file_tags[regime]

    csv_path = table_dir / f"{file_prefix}_{file_tag}_significance_tests.csv"
    tex_path = table_dir / f"{file_prefix}_{file_tag}_significance_tests.tex"

    save_dataframe(table, csv_path)

    export_latex_table(
        dataframe=table,
        path=tex_path,
        caption=(
            f"{caption_prefix} {regime_labels[regime].lower()} regime "
            f"at round {final_t}."
        ),
        label=f"{label_prefix}_{file_tag}_significance_tests",
    )

    print("Saved:", csv_path)
    print("Saved:", tex_path)
    print()
    print(table.to_string(index=False))

    return table


def export_synthetic_significance_tables(
    *,
    final_endpoint_df: pd.DataFrame,
    regimes: list[str],
    table_dir: str | Path,
    comparisons_by_regime: dict[str, list[dict]],
    significance_metrics: list[tuple[str, str]],
    regime_labels: dict[str, str],
    regime_file_tags: dict[str, str],
    final_t: int,
    policy_labels: dict[str, str] | None = None,
    preprocessing_labels: dict[str, str] | None = None,
    caption_prefix: str = "Paired Wilcoxon tests with Holm correction for the synthetic",
    label_prefix: str = "tab:synthetic",
    file_prefix: str = "synthetic",
    seed_col: str = "seed",
    alpha: float = 0.05,
    value_digits: int = 3,
) -> dict[str, pd.DataFrame]:
    """
    Export paired significance tables for several regimes.
    """
    tables: dict[str, pd.DataFrame] = {}

    for regime in regimes:
        print("Significance table:", regime)

        tables[regime] = export_synthetic_significance_table(
            final_endpoint_df=final_endpoint_df,
            regime=regime,
            table_dir=table_dir,
            comparisons_by_regime=comparisons_by_regime,
            significance_metrics=significance_metrics,
            regime_labels=regime_labels,
            regime_file_tags=regime_file_tags,
            final_t=final_t,
            policy_labels=policy_labels,
            preprocessing_labels=preprocessing_labels,
            caption_prefix=caption_prefix,
            label_prefix=label_prefix,
            file_prefix=file_prefix,
            seed_col=seed_col,
            alpha=alpha,
            value_digits=value_digits,
        )

    return tables