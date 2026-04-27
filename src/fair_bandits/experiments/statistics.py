from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon


def holm_correction(p_values: list[float]) -> list[float]:
    """
    Holm-Bonferroni correction.
    Returns adjusted p-values in the original order.
    """
    m = len(p_values)
    if m == 0:
        return []

    order = np.argsort(p_values)
    sorted_p = np.asarray(p_values, dtype=float)[order]

    adjusted = np.empty(m, dtype=float)
    running_max = 0.0

    for i, p in enumerate(sorted_p):
        adj = (m - i) * p
        running_max = max(running_max, adj)
        adjusted[i] = min(running_max, 1.0)

    out = np.empty(m, dtype=float)
    out[order] = adjusted
    return out.tolist()


def _iter_condition_slices(
    df: pd.DataFrame,
    group_cols: list[str] | None,
):
    if not group_cols:
        yield {}, df.copy()
        return

    for keys, gdf in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        cond = {col: val for col, val in zip(group_cols, keys)}
        yield cond, gdf.copy()


def friedman_test(
    df: pd.DataFrame,
    *,
    value_col: str,
    method_col: str = "policy",
    seed_col: str = "seed",
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Friedman test across methods, paired by seed.
    """
    rows: list[dict[str, Any]] = []

    for cond, gdf in _iter_condition_slices(df, group_cols):
        pivot = gdf.pivot_table(
            index=seed_col,
            columns=method_col,
            values=value_col,
            aggfunc="first",
        ).dropna(axis=0, how="any")

        methods = pivot.columns.tolist()
        if len(methods) < 3 or len(pivot) < 2:
            continue

        arrays = [pivot[m].to_numpy(dtype=float) for m in methods]
        stat, pval = friedmanchisquare(*arrays)

        row = dict(cond)
        row["value_col"] = value_col
        row["n_seeds"] = int(len(pivot))
        row["n_methods"] = int(len(methods))
        row["methods"] = ", ".join(methods)
        row["statistic"] = float(stat)
        row["p_value"] = float(pval)
        rows.append(row)

    return pd.DataFrame(rows)


def pairwise_wilcoxon(
    df: pd.DataFrame,
    *,
    value_col: str,
    method_col: str = "policy",
    seed_col: str = "seed",
    group_cols: list[str] | None = None,
    alternative: str = "two-sided",
    lower_is_better: bool = False,
) -> pd.DataFrame:
    """
    Pairwise Wilcoxon signed-rank tests across methods, paired by seed.
    Holm correction is applied separately within each condition slice.
    """
    rows: list[dict[str, Any]] = []

    for cond, gdf in _iter_condition_slices(df, group_cols):
        pivot = gdf.pivot_table(
            index=seed_col,
            columns=method_col,
            values=value_col,
            aggfunc="first",
        )

        local_rows: list[dict[str, Any]] = []

        for a, b in combinations(pivot.columns.tolist(), 2):
            pair = pivot[[a, b]].dropna()
            if len(pair) == 0:
                continue

            x = pair[a].to_numpy(dtype=float)
            y = pair[b].to_numpy(dtype=float)
            diff = x - y

            if np.allclose(diff, 0.0):
                stat = 0.0
                pval = 1.0
            else:
                stat, pval = wilcoxon(
                    x,
                    y,
                    alternative=alternative,
                    zero_method="wilcox",
                    method="auto",
                )

            mean_a = float(x.mean())
            mean_b = float(y.mean())

            if np.isclose(mean_a, mean_b):
                winner = "tie"
            else:
                if lower_is_better:
                    winner = a if mean_a < mean_b else b
                else:
                    winner = a if mean_a > mean_b else b

            row = dict(cond)
            row["value_col"] = value_col
            row["method_a"] = a
            row["method_b"] = b
            row["n_pairs"] = int(len(pair))
            row["mean_a"] = mean_a
            row["mean_b"] = mean_b
            row["mean_diff_a_minus_b"] = float(mean_a - mean_b)
            row["statistic"] = float(stat)
            row["p_raw"] = float(pval)
            row["winner"] = winner
            local_rows.append(row)

        if local_rows:
            adjusted = holm_correction([r["p_raw"] for r in local_rows])
            for r, p_adj in zip(local_rows, adjusted):
                r["p_holm"] = float(p_adj)
            rows.extend(local_rows)

    return pd.DataFrame(rows)


def compare_methods(
    df: pd.DataFrame,
    *,
    value_col: str,
    method_col: str = "policy",
    seed_col: str = "seed",
    group_cols: list[str] | None = None,
    alternative: str = "two-sided",
    lower_is_better: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    friedman_df = friedman_test(
        df,
        value_col=value_col,
        method_col=method_col,
        seed_col=seed_col,
        group_cols=group_cols,
    )

    wilcoxon_df = pairwise_wilcoxon(
        df,
        value_col=value_col,
        method_col=method_col,
        seed_col=seed_col,
        group_cols=group_cols,
        alternative=alternative,
        lower_is_better=lower_is_better,
    )
    return friedman_df, wilcoxon_df
