from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def fmt_mean_sd(values) -> str:
    """
    Format the mean and standard deviation of a list of values as a LaTeX string.
    The output format is: "mean ± std", where mean and std are rounded to 4 decimal places.
    """
    array = np.asarray(list(values), dtype=float)

    return (
        f"{np.nanmean(array):.4f} "
        "$\\pm$ "
        f"{np.nanstd(array, ddof=1):.4f}"
    )


def export_latex_table(
    dataframe: pd.DataFrame,
    path: Path,
    caption: str,
    label: str,
) -> Path:
    """
    Export a pandas DataFrame as a LaTeX table to a file.
    """
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    latex = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\scriptsize\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        + dataframe.to_latex(
            index=False,
            escape=False,
        )
        + "}\n"
        "\\end{table}\n"
    )

    path.write_text(
        latex,
        encoding="utf-8",
    )

    return path