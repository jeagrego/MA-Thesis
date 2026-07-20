from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def fmt_mean_sd(values: Iterable[float], digits: int = 4) -> str:
    """
    Format values as mean ± standard deviation for LaTeX tables.
    """
    array = np.asarray(list(values), dtype=float)

    return (
        f"{np.nanmean(array):.{digits}f} "
        "$\\pm$ "
        f"{np.nanstd(array, ddof=1):.{digits}f}"
    )


def dataframe_to_latex_tabular(
    dataframe: pd.DataFrame,
    *,
    column_alignment: str | None = None,
) -> str:
    """
    Convert a small DataFrame to a LaTeX tabular without pandas.to_latex.
    """
    columns = [str(column) for column in dataframe.columns]

    if column_alignment is None:
        column_alignment = "l" * len(columns)

    lines = [
        "\\begin{tabular}{" + column_alignment + "}",
        "\\toprule",
        " & ".join(columns) + r" \\",
        "\\midrule",
    ]

    for _, row in dataframe.iterrows():
        values = [str(row[column]) for column in dataframe.columns]
        lines.append(" & ".join(values) + r" \\")

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
        ]
    )

    return "\n".join(lines)


def export_latex_table(
    dataframe: pd.DataFrame,
    path: str | Path,
    caption: str,
    label: str,
    *,
    scriptsize: bool = True,
    resize_to_textwidth: bool = True,
) -> Path:
    """
    Export a DataFrame as a LaTeX table without requiring Jinja2.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    size_line = "\\scriptsize\n" if scriptsize else ""

    tabular = dataframe_to_latex_tabular(dataframe)

    if resize_to_textwidth:
        body = "\\resizebox{\\textwidth}{!}{%\n" + tabular + "\n}%\n"
    else:
        body = tabular + "\n"

    latex = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        f"{size_line}"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"{body}"
        "\\end{table}\n"
    )

    out_path.write_text(latex, encoding="utf-8")

    return out_path