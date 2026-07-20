from __future__ import annotations

import pandas as pd


COLUMN_RENAMES: dict[str, str] = {
    "regret": "prediction_error_increment",
    "cumulative_regret": "cumulative_prediction_error",
    "cum_regret": "cumulative_prediction_error",
    "CumulativeRegret": "cumulative_prediction_error",
}


def normalize_metric_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize legacy regret terminology to prediction-error terminology.

    This keeps old cached outputs readable while making new outputs use the
    thesis terminology:
        cumulative_regret -> cumulative_prediction_error
    """
    out = dataframe.copy()

    rename_map = {
        old: new
        for old, new in COLUMN_RENAMES.items()
        if old in out.columns and new not in out.columns
    }

    if rename_map:
        out = out.rename(columns=rename_map)

    return out