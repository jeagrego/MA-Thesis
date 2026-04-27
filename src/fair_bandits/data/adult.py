from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

from .preprocessing import find_first_existing_column, prepare_contextual_frame

ADULT_DATASET_NAME = "adult"


def load_adult() -> pd.DataFrame:
    """
    Load the Adult dataset from OpenML.
    """
    X, y = fetch_openml(
        ADULT_DATASET_NAME,
        version=2,
        as_frame=True,
        return_X_y=True,
    )
    df = X.copy()
    df["income"] = y
    return df


def to_binary_income(y: pd.Series) -> np.ndarray:
    """
    Adult label: >50K -> 1, otherwise 0.
    """
    y_str = y.astype(str).str.strip()
    return (y_str == ">50K").astype(int).to_numpy()


def prepare_adult_contextual(
    df: pd.DataFrame,
    sensitive_col: str = "sex",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict]:
    """
    Prepare Adult for contextual bandit experiments.

    Accepted sensitive attributes typically include:
    - sex
    - race
    """
    label_col = find_first_existing_column(
        df,
        candidates=["income", "class", "target"],
        what="Adult label column",
    )

    return prepare_contextual_frame(
        df,
        label_col=label_col,
        sensitive_col=sensitive_col,
        label_transform=to_binary_income,
        drop_cols=[],
    )