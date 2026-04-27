from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd


def find_first_existing_column(
    df: pd.DataFrame,
    candidates: list[str],
    what: str,
) -> str:
    """
    Find the first existing column in the dataframe from a list of candidates. 
    """
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Could not find {what}. Tried: {candidates}")


def prepare_contextual_frame(
    df: pd.DataFrame,
    *,
    label_col: str,
    sensitive_col: str,
    label_transform: Callable[[pd.Series], np.ndarray],
    drop_cols: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    """
    Convert a dataframe into a contextual-bandit design matrix.

    The sensitive attribute is returned separately as "group" and is excluded from X by construction.
    """
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found.")
    if sensitive_col not in df.columns:
        raise ValueError(f"Sensitive column '{sensitive_col}' not found.")

    data = df.copy()

    y = np.asarray(label_transform(data[label_col]), dtype=int)
    group = data[sensitive_col].astype(str).fillna("Unknown").to_numpy()

    exclude = {label_col, sensitive_col}
    if drop_cols is not None:
        exclude.update(c for c in drop_cols if c in data.columns)

    X_df = data.drop(columns=[c for c in exclude if c in data.columns]).copy()

    for col in X_df.columns:
        if pd.api.types.is_numeric_dtype(X_df[col]):
            X_df[col] = pd.to_numeric(X_df[col], errors="coerce")
            fill_value = X_df[col].median() if X_df[col].notna().any() else 0.0
            X_df[col] = X_df[col].fillna(fill_value)
        else:
            X_df[col] = X_df[col].astype(str).fillna("Unknown")

    X_df = pd.get_dummies(X_df, drop_first=False)
    X_df = X_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    X = X_df.to_numpy(dtype=float)

    # Lightweight standardization for numerical stability in LinUCB
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True)
    sigma[sigma < 1e-12] = 1.0
    X = (X - mu) / sigma

    feature_names = X_df.columns.tolist()
    meta = {
        "label_col": label_col,
        "sensitive_col": sensitive_col,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
    }
    return X, y, group, feature_names, meta


def make_group_label_sampling_probs(group: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Reweight sampling by inverse frequency of each (group, label) stratum.
    """
    df = pd.DataFrame(
        {
            "group": pd.Series(group).astype(str),
            "y": np.asarray(y, dtype=int),
        }
    )

    counts = df.value_counts(["group", "y"]).to_dict()

    weights = np.zeros(len(df), dtype=float)
    for i, row in df.iterrows():
        weights[i] = 1.0 / counts[(row["group"], row["y"])]

    weights /= weights.sum()
    return weights
