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

def make_group_label_update_weights(
    *,
    groups: np.ndarray,
    labels: np.ndarray,
    label_name: str = "label",
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Compute inverse-frequency update weights for each group-label stratum.

    The returned weights have mean 1. This is important for online bandit replay:
    reweighing changes the relative importance of observations without changing
    the overall update scale.

    groups:
        Sensitive group observed at each round.

    labels:
        Label used to define the strata. In the synthetic CMAB benchmark this is
        usually the oracle action.

    label_name:
        Name used for the label column in the support table.

    Returns
    weights:
        One update weight per observation.

    support:
        Compact table with stratum counts and weights.
    """
    groups = np.asarray(groups).astype(str)
    labels = np.asarray(labels, dtype=int)

    if len(groups) != len(labels):
        raise ValueError("groups and labels must have the same length.")

    support = (
        pd.DataFrame(
            {
                "group": groups,
                label_name: labels,
            }
        )
        .value_counts(["group", label_name])
        .rename("n")
        .reset_index()
    )

    n_total = int(support["n"].sum())
    n_cells = int(len(support))

    support["raw_weight"] = n_total / (n_cells * support["n"])

    lookup = {
        (str(row["group"]), int(row[label_name])): float(row["raw_weight"])
        for row in support.to_dict("records")
    }

    weights = np.asarray(
        [
            lookup[(str(group), int(label))]
            for group, label in zip(groups, labels)
        ],
        dtype=float,
    )

    mean_weight = float(weights.mean())

    if mean_weight <= 0:
        raise ValueError("Mean update weight must be positive.")

    weights = weights / mean_weight
    support["normalized_weight"] = support["raw_weight"] / mean_weight

    support = (
        support
        .sort_values(["group", label_name])
        .reset_index(drop=True)
    )

    return weights, support


def make_uniform_update_weights(
    *,
    groups: np.ndarray,
    labels: np.ndarray,
    label_name: str = "label",
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Return unit update weights and the corresponding group-label support table.
    """
    groups = np.asarray(groups).astype(str)
    labels = np.asarray(labels, dtype=int)

    if len(groups) != len(labels):
        raise ValueError("groups and labels must have the same length.")

    weights = np.ones(len(groups), dtype=float)

    support = (
        pd.DataFrame(
            {
                "group": groups,
                label_name: labels,
            }
        )
        .value_counts(["group", label_name])
        .rename("n")
        .reset_index()
        .sort_values(["group", label_name])
        .reset_index(drop=True)
    )

    support["raw_weight"] = 1.0
    support["normalized_weight"] = 1.0

    return weights, support


def make_preprocessing_weights(
    *,
    preprocessing: str,
    groups: np.ndarray,
    oracle_actions: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Build the online update weights used by the synthetic CMAB benchmark.

    Supported modes:
    - "uniform": all observations receive weight 1.
    - "reweigh_group_label": inverse-frequency weighting by group and oracle action.
    """
    if preprocessing == "uniform":
        return make_uniform_update_weights(
            groups=groups,
            labels=oracle_actions,
            label_name="oracle_action",
        )

    if preprocessing == "reweigh_group_label":
        return make_group_label_update_weights(
            groups=groups,
            labels=oracle_actions,
            label_name="oracle_action",
        )

    raise ValueError(f"Unknown preprocessing mode: {preprocessing}")