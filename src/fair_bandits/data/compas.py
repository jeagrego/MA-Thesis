from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .preprocessing import prepare_contextual_frame

COMPAS_URL = (
    "https://raw.githubusercontent.com/propublica/compas-analysis/"
    "master/compas-scores-two-years.csv"
)


def _download_bytes(url: str) -> bytes:
    with urllib.request.urlopen(url) as response:
        return response.read()


def load_compas(cache_dir: Path | None = None) -> pd.DataFrame:
    """
    Download-once loader for the public ProPublica COMPAS CSV.
    """
    cache_root = Path("data_cache") if cache_dir is None else Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    file_path = cache_root / "compas-scores-two-years.csv"
    if not file_path.exists():
        file_path.write_bytes(_download_bytes(COMPAS_URL))

    return pd.read_csv(file_path)


def clean_compas_minimal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal COMPAS cleaning aligned with your notebook.

    Keeps a stable subset of interpretable variables.
    """
    keep_cols = [
        "sex",
        "race",
        "age",
        "juv_fel_count",
        "juv_misd_count",
        "juv_other_count",
        "priors_count",
        "c_charge_degree",
        "two_year_recid",
    ]

    out = df.copy()
    out = out[keep_cols].dropna()

    out["two_year_recid"] = (
        pd.to_numeric(out["two_year_recid"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    out["sex"] = out["sex"].astype(str)
    out["race"] = out["race"].astype(str)
    out["c_charge_degree"] = out["c_charge_degree"].astype(str)
    return out


def binarize_race_compas(group: np.ndarray | pd.Series) -> np.ndarray:
    """
    Binary race grouping for stability:
    Caucasian -> White
    everything else -> Non-White
    """
    g = np.asarray(group).astype(str)
    return np.where(g == "Caucasian", "White", "Non-White")


def ensure_compas_race_binary(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "race_binary" not in out.columns:
        if "race" not in out.columns:
            raise ValueError(
                "The dataframe has no 'race' column, so 'race_binary' cannot be created."
            )
        out["race_binary"] = binarize_race_compas(out["race"])
    else:
        out["race_binary"] = out["race_binary"].astype(str)
    return out


def compas_reward_from_recid(two_year_recid: pd.Series) -> np.ndarray:
    """
    Reward = 1 for non-recidivism, 0 for recidivism.
    """
    z = (
        pd.to_numeric(two_year_recid, errors="coerce")
        .fillna(0)
        .astype(int)
        .to_numpy()
    )
    return 1 - z


def prepare_compas_contextual(
    df: pd.DataFrame,
    sensitive_col: str = "sex",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict]:
    """
    Prepare COMPAS for contextual bandit experiments.

    Expected workflow:
        df = load_compas(...)
        df = clean_compas_minimal(df)
        df = ensure_compas_race_binary(df)
        X, y, group, feature_names, meta = prepare_compas_contextual(df, sensitive_col="sex")
    """
    if sensitive_col not in df.columns:
        raise ValueError(f"Sensitive column '{sensitive_col}' not found in COMPAS dataframe.")

    return prepare_contextual_frame(
        df,
        label_col="two_year_recid",
        sensitive_col=sensitive_col,
        label_transform=compas_reward_from_recid,
        drop_cols=[],
    )
