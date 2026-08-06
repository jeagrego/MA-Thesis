from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


ADULT_COLUMN_NAMES = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]


@dataclass(frozen=True)
class AdultSexPreparedData:
    """
    Adult-sex dataset prepared for classification-based CMAB experiments.

    The reward used by the downstream runners is 1(action == y_true), so
    average reward is equivalent to accuracy.
    """

    X_train: np.ndarray
    y_train: np.ndarray
    g_train: np.ndarray
    X_cal: np.ndarray
    y_cal: np.ndarray
    g_cal: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    g_test: np.ndarray
    feature_names: list[str]
    group_names: list[str]
    scaler: StandardScaler
    raw_frame: pd.DataFrame
    design_frame: pd.DataFrame


def normalize_adult_column(column: str) -> str:
    """
    Normalize Adult column names to a stable snake-case convention.
    """
    return (
        str(column)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def find_adult_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
    *,
    what: str,
) -> str:
    """
    Find the first matching Adult column after normalization.
    """
    lookup = {
        normalize_adult_column(column): column
        for column in dataframe.columns
    }

    for candidate in candidates:
        normalized = normalize_adult_column(candidate)
        if normalized in lookup:
            return lookup[normalized]

    raise KeyError(
        f"Missing {what}. Expected one of {candidates}. "
        f"Available columns are: {list(dataframe.columns)}"
    )


def discover_adult_csv(
    adult_csv_path: str | Path | None = None,
) -> Path:
    """
    Find a local Adult dataset CSV/data file.
    """
    if adult_csv_path is not None:
        path = Path(adult_csv_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    candidates = [
        Path.cwd() / ".." / "data" / "adult.csv",
        Path.cwd() / ".." / "data" / "adult" / "adult.csv",
        Path.cwd() / ".." / "datasets" / "adult.csv",
        Path.cwd() / "data" / "adult.csv",
        Path.cwd() / "adult.csv",
        Path.cwd() / ".." / "data" / "adult.data",
        Path.cwd() / "adult.data",
        Path.home() / "Downloads" / "adult.csv",
        Path.home() / "Downloads" / "adult.data",
    ]

    for path in candidates:
        path = path.expanduser().resolve()
        if path.exists():
            return path

    raise FileNotFoundError(
        "Adult dataset not found. Set ADULT_CSV_PATH explicitly or place "
        "adult.csv/adult.data in data/, datasets/, the working directory, or Downloads."
    )


def load_adult_dataframe(
    *,
    adult_csv_path: str | Path | None = None,
    in_memory_frames: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """
    Load the Adult dataframe from memory or from a local CSV/data file.
    """
    if in_memory_frames:
        for name in ["adult_df", "adult_raw_df", "df"]:
            obj = in_memory_frames.get(name)
            if isinstance(obj, pd.DataFrame) and len(obj) > 1000:
                print("Using in-memory dataframe:", name)
                return obj.copy()

    path = discover_adult_csv(adult_csv_path)
    print("Loading:", path)

    if path.suffix.lower() == ".data":
        return pd.read_csv(
            path,
            names=ADULT_COLUMN_NAMES,
            skipinitialspace=True,
        )

    return pd.read_csv(path)


def encode_adult_income(series: pd.Series) -> pd.Series:
    """
    Encode Adult income label as >50K -> 1, otherwise 0.
    """
    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="coerce")
        observed = set(values.dropna().astype(int).unique())
        if observed.issubset({0, 1}):
            return values.astype(int)

    text = (
        series.astype(str)
        .str.strip()
        .str.replace(".", "", regex=False)
    )

    return text.str.contains(">50K", case=False, regex=False).astype(int)


def make_adult_design_matrix(
    raw_dataframe: pd.DataFrame,
    *,
    target_col: str,
    sensitive_col: str,
) -> pd.DataFrame:
    """
    Build the one-hot encoded design matrix.
    """
    X_df = raw_dataframe.drop(
        columns=[target_col, sensitive_col],
        errors="ignore",
    )

    X_df = X_df.drop(
        columns=[
            column
            for column in X_df.columns
            if normalize_adult_column(column) in {"id", "index", "fnlwgt"}
        ],
        errors="ignore",
    )

    X_df = X_df.replace(
        {
            "?": np.nan,
            " ?": np.nan,
            "": np.nan,
        }
    )

    X_df = pd.get_dummies(
        X_df,
        dummy_na=True,
        drop_first=False,
    )

    X_df = (
        X_df.apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
    )

    return X_df


def prepare_adult_sex_splits(
    *,
    adult_csv_path: str | Path | None = None,
    in_memory_frames: dict[str, pd.DataFrame] | None = None,
    test_size: float = 0.20,
    calibration_size_within_remaining: float = 0.20,
    random_state_test: int = 42,
    random_state_calibration: int = 43,
) -> AdultSexPreparedData:
    """
    Prepare Adult with sex as sensitive attribute and produce train/cal/test splits.
    The split is stratified by group-label strata.
    """
    raw = load_adult_dataframe(
        adult_csv_path=adult_csv_path,
        in_memory_frames=in_memory_frames,
    )

    raw = raw.copy()
    raw.columns = [normalize_adult_column(column) for column in raw.columns]

    target_col = find_adult_column(
        raw,
        ["income", "class", "label", "target", "y", "income_bracket"],
        what="Adult target column",
    )

    sex_col = find_adult_column(
        raw,
        ["sex", "gender"],
        what="Adult sex column",
    )

    y_series = encode_adult_income(raw[target_col])
    group_series = raw[sex_col].astype(str).str.strip()
    X_df = make_adult_design_matrix(
        raw,
        target_col=target_col,
        sensitive_col=sex_col,
    )

    strata = group_series.astype(str) + "||" + y_series.astype(str)

    (
        X_train_df,
        X_test_df,
        y_train_s,
        y_test_s,
        g_train_s,
        g_test_s,
    ) = train_test_split(
        X_df,
        y_series,
        group_series,
        test_size=test_size,
        random_state=random_state_test,
        stratify=strata,
    )

    strata_remaining = g_train_s.astype(str) + "||" + y_train_s.astype(str)

    (
        X_train_df,
        X_cal_df,
        y_train_s,
        y_cal_s,
        g_train_s,
        g_cal_s,
    ) = train_test_split(
        X_train_df,
        y_train_s,
        g_train_s,
        test_size=calibration_size_within_remaining,
        random_state=random_state_calibration,
        stratify=strata_remaining,
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train_df)
    X_cal = scaler.transform(X_cal_df)
    X_test = scaler.transform(X_test_df)

    X_train = np.nan_to_num(
        X_train,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    X_cal = np.nan_to_num(
        X_cal,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    X_test = np.nan_to_num(
        X_test,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    y_train = y_train_s.to_numpy(dtype=int)
    y_cal = y_cal_s.to_numpy(dtype=int)
    y_test = y_test_s.to_numpy(dtype=int)

    g_train = g_train_s.astype(str).to_numpy()
    g_cal = g_cal_s.astype(str).to_numpy()
    g_test = g_test_s.astype(str).to_numpy()

    print("Prepared X:", X_df.shape)
    print("Train:", X_train.shape, "Calibration:", X_cal.shape, "Test:", X_test.shape)
    print("Groups:", sorted(np.unique(g_train)))

    return AdultSexPreparedData(
        X_train=X_train,
        y_train=y_train,
        g_train=g_train,
        X_cal=X_cal,
        y_cal=y_cal,
        g_cal=g_cal,
        X_test=X_test,
        y_test=y_test,
        g_test=g_test,
        feature_names=[str(column) for column in X_df.columns],
        group_names=sorted(np.unique(g_train).astype(str).tolist()),
        scaler=scaler,
        raw_frame=raw,
        design_frame=X_df,
    )
