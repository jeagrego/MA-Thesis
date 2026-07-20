from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..experiments.runner import ExperimentBundle

from ..metrics import normalize_metric_columns


def experiment_bundle_exists(path: str | Path) -> bool:
    """
    Check if a complete experiment bundle exists at the given path.
    """
    root = Path(path)
    return (
        (root / "logs.csv").exists()
        and (root / "seed_summary.csv").exists()
        and (root / "temporal_summary.csv").exists()
        and (root / "metadata.json").exists()
    )


def load_experiment_bundle(path: str | Path) -> ExperimentBundle:
    """
    Load an experiment bundle from the given path.
    """
    root = Path(path)
    if not experiment_bundle_exists(root):
        raise FileNotFoundError(
            f"Could not find a complete experiment bundle in: {root}"
        )

    logs = pd.read_csv(root / "logs.csv")
    seed_summary = pd.read_csv(root / "seed_summary.csv")
    temporal_summary = pd.read_csv(root / "temporal_summary.csv")

    with open(root / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return ExperimentBundle(
        logs=logs,
        seed_summary=seed_summary,
        temporal_summary=temporal_summary,
        metadata=metadata,
    )
    
DEFAULT_SYNTHETIC_TEMPORAL_COLUMNS = [
    "t",
    "avg_reward",
    "cumulative_prediction_error",
    "DP_gap_over_time",
    "EO_gap_over_time",
    "UtilityGap_over_time",
]

LEGACY_SYNTHETIC_TEMPORAL_COLUMNS = [
    "cumulative_regret",
    "cum_regret",
    "regret",
]


def read_csv_or_empty(path: str | Path) -> pd.DataFrame:
    """
    Read a CSV file if it exists, otherwise return an empty DataFrame.
    """
    csv_path = Path(path)

    if not csv_path.exists():
        return pd.DataFrame()

    return pd.read_csv(csv_path)


def resolve_synthetic_trajectory_path(
    row: pd.Series,
    *,
    run_dir: str | Path,
) -> Path:
    """
    Resolve a synthetic trajectory path.

    Prefer the exact path saved in run_index.csv. If the path belongs to another
    computer, reconstruct the path relative to the current run_dir.
    """
    original_path = Path(str(row["trajectory_file"]))

    if original_path.exists():
        return original_path

    return (
        Path(run_dir)
        / "trajectories"
        / str(row["regime"])
        / str(row["preprocessing"])
        / str(row["policy"])
        / f"seed_{int(row['seed']):03d}.csv.gz"
    )


def read_synthetic_temporal_file(
    path: str | Path,
    *,
    temporal_columns: list[str] | None = None,
    legacy_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Read one synthetic trajectory file and support both current and legacy names.
    """
    trajectory_path = Path(path)

    temporal_columns = (
        DEFAULT_SYNTHETIC_TEMPORAL_COLUMNS
        if temporal_columns is None
        else temporal_columns
    )

    legacy_columns = (
        LEGACY_SYNTHETIC_TEMPORAL_COLUMNS
        if legacy_columns is None
        else legacy_columns
    )

    header = pd.read_csv(
        trajectory_path,
        compression="infer",
        nrows=0,
    )

    available_columns = set(header.columns)

    requested_columns = [
        column
        for column in temporal_columns + legacy_columns
        if column in available_columns
    ]

    trajectory = pd.read_csv(
        trajectory_path,
        compression="infer",
        usecols=requested_columns,
    )

    return normalize_metric_columns(trajectory)


def load_downsampled_synthetic_temporal_logs(
    run_index: pd.DataFrame,
    *,
    run_dir: str | Path,
    plot_every: int,
    temporal_columns: list[str] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load synthetic trajectory logs listed in run_index.csv and downsample them
    for plotting.
    """
    if run_index.empty:
        return pd.DataFrame()

    temporal_columns = (
        DEFAULT_SYNTHETIC_TEMPORAL_COLUMNS
        if temporal_columns is None
        else temporal_columns
    )

    frames: list[pd.DataFrame] = []
    total = len(run_index)

    for number, (_, row) in enumerate(run_index.iterrows(), start=1):
        path = resolve_synthetic_trajectory_path(
            row,
            run_dir=run_dir,
        )

        if not path.exists():
            raise FileNotFoundError(f"Missing trajectory: {path}")

        trajectory = read_synthetic_temporal_file(
            path,
            temporal_columns=temporal_columns,
        )

        missing_columns = [
            column
            for column in temporal_columns
            if column not in trajectory.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing temporal columns in {path}: {missing_columns}"
            )

        final_t = int(trajectory["t"].max())

        keep_mask = (
            (trajectory["t"] % int(plot_every) == 0)
            | (trajectory["t"] == final_t)
        )

        trajectory = trajectory.loc[keep_mask].copy()

        trajectory["regime"] = str(row["regime"])
        trajectory["preprocessing"] = str(row["preprocessing"])
        trajectory["policy"] = str(row["policy"])
        trajectory["seed"] = int(row["seed"])

        frames.append(trajectory)

        if verbose and (number % 100 == 0 or number == total):
            print(f"Loaded {number}/{total} trajectories")

    return pd.concat(frames, ignore_index=True)
