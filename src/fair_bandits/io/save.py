from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..experiments.runner import ExperimentBundle


def save_dataframe(df: pd.DataFrame, path: str | Path) -> Path:
    """
    Save a DataFrame to CSV at the specified path, creating parent directories if needed.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path


def export_experiment_bundle(
    bundle: ExperimentBundle,
    out_dir: str | Path,
) -> Path:
    """
    Save a complete experiment bundle to disk.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    save_dataframe(bundle.logs, out / "logs.csv")
    save_dataframe(bundle.seed_summary, out / "seed_summary.csv")
    save_dataframe(bundle.temporal_summary, out / "temporal_summary.csv")

    with open(out / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(bundle.metadata, f, indent=2)

    return out
