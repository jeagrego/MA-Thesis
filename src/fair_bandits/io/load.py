from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..experiments.runner import ExperimentBundle


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
