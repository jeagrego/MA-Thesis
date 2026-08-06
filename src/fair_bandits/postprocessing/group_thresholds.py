from __future__ import annotations

from collections.abc import Callable
from itertools import product

import numpy as np
import pandas as pd


MetricFunction = Callable[[np.ndarray, np.ndarray, np.ndarray], dict[str, float]]


def actions_from_group_thresholds(
    table: pd.DataFrame,
    thresholds: dict[str, float],
) -> np.ndarray:
    """
    Convert group-specific scores into binary actions using thresholds.
    """
    required = {"group", "score"}
    missing = required - set(table.columns)
    if missing:
        raise KeyError(f"Missing columns in score table: {sorted(missing)}")

    return np.asarray(
        [
            int(float(row.score) >= float(thresholds.get(str(row.group), 0.0)))
            for row in table.itertuples()
        ],
        dtype=int,
    )


def optimize_group_thresholds(
    calibration_table: pd.DataFrame,
    *,
    metric_fn: MetricFunction,
    max_accuracy_drop: float = 0.01,
    threshold_grid_size: int = 31,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """
    Optimize group-specific thresholds for post-processing.

    Objective order:
    1. stay within the allowed accuracy drop if possible;
    2. minimize DP gap;
    3. minimize EO gap;
    4. maximize accuracy.
    """
    required = {"group", "y_true", "score"}
    missing = required - set(calibration_table.columns)
    if missing:
        raise KeyError(f"Missing columns in calibration table: {sorted(missing)}")

    groups = sorted(calibration_table["group"].astype(str).unique().tolist())

    if len(groups) != 2:
        raise ValueError(f"Expected exactly two groups, observed {groups}.")

    raw_thresholds = {group: 0.0 for group in groups}

    raw_actions = actions_from_group_thresholds(
        calibration_table,
        raw_thresholds,
    )

    raw_metrics = metric_fn(
        calibration_table["y_true"].to_numpy(dtype=int),
        raw_actions,
        calibration_table["group"].astype(str).to_numpy(),
    )

    scores = calibration_table["score"].to_numpy(dtype=float)
    low = min(float(np.quantile(scores, 0.02)), 0.0)
    high = max(float(np.quantile(scores, 0.98)), 0.0)

    if np.isclose(low, high):
        low -= 1e-6
        high += 1e-6

    grid = np.unique(np.append(np.linspace(low, high, int(threshold_grid_size)), 0.0))
    minimum_accuracy = float(raw_metrics["accuracy"]) - float(max_accuracy_drop)

    best = None

    for threshold_0, threshold_1 in product(grid, repeat=2):
        thresholds = {
            groups[0]: float(threshold_0),
            groups[1]: float(threshold_1),
        }

        actions = actions_from_group_thresholds(calibration_table, thresholds)

        metrics = metric_fn(
            calibration_table["y_true"].to_numpy(dtype=int),
            actions,
            calibration_table["group"].astype(str).to_numpy(),
        )

        objective = (
            0 if metrics["accuracy"] >= minimum_accuracy else 1,
            metrics["DP_gap"],
            metrics["EO_gap"],
            -metrics["accuracy"],
        )

        if best is None or objective < best[0]:
            best = (objective, thresholds, metrics)

    if best is None:
        raise RuntimeError("Threshold optimization failed.")

    return best[1], raw_metrics, best[2]
