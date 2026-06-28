from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd

from fair_bandits.metrics import summarize_classification_bandit


def policy_margin(
    policy,
    advice,
    group: str | None = None,
    *,
    fair: bool = True,
    positive_class: int = 1,
) -> float:
    """
    Return the EXP4 decision margin:

        P(positive_class) - P(other_class)

    This reproduces the old notebook behavior.
    """
    try:
        if fair and group is not None:
            probabilities = policy.action_probabilities(
                advice,
                group=str(group),
            )
        else:
            probabilities = policy.action_probabilities(
                advice,
            )
    except TypeError:
        probabilities = policy.action_probabilities(
            advice,
        )

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    ).ravel()

    negative_class = 1 - positive_class

    return float(
        probabilities[positive_class]
        - probabilities[negative_class]
    )


def score_table(
    policy,
    advice,
    y,
    groups,
    *,
    positive_class: int = 1,
    fair: bool | None = True,
) -> pd.DataFrame:
    """
    Build a calibration/test score table using the old notebook convention:

        score = P(action=1) - P(action=0)

    The table includes both old and new column aliases:
    - old: group, y_true
    - new: g, y
    """
    group_array = np.asarray(groups).astype(str)
    y_array = np.asarray(y, dtype=int)

    scores = [
        policy_margin(
            policy,
            one_advice,
            group=group,
            fair=bool(fair),
            positive_class=positive_class,
        )
        for one_advice, group in zip(
            np.asarray(advice),
            group_array,
        )
    ]

    return pd.DataFrame(
        {
            "group": group_array,
            "g": group_array,
            "y_true": y_array,
            "y": y_array,
            "score": np.asarray(scores, dtype=float),
        }
    )


def actions_from_thresholds(
    table: pd.DataFrame,
    thresholds: dict[str, float],
    *,
    default_threshold: float = 0.0,
) -> np.ndarray:
    """
    Predict action 1 if margin score >= group-specific threshold.
    The table includes both old and new column aliases:
    - old: group, y_true
    - new: g, y
    """
    group_col = "group" if "group" in table.columns else "g"

    return np.asarray(
        [
            int(
                float(row.score)
                >= float(
                    thresholds.get(
                        str(getattr(row, group_col)),
                        default_threshold,
                    )
                )
            )
            for row in table.itertuples()
        ],
        dtype=int,
    )


def optimize_thresholds(
    calibration_table: pd.DataFrame,
    *,
    max_accuracy_drop: float = 0.01,
    threshold_grid_size: int = 31,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """
    Optimize group-specific thresholds to minimize fairness gaps while
    maintaining accuracy within the specified drop from the raw accuracy.
    """
    group_col = "group" if "group" in calibration_table.columns else "g"
    y_col = "y_true" if "y_true" in calibration_table.columns else "y"

    groups = sorted(
        calibration_table[group_col].astype(str).unique()
    )

    if len(groups) != 2:
        raise ValueError(
            f"Expected two groups, observed {groups}"
        )

    raw_thresholds = {
        group: 0.0
        for group in groups
    }

    raw_actions = actions_from_thresholds(
        calibration_table,
        raw_thresholds,
        default_threshold=0.0,
    )

    raw_metrics = summarize_classification_bandit(
        calibration_table[y_col],
        raw_actions,
        calibration_table[group_col],
    )

    scores = calibration_table["score"].to_numpy(dtype=float)

    low = min(
        float(np.quantile(scores, 0.02)),
        0.0,
    )

    high = max(
        float(np.quantile(scores, 0.98)),
        0.0,
    )

    if np.isclose(low, high):
        low -= 1e-6
        high += 1e-6

    grid = np.unique(
        np.append(
            np.linspace(
                low,
                high,
                threshold_grid_size,
            ),
            0.0,
        )
    )

    minimum_accuracy = (
        raw_metrics["accuracy"]
        - max_accuracy_drop
    )

    best = None

    for threshold_0, threshold_1 in product(
        grid,
        repeat=2,
    ):
        thresholds = {
            groups[0]: float(threshold_0),
            groups[1]: float(threshold_1),
        }

        actions = actions_from_thresholds(
            calibration_table,
            thresholds,
            default_threshold=0.0,
        )

        metrics = summarize_classification_bandit(
            calibration_table[y_col],
            actions,
            calibration_table[group_col],
        )

        objective = (
            0 if metrics["accuracy"] >= minimum_accuracy else 1,
            metrics["DP_gap"],
            metrics["EO_gap"],
            -metrics["accuracy"],
        )

        if best is None or objective < best[0]:
            best = (
                objective,
                thresholds,
                metrics,
            )

    return (
        best[1],
        raw_metrics,
        best[2],
    )