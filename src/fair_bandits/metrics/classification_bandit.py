from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def safe_gap(values: Iterable[float]) -> float:
    """
    Compute the gap between the maximum and minimum values in a list, handling infinite values.
    If the list has fewer than 2 finite values, return NaN.
    """
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]

    return float(array.max() - array.min()) if len(array) >= 2 else np.nan


def summarize_classification_bandit(
    y_true,
    actions,
    groups,
    *,
    positive_class: int = 1,
) -> dict[str, float]:
    """
    Summarize the performance of a classification bandit across different groups.
    """
    y_true = np.asarray(y_true, dtype=int)
    actions = np.asarray(actions, dtype=int)
    groups = np.asarray(groups).astype(str)

    rewards = (actions == y_true).astype(float)

    positive_rates = []
    tpr_values = []
    fpr_values = []
    utility_values = []

    for group in sorted(np.unique(groups)):
        mask = groups == group

        y_group = y_true[mask]
        actions_group = actions[mask]
        rewards_group = rewards[mask]

        positive_rates.append(
            float(np.mean(actions_group == positive_class))
        )

        utility_values.append(
            float(rewards_group.mean())
        )

        positives = y_group == positive_class
        negatives = y_group != positive_class

        if np.any(positives):
            tpr_values.append(
                float(np.mean(actions_group[positives] == positive_class))
            )

        if np.any(negatives):
            fpr_values.append(
                float(np.mean(actions_group[negatives] == positive_class))
            )

    tpr_gap = safe_gap(tpr_values)
    fpr_gap = safe_gap(fpr_values)

    eo_candidates = [
        value
        for value in [tpr_gap, fpr_gap]
        if np.isfinite(value)
    ]

    eo_gap = float(max(eo_candidates)) if eo_candidates else np.nan
    cumulative_prediction_error = float(len(rewards) - rewards.sum())

    return {
        "accuracy": float(rewards.mean()),
        "average_reward": float(rewards.mean()),
        "cumulative_prediction_error": cumulative_prediction_error,

        # Temporary backward-compatible alias.
        # Use cumulative_prediction_error in tables/figures.
        "cumulative_regret": cumulative_prediction_error,

        "DP_gap": safe_gap(positive_rates),
        "TPR_gap": tpr_gap,
        "FPR_gap": fpr_gap,
        "EO_gap": eo_gap,
        "UtilityGap": safe_gap(utility_values),
    }