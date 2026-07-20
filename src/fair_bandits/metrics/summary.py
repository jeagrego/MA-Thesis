from __future__ import annotations

import numpy as np

from .fairness import (
    demographic_parity_gap,
    equalized_odds_gap,
    fpr_gap,
    ppv_gap,
    tpr_gap,
)
from .utility import cumulative_prediction_error, utility_gap


def summarize_metrics(
    y_true: np.ndarray,
    y_hat: np.ndarray,
    group: np.ndarray,
    reward: np.ndarray | None = None,
    oracle_reward: float | np.ndarray = 1.0,
) -> dict[str, float]:
    """
    Summarize final utility and fairness metrics for one run.
    """
    yt = np.asarray(y_true, dtype=int)
    yh = np.asarray(y_hat, dtype=int)
    grp = np.asarray(group).astype(str)

    reward_arr = (
        np.asarray(reward, dtype=float)
        if reward is not None
        else (yt == yh).astype(float)
    )

    cpe = cumulative_prediction_error(
        reward=reward_arr,
        oracle_reward=oracle_reward,
    )

    average_reward = float(reward_arr.mean())

    return {
        "accuracy": float((yt == yh).mean()),
        "average_reward": average_reward,
        "avg_reward": average_reward,
        "DP_gap": demographic_parity_gap(yh, grp),
        "TPR_gap": tpr_gap(yt, yh, grp),
        "FPR_gap": fpr_gap(yt, yh, grp),
        "EO_gap": equalized_odds_gap(yt, yh, grp),
        "PPV_gap": ppv_gap(yt, yh, grp),
        "UtilityGap": utility_gap(reward_arr, grp),
        "cumulative_prediction_error": cpe,

        # Backward-compatible alias only.
        "cumulative_regret": cpe,
    }


def mean_ci(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float, float]:
    """
    Simple nonparametric CI from empirical quantiles across seeds.
    """
    v = np.asarray(values, dtype=float)
    return (
        float(v.mean()),
        float(np.quantile(v, alpha / 2)),
        float(np.quantile(v, 1 - alpha / 2)),
    )
