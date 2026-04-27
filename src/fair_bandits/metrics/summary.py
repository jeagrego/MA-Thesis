from __future__ import annotations

import numpy as np

from .fairness import demographic_parity_gap, equalized_odds_gap, fpr_gap, ppv_gap, tpr_gap
from .utility import cumulative_regret, utility_gap


def summarize_metrics(
    y_true: np.ndarray,
    y_hat: np.ndarray,
    group: np.ndarray,
    reward: np.ndarray | None = None,
    oracle_reward: float | np.ndarray = 1.0,
) -> dict[str, float]:
    """
    Summarizes a set of metrics for a single run of a bandit algorithm.
    """
    yt = np.asarray(y_true, dtype=int)
    yh = np.asarray(y_hat, dtype=int)
    grp = np.asarray(group).astype(str)

    reward_arr = (
        np.asarray(reward, dtype=float)
        if reward is not None
        else (yt == yh).astype(float)
    )

    return {
        "accuracy": float((yt == yh).mean()),
        "DP_gap": demographic_parity_gap(yh, grp),
        "TPR_gap": tpr_gap(yt, yh, grp),
        "FPR_gap": fpr_gap(yt, yh, grp),
        "EO_gap": equalized_odds_gap(yt, yh, grp),
        "PPV_gap": ppv_gap(yt, yh, grp),
        "UtilityGap": utility_gap(reward_arr, grp),
        "CumulativeRegret": cumulative_regret(reward_arr, oracle_reward=oracle_reward),
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
