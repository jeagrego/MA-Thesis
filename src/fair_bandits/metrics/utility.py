from __future__ import annotations

import numpy as np

from .fairness import group_mean, max_group_gap


def utility_gap(reward: np.ndarray, group: np.ndarray) -> float:
    """
    Bandit-native utility disparity:
    max_g mean_reward(g) - min_g mean_reward(g)

    reward = 1[action == y_true], so this is effectively a per-group accuracy gap.
    """
    return max_group_gap(group_mean(reward, group))


def cumulative_regret(reward: np.ndarray, oracle_reward: float | np.ndarray = 1.0,) -> float:
    """
    Cumulative regret relative to an oracle reward stream.
    """
    r = np.asarray(reward, dtype=float)

    if np.isscalar(oracle_reward):
        oracle = np.full(r.shape[0], float(oracle_reward), dtype=float)
    else:
        oracle = np.asarray(oracle_reward, dtype=float)
        if oracle.shape[0] != r.shape[0]:
            raise ValueError("oracle_reward must be either a scalar or an array with the same length as reward.")

    return float(np.sum(oracle - r))
