from __future__ import annotations

import numpy as np

from .fairness import group_mean, max_group_gap


def utility_gap(reward: np.ndarray, group: np.ndarray) -> float:
    """
    Bandit-native utility disparity:
    max_g mean_reward(g) - min_g mean_reward(g)

    In classification-based bandits, reward = 1[action == y_true], so this is
    effectively a per-group accuracy gap.
    """
    return max_group_gap(group_mean(reward, group))


def cumulative_prediction_error(
    reward: np.ndarray,
    oracle_reward: float | np.ndarray = 1.0,
) -> float:
    """
    Cumulative prediction error relative to an oracle reward stream.

    This is the preferred thesis terminology for classification-based bandit
    experiments where reward = 1[action == label].
    """
    reward_arr = np.asarray(reward, dtype=float)

    if np.isscalar(oracle_reward):
        oracle = np.full(reward_arr.shape[0], float(oracle_reward), dtype=float)
    else:
        oracle = np.asarray(oracle_reward, dtype=float)

        if oracle.shape[0] != reward_arr.shape[0]:
            raise ValueError(
                "oracle_reward must be either a scalar or an array with the "
                "same length as reward."
            )

    return float(np.sum(oracle - reward_arr))


def cumulative_regret(
    reward: np.ndarray,
    oracle_reward: float | np.ndarray = 1.0,
) -> float:
    """
    Backward-compatible alias.
    """
    return cumulative_prediction_error(
        reward=reward,
        oracle_reward=oracle_reward,
    )