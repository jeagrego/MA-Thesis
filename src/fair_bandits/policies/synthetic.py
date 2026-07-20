from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dp_penalized_linucb import GroupAwareDPLinUCB
from .exp4 import EXP4, GroupAwareDPEXP4
from .linear_ts import (
    GroupAwareDPLinearThompsonSampling,
    LinearThompsonSampling,
)
from .linucb import LinUCB


@dataclass(frozen=True)
class SyntheticPolicyParams:
    """
    Hyperparameters used to instantiate regime-matched synthetic CMAB policies.
    """
    d: int = 10
    alpha_linucb: float = 1.5
    lambda_ridge: float = 1.0
    ts_v: float = 0.5
    exp4_gamma: float = 0.07
    dp_tau: float = 0.02
    dp_lambda_linear: float = 2.0
    dp_lambda_exp4: float = 0.20
    beta_smooth: float = 1.0
    min_group_count: int = 20
    n_experts: int = 6


def instantiate_synthetic_policy(
    *,
    policy_name: str,
    groups: np.ndarray,
    seed: int,
    params: SyntheticPolicyParams,
):
    """
    Build one regime-matched policy for the synthetic benchmark.
    """
    groups = np.asarray(groups).astype(str)

    if policy_name == "LinUCB":
        return LinUCB(
            d=params.d,
            alpha=params.alpha_linucb,
            lam=params.lambda_ridge,
            n_actions=2,
        )

    if policy_name == "FairLinUCB_DP":
        return GroupAwareDPLinUCB(
            d=params.d,
            groups=groups,
            alpha=params.alpha_linucb,
            lam=params.lambda_ridge,
            tau=params.dp_tau,
            lambda_fair=params.dp_lambda_linear,
            beta_smooth=params.beta_smooth,
            min_group_count=params.min_group_count,
            n_actions=2,
        )

    if policy_name == "LinTS":
        return LinearThompsonSampling(
            d=params.d,
            v=params.ts_v,
            lam=params.lambda_ridge,
            n_actions=2,
            seed=int(seed),
        )

    if policy_name == "FairLinTS_DP":
        return GroupAwareDPLinearThompsonSampling(
            d=params.d,
            groups=groups,
            v=params.ts_v,
            lam=params.lambda_ridge,
            tau=params.dp_tau,
            lambda_fair=params.dp_lambda_linear,
            beta_smooth=params.beta_smooth,
            min_group_count=params.min_group_count,
            n_actions=2,
            seed=int(seed),
        )

    if policy_name == "EXP4":
        return EXP4(
            n_experts=params.n_experts,
            n_actions=2,
            gamma=params.exp4_gamma,
            seed=int(seed),
        )

    if policy_name == "FairEXP4_DP":
        return GroupAwareDPEXP4(
            n_experts=params.n_experts,
            groups=groups,
            n_actions=2,
            gamma=params.exp4_gamma,
            tau=params.dp_tau,
            lambda_fair=params.dp_lambda_exp4,
            beta_smooth=params.beta_smooth,
            min_group_count=params.min_group_count,
            positive_action=1,
            seed=int(seed),
        )

    raise ValueError(f"Unknown synthetic policy: {policy_name}")


def weighted_linear_update(
    *,
    policy,
    x: np.ndarray,
    action: int,
    reward: float,
    weight: float,
    group: str | None,
) -> None:
    """
    Apply one weighted linear-policy update.

    Scaling both x and reward by sqrt(weight) preserves the weighted least-squares
    interpretation used by LinUCB and linear Thompson Sampling.
    """
    if weight <= 0:
        raise ValueError(f"weight must be positive, got {weight}")

    scale = float(np.sqrt(weight))
    x_weighted = scale * np.asarray(x, dtype=float)
    reward_weighted = scale * float(reward)

    if group is None:
        policy.update(
            x_weighted,
            int(action),
            reward_weighted,
        )
    else:
        policy.update(
            x=x_weighted,
            action=int(action),
            reward=reward_weighted,
            group=str(group),
        )


def weighted_exp4_update(
    *,
    policy,
    action: int,
    reward: float,
    expert_advice: np.ndarray,
    weight: float,
    group: str | None,
) -> None:
    """
    Apply one weighted EXP4 update.

    The reward is weighted, while fairness-aware group counters still represent
    real observed decisions.
    """
    if weight <= 0:
        raise ValueError(f"weight must be positive, got {weight}")

    reward_weighted = float(weight) * float(reward)

    if group is None:
        policy.update_from_advice(
            action=int(action),
            reward=reward_weighted,
            expert_advice=expert_advice,
        )
    else:
        policy.update_from_advice(
            action=int(action),
            reward=reward_weighted,
            expert_advice=expert_advice,
            group=str(group),
        )