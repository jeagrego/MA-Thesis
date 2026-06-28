from __future__ import annotations

import numpy as np

from .exp4 import EXP4, FairEXP4_DP


class EXP4Policy:
    """
    Thin adapter around EXP4.
    So that it can be used in the same way as other policies in the fair_bandits framework.
    """

    def __init__(
        self,
        *,
        n_experts: int,
        n_actions: int = 2,
        gamma: float = 0.07,
        eta: float | None = None,
        seed: int | None = None,
    ) -> None:
        self.n_experts = int(n_experts)
        self.n_actions = int(n_actions)
        self.gamma = float(gamma)
        self.eta = eta

        self.model = EXP4(
            n_experts=self.n_experts,
            n_actions=self.n_actions,
            gamma=self.gamma,
            eta=self.eta,
            seed=seed,
        )

    def action_probabilities(
        self,
        advice,
        group: str | None = None,
    ) -> np.ndarray:
        """
        Get the action probabilities for the given advice.
        The group parameter is ignored, as EXP4 does not use group information.
        """
        return self.model.action_probabilities(advice)

    def select(
        self,
        advice,
        group: str | None = None,
    ) -> int:
        """
        Select an action based on the given advice.
        The group parameter is ignored, as EXP4 does not use group information.
        """
        return int(
            self.model.select_from_advice(
                advice,
                group=group,
            )
        )

    def update(
        self,
        *,
        advice,
        action: int,
        reward: float,
        group: str | None = None,
    ) -> None:
        """
        Update the model with the given action, reward, and advice.
        The group parameter is ignored, as EXP4 does not use group information.
        """
        self.model.update_from_advice(
            action=action,
            reward=reward,
            expert_advice=advice,
            group=group,
        )


class FairEXP4Policy(EXP4Policy):
    """
    Thin adapter around FairEXP4_DP.
    So that it can be used in the same way as other policies in the fair_bandits framework.
    """
    def __init__(
        self,
        *,
        n_experts: int,
        groups,
        n_actions: int = 2,
        gamma: float = 0.07,
        eta: float | None = None,
        lambda_fair: float = 2.0,
        tau: float = 0.02,
        beta_smooth: float = 1.0,
        min_group_count: int = 20,
        seed: int | None = None,
    ) -> None:
        self.n_experts = int(n_experts)
        self.n_actions = int(n_actions)
        self.gamma = float(gamma)
        self.eta = eta

        self.model = FairEXP4_DP(
            n_experts=n_experts,
            groups=np.asarray(groups).astype(str),
            n_actions=n_actions,
            gamma=gamma,
            eta=eta,
            lambda_fair=lambda_fair,
            tau=tau,
            beta_smooth=beta_smooth,
            min_group_count=min_group_count,
            seed=seed,
        )

    def action_probabilities(
        self,
        advice,
        group: str | None = None,
    ) -> np.ndarray:
        """
        Get the action probabilities for the given advice and group.
        If group is None, the probabilities are computed without group information.
        """
        return self.model.action_probabilities(
            advice,
            group=group,
        )

    def select(
        self,
        advice,
        group: str | None = None,
    ) -> int:
        """
        Select an action based on the given advice and group.
        If group is None, the action is selected without group information.
        """
        return int(
            self.model.select_from_advice(
                advice,
                group=group,
            )
        )

    def update(
        self,
        *,
        advice,
        action: int,
        reward: float,
        group: str | None = None,
    ) -> None:
        """
        Update the model with the given action, reward, and advice.
        The group parameter is ignored, as EXP4 does not use group information.
        """
        self.model.update_from_advice(
            action=action,
            reward=reward,
            expert_advice=advice,
            group=group,
        )