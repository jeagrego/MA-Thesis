from __future__ import annotations

import numpy as np


class AdultEXP4Policy:
    """
    EXP4 policy for binary Adult classification-bandit experiments.
    """

    def __init__(
        self,
        *,
        n_experts: int,
        n_actions: int = 2,
        gamma: float = 0.07,
        eta: float | None = None,
        horizon: int = 30000,
        seed: int | None = None,
    ) -> None:
        self.n_experts = int(n_experts)
        self.n_actions = int(n_actions)
        self.gamma = float(gamma)
        self.eta = (
            float(eta)
            if eta is not None
            else float(
                np.sqrt(
                    np.log(max(self.n_experts, 2))
                    / max(int(horizon) * self.n_actions, 1)
                )
            )
        )
        self.weights = np.ones(self.n_experts, dtype=float)
        self.rng = np.random.default_rng(seed)

    def _normalize_advice(self, advice) -> np.ndarray:
        """
        Normalize the advice from experts to ensure it is a valid probability distribution.
        """
        advice = np.asarray(advice, dtype=float)

        expected = (self.n_experts, self.n_actions)
        if advice.shape != expected:
            raise ValueError(f"Expected advice shape {expected}, got {advice.shape}.")

        advice = np.clip(advice, 1e-12, np.inf)
        advice = advice / advice.sum(axis=1, keepdims=True)
        return advice

    def action_probabilities(
        self,
        advice,
        group: str | None = None,
    ) -> np.ndarray:
        """
        Compute action probabilities for the given advice and group.
        """
        advice = self._normalize_advice(advice)
        normalized_weights = self.weights / self.weights.sum()
        mixed = normalized_weights @ advice
        probabilities = (1.0 - self.gamma) * mixed + self.gamma / self.n_actions
        probabilities = np.clip(probabilities, 1e-12, np.inf)
        probabilities = probabilities / probabilities.sum()
        return probabilities

    def select(
        self,
        advice,
        group: str | None = None,
    ) -> int:
        """
        Select an action based on the given advice and group.
        """
        probabilities = self.action_probabilities(advice, group=group)
        return int(self.rng.choice(self.n_actions, p=probabilities))

    def update(
        self,
        *,
        advice,
        action: int,
        reward: float,
        group: str | None = None,
    ) -> None:
        """
        Update the policy based on the observed reward and action.
        """
        advice = self._normalize_advice(advice)
        probabilities = self.action_probabilities(advice, group=group)
        action = int(action)
        selected_probability = max(float(probabilities[action]), 1e-12)

        expert_reward_estimates = advice[:, action] * float(reward) / selected_probability
        self.weights *= np.exp(self.eta * expert_reward_estimates)
        self.weights = np.clip(self.weights, 1e-300, 1e300)
        self.weights /= self.weights.mean()


class AdultFairEXP4Policy(AdultEXP4Policy):
    """
    EXP4 with a group-aware demographic-parity penalty on action probabilities.
    """

    def __init__(
        self,
        *,
        n_experts: int,
        groups,
        n_actions: int = 2,
        gamma: float = 0.07,
        eta: float | None = None,
        horizon: int = 30000,
        lambda_fair: float = 2.0,
        tau: float = 0.02,
        beta_smooth: float = 1.0,
        min_group_count: int = 20,
        positive_class: int = 1,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            n_experts=n_experts,
            n_actions=n_actions,
            gamma=gamma,
            eta=eta,
            horizon=horizon,
            seed=seed,
        )

        self.lambda_fair = float(lambda_fair)
        self.tau = float(tau)
        self.beta_smooth = float(beta_smooth)
        self.min_group_count = int(min_group_count)
        self.positive_class = int(positive_class)

        known_groups = sorted({str(group) for group in np.asarray(groups).astype(str)})
        self.count_g = {group: 0 for group in known_groups}
        self.pos_g = {group: 0 for group in known_groups}
        self.total_count = 0
        self.total_positive = 0

    def _ensure_group_known(self, group: str) -> None:
        """
        Ensure that the given group is known and initialized in the counts.
        """
        group = str(group)
        if group not in self.count_g:
            self.count_g[group] = 0
            self.pos_g[group] = 0

    def _rates(self, group: str) -> tuple[float, float]:
        """
        Compute the rates for the given group.
        """
        self._ensure_group_known(group)
        group_count = self.count_g[group]
        group_positive = self.pos_g[group]

        group_rate = (group_positive + self.beta_smooth) / (
            group_count + 2.0 * self.beta_smooth
        )

        global_rate = (self.total_positive + self.beta_smooth * len(self.count_g)) / (
            self.total_count + 2.0 * self.beta_smooth * len(self.count_g)
        )

        return float(group_rate), float(global_rate)

    def _fair_penalty(self, group: str, action: int) -> float:
        """
        Compute the fairness penalty for the given group and action.
        """
        self._ensure_group_known(group)

        if self.total_count < self.min_group_count or self.count_g[group] < self.min_group_count:
            return 0.0

        group_rate, global_rate = self._rates(group)
        disparity = group_rate - global_rate

        if int(action) == self.positive_class:
            return float(max(0.0, disparity - self.tau))

        return float(max(0.0, -disparity - self.tau))

    def action_probabilities(
        self,
        advice,
        group: str | None = None,
    ) -> np.ndarray:
        """
        Compute action probabilities for the given advice and group, applying a fairness penalty if a group is specified.
        """
        base_probabilities = super().action_probabilities(advice, group=None)

        if group is None:
            return base_probabilities

        penalties = np.asarray(
            [self._fair_penalty(str(group), action) for action in range(self.n_actions)],
            dtype=float,
        )

        adjusted = base_probabilities * np.exp(-self.lambda_fair * penalties)
        adjusted = np.clip(adjusted, 1e-12, np.inf)
        adjusted = adjusted / adjusted.sum()
        return adjusted

    def update(
        self,
        *,
        advice,
        action: int,
        reward: float,
        group: str | None = None,
    ) -> None:
        """
        Update the policy based on the observed reward and action.
        """
        super().update(
            advice=advice,
            action=action,
            reward=reward,
            group=group,
        )

        if group is not None:
            group = str(group)
            self._ensure_group_known(group)
            self.count_g[group] += 1
            self.total_count += 1

            if int(action) == self.positive_class:
                self.pos_g[group] += 1
                self.total_positive += 1
