from __future__ import annotations

import numpy as np


class EXP4:
    """
    EXP4 for adversarial contextual bandits with expert advice.

    At each round, the policy receives expert advice as a matrix:
        expert_advice.shape = (n_experts, n_actions)

    Each row is a probability distribution over actions.
    EXP4 combines experts using exponential weights.
    """

    def __init__(
        self,
        n_experts: int,
        n_actions: int = 2,
        gamma: float = 0.07,
        eta: float | None = None,
        seed: int | None = None,
    ) -> None:
        if n_experts <= 0:
            raise ValueError("n_experts must be positive.")
        if n_actions <= 1:
            raise ValueError("n_actions must be at least 2.")
        if not (0.0 < gamma <= 1.0):
            raise ValueError("gamma must be in (0, 1].")

        self.n_experts = int(n_experts)
        self.n_actions = int(n_actions)
        self.gamma = float(gamma)
        self.eta = eta if eta is not None else self.gamma / self.n_actions
        self.rng = np.random.default_rng(seed)

        self.weights = np.ones(self.n_experts, dtype=float)
        self.last_action_probs: np.ndarray | None = None
        self.last_expert_advice: np.ndarray | None = None

    def _normalize_advice(self, expert_advice: np.ndarray) -> np.ndarray:
        """
        Normalize expert advice to ensure valid probability distributions."""
        advice = np.asarray(expert_advice, dtype=float)

        if advice.shape != (self.n_experts, self.n_actions):
            raise ValueError(
                f"expert_advice must have shape "
                f"({self.n_experts}, {self.n_actions}), got {advice.shape}."
            )

        advice = np.clip(advice, 0.0, None)
        row_sums = advice.sum(axis=1, keepdims=True)
        bad_rows = row_sums.squeeze() <= 0.0

        if np.any(bad_rows):
            advice[bad_rows, :] = 1.0 / self.n_actions
            row_sums = advice.sum(axis=1, keepdims=True)

        return advice / row_sums

    def action_probabilities(self, expert_advice: np.ndarray) -> np.ndarray:
        """
        Compute the EXP4 action probabilities given expert advice.
        """
        advice = self._normalize_advice(expert_advice)

        q = self.weights / self.weights.sum()
        p = q @ advice

        # Explicit exploration
        p = (1.0 - self.gamma) * p + self.gamma / self.n_actions
        p = np.clip(p, 1e-12, 1.0)
        p = p / p.sum()

        return p

    def select_from_advice(
        self,
        expert_advice: np.ndarray,
        group: str | None = None,
    ) -> int:
        """
        Select an action based on the given expert advice.
        """
        advice = self._normalize_advice(expert_advice)
        p = self.action_probabilities(advice)

        action = int(self.rng.choice(self.n_actions, p=p))

        self.last_action_probs = p
        self.last_expert_advice = advice
        return action

    def update_from_advice(
        self,
        action: int,
        reward: float,
        expert_advice: np.ndarray | None = None,
        action_probs: np.ndarray | None = None,
        group: str | None = None,
    ) -> None:
        """
        Update the policy based on the observed reward and expert advice.
        """
        if expert_advice is None:
            if self.last_expert_advice is None:
                raise ValueError("No expert advice available for update.")
            advice = self.last_expert_advice
        else:
            advice = self._normalize_advice(expert_advice)

        if action_probs is None:
            if self.last_action_probs is None:
                raise ValueError("No action probabilities available for update.")
            p = self.last_action_probs
        else:
            p = np.asarray(action_probs, dtype=float)

        a = int(action)
        r = float(reward)

        estimated_action_rewards = np.zeros(self.n_actions, dtype=float)
        estimated_action_rewards[a] = r / max(float(p[a]), 1e-12)

        estimated_expert_rewards = advice @ estimated_action_rewards

        self.weights *= np.exp(self.eta * estimated_expert_rewards)
        self.weights = np.clip(self.weights, 1e-12, 1e12)


class GroupAwareDPEXP4(EXP4):
    """
    EXP4 with a group-aware demographic-parity correction.

    The EXP4 distribution is first computed from expert advice.
    Then a soft DP correction modifies the probability of the positive action.

    If a group is over-selected for action 1:
        reduce P(action=1)
    If a group is under-selected for action 1:
        increase P(action=1)

    This is an empirical DP-regularized variant, not a theoretical EXP4 fairness
    guarantee.
    """

    def __init__(
        self,
        n_experts: int,
        groups: np.ndarray,
        n_actions: int = 2,
        gamma: float = 0.07,
        eta: float | None = None,
        tau: float = 0.02,
        lambda_fair: float = 0.20,
        beta_smooth: float = 1.0,
        min_group_count: int = 20,
        positive_action: int = 1,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            n_experts=n_experts,
            n_actions=n_actions,
            gamma=gamma,
            eta=eta,
            seed=seed,
        )

        self.uniq_groups = np.unique(np.asarray(groups).astype(str))
        self.tau = float(tau)
        self.lambda_fair = float(lambda_fair)
        self.beta_smooth = float(beta_smooth)
        self.min_group_count = int(min_group_count)
        self.positive_action = int(positive_action)

        self.count_g = {g: 0 for g in self.uniq_groups}
        self.count_g1 = {g: 0 for g in self.uniq_groups}

    def _ensure_group_known(self, group: str) -> None:
        """
        Ensure the group is known and initialized in the counts.
        """
        g = str(group)
        if g not in self.count_g:
            self.count_g[g] = 0
            self.count_g1[g] = 0
            self.uniq_groups = np.array(sorted(self.count_g.keys()), dtype=str)

    def _smoothed_rate(self, group: str) -> float:
        """
        Compute the smoothed selection rate for the given group.
        """
        g = str(group)
        b = self.beta_smooth
        return float((self.count_g1[g] + b) / (self.count_g[g] + 2.0 * b))

    def _target_rate(self) -> float:
        """
        Compute the target selection rate across all groups.
        """
        rates = [self._smoothed_rate(g) for g in self.uniq_groups]
        return float(np.mean(rates)) if rates else 0.5

    def _apply_dp_probability_correction(
        self,
        p: np.ndarray,
        group: str,
    ) -> np.ndarray:
        """
        Apply demographic-parity probability correction to the given action probabilities.
        """
        g = str(group)
        self._ensure_group_known(g)

        corrected = np.asarray(p, dtype=float).copy()

        if self.count_g[g] < self.min_group_count:
            return corrected / corrected.sum()

        p_g = self._smoothed_rate(g)
        p_target = self._target_rate()

        if p_g > p_target + self.tau:
            # Over-selected group: decrease probability of positive action.
            shift = self.lambda_fair * (p_g - p_target - self.tau)
            corrected[self.positive_action] -= shift

        elif p_g < p_target - self.tau:
            # Under-selected group: increase probability of positive action.
            shift = self.lambda_fair * (p_target - p_g - self.tau)
            corrected[self.positive_action] += shift

        corrected = np.clip(corrected, 1e-8, 1.0)
        corrected = corrected / corrected.sum()
        return corrected

    def action_probabilities(
        self,
        expert_advice: np.ndarray,
        group: str | None = None,
    ) -> np.ndarray:
        """
        Compute the action probabilities with an optional DP correction based on group.
        """
        base_p = super().action_probabilities(expert_advice)

        if group is None:
            return base_p

        return self._apply_dp_probability_correction(base_p, str(group))

    def select_from_advice(
        self,
        expert_advice: np.ndarray,
        group: str | None = None,
    ) -> int:
        """
        Select an action based on the given expert advice and group information.
        """
        if group is None:
            raise ValueError("GroupAwareDPEXP4 requires group information.")

        advice = self._normalize_advice(expert_advice)
        p = self.action_probabilities(advice, group=str(group))

        action = int(self.rng.choice(self.n_actions, p=p))

        self.last_action_probs = p
        self.last_expert_advice = advice
        return action

    def update_from_advice(
        self,
        action: int,
        reward: float,
        expert_advice: np.ndarray | None = None,
        action_probs: np.ndarray | None = None,
        group: str | None = None,
    ) -> None:
        """
        Update the policy based on the observed reward, expert advice, and group information.
        """
        if group is None:
            raise ValueError("GroupAwareDPEXP4.update_from_advice requires group information.")

        super().update_from_advice(
            action=action,
            reward=reward,
            expert_advice=expert_advice,
            action_probs=action_probs,
            group=group,
        )

        g = str(group)
        self._ensure_group_known(g)
        self.count_g[g] += 1

        if int(action) == self.positive_action:
            self.count_g1[g] += 1

FairEXP4_DP = GroupAwareDPEXP4