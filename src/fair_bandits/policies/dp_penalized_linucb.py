from __future__ import annotations

import numpy as np

from .linucb import LinUCB


class GroupAwareDPLinUCB(LinUCB):
    """
    LinUCB with a group-aware soft demographic-parity penalty.

    Mechanism:
    - track the empirical positive-action rate per group
    - compare the current group's rate to the average target rate
    - penalize action 1 when the group is over-selected
    - penalize action 0 when the group is under-selected
    """

    def __init__(
        self,
        d: int,
        groups: np.ndarray,
        alpha: float = 1.0,
        lam: float = 1.0,
        tau: float = 0.02,
        lambda_fair: float = 2.0,
        beta_smooth: float = 1.0,
        min_group_count: int = 20,
        n_actions: int = 2,
    ) -> None:
        """
        Initializes the GroupAwareDPLinUCB policy.
        """
        super().__init__(d=d, alpha=alpha, lam=lam, n_actions=n_actions)

        group_array = np.asarray(groups).astype(str)
        self.uniq_groups = np.unique(group_array)

        self.tau = float(tau)
        self.lambda_fair = float(lambda_fair)
        self.beta_smooth = float(beta_smooth)
        self.min_group_count = int(min_group_count)

        self.count_g = {g: 0 for g in self.uniq_groups}
        self.count_g1 = {g: 0 for g in self.uniq_groups}

    def _ensure_group_known(self, group: str) -> None:
        """
        Ensures that the group is known and initialized in the counts.
        """
        g = str(group)
        if g not in self.count_g:
            self.count_g[g] = 0
            self.count_g1[g] = 0
            self.uniq_groups = np.array(sorted(self.count_g.keys()), dtype=str)

    def _smoothed_rate(self, group: str) -> float:
        """
        Returns the smoothed positive-action rate for the given group.
        """
        g = str(group)
        n = self.count_g[g]
        n1 = self.count_g1[g]
        b = self.beta_smooth
        return float((n1 + b) / (n + 2.0 * b))

    def _target_rate(self) -> float:
        """
        Returns the average target positive-action rate across all groups.
        """
        rates = [self._smoothed_rate(g) for g in self.uniq_groups]
        return float(np.mean(rates)) if rates else 0.5

    def _fair_penalty(self, group: str, action: int) -> float:
        """
        Returns the fairness penalty for the given group and action.
        """
        g = str(group)

        if self.count_g[g] < self.min_group_count:
            return 0.0

        p_g = self._smoothed_rate(g)
        p_target = self._target_rate()

        # Penalize action 1 when the group is over-selected
        if action == 1 and p_g > p_target + self.tau:
            return self.lambda_fair * (p_g - p_target - self.tau)

        # Penalize action 0 when the group is under-selected
        if action == 0 and p_g < p_target - self.tau:
            return self.lambda_fair * (p_target - p_g - self.tau)

        return 0.0

    def score(self, x: np.ndarray, action: int, group: str | None = None) -> float:
        """
        Scores the given context and action, applying the fairness penalty if a group is provided.
        """
        base_score = super().score(x, action)
        if group is None:
            return base_score

        self._ensure_group_known(str(group))
        penalty = self._fair_penalty(str(group), action)
        return base_score - penalty

    def select(self, x: np.ndarray, group: str | None = None) -> int:
        """
        Selects an action for the given context, applying the fairness penalty if a group is provided.
        """
        if group is None:
            raise ValueError("GroupAwareDPLinUCB.select requires a group value.")

        g = str(group)
        self._ensure_group_known(g)

        scores = np.array(
            [self.score(x, a, group=g) for a in range(self.n_actions)],
            dtype=np.float64,
        )
        return int(np.argmax(scores))

    def update(
        self,
        x: np.ndarray,
        action: int,
        reward: float,
        group: str | None = None,
    ) -> None:
        """
        Updates the policy with the given context, action, reward, and group.
        """
        if group is None:
            raise ValueError("GroupAwareDPLinUCB.update requires a group value.")

        g = str(group)
        self._ensure_group_known(g)

        super().update(x=x, action=action, reward=reward, group=group)

        self.count_g[g] += 1
        if int(action) == 1:
            self.count_g1[g] += 1

FairLinUCB_DP_GroupAware = GroupAwareDPLinUCB
