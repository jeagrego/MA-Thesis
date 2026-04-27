from __future__ import annotations

import numpy as np

from .base import BaseContextualPolicy


class LinearThompsonSampling(BaseContextualPolicy):
    """
    Linear Thompson Sampling for a finite action set, defaulting to binary actions {0, 1}.

    The posterior approximation is a Gaussian centered on the ridge estimate with
    covariance proportional to A^{-1}.
    """

    def __init__(
        self,
        d: int,
        v: float = 0.5,
        lam: float = 1.0,
        n_actions: int = 2,
        seed: int | None = None,
    ) -> None:
        if d <= 0:
            raise ValueError("d must be positive.")
        if n_actions <= 1:
            raise ValueError("n_actions must be at least 2.")
        if lam <= 0:
            raise ValueError("lam must be strictly positive.")
        if v <= 0:
            raise ValueError("v must be strictly positive.")

        self.d = int(d)
        self.v = float(v)
        self.lam = float(lam)
        self.n_actions = int(n_actions)
        self.rng = np.random.default_rng(seed)

        self.A_inv = [
            np.eye(self.d, dtype=np.float64) / self.lam
            for _ in range(self.n_actions)
        ]
        self.b = [
            np.zeros(self.d, dtype=np.float64)
            for _ in range(self.n_actions)
        ]

    def _theta_hat(self, action: int) -> np.ndarray:
        """
        Compute the ridge regression estimate for the given action.
        """
        return self.A_inv[action] @ self.b[action]

    def action_scores(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the expected reward scores for each action given context x.
        """
        x_vec = np.asarray(x, dtype=np.float64).reshape(-1)
        scores = np.zeros(self.n_actions, dtype=np.float64)
        for a in range(self.n_actions):
            scores[a] = float(self._theta_hat(a) @ x_vec)
        return scores

    def _sample_theta(self, action: int) -> np.ndarray:
        """
        Sample a parameter vector from the approximate posterior for the given action.
        """
        mean = self._theta_hat(action)
        cov = (self.v ** 2) * 0.5 * (self.A_inv[action] + self.A_inv[action].T)
        cov += 1e-10 * np.eye(self.d, dtype=np.float64)
        return self.rng.multivariate_normal(mean=mean, cov=cov)

    def score(self, x: np.ndarray, action: int) -> float:
        """
        Compute the Thompson-sampled score for the given context and action.
        """
        x_vec = np.asarray(x, dtype=np.float64).reshape(-1)
        theta_tilde = self._sample_theta(action)
        return float(theta_tilde @ x_vec)

    def select(self, x: np.ndarray, group: str | None = None) -> int:
        """
        Select an action for the given context by sampling from the posterior and choosing the best action.
        """
        scores = np.array([self.score(x, a) for a in range(self.n_actions)], dtype=np.float64)
        return int(np.argmax(scores))

    def update(
        self,
        x: np.ndarray,
        action: int,
        reward: float,
        group: str | None = None,
    ) -> None:
        """
        Update the model parameters based on the observed context, action, and reward.
        """
        x_vec = np.asarray(x, dtype=np.float64).reshape(-1)
        a = int(action)
        r = float(reward)

        if not (0 <= a < self.n_actions):
            raise ValueError(f"Invalid action index {a} for n_actions={self.n_actions}.")

        A_inv = self.A_inv[a]
        Ax = A_inv @ x_vec
        denom = 1.0 + float(x_vec @ Ax)

        self.A_inv[a] = A_inv - np.outer(Ax, Ax) / denom
        self.b[a] += r * x_vec


class GroupAwareDPLinearThompsonSampling(LinearThompsonSampling):
    """
    Linear Thompson Sampling with a group-aware soft demographic-parity penalty.

    The penalty mirrors the DP-aware LinUCB variant already used in the project:
    - action 1 is penalized when the current group is over-selected
    - action 0 is penalized when the current group is under-selected
    """
    def __init__(
        self,
        d: int,
        groups: np.ndarray,
        v: float = 0.5,
        lam: float = 1.0,
        tau: float = 0.02,
        lambda_fair: float = 2.0,
        beta_smooth: float = 1.0,
        min_group_count: int = 20,
        n_actions: int = 2,
        seed: int | None = None,
    ) -> None:
        super().__init__(d=d, v=v, lam=lam, n_actions=n_actions, seed=seed)

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
        Ensure the group is known and initialized in the counts.
        """
        g = str(group)
        if g not in self.count_g:
            self.count_g[g] = 0
            self.count_g1[g] = 0
            self.uniq_groups = np.array(sorted(self.count_g.keys()), dtype=str)

    def _smoothed_rate(self, group: str) -> float:
        """
        Compute the smoothed selection rate for action 1 in the given group.
        """
        g = str(group)
        n = self.count_g[g]
        n1 = self.count_g1[g]
        b = self.beta_smooth
        return float((n1 + b) / (n + 2.0 * b))

    def _target_rate(self) -> float:
        """
        Compute the target selection rate for action 1 across all groups.
        """
        rates = [self._smoothed_rate(g) for g in self.uniq_groups]
        return float(np.mean(rates)) if rates else 0.5

    def _fair_penalty(self, group: str, action: int) -> float:
        """
        Compute the fairness penalty for the given group and action.
        """
        g = str(group)
        if self.count_g[g] < self.min_group_count:
            return 0.0

        p_g = self._smoothed_rate(g)
        p_target = self._target_rate()

        if action == 1 and p_g > p_target + self.tau:
            return self.lambda_fair * (p_g - p_target - self.tau)
        if action == 0 and p_g < p_target - self.tau:
            return self.lambda_fair * (p_target - p_g - self.tau)

        return 0.0

    def score(self, x: np.ndarray, action: int, group: str | None = None) -> float:
        """
        Compute the Thompson-sampled score for the given context, action, and group,
        applying the fairness penalty if applicable.
        """
        base_score = super().score(x, action)
        if group is None:
            return base_score
        self._ensure_group_known(str(group))
        penalty = self._fair_penalty(str(group), action)
        return base_score - penalty

    def select(self, x: np.ndarray, group: str | None = None) -> int:
        """
        Select an action for the given context and group by sampling from the posterior
        and choosing the best action with fairness penalty.
        """
        if group is None:
            raise ValueError("GroupAwareDPLinearThompsonSampling.select requires a group value.")
        g = str(group)
        self._ensure_group_known(g)

        scores = np.array([self.score(x, a, group=g) for a in range(self.n_actions)], dtype=np.float64)
        return int(np.argmax(scores))

    def update(
        self,
        x: np.ndarray,
        action: int,
        reward: float,
        group: str | None = None,
    ) -> None:
        """
        Update the model parameters and group counts based on the observed context, action, reward, and group.
        """
        if group is None:
            raise ValueError("GroupAwareDPLinearThompsonSampling.update requires a group value.")

        g = str(group)
        self._ensure_group_known(g)

        super().update(x=x, action=action, reward=reward, group=group)

        self.count_g[g] += 1
        if int(action) == 1:
            self.count_g1[g] += 1

FairLinearThompsonSampling_DP_GroupAware = GroupAwareDPLinearThompsonSampling