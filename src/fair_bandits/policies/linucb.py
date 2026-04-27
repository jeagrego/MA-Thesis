from __future__ import annotations

import numpy as np

from .base import BaseContextualPolicy


class LinUCB(BaseContextualPolicy):
    """
    Linear UCB for a finite action set, defaulting to binary actions {0, 1}.

    This implementation uses the Sherman-Morrison update on A^-1 to avoid repeated matrix inversion.
    """

    def __init__(
        self,
        d: int,
        alpha: float = 1.0,
        lam: float = 1.0,
        n_actions: int = 2,
    ) -> None:
        if d <= 0:
            raise ValueError("d must be positive.")
        if n_actions <= 1:
            raise ValueError("n_actions must be at least 2.")
        if lam <= 0:
            raise ValueError("lambda must be strictly positive.")

        self.d = int(d)
        self.alpha = float(alpha)
        self.lam = float(lam)
        self.n_actions = int(n_actions)

        self.A_inv = [
            np.eye(self.d, dtype=np.float64) / self.lam
            for _ in range(self.n_actions)
        ]
        self.b = [
            np.zeros(self.d, dtype=np.float64)
            for _ in range(self.n_actions)
        ]

    def _theta(self, action: int) -> np.ndarray:
        """
        Returns the parameter estimate for the given action.
        """
        return self.A_inv[action] @ self.b[action]

    def score(self, x: np.ndarray, action: int) -> float:
        """
        Scores the given context and action.
        """
        x_vec = np.asarray(x, dtype=np.float64).reshape(-1)
        theta = self._theta(action)
        mean = float(theta @ x_vec)
        var = float(x_vec @ (self.A_inv[action] @ x_vec))
        bonus = self.alpha * np.sqrt(max(var, 0.0))
        return mean + bonus

    def select(self, x: np.ndarray, group: str | None = None) -> int:
        """
        Selects an action for the given context.
        """
        scores = np.array(
            [self.score(x, a) for a in range(self.n_actions)],
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
        Updates the policy with the given context, action, and reward.
        """
        x_vec = np.asarray(x, dtype=np.float64).reshape(-1)
        a = int(action)
        r = float(reward)

        if not (0 <= a < self.n_actions):
            raise ValueError(f"Invalid action index {a} for n_actions={self.n_actions}.")

        A_inv = self.A_inv[a]   # get current A^-1 for the action
        Ax = A_inv @ x_vec  # compute A^-1 x for the action's context vector
        denom = 1.0 + float(x_vec @ Ax) # compute denominator for Sherman-Morrison update

        self.A_inv[a] = A_inv - np.outer(Ax, Ax) / denom
        self.b[a] += r * x_vec
