from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression


@dataclass
class AdultExpertPool:
    """
    Pool of probabilistic binary classifiers used as EXP4 experts.
    """

    models: list
    epsilon: float = 1e-4

    def predict_advice(self, X: np.ndarray) -> np.ndarray:
        """
        Return EXP4 advice with shape (n_samples, n_experts, 2).
        """
        advice = []

        for model in self.models:
            probabilities = model.predict_proba(X)

            if probabilities.shape[1] == 1:
                p1 = np.repeat(
                    float(model.classes_[0] == 1),
                    X.shape[0],
                )
            else:
                class_index = list(model.classes_).index(1)
                p1 = probabilities[:, class_index]

            p1 = np.clip(p1, self.epsilon, 1.0 - self.epsilon)
            advice.append(np.column_stack([1.0 - p1, p1]))

        return np.stack(advice, axis=1)


def train_adult_expert_pool(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_experts: int = 20,
    bootstrap_size: int = 12000,
    seed: int = 2026,
    max_iter: int = 1000,
) -> AdultExpertPool:
    """
    Train a bootstrapped logistic-regression expert pool for Adult EXP4.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows.")

    models = []
    n = X.shape[0]
    sample_size = min(int(bootstrap_size), n)

    for expert_id in range(int(n_experts)):
        indices = rng.choice(
            n,
            size=sample_size,
            replace=True,
        )

        C = float(10 ** rng.uniform(-1.0, 1.0))

        model = LogisticRegression(
            C=C,
            solver="lbfgs",
            max_iter=max_iter,
            class_weight=None,
            random_state=int(seed) + expert_id,
        )

        model.fit(X[indices], y[indices])
        models.append(model)

    return AdultExpertPool(models=models)
