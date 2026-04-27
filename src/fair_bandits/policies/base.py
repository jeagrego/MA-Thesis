from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseContextualPolicy(ABC):
    """
    Shared interface for contextual policies.

    The "group" argument is optional.
    """

    @abstractmethod
    def select(self, x: np.ndarray, group: str | None = None) -> int:
        raise NotImplementedError

    @abstractmethod
    def update(
        self,
        x: np.ndarray,
        action: int,
        reward: float,
        group: str | None = None,
    ) -> None:
        raise NotImplementedError
