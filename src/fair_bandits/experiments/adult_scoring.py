from __future__ import annotations

import numpy as np
import pandas as pd

from ..policies import AdultEXP4Policy, AdultFairEXP4Policy


def adult_lints_margin(policy, x: np.ndarray, group: str | None = None) -> float:
    """
    Margin score for LinTS/FairLinTS post-processing.
    """
    scores = np.asarray(policy.action_scores(x), dtype=float)

    if group is None:
        return float(scores[1] - scores[0])

    group = str(group)
    policy._ensure_group_known(group)

    return float(
        (scores[1] - policy._fair_penalty(group, 1))
        - (scores[0] - policy._fair_penalty(group, 0))
    )

def adult_linucb_margin(
    policy,
    x: np.ndarray,
    group: str | None = None,
    *,
    fair: bool = True,
) -> float:
    """
    Margin score for LinUCB/FairLinUCB post-processing.

    The score is action-1 score minus action-0 score. For the fair policy,
    group-aware penalties are included through the policy's score method.
    """
    if fair and group is not None:
        group = str(group)
        policy._ensure_group_known(group)

        score_1 = policy.score(x, 1, group=group)
        score_0 = policy.score(x, 0, group=group)

    else:
        score_1 = policy.score(x, 1)
        score_0 = policy.score(x, 0)

    return float(score_1 - score_0)


def adult_linucb_score_table(
    policy,
    X: np.ndarray,
    y,
    groups,
    *,
    fair: bool = True,
) -> pd.DataFrame:
    """
    Build a held-out score table for LinUCB/FairLinUCB.
    """
    return pd.DataFrame(
        {
            "group": np.asarray(groups).astype(str),
            "y_true": np.asarray(y, dtype=int),
            "score": [
                adult_linucb_margin(
                    policy,
                    x,
                    str(group),
                    fair=fair,
                )
                for x, group in zip(X, groups)
            ],
        }
    )


def adult_lints_score_table(
    policy,
    X: np.ndarray,
    y,
    groups,
    *,
    fair: bool = True,
) -> pd.DataFrame:
    """
    Build a held-out score table for LinTS/FairLinTS.
    """
    return pd.DataFrame(
        {
            "group": np.asarray(groups).astype(str),
            "y_true": np.asarray(y, dtype=int),
            "score": [
                adult_lints_margin(
                    policy,
                    x,
                    str(group) if fair else None,
                )
                for x, group in zip(X, groups)
            ],
        }
    )


def adult_exp4_margin(
    policy,
    advice,
    group: str | None = None,
    *,
    fair: bool = True,
) -> float:
    """
    Margin score for EXP4/FairEXP4 post-processing.
    """
    if fair and isinstance(policy, AdultFairEXP4Policy) and group is not None:
        probabilities = policy.action_probabilities(advice, group=str(group))
    else:
        probabilities = AdultEXP4Policy.action_probabilities(policy, advice, group=None)

    return float(probabilities[1] - probabilities[0])


def adult_exp4_score_table(
    policy,
    advice_array: np.ndarray,
    y,
    groups,
    *,
    fair: bool = True,
) -> pd.DataFrame:
    """
    Build a held-out score table for EXP4/FairEXP4.
    """
    return pd.DataFrame(
        {
            "group": np.asarray(groups).astype(str),
            "y_true": np.asarray(y, dtype=int),
            "score": [
                adult_exp4_margin(
                    policy,
                    advice,
                    group=str(group),
                    fair=fair,
                )
                for advice, group in zip(advice_array, groups)
            ],
        }
    )
