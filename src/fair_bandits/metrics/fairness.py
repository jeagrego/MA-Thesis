from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix


def safe_rate(num: int | float, den: int | float) -> float:
    """
    Compute a rate = num / den, returning 0.0 if den is zero.
    """
    return float(num / den) if den > 0 else 0.0


def group_mean(values: np.ndarray, group: np.ndarray) -> dict[Any, float]:
    """
    Compute the mean of values for each unique group.
    """
    vals = np.asarray(values, dtype=float)
    grp = np.asarray(group).astype(str)

    out: dict[Any, float] = {}
    for g in np.unique(grp):
        mask = grp == g
        out[g] = float(vals[mask].mean()) if mask.any() else 0.0
    return out


def max_group_gap(group_values: dict[Any, float]) -> float:
    """
    Compute the maximum gap between any two groups.
    """
    vals = [float(v) for v in group_values.values()]
    return (max(vals) - min(vals)) if vals else 0.0


def demographic_parity_gap(y_hat: np.ndarray, group: np.ndarray) -> float:
    """
    DP-gap = max_g,g' |P(y_hat = 1 | g) - P(y_hat = 1 | g')|
    """
    return max_group_gap(group_mean(y_hat, group))


def per_group_confusion_rates(
    y_true: np.ndarray,
    y_hat: np.ndarray,
    group: np.ndarray,
) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
    """
    Returns per-group TPR, FPR, PPV, and NPV.
    """
    grp = np.asarray(group).astype(str)
    yt = np.asarray(y_true, dtype=int)
    yh = np.asarray(y_hat, dtype=int)

    tpr: dict[str, float] = {}
    fpr: dict[str, float] = {}
    ppv: dict[str, float] = {}
    npv: dict[str, float] = {}

    for g in np.unique(grp):
        mask = grp == g
        yt_g = yt[mask]
        yh_g = yh[mask]

        if yt_g.size == 0:
            tpr[g] = 0.0
            fpr[g] = 0.0
            ppv[g] = 0.0
            npv[g] = 0.0
            continue

        tn, fp, fn, tp = confusion_matrix(yt_g, yh_g, labels=[0, 1]).ravel()

        tpr[g] = safe_rate(tp, tp + fn)
        fpr[g] = safe_rate(fp, fp + tn)
        ppv[g] = safe_rate(tp, tp + fp)
        npv[g] = safe_rate(tn, tn + fn)

    return tpr, fpr, ppv, npv


def tpr_gap(y_true: np.ndarray, y_hat: np.ndarray, group: np.ndarray) -> float:
    """
    TPR-gap = max_g,g' |TPR(g) - TPR(g')|
    """
    tpr, _, _, _ = per_group_confusion_rates(y_true, y_hat, group)
    return max_group_gap(tpr)


def fpr_gap(y_true: np.ndarray, y_hat: np.ndarray, group: np.ndarray) -> float:
    """
    FPR-gap = max_g,g' |FPR(g) - FPR(g')|
    """
    _, fpr, _, _ = per_group_confusion_rates(y_true, y_hat, group)
    return max_group_gap(fpr)


def equalized_odds_gap(y_true: np.ndarray, y_hat: np.ndarray, group: np.ndarray) -> float:
    """
    EO-gap = max(TPR-gap, FPR-gap)
    """
    return max(
        tpr_gap(y_true, y_hat, group),
        fpr_gap(y_true, y_hat, group),
    )


def ppv_gap(y_true: np.ndarray, y_hat: np.ndarray, group: np.ndarray) -> float:
    """
    PPV-gap = max_g,g' |PPV(g) - PPV(g')|
    """
    _, _, ppv, _ = per_group_confusion_rates(y_true, y_hat, group)
    return max_group_gap(ppv)
