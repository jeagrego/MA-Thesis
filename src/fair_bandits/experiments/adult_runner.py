from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..data import make_preprocessing_weights
from ..metrics import normalize_metric_columns, summarize_classification_bandit

from ..policies import (
    AdultEXP4Policy,
    AdultFairEXP4Policy,
    GroupAwareDPLinUCB,
    GroupAwareDPLinearThompsonSampling,
    LinearThompsonSampling,
    LinUCB,
)

from ..postprocessing import actions_from_group_thresholds, optimize_group_thresholds

from .adult_scoring import (
    adult_exp4_score_table,
    adult_lints_score_table,
    adult_linucb_score_table,
)


@dataclass(frozen=True)
class AdultBanditParams:
    """
    Common Adult CMAB experiment parameters.
    """
    d: int
    t_max: int
    log_every: int
    alpha_linucb: float = 1.5
    ts_v: float = 0.5
    lambda_ridge: float = 1.0
    exp4_gamma: float = 0.07
    exp4_eta: float | None = None
    n_experts: int = 20
    dp_tau: float = 0.02
    dp_lambda: float = 2.0
    beta_smooth: float = 1.0
    min_group_count: int = 20
    max_accuracy_drop: float = 0.01
    threshold_grid_size: int = 31


ADULT_FAMILY_LABELS = {
    "linucb": "LinUCB",
    "linear_ts": "Linear Thompson Sampling",
    "exp4": "EXP4",
}


ADULT_POLICIES_BY_FAMILY = {
    "linucb": ["LinUCB", "FairLinUCB"],
    "linear_ts": ["LinTS", "FairLinTS"],
    "exp4": ["EXP4", "FairEXP4"],
}


def read_csv_or_empty(path: str | Path) -> pd.DataFrame:
    """
    Read a CSV file if it exists, otherwise return an empty dataframe.
    """
    path = Path(path)
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _append_family(dataframe: pd.DataFrame, family: str) -> pd.DataFrame:
    """
    Append a family column to a dataframe if it is not already present.
    """
    if dataframe.empty:
        return dataframe
    out = dataframe.copy()
    if "family" not in out.columns:
        out.insert(0, "family", family)
    return out


def adult_policy_is_fair(policy_name: str) -> bool:
    """
    Return True if the Adult policy name corresponds to a fair policy.
    """
    return str(policy_name) in {"FairLinUCB", "FairLinTS", "FairEXP4"}


def adult_postprocessed_policy_name(family: str) -> str:
    """
    Return the name of the post-processed policy for a given family.
    """
    if family == "linucb":
        return "FairLinUCB+PP"
    if family == "linear_ts":
        return "FairLinTS+PP"
    if family == "exp4":
        return "FairEXP4+PP"
    raise ValueError(f"Unknown Adult family: {family}")


def adult_fair_policy_name(family: str) -> str:
    """
    Return the name of the fair policy for a given family.
    """
    if family == "linucb":
        return "FairLinUCB"
    if family == "linear_ts":
        return "FairLinTS"
    if family == "exp4":
        return "FairEXP4"
    raise ValueError(f"Unknown Adult family: {family}")


def make_adult_policy(
    *,
    family: str,
    policy_name: str,
    groups,
    params: AdultBanditParams,
    seed: int,
):
    """
    Instantiate one Adult policy.
    """
    groups = np.asarray(groups).astype(str)

    if family == "linucb":
        if policy_name == "LinUCB":
            return LinUCB(
                d=params.d,
                alpha=params.alpha_linucb,
                lam=params.lambda_ridge,
                n_actions=2,
            )

        if policy_name == "FairLinUCB":
            return GroupAwareDPLinUCB(
                d=params.d,
                groups=groups,
                alpha=params.alpha_linucb,
                lam=params.lambda_ridge,
                tau=params.dp_tau,
                lambda_fair=params.dp_lambda,
                beta_smooth=params.beta_smooth,
                min_group_count=params.min_group_count,
                n_actions=2,
            )

    if family == "linear_ts":
        if policy_name == "LinTS":
            return LinearThompsonSampling(
                d=params.d,
                v=params.ts_v,
                lam=params.lambda_ridge,
                n_actions=2,
                seed=int(seed),
            )

        if policy_name == "FairLinTS":
            return GroupAwareDPLinearThompsonSampling(
                d=params.d,
                groups=groups,
                v=params.ts_v,
                lam=params.lambda_ridge,
                tau=params.dp_tau,
                lambda_fair=params.dp_lambda,
                beta_smooth=params.beta_smooth,
                min_group_count=params.min_group_count,
                n_actions=2,
                seed=int(seed),
            )

    if family == "exp4":
        if policy_name == "EXP4":
            return AdultEXP4Policy(
                n_experts=params.n_experts,
                n_actions=2,
                gamma=params.exp4_gamma,
                eta=params.exp4_eta,
                horizon=params.t_max,
                seed=int(seed),
            )

        if policy_name == "FairEXP4":
            return AdultFairEXP4Policy(
                n_experts=params.n_experts,
                groups=groups,
                n_actions=2,
                gamma=params.exp4_gamma,
                eta=params.exp4_eta,
                horizon=params.t_max,
                lambda_fair=params.dp_lambda,
                tau=params.dp_tau,
                beta_smooth=params.beta_smooth,
                min_group_count=params.min_group_count,
                seed=int(seed),
            )

    raise ValueError(f"Unknown family/policy combination: {family} / {policy_name}")


def weighted_linear_update(
    policy,
    *,
    x: np.ndarray,
    action: int,
    reward: float,
    weight: float,
    group: str | None = None,
) -> None:
    """
    Update a linear bandit policy with a weighted observation.
    """
    scale = float(np.sqrt(weight))
    x_weighted = scale * np.asarray(x, dtype=float)
    reward_weighted = scale * float(reward)

    if group is None:
        policy.update(x_weighted, int(action), reward_weighted)
    else:
        policy.update(
            x=x_weighted,
            action=int(action),
            reward=reward_weighted,
            group=str(group),
        )


def make_linear_ts_stream_indices(
    *,
    n_train: int,
    t_max: int,
    seed: int,
) -> np.ndarray:
    """
    LinTS uses a uniform stream; reweighting affects update weights.
    """
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(int(n_train))

    if int(t_max) > len(order):
        extra = rng.choice(
            order,
            size=int(t_max) - len(order),
            replace=True,
        )
        order = np.concatenate([order, extra])
    else:
        order = order[: int(t_max)]

    return order


def make_exp4_stream_indices(
    *,
    y_train: np.ndarray,
    g_train: np.ndarray,
    preprocessing: str,
    t_max: int,
    seed: int,
) -> np.ndarray:
    """
    Make a stream of indices for EXP4/FairEXP4, optionally reweighted by preprocessing.
    """
    rng = np.random.default_rng(int(seed))

    if preprocessing == "uniform":
        order = rng.permutation(len(y_train))
        if int(t_max) > len(order):
            extra = rng.choice(
                order,
                size=int(t_max) - len(order),
                replace=True,
            )
            order = np.concatenate([order, extra])
        else:
            order = order[: int(t_max)]
        return order

    weights, _ = make_preprocessing_weights(
        preprocessing=preprocessing,
        groups=g_train,
        oracle_actions=y_train,
    )

    probabilities = weights / weights.sum()

    return rng.choice(
        len(y_train),
        size=int(t_max),
        replace=True,
        p=probabilities,
    )


def run_adult_linear_ts_trajectory(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    g_train: np.ndarray,
    seed: int,
    policy_name: str,
    preprocessing: str,
    params: AdultBanditParams,
) -> tuple[pd.DataFrame, Any]:
    """
    Run one Adult LinTS/FairLinTS trajectory.
    """
    order = make_linear_ts_stream_indices(
        n_train=len(y_train),
        t_max=params.t_max,
        seed=seed,
    )

    X_stream = X_train[order]
    y_stream = y_train[order]
    g_stream = g_train[order]

    weights, _ = make_preprocessing_weights(
        preprocessing=preprocessing,
        groups=g_stream,
        oracle_actions=y_stream,
    )

    policy = make_adult_policy(
        family="linear_ts",
        policy_name=policy_name,
        groups=g_stream,
        params=params,
        seed=seed,
    )

    rows = []
    seen_y = []
    seen_actions = []
    seen_groups = []

    for index in range(int(params.t_max)):
        x = X_stream[index]
        y = int(y_stream[index])
        group = str(g_stream[index])
        weight = float(weights[index])

        if policy_name == "FairLinTS":
            action = int(policy.select(x, group=group))
        else:
            action = int(policy.select(x))

        reward = float(action == y)

        weighted_linear_update(
            policy,
            x=x,
            action=action,
            reward=reward,
            weight=weight,
            group=group if policy_name == "FairLinTS" else None,
        )

        seen_y.append(y)
        seen_actions.append(action)
        seen_groups.append(group)

        t = index + 1
        if t == 1 or t % int(params.log_every) == 0 or t == int(params.t_max):
            metrics = summarize_classification_bandit(seen_y, seen_actions, seen_groups)
            rows.append(
                {
                    "family": "linear_ts",
                    "t": t,
                    "seed": int(seed),
                    "policy": policy_name,
                    "preprocessing": preprocessing,
                    **metrics,
                }
            )

    return normalize_metric_columns(pd.DataFrame(rows)), policy

def run_adult_linucb_trajectory(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    g_train: np.ndarray,
    seed: int,
    policy_name: str,
    preprocessing: str,
    params: AdultBanditParams,
) -> tuple[pd.DataFrame, Any]:
    """
    Run one Adult LinUCB/FairLinUCB trajectory.

    As for the LinTS family, the stream is uniform and reweighting affects
    the update weights.
    """
    order = make_linear_ts_stream_indices(
        n_train=len(y_train),
        t_max=params.t_max,
        seed=seed,
    )

    X_stream = X_train[order]
    y_stream = y_train[order]
    g_stream = g_train[order]

    weights, _ = make_preprocessing_weights(
        preprocessing=preprocessing,
        groups=g_stream,
        oracle_actions=y_stream,
    )

    policy = make_adult_policy(
        family="linucb",
        policy_name=policy_name,
        groups=g_stream,
        params=params,
        seed=seed,
    )

    rows = []
    seen_y = []
    seen_actions = []
    seen_groups = []

    for index in range(int(params.t_max)):
        x = X_stream[index]
        y = int(y_stream[index])
        group = str(g_stream[index])
        weight = float(weights[index])

        if policy_name == "FairLinUCB":
            action = int(policy.select(x, group=group))
        else:
            action = int(policy.select(x))

        reward = float(action == y)

        weighted_linear_update(
            policy,
            x=x,
            action=action,
            reward=reward,
            weight=weight,
            group=group if policy_name == "FairLinUCB" else None,
        )

        seen_y.append(y)
        seen_actions.append(action)
        seen_groups.append(group)

        t = index + 1

        if t == 1 or t % int(params.log_every) == 0 or t == int(params.t_max):
            metrics = summarize_classification_bandit(
                seen_y,
                seen_actions,
                seen_groups,
            )

            rows.append(
                {
                    "family": "linucb",
                    "t": t,
                    "seed": int(seed),
                    "policy": policy_name,
                    "preprocessing": preprocessing,
                    **metrics,
                }
            )

    return normalize_metric_columns(pd.DataFrame(rows)), policy

def run_adult_exp4_trajectory(
    *,
    advice_train: np.ndarray,
    y_train: np.ndarray,
    g_train: np.ndarray,
    seed: int,
    policy_name: str,
    preprocessing: str,
    params: AdultBanditParams,
) -> tuple[pd.DataFrame, Any]:
    """
    Run one Adult EXP4/FairEXP4 trajectory.
    """
    order = make_exp4_stream_indices(
        y_train=y_train,
        g_train=g_train,
        preprocessing=preprocessing,
        t_max=params.t_max,
        seed=seed,
    )

    y_stream = y_train[order]
    g_stream = g_train[order]
    advice_stream = advice_train[order]

    policy = make_adult_policy(
        family="exp4",
        policy_name=policy_name,
        groups=g_stream,
        params=params,
        seed=seed,
    )

    rows = []
    seen_y = []
    seen_actions = []
    seen_groups = []

    for index in range(int(params.t_max)):
        y = int(y_stream[index])
        group = str(g_stream[index])
        advice = advice_stream[index]

        if policy_name == "FairEXP4":
            action = int(policy.select(advice, group=group))
            policy.update(
                advice=advice,
                action=action,
                reward=float(action == y),
                group=group,
            )
        else:
            action = int(policy.select(advice))
            policy.update(
                advice=advice,
                action=action,
                reward=float(action == y),
            )

        seen_y.append(y)
        seen_actions.append(action)
        seen_groups.append(group)

        t = index + 1
        if t == 1 or t % int(params.log_every) == 0 or t == int(params.t_max):
            metrics = summarize_classification_bandit(seen_y, seen_actions, seen_groups)
            rows.append(
                {
                    "family": "exp4",
                    "t": t,
                    "seed": int(seed),
                    "policy": policy_name,
                    "preprocessing": preprocessing,
                    **metrics,
                }
            )

    return normalize_metric_columns(pd.DataFrame(rows)), policy


def score_fair_policy_on_holdout(
    *,
    family: str,
    policy,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    g_cal: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    g_test: np.ndarray,
    advice_cal: np.ndarray | None = None,
    advice_test: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build calibration and test score tables for a trained fair policy.
    """
    if family == "linucb":
        return (
            adult_linucb_score_table(policy, X_cal, y_cal, g_cal, fair=True),
            adult_linucb_score_table(policy, X_test, y_test, g_test, fair=True),
        )
    if family == "linear_ts":
        return (
            adult_lints_score_table(policy, X_cal, y_cal, g_cal, fair=True),
            adult_lints_score_table(policy, X_test, y_test, g_test, fair=True),
        )

    if family == "exp4":
        if advice_cal is None or advice_test is None:
            raise ValueError("EXP4 scoring requires advice_cal and advice_test.")
        return (
            adult_exp4_score_table(policy, advice_cal, y_cal, g_cal, fair=True),
            adult_exp4_score_table(policy, advice_test, y_test, g_test, fair=True),
        )

    raise ValueError(f"Unknown Adult family: {family}")


def run_adult_family_trajectory(
    *,
    family: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    g_train: np.ndarray,
    advice_train: np.ndarray | None,
    seed: int,
    policy_name: str,
    preprocessing: str,
    params: AdultBanditParams,
) -> tuple[pd.DataFrame, Any]:
    """
    Dispatch one trajectory by Adult policy family.
    """
    if family == "linucb":
        return run_adult_linucb_trajectory(
            X_train=X_train,
            y_train=y_train,
            g_train=g_train,
            seed=seed,
            policy_name=policy_name,
            preprocessing=preprocessing,
            params=params,
        )
    
    if family == "linear_ts":
        return run_adult_linear_ts_trajectory(
            X_train=X_train,
            y_train=y_train,
            g_train=g_train,
            seed=seed,
            policy_name=policy_name,
            preprocessing=preprocessing,
            params=params,
        )

    if family == "exp4":
        if advice_train is None:
            raise ValueError("EXP4 family requires advice_train.")
        return run_adult_exp4_trajectory(
            advice_train=advice_train,
            y_train=y_train,
            g_train=g_train,
            seed=seed,
            policy_name=policy_name,
            preprocessing=preprocessing,
            params=params,
        )

    raise ValueError(f"Unknown Adult family: {family}")

def run_adult_family_benchmark(
    *,
    family: str,
    run_dir: str | Path,
    X_train: np.ndarray,
    y_train: np.ndarray,
    g_train: np.ndarray,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    g_cal: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    g_test: np.ndarray,
    advice_train: np.ndarray | None = None,
    advice_cal: np.ndarray | None = None,
    advice_test: np.ndarray | None = None,
    preprocessings: list[str] | tuple[str, ...] = ("uniform", "reweigh_group_label"),
    policies: list[str] | tuple[str, ...] | None = None,
    seeds: list[int] | tuple[int, ...] = (0,),
    params: AdultBanditParams,
    force_rerun: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run or resume one Adult policy-family benchmark.
    """
    if policies is None:
        policies = ADULT_POLICIES_BY_FAMILY[family]

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    temporal_path = run_dir / "temporal.csv"
    endpoint_path = run_dir / "endpoint.csv"
    postproc_path = run_dir / "postprocessing.csv"
    parameters_path = run_dir / "postprocessing_parameters.csv"

    temporal = read_csv_or_empty(temporal_path)
    endpoint = read_csv_or_empty(endpoint_path)
    postproc = read_csv_or_empty(postproc_path)
    parameters = read_csv_or_empty(parameters_path)

    if force_rerun:
        temporal = pd.DataFrame()
        endpoint = pd.DataFrame()
        postproc = pd.DataFrame()
        parameters = pd.DataFrame()

    done = (
        set()
        if endpoint.empty
        else {
            (int(row.seed), str(row.preprocessing), str(row.policy))
            for row in endpoint.itertuples()
        }
    )

    done_postprocessing = (
        set()
        if parameters.empty
        else {
            (int(row.seed), str(row.preprocessing))
            for row in parameters.itertuples()
        }
    )

    total = len(seeds) * len(preprocessings) * len(policies)
    completed = len(done)
    start_time = time.time()

    for seed in seeds:
        for preprocessing in preprocessings:
            trained: dict[str, Any] = {}

            for policy_name in policies:
                key = (int(seed), str(preprocessing), str(policy_name))

                if key in done:
                    print("Cached:", family, key)
                    continue

                print("Running:", family, key)

                trajectory, policy = run_adult_family_trajectory(
                    family=family,
                    X_train=X_train,
                    y_train=y_train,
                    g_train=g_train,
                    advice_train=advice_train,
                    seed=int(seed),
                    policy_name=str(policy_name),
                    preprocessing=str(preprocessing),
                    params=params,
                )

                trained[str(policy_name)] = policy
                temporal = pd.concat([temporal, trajectory], ignore_index=True)
                endpoint = pd.concat(
                    [endpoint, trajectory.sort_values("t").tail(1)],
                    ignore_index=True,
                )

                temporal.to_csv(temporal_path, index=False)
                endpoint.to_csv(endpoint_path, index=False)

                done.add(key)
                completed += 1

                print(
                    f"Completed {completed}/{total} | elapsed="
                    f"{(time.time() - start_time) / 60:.2f} min"
                )

            postprocessing_key = (int(seed), str(preprocessing))

            if postprocessing_key in done_postprocessing:
                print("Cached post-processing:", family, postprocessing_key)
                continue

            fair_name = adult_fair_policy_name(family)
            fair_policy = trained.get(fair_name)

            if fair_policy is None:
                print("Retraining fair policy for post-processing:", family, postprocessing_key)
                _, fair_policy = run_adult_family_trajectory(
                    family=family,
                    X_train=X_train,
                    y_train=y_train,
                    g_train=g_train,
                    advice_train=advice_train,
                    seed=int(seed),
                    policy_name=fair_name,
                    preprocessing=str(preprocessing),
                    params=params,
                )

            calibration_table, test_table = score_fair_policy_on_holdout(
                family=family,
                policy=fair_policy,
                X_cal=X_cal,
                y_cal=y_cal,
                g_cal=g_cal,
                X_test=X_test,
                y_test=y_test,
                g_test=g_test,
                advice_cal=advice_cal,
                advice_test=advice_test,
            )

            thresholds, calibration_raw, calibration_postprocessed = optimize_group_thresholds(
                calibration_table,
                metric_fn=summarize_classification_bandit,
                max_accuracy_drop=params.max_accuracy_drop,
                threshold_grid_size=params.threshold_grid_size,
            )

            raw_actions = actions_from_group_thresholds(
                test_table,
                {group: 0.0 for group in sorted(np.unique(g_test).astype(str))},
            )

            postprocessed_actions = actions_from_group_thresholds(
                test_table,
                thresholds,
            )

            for policy_name, actions in [
                (fair_name, raw_actions),
                (adult_postprocessed_policy_name(family), postprocessed_actions),
            ]:
                row = {
                    "family": family,
                    "seed": int(seed),
                    "preprocessing": str(preprocessing),
                    "policy": policy_name,
                    "t": int(params.t_max),
                    **summarize_classification_bandit(y_test, actions, g_test),
                }
                postproc = pd.concat([postproc, pd.DataFrame([row])], ignore_index=True)

            parameter_row = {
                "family": family,
                "seed": int(seed),
                "preprocessing": str(preprocessing),
                "thresholds_json": json.dumps(thresholds, sort_keys=True),
                "calibration_raw_DP_gap": calibration_raw["DP_gap"],
                "calibration_postproc_DP_gap": calibration_postprocessed["DP_gap"],
                "calibration_raw_EO_gap": calibration_raw["EO_gap"],
                "calibration_postproc_EO_gap": calibration_postprocessed["EO_gap"],
                "calibration_raw_accuracy": calibration_raw["accuracy"],
                "calibration_postproc_accuracy": calibration_postprocessed["accuracy"],
            }

            parameters = pd.concat([parameters, pd.DataFrame([parameter_row])], ignore_index=True)
            postproc.to_csv(postproc_path, index=False)
            parameters.to_csv(parameters_path, index=False)
            done_postprocessing.add(postprocessing_key)

            print("Completed post-processing:", family, postprocessing_key)

    return (
        normalize_metric_columns(_append_family(temporal, family)),
        normalize_metric_columns(_append_family(endpoint, family)),
        normalize_metric_columns(_append_family(postproc, family)),
        _append_family(parameters, family),
    )


def load_adult_family_outputs(
    *,
    family: str,
    run_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load cached outputs for one Adult policy family.
    """
    run_dir = Path(run_dir)

    temporal = read_csv_or_empty(run_dir / "temporal.csv")
    endpoint = read_csv_or_empty(run_dir / "endpoint.csv")
    postproc = read_csv_or_empty(run_dir / "postprocessing.csv")
    parameters = read_csv_or_empty(run_dir / "postprocessing_parameters.csv")

    if temporal.empty or endpoint.empty or postproc.empty or parameters.empty:
        raise FileNotFoundError(
            f"Missing cached Adult outputs in {run_dir}. "
            "Set RUN_BENCHMARK = True to generate them."
        )

    return (
        normalize_metric_columns(_append_family(temporal, family)),
        normalize_metric_columns(_append_family(endpoint, family)),
        normalize_metric_columns(_append_family(postproc, family)),
        _append_family(parameters, family),
    )
