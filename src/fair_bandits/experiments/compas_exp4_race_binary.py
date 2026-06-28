from __future__ import annotations

import json
import time

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

from fair_bandits.metrics import summarize_classification_bandit
from fair_bandits.policies import EXP4Policy, FairEXP4Policy
from fair_bandits.postprocessing.thresholds import (
    actions_from_thresholds,
    optimize_thresholds,
    score_table,
)

@dataclass
class ExpertPool:
    """
    A pool of trained experts (e.g., logistic regression models).
    Each expert can provide advice in the form of a probability distribution over actions.
    """
    models: list
    epsilon: float = 1e-4

    def predict_advice(self, X: np.ndarray) -> np.ndarray:
        """
        Return an array of shape (n_samples, n_experts, 2).
        Each expert supplies a probability distribution over actions {0, 1}.
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
    
def train_expert_pool(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_experts: int,
    bootstrap_size: int,
    seed: int = 2026,
    ) -> ExpertPool:
    """
    Train a pool of experts (logistic regression models) using bootstrapped samples of the training data.
    Each expert is trained on a different bootstrapped sample of the data.
    """
    rng = np.random.default_rng(seed)
    models = []

    class_weight_options = [None, "balanced"]
    c_options = [0.25, 0.5, 1.0, 2.0]

    for expert_index in range(n_experts):
        sample_size = min(int(bootstrap_size), len(y))
        sample_indices = rng.choice(len(y), size=sample_size, replace=True)

        class_weight = class_weight_options[expert_index % len(class_weight_options)]
        c_value = c_options[expert_index % len(c_options)]

        model = LogisticRegression(
            C=c_value,
            class_weight=class_weight,
            max_iter=500,
            solver="liblinear",
            random_state=int(seed + expert_index),
        )

        model.fit(X[sample_indices], y[sample_indices])
        models.append(model)

    return ExpertPool(models=models)

def preprocessing_weights(mode: str, groups, labels) -> np.ndarray:
    """
    Compute preprocessing weights for the training data based on the specified mode.
    Supported modes:
    - "uniform": All samples have equal weight.
    - "reweigh_group_label": Weights are inversely proportional to the frequency of each (group, label) combination.
    """
    groups = np.asarray(groups).astype(str)
    labels = np.asarray(labels, dtype=int)

    if mode == "uniform":
        return np.ones(len(labels), dtype=float)

    if mode != "reweigh_group_label":
        raise ValueError(mode)

    table = (
        pd.DataFrame({"group": groups, "label": labels})
        .value_counts(["group", "label"])
        .rename("n")
        .reset_index()
    )

    total = int(table["n"].sum())
    cells = len(table)

    lookup = {
        (str(row.group), int(row.label)): total / (cells * int(row.n))
        for row in table.itertuples()
    }

    weights = np.asarray(
        [lookup[(str(group), int(label))] for group, label in zip(groups, labels)],
        dtype=float,
    )

    return weights / weights.mean()

def stream_indices(
    *,
    seed: int,
    preprocessing: str,
    y_train,
    g_train,
    t_max: int,
) -> np.ndarray:
    """
    Generate a stream of indices for training samples based on the specified preprocessing method.
    """
    rng = np.random.default_rng(int(seed))

    n_train = len(y_train)
    base_indices = np.arange(n_train)

    if preprocessing == "uniform":
        order = rng.permutation(base_indices)

        if t_max > len(order):
            extra = rng.choice(
                base_indices,
                size=t_max - len(order),
                replace=True,
            )
            order = np.concatenate([order, extra])
        else:
            order = order[:t_max]

        return order

    weights = preprocessing_weights(
        preprocessing,
        g_train,
        y_train,
    )

    probabilities = weights / weights.sum()

    return rng.choice(
        base_indices,
        size=t_max,
        replace=True,
        p=probabilities,
    )
    
def make_policy(
    *,
    policy_name: str,
    groups,
    seed: int,
    n_experts: int,
    exp4_gamma: float,
    exp4_eta: float | None,
    dp_lambda: float,
    dp_tau: float,
    beta_smooth: float,
    min_group_count: int,
):
    """
    Create a policy based on the specified policy name.
    Supported policy names:
    - "EXP4": Standard EXP4 policy.
    - "FairEXP4": Fair EXP4 policy with group fairness constraints.
    """
    if policy_name == "EXP4":
        return EXP4Policy(
            n_experts=n_experts,
            n_actions=2,
            gamma=exp4_gamma,
            eta=exp4_eta,
            seed=seed,
        )

    if policy_name == "FairEXP4":
        return FairEXP4Policy(
            n_experts=n_experts,
            groups=np.asarray(groups).astype(str),
            n_actions=2,
            gamma=exp4_gamma,
            eta=exp4_eta,
            lambda_fair=dp_lambda,
            tau=dp_tau,
            beta_smooth=beta_smooth,
            min_group_count=min_group_count,
            seed=seed,
        )

    raise ValueError(f"Unknown policy_name: {policy_name}")

def run_trajectory(
    *,
    seed: int,
    policy_name: str,
    preprocessing: str,
    y_train,
    g_train,
    advice_train,
    t_max: int,
    log_every: int,
    n_experts: int,
    exp4_gamma: float,
    exp4_eta: float | None,
    dp_lambda: float,
    dp_tau: float,
    beta_smooth: float,
    min_group_count: int,
) -> tuple[pd.DataFrame, object]:
    """
    Run a trajectory for the specified policy and return the results.
    The trajectory is run for t_max time steps, and results are logged every log_every steps.
    """
    order = stream_indices(
        seed=seed,
        preprocessing=preprocessing,
        y_train=y_train,
        g_train=g_train,
        t_max=t_max,
    )

    y_stream = np.asarray(y_train)[order]
    g_stream = np.asarray(g_train).astype(str)[order]
    advice_stream = np.asarray(advice_train)[order]

    policy = make_policy(
        policy_name=policy_name,
        groups=g_stream,
        seed=seed,
        n_experts=n_experts,
        exp4_gamma=exp4_gamma,
        exp4_eta=exp4_eta,
        dp_lambda=dp_lambda,
        dp_tau=dp_tau,
        beta_smooth=beta_smooth,
        min_group_count=min_group_count,
    )

    rows = []
    seen_y = []
    seen_actions = []
    seen_groups = []

    for index in range(t_max):
        y_true = int(y_stream[index])
        group = str(g_stream[index])
        advice = advice_stream[index]

        if policy_name == "FairEXP4":
            action = int(
                policy.select(
                    advice,
                    group=group,
                )
            )
        else:
            action = int(
                policy.select(
                    advice,
                )
            )

        reward = float(action == y_true)

        if policy_name == "FairEXP4":
            policy.update(
                advice=advice,
                action=action,
                reward=reward,
                group=group,
            )
        else:
            policy.update(
                advice=advice,
                action=action,
                reward=reward,
            )

        seen_y.append(y_true)
        seen_actions.append(action)
        seen_groups.append(group)

        t = index + 1

        if t == 1 or t % log_every == 0 or t == t_max:
            rows.append(
                {
                    "t": t,
                    "seed": int(seed),
                    "policy": policy_name,
                    "preprocessing": preprocessing,
                    **summarize_classification_bandit(
                        seen_y,
                        seen_actions,
                        seen_groups,
                    ),
                }
            )

    return pd.DataFrame(rows), policy

def read_csv_or_empty(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()

def run_benchmark(
    *,
    y_train,
    g_train,
    advice_train,
    y_cal,
    g_cal,
    advice_cal,
    y_test,
    g_test,
    advice_test,
    temporal_path: Path,
    endpoint_path: Path,
    postproc_path: Path,
    parameters_path: Path,
    seeds: list[int],
    preprocessings: list[str],
    policies: list[str],
    t_max: int,
    log_every: int,
    n_experts: int,
    exp4_gamma: float,
    exp4_eta: float | None,
    dp_lambda: float,
    dp_tau: float,
    beta_smooth: float,
    min_group_count: int,
    force_rerun: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run a benchmark experiment with the specified parameters.
    Results are written incrementally to:
    - temporal_path
    - endpoint_path
    - postproc_path
    - parameters_path
    """
    for path in [
        temporal_path,
        endpoint_path,
        postproc_path,
        parameters_path,
    ]:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    temporal = read_csv_or_empty(temporal_path)
    endpoint = read_csv_or_empty(endpoint_path)
    postproc = read_csv_or_empty(postproc_path)
    parameters = read_csv_or_empty(parameters_path)

    if force_rerun:
        temporal = pd.DataFrame()
        endpoint = pd.DataFrame()
        postproc = pd.DataFrame()
        parameters = pd.DataFrame()

    if endpoint.empty:
        done = set()
    else:
        done = {
            (
                int(row.seed),
                str(row.preprocessing),
                str(row.policy),
            )
            for row in endpoint.itertuples()
        }

    if parameters.empty:
        done_postproc = set()
    else:
        done_postproc = {
            (
                int(row.seed),
                str(row.preprocessing),
            )
            for row in parameters.itertuples()
        }

    total = len(seeds) * len(preprocessings) * len(policies)
    completed = len(done)
    start = time.time()

    for seed in seeds:
        for preprocessing in preprocessings:
            trained = {}

            for policy_name in policies:
                key = (
                    int(seed),
                    preprocessing,
                    policy_name,
                )

                if key in done:
                    print("Cached:", key)
                    continue

                print("Running:", key)

                trajectory_df, policy = run_trajectory(
                    seed=seed,
                    policy_name=policy_name,
                    preprocessing=preprocessing,
                    y_train=y_train,
                    g_train=g_train,
                    advice_train=advice_train,
                    t_max=t_max,
                    log_every=log_every,
                    n_experts=n_experts,
                    exp4_gamma=exp4_gamma,
                    exp4_eta=exp4_eta,
                    dp_lambda=dp_lambda,
                    dp_tau=dp_tau,
                    beta_smooth=beta_smooth,
                    min_group_count=min_group_count,
                )

                trained[policy_name] = policy

                temporal = pd.concat(
                    [
                        temporal,
                        trajectory_df,
                    ],
                    ignore_index=True,
                )

                endpoint = pd.concat(
                    [
                        endpoint,
                        trajectory_df.sort_values("t").tail(1),
                    ],
                    ignore_index=True,
                )

                temporal.to_csv(
                    temporal_path,
                    index=False,
                )

                endpoint.to_csv(
                    endpoint_path,
                    index=False,
                )

                done.add(key)
                completed += 1

                elapsed_minutes = (
                    time.time()
                    - start
                ) / 60.0

                print(
                    f"Completed {completed}/{total} "
                    f"| elapsed={elapsed_minutes:.2f} min"
                )

            postproc_key = (
                int(seed),
                preprocessing,
            )

            if postproc_key in done_postproc:
                print("Cached PP:", postproc_key)
                continue

            fair_policy = trained.get("FairEXP4")

            if fair_policy is None:
                print("Retraining FairEXP4 for PP:", postproc_key)

                _, fair_policy = run_trajectory(
                    seed=seed,
                    policy_name="FairEXP4",
                    preprocessing=preprocessing,
                    y_train=y_train,
                    g_train=g_train,
                    advice_train=advice_train,
                    t_max=t_max,
                    log_every=log_every,
                    n_experts=n_experts,
                    exp4_gamma=exp4_gamma,
                    exp4_eta=exp4_eta,
                    dp_lambda=dp_lambda,
                    dp_tau=dp_tau,
                    beta_smooth=beta_smooth,
                    min_group_count=min_group_count,
                )

            calibration_table = score_table(
                fair_policy,
                advice_cal,
                y_cal,
                g_cal,
            )

            test_table = score_table(
                fair_policy,
                advice_test,
                y_test,
                g_test,
            )

            thresholds, calibration_raw, calibration_postproc = optimize_thresholds(
                calibration_table,
            )

            raw_actions = actions_from_thresholds(
                test_table,
                {
                    str(group): 0.0
                    for group in sorted(np.unique(np.asarray(g_test).astype(str)))
                },
                default_threshold=0.0,
            )

            postproc_actions = actions_from_thresholds(
                test_table,
                thresholds,
            )

            for policy_label, actions in [
                (
                    "FairEXP4",
                    raw_actions,
                ),
                (
                    "FairEXP4+PP",
                    postproc_actions,
                ),
            ]:
                postproc = pd.concat(
                    [
                        postproc,
                        pd.DataFrame(
                            [
                                {
                                    "seed": int(seed),
                                    "policy": policy_label,
                                    "preprocessing": preprocessing,
                                    "t": int(t_max),
                                    **summarize_classification_bandit(
                                        y_test,
                                        actions,
                                        g_test,
                                    ),
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )

            parameters = pd.concat(
                [
                    parameters,
                    pd.DataFrame(
                        [
                            {
                                "seed": int(seed),
                                "preprocessing": preprocessing,
                                "thresholds_json": json.dumps(
                                    thresholds,
                                    sort_keys=True,
                                ),
                                "calibration_raw_DP_gap": calibration_raw["DP_gap"],
                                "calibration_postproc_DP_gap": calibration_postproc["DP_gap"],
                                "calibration_raw_EO_gap": calibration_raw["EO_gap"],
                                "calibration_postproc_EO_gap": calibration_postproc["EO_gap"],
                                "calibration_raw_accuracy": calibration_raw["accuracy"],
                                "calibration_postproc_accuracy": calibration_postproc["accuracy"],
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )

            postproc.to_csv(
                postproc_path,
                index=False,
            )

            parameters.to_csv(
                parameters_path,
                index=False,
            )

            done_postproc.add(postproc_key)

    return temporal, endpoint, postproc, parameters

def evaluate_fair_exp4_with_postprocessing(
    *,
    policy,
    seed: int,
    preprocessing: str,
    horizon: int,
    y_cal,
    g_cal,
    advice_cal,
    y_test,
    g_test,
    advice_test,
) -> tuple[list[dict], dict]:
    """
    Evaluate a trained FairEXP4 policy before and after post-processing
    on the held-out test set.
    """
    calibration_table = score_table(
        policy,
        advice_cal,
        y_cal,
        g_cal,
    )

    test_table = score_table(
        policy,
        advice_test,
        y_test,
        g_test,
    )

    thresholds, calibration_raw, calibration_postproc = optimize_thresholds(
        calibration_table,
    )

    raw_actions = actions_from_thresholds(
        test_table,
        {
            str(group): 0.0
            for group in sorted(np.unique(np.asarray(g_test).astype(str)))
        },
        default_threshold=0.0,
    )

    postproc_actions = actions_from_thresholds(
        test_table,
        thresholds,
    )

    rows = []

    for policy_label, actions in [
        (
            "FairEXP4",
            raw_actions,
        ),
        (
            "FairEXP4+PP",
            postproc_actions,
        ),
    ]:
        rows.append(
            {
                "seed": int(seed),
                "preprocessing": preprocessing,
                "policy": policy_label,
                "horizon": int(horizon),
                **summarize_classification_bandit(
                    y_test,
                    actions,
                    g_test,
                ),
            }
        )

    parameter_row = {
        "seed": int(seed),
        "preprocessing": preprocessing,
        "horizon": int(horizon),
        "thresholds_json": json.dumps(
            thresholds,
            sort_keys=True,
        ),
        "calibration_raw_DP_gap": calibration_raw["DP_gap"],
        "calibration_postproc_DP_gap": calibration_postproc["DP_gap"],
        "calibration_raw_EO_gap": calibration_raw["EO_gap"],
        "calibration_postproc_EO_gap": calibration_postproc["EO_gap"],
        "calibration_raw_accuracy": calibration_raw["accuracy"],
        "calibration_postproc_accuracy": calibration_postproc["accuracy"],
    }

    return rows, parameter_row


def run_exp4_postprocessing_horizon_benchmark(
    *,
    y_train,
    g_train,
    advice_train,
    y_cal,
    g_cal,
    advice_cal,
    y_test,
    g_test,
    advice_test,
    postproc_long_path: Path,
    postproc_long_parameters_path: Path,
    seeds: list[int],
    preprocessings: list[str],
    postproc_horizons: list[int],
    n_experts: int,
    exp4_gamma: float,
    exp4_eta: float | None,
    dp_lambda: float,
    dp_tau: float,
    beta_smooth: float,
    min_group_count: int,
    force_rerun: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train one FairEXP4 trajectory for each seed × preprocessing pair.

    At selected training horizons, evaluate:
    - FairEXP4 without post-processing
    - FairEXP4 with group-specific post-processing thresholds

    Results are written incrementally to:
    - postproc_long_path
    - postproc_long_parameters_path
    """
    postproc_long_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    postproc_long_parameters_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    horizons = sorted(
        {
            int(horizon)
            for horizon in postproc_horizons
        }
    )

    if not horizons:
        raise ValueError("postproc_horizons cannot be empty.")

    max_horizon = max(horizons)

    if force_rerun:
        long_df = pd.DataFrame()
        parameters_long_df = pd.DataFrame()
    else:
        long_df = read_csv_or_empty(
            postproc_long_path
        )

        parameters_long_df = read_csv_or_empty(
            postproc_long_parameters_path
        )

    if parameters_long_df.empty:
        completed = set()
    else:
        completed = {
            (
                int(row.seed),
                str(row.preprocessing),
                int(row.horizon),
            )
            for row in parameters_long_df.itertuples()
        }

    total = (
        len(seeds)
        * len(preprocessings)
        * len(horizons)
    )

    completed_count = len(completed)
    start_time = time.time()

    for seed in seeds:
        for preprocessing in preprocessings:
            print(
                "\nTraining FairEXP4:",
                (
                    seed,
                    preprocessing,
                ),
            )

            order = stream_indices(
                seed=seed,
                preprocessing=preprocessing,
                y_train=y_train,
                g_train=g_train,
                t_max=max_horizon,
            )

            y_stream = np.asarray(y_train)[order]
            g_stream = np.asarray(g_train).astype(str)[order]
            advice_stream = np.asarray(advice_train)[order]

            policy = make_policy(
                policy_name="FairEXP4",
                groups=g_stream,
                seed=seed,
                n_experts=n_experts,
                exp4_gamma=exp4_gamma,
                exp4_eta=exp4_eta,
                dp_lambda=dp_lambda,
                dp_tau=dp_tau,
                beta_smooth=beta_smooth,
                min_group_count=min_group_count,
            )

            requested_horizons = set(horizons)

            for index in range(max_horizon):
                y_true = int(y_stream[index])
                group = str(g_stream[index])
                advice = advice_stream[index]

                action = int(
                    policy.select(
                        advice,
                        group=group,
                    )
                )

                reward = float(action == y_true)

                policy.update(
                    advice=advice,
                    action=action,
                    reward=reward,
                    group=group,
                )

                horizon = int(index + 1)

                key = (
                    int(seed),
                    preprocessing,
                    horizon,
                )

                if (
                    horizon in requested_horizons
                    and key not in completed
                ):
                    rows, parameter_row = evaluate_fair_exp4_with_postprocessing(
                        policy=policy,
                        seed=seed,
                        preprocessing=preprocessing,
                        horizon=horizon,
                        y_cal=y_cal,
                        g_cal=g_cal,
                        advice_cal=advice_cal,
                        y_test=y_test,
                        g_test=g_test,
                        advice_test=advice_test,
                    )

                    long_df = pd.concat(
                        [
                            long_df,
                            pd.DataFrame(rows),
                        ],
                        ignore_index=True,
                    )

                    parameters_long_df = pd.concat(
                        [
                            parameters_long_df,
                            pd.DataFrame([parameter_row]),
                        ],
                        ignore_index=True,
                    )

                    long_df.to_csv(
                        postproc_long_path,
                        index=False,
                    )

                    parameters_long_df.to_csv(
                        postproc_long_parameters_path,
                        index=False,
                    )

                    completed.add(key)
                    completed_count += 1

                    elapsed_minutes = (
                        time.time()
                        - start_time
                    ) / 60.0

                    print(
                        f"Completed {completed_count}/{total} "
                        f"| seed={seed} "
                        f"| preprocessing={preprocessing} "
                        f"| horizon={horizon} "
                        f"| elapsed={elapsed_minutes:.2f} min"
                    )

    return long_df, parameters_long_df