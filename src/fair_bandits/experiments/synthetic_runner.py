from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import time
from dataclasses import dataclass
from pathlib import Path

from ..metrics.summary import summarize_metrics
from ..data.synthetic import make_synthetic_cmab_dataset
from ..policies import LinUCB, GroupAwareDPLinUCB
from ..policies.linear_ts import (
    LinearThompsonSampling,
    GroupAwareDPLinearThompsonSampling,
)

from ..policies import (
    EXP4,
    GroupAwareDPEXP4,
    GroupAwareDPLinUCB,
    GroupAwareDPLinearThompsonSampling,
    LinUCB,
    LinearThompsonSampling,
    SyntheticPolicyParams,
    instantiate_synthetic_policy,
    weighted_exp4_update,
    weighted_linear_update,
)

from ..data.synthetic import make_synthetic_cmab_dataset, make_synthetic_expert_advice

from ..data import (
    make_preprocessing_weights,
    make_synthetic_cmab_dataset,
    make_synthetic_expert_advice,
    make_synthetic_potential_rewards,
)

from ..metrics import (
    add_synthetic_temporal_metrics,
    normalize_metric_columns,
)

def replay_bandit_on_synthetic_env(
    policy,
    dataset: dict,
    fair_policy_types: tuple[type, ...] | None = None,
    action_col_positive: int = 1,
) -> pd.DataFrame:
    """
    Replay one policy on one generated synthetic CMAB environment.
    """
    df = dataset["df"].copy()
    gen = dataset["generator"]
    X = dataset["X"]
    group = dataset["group"]

    fair_policy_types = fair_policy_types or tuple()
    logs: list[dict[str, Any]] = []

    for t in range(len(df)):
        x = X[t]
        g = int(group[t])
        probs = np.array(
            [float(df.iloc[t]["p_action_0"]), float(df.iloc[t]["p_action_1"])],
            dtype=np.float64,
        )

        if isinstance(policy, fair_policy_types):
            a = int(policy.select(x, str(g)))
        else:
            a = int(policy.select(x))

        true_prob = float(probs[a])
        reward = gen.sample_reward(chosen_action=a, true_prob=true_prob, group=g)

        oracle_action = int(df.iloc[t]["oracle_action"])
        oracle_prob = float(df.iloc[t]["oracle_prob"])
        regret = oracle_prob - true_prob

        if isinstance(policy, fair_policy_types):
            policy.update(x=x, action=a, reward=reward, group=str(g))
        else:
            policy.update(x, a, reward)

        logs.append(
            {
                "t": int(t + 1),
                "group": int(g),
                "action": int(a),
                "reward": float(reward),
                "oracle_action": oracle_action,
                "oracle_prob": oracle_prob,
                "chosen_prob": true_prob,
                "regret": float(regret),
                "is_correct_oracle": int(a == oracle_action),
                "positive_action": int(a == action_col_positive),
            }
        )

    out = pd.DataFrame(logs)
    out["cum_reward"] = out["reward"].cumsum()
    out["avg_reward"] = out["cum_reward"] / np.arange(1, len(out) + 1)
    out["cumulative_regret"] = out["regret"].cumsum()
    out["rolling_reward"] = out["reward"].rolling(window=50, min_periods=1).mean()

    # cumulative DP-gap on positive-action exposure
    grp_vals = out["group"].to_numpy()
    pos_vals = out["positive_action"].to_numpy()
    uniq = np.unique(grp_vals)
    n_g = {g: 0 for g in uniq}
    n1_g = {g: 0 for g in uniq}
    dp_gap = []

    for g, p in zip(grp_vals, pos_vals):
        n_g[g] += 1
        if p == 1:
            n1_g[g] += 1

        rates = [(n1_g[gg] / n_g[gg]) if n_g[gg] > 0 else 0.0 for gg in uniq]
        dp_gap.append(float(max(rates) - min(rates)))

    out["DP_gap_over_time"] = dp_gap
    return out


def summarize_synth_logs(logs_df: pd.DataFrame) -> dict[str, float]:
    """
    Summary metrics for one synthetic run.
    """
    y_true = logs_df["oracle_action"].to_numpy(dtype=int)
    y_hat = logs_df["action"].to_numpy(dtype=int)
    group = logs_df["group"].astype(str).to_numpy()
    reward = logs_df["reward"].to_numpy(dtype=float)

    metrics = summarize_metrics(
        y_true=y_true,
        y_hat=y_hat,
        group=group,
        reward=reward,
        oracle_reward=logs_df["oracle_prob"].to_numpy(dtype=float),
    )

    metrics["avg_reward"] = float(logs_df["reward"].mean())
    metrics["accuracy"] = float((logs_df["action"] == logs_df["oracle_action"]).mean())
    metrics["cumulative_regret"] = float(logs_df["regret"].sum())
    metrics["final_DP_gap_over_time"] = float(logs_df["DP_gap_over_time"].iloc[-1])

    return metrics


def _synth_policy_factories(
    d: int,
    groups: np.ndarray,
    seed: int,
    alpha_linucb: float = 1.5,
    lambda_ridge: float = 1.0,
    ts_v: float = 0.5,
    dp_tau: float = 0.02,
    dp_lambda: float = 2.0,
    beta_smooth: float = 1.0,
    min_group_count: int = 20,
) -> dict[str, Any]:
    """
    Create policy instances for the synthetic benchmark with specified parameters.
    The policies include LinUCB, FairLinUCB-DP, LinTS, and FairLinTS-DP.
    The parameters control the behavior of the policies, such as exploration strength (alpha_linucb, ts_v),
    """
    return {
        "LinUCB": LinUCB(
            d=d,
            alpha=alpha_linucb,
            lam=lambda_ridge,
            n_actions=2,
        ),
        "FairLinUCB_DP_GroupAware": GroupAwareDPLinUCB(
            d=d,
            groups=groups.astype(str),
            alpha=alpha_linucb,
            lam=lambda_ridge,
            tau=dp_tau,
            lambda_fair=dp_lambda,
            beta_smooth=beta_smooth,
            min_group_count=min_group_count,
            n_actions=2,
        ),
        "LinTS": LinearThompsonSampling(
            d=d,
            v=ts_v,
            lam=lambda_ridge,
            n_actions=2,
            seed=seed,
        ),
        "FairTS_DP_GroupAware": GroupAwareDPLinearThompsonSampling(
            d=d,
            groups=groups.astype(str),
            v=ts_v,
            lam=lambda_ridge,
            tau=dp_tau,
            lambda_fair=dp_lambda,
            beta_smooth=beta_smooth,
            min_group_count=min_group_count,
            n_actions=2,
            seed=seed,
        ),
    }


def run_synth_benchmark(
    *,
    regimes: tuple[str, ...] = ("stationary_deterministic", "stationary_stochastic", "adversarial_switching"),
    horizons: tuple[int, ...] = (50, 100, 250, 500),
    d: int = 10,
    seeds: tuple[int, ...] = tuple(range(10)),
    keep_logs: bool = True,
    dataset_overrides: dict[str, Any] | None = None,
    alpha_linucb: float = 1.5,
    lambda_ridge: float = 1.0,
    ts_v: float = 0.5,
    dp_tau: float = 0.02,
    dp_lambda: float = 2.0,
    beta_smooth: float = 1.0,
    min_group_count: int = 20,
) -> dict[str, pd.DataFrame]:
    """
    Run the synthetic benchmark over regimes, horizons, policies, and seeds.
    """
    dataset_overrides = dataset_overrides or {}
    fair_policy_types = (GroupAwareDPLinUCB, GroupAwareDPLinearThompsonSampling)

    perseed_rows: list[dict[str, Any]] = []
    logs_all: list[pd.DataFrame] = []

    for regime in regimes:
        for T in horizons:
            for seed in seeds:
                dataset = make_synthetic_cmab_dataset(
                    T=T,
                    d=d,
                    regime=regime,
                    seed=int(seed),
                    **dataset_overrides,
                )
                X = dataset["X"]
                groups = dataset["group"]

                policies = _synth_policy_factories(
                    d=d,
                    groups=groups,
                    seed=int(seed),
                    alpha_linucb=alpha_linucb,
                    lambda_ridge=lambda_ridge,
                    ts_v=ts_v,
                    dp_tau=dp_tau,
                    dp_lambda=dp_lambda,
                    beta_smooth=beta_smooth,
                    min_group_count=min_group_count,
                )

                for policy_name, policy in policies.items():
                    logs = replay_bandit_on_synthetic_env(
                        policy=policy,
                        dataset=dataset,
                        fair_policy_types=fair_policy_types,
                    )
                    logs["policy"] = policy_name
                    logs["regime"] = regime
                    logs["T"] = int(T)
                    logs["seed"] = int(seed)

                    summary = summarize_synth_logs(logs)
                    summary.update(
                        {
                            "policy": policy_name,
                            "regime": regime,
                            "T": int(T),
                            "seed": int(seed),
                        }
                    )
                    perseed_rows.append(summary)

                    if keep_logs:
                        logs_all.append(logs)

    perseed = pd.DataFrame(perseed_rows)

    metric_cols = [
        c for c in perseed.columns
        if c not in {"policy", "regime", "T", "seed"} and pd.api.types.is_numeric_dtype(perseed[c])
    ]

    ci_rows = []
    for keys, gdf in perseed.groupby(["regime", "T", "policy"], dropna=False):
        regime, T, policy = keys
        for metric in metric_cols:
            vals = gdf[metric].to_numpy(dtype=float)
            ci_rows.append(
                {
                    "regime": regime,
                    "T": int(T),
                    "policy": policy,
                    "metric": metric,
                    "mean": float(vals.mean()),
                    "ci_low": float(np.quantile(vals, 0.025)),
                    "ci_high": float(np.quantile(vals, 0.975)),
                    "n_seeds": int(len(vals)),
                }
            )

    ci_long = pd.DataFrame(ci_rows)

    logs_df = pd.concat(logs_all, ignore_index=True) if logs_all else pd.DataFrame()
    return {
        "perseed": perseed,
        "ci_long": ci_long,
        "logs": logs_df,
    }


def make_synth_temporal_ci(logs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate temporal metrics across seeds.
    """
    if logs_df.empty:
        return pd.DataFrame()

    metric_map = {
        "avg_reward": "avg_reward",
        "rolling_reward": "rolling_reward",
        "DP_gap_over_time": "DP_gap",
        "cumulative_regret": "cum_regret",
    }

    rows = []
    for (regime, T, policy, t), gdf in logs_df.groupby(["regime", "T", "policy", "t"], dropna=False):
        for src_col, metric_name in metric_map.items():
            vals = gdf[src_col].to_numpy(dtype=float)
            rows.append(
                {
                    "regime": regime,
                    "T": int(T),
                    "policy": policy,
                    "t": int(t),
                    "metric": metric_name,
                    "mean": float(vals.mean()),
                    "ci_low": float(np.quantile(vals, 0.025)),
                    "ci_high": float(np.quantile(vals, 0.975)),
                    "n_seeds": int(len(vals)),
                }
            )

    return pd.DataFrame(rows).sort_values(["regime", "T", "metric", "policy", "t"]).reset_index(drop=True)


def make_synth_summary_table(ci_long_df: pd.DataFrame, T: int = 500) -> pd.DataFrame:
    """
    Create a summary table of key metrics for a specific horizon T.
    """
    metrics_keep = [
        "accuracy", "avg_reward", "cumulative_regret",
        "DP_gap", "EO_gap", "PPV_gap", "UtilityGap",
    ]
    sub = ci_long_df[(ci_long_df["T"] == T) & (ci_long_df["metric"].isin(metrics_keep))].copy()
    if sub.empty:
        return pd.DataFrame()

    sub["mean_pm_ci"] = sub.apply(
        lambda r: f"{r['mean']:.3f} ± {(r['ci_high'] - r['mean']):.3f}",
        axis=1,
    )
    return sub.pivot_table(
        index=["regime", "policy"],
        columns="metric",
        values="mean_pm_ci",
        aggfunc="first",
    ).reset_index()

def replay_exp4_on_synthetic_env(
    policy,
    dataset: dict,
    expert_advice: np.ndarray,
    fair: bool = False,
) -> pd.DataFrame:
    """
    Replay EXP4 or FairEXP4-DP on a synthetic expert-advice environment.
    """
    df = dataset["df"].copy()
    gen = dataset["generator"]
    group = dataset["group"]

    logs = []

    for t in range(len(df)):
        g = str(int(group[t]))
        advice_t = expert_advice[t]

        if fair:
            action = int(policy.select_from_advice(advice_t, group=g))
        else:
            action = int(policy.select_from_advice(advice_t))

        probs = np.array(
            [float(df.iloc[t]["p_action_0"]), float(df.iloc[t]["p_action_1"])],
            dtype=float,
        )

        true_prob = float(probs[action])
        reward = gen.sample_reward(
            chosen_action=action,
            true_prob=true_prob,
            group=int(g),
        )

        oracle_action = int(df.iloc[t]["oracle_action"])
        oracle_prob = float(df.iloc[t]["oracle_prob"])
        regret = oracle_prob - true_prob

        if fair:
            policy.update_from_advice(
                action=action,
                reward=reward,
                expert_advice=advice_t,
                group=g,
            )
        else:
            policy.update_from_advice(
                action=action,
                reward=reward,
                expert_advice=advice_t,
            )

        logs.append(
            {
                "t": int(t + 1),
                "group": int(g),
                "action": int(action),
                "reward": float(reward),
                "oracle_action": oracle_action,
                "oracle_prob": oracle_prob,
                "chosen_prob": true_prob,
                "regret": float(regret),
                "is_correct_oracle": int(action == oracle_action),
                "positive_action": int(action == 1),
            }
        )

    out = pd.DataFrame(logs)
    out["cum_reward"] = out["reward"].cumsum()
    out["avg_reward"] = out["cum_reward"] / np.arange(1, len(out) + 1)
    out["cumulative_regret"] = out["regret"].cumsum()
    out["rolling_reward"] = out["reward"].rolling(window=50, min_periods=1).mean()

    # Cumulative DP-gap
    grp_vals = out["group"].to_numpy()
    pos_vals = out["positive_action"].to_numpy()
    uniq = np.unique(grp_vals)
    n_g = {g: 0 for g in uniq}
    n1_g = {g: 0 for g in uniq}
    dp_gap = []

    for g, p in zip(grp_vals, pos_vals):
        n_g[g] += 1
        if p == 1:
            n1_g[g] += 1

        rates = [(n1_g[gg] / n_g[gg]) if n_g[gg] > 0 else 0.0 for gg in uniq]
        dp_gap.append(float(max(rates) - min(rates)))

    out["DP_gap_over_time"] = dp_gap
    return out


def run_regime_matched_synth_benchmark(
    *,
    horizons: tuple[int, ...] = (50, 100, 250, 500),
    seeds: tuple[int, ...] = tuple(range(10)),
    d: int = 10,
    dataset_overrides: dict | None = None,
    alpha_linucb: float = 1.5,
    lambda_ridge: float = 1.0,
    ts_v: float = 0.5,
    exp4_gamma: float = 0.07,
    dp_tau: float = 0.02,
    dp_lambda_lin: float = 2.0,
    dp_lambda_exp4: float = 0.20,
    beta_smooth: float = 1.0,
    min_group_count: int = 20,
    n_experts: int = 6,
    keep_logs: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Environment-to-method map:
    - deterministic: LinUCB vs FairLinUCB-DP
    - stochastic: LinTS vs FairLinTS-DP
    - adversarial: EXP4 vs FairEXP4-DP
    """
    dataset_overrides = dataset_overrides or {}

    regime_map = {
        "stationary_deterministic": ["LinUCB", "FairLinUCB_DP"],
        "stationary_stochastic": ["LinTS", "FairLinTS_DP"],
        "adversarial_switching": ["EXP4", "FairEXP4_DP"],
    }

    perseed_rows = []
    logs_all = []

    for regime, methods in regime_map.items():
        for T in horizons:
            for seed in seeds:
                dataset = make_synthetic_cmab_dataset(
                    T=T,
                    d=d,
                    regime=regime,
                    seed=int(seed),
                    **dataset_overrides,
                )

                X = dataset["X"]
                groups = dataset["group"].astype(str)

                expert_advice = None
                if regime == "adversarial_switching":
                    expert_advice, _ = make_synthetic_expert_advice(
                        dataset,
                        n_experts=n_experts,
                        seed=int(seed),
                    )

                policies = {}

                if "LinUCB" in methods:
                    policies["LinUCB"] = LinUCB(
                        d=d,
                        alpha=alpha_linucb,
                        lam=lambda_ridge,
                        n_actions=2,
                    )

                if "FairLinUCB_DP" in methods:
                    policies["FairLinUCB_DP"] = GroupAwareDPLinUCB(
                        d=d,
                        groups=groups,
                        alpha=alpha_linucb,
                        lam=lambda_ridge,
                        tau=dp_tau,
                        lambda_fair=dp_lambda_lin,
                        beta_smooth=beta_smooth,
                        min_group_count=min_group_count,
                        n_actions=2,
                    )

                if "LinTS" in methods:
                    policies["LinTS"] = LinearThompsonSampling(
                        d=d,
                        v=ts_v,
                        lam=lambda_ridge,
                        n_actions=2,
                        seed=int(seed),
                    )

                if "FairLinTS_DP" in methods:
                    policies["FairLinTS_DP"] = GroupAwareDPLinearThompsonSampling(
                        d=d,
                        groups=groups,
                        v=ts_v,
                        lam=lambda_ridge,
                        tau=dp_tau,
                        lambda_fair=dp_lambda_lin,
                        beta_smooth=beta_smooth,
                        min_group_count=min_group_count,
                        n_actions=2,
                        seed=int(seed),
                    )

                if "EXP4" in methods:
                    policies["EXP4"] = EXP4(
                        n_experts=n_experts,
                        n_actions=2,
                        gamma=exp4_gamma,
                        seed=int(seed),
                    )

                if "FairEXP4_DP" in methods:
                    policies["FairEXP4_DP"] = GroupAwareDPEXP4(
                        n_experts=n_experts,
                        groups=groups,
                        n_actions=2,
                        gamma=exp4_gamma,
                        tau=dp_tau,
                        lambda_fair=dp_lambda_exp4,
                        beta_smooth=beta_smooth,
                        min_group_count=min_group_count,
                        seed=int(seed),
                    )

                for policy_name, policy in policies.items():
                    if policy_name in {"EXP4", "FairEXP4_DP"}:
                        logs = replay_exp4_on_synthetic_env(
                            policy=policy,
                            dataset=dataset,
                            expert_advice=expert_advice,
                            fair=(policy_name == "FairEXP4_DP"),
                        )
                    else:
                        fair_types = (
                            GroupAwareDPLinUCB,
                            GroupAwareDPLinearThompsonSampling,
                        )
                        logs = replay_bandit_on_synthetic_env(
                            policy=policy,
                            dataset=dataset,
                            fair_policy_types=fair_types,
                        )

                    logs["policy"] = policy_name
                    logs["regime"] = regime
                    logs["T"] = int(T)
                    logs["seed"] = int(seed)

                    summary = summarize_synth_logs(logs)
                    summary.update(
                        {
                            "policy": policy_name,
                            "regime": regime,
                            "T": int(T),
                            "seed": int(seed),
                            "algorithm_family": (
                                "UCB"
                                if "UCB" in policy_name
                                else "TS"
                                if "TS" in policy_name
                                else "EXP4"
                            ),
                        }
                    )
                    perseed_rows.append(summary)

                    if keep_logs:
                        logs_all.append(logs)

    perseed = pd.DataFrame(perseed_rows)

    metric_cols = [
        c
        for c in perseed.columns
        if c
        not in {
            "policy",
            "regime",
            "T",
            "seed",
            "algorithm_family",
        }
        and pd.api.types.is_numeric_dtype(perseed[c])
    ]

    ci_rows = []
    for keys, gdf in perseed.groupby(["regime", "T", "policy"], dropna=False):
        regime, T, policy = keys
        for metric in metric_cols:
            vals = gdf[metric].to_numpy(dtype=float)
            ci_rows.append(
                {
                    "regime": regime,
                    "T": int(T),
                    "policy": policy,
                    "metric": metric,
                    "mean": float(vals.mean()),
                    "ci_low": float(np.quantile(vals, 0.025)),
                    "ci_high": float(np.quantile(vals, 0.975)),
                    "n_seeds": int(len(vals)),
                }
            )

    ci_long = pd.DataFrame(ci_rows)
    logs_df = pd.concat(logs_all, ignore_index=True) if logs_all else pd.DataFrame()

    return {
        "perseed": perseed,
        "ci_long": ci_long,
        "logs": logs_df,
    }

def replay_one_synthetic_trajectory(
    *,
    dataset: dict,
    potential_rewards: np.ndarray,
    policy_name: str,
    preprocessing: str,
    seed: int,
    params: SyntheticPolicyParams,
    expert_advice: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Replay one policy on one fixed synthetic environment.
    """
    df_environment = dataset["df"].copy()

    x_values = np.asarray(dataset["X"], dtype=float)
    groups = np.asarray(dataset["group"]).astype(str)
    oracle_actions = np.asarray(dataset["y_opt"], dtype=int)

    weights, weight_support = make_preprocessing_weights(
        preprocessing=preprocessing,
        groups=groups,
        oracle_actions=oracle_actions,
    )

    policy = instantiate_synthetic_policy(
        policy_name=policy_name,
        groups=groups,
        seed=int(seed),
        params=params,
    )

    logs = []

    for t in range(len(df_environment)):
        x_t = x_values[t]
        group = str(groups[t])

        probabilities = df_environment.iloc[t][
            ["p_action_0", "p_action_1"]
        ].to_numpy(dtype=float)

        if policy_name in {"EXP4", "FairEXP4_DP"}:
            if expert_advice is None:
                raise ValueError("EXP4 policies require expert_advice.")

            advice_t = expert_advice[t]

            if policy_name == "FairEXP4_DP":
                action = int(policy.select_from_advice(advice_t, group=group))
            else:
                action = int(policy.select_from_advice(advice_t))

        else:
            if policy_name in {"FairLinUCB_DP", "FairLinTS_DP"}:
                action = int(policy.select(x_t, group=group))
            else:
                action = int(policy.select(x_t))

        reward = float(potential_rewards[t, action])
        chosen_probability = float(probabilities[action])

        oracle_action = int(df_environment.iloc[t]["oracle_action"])
        oracle_probability = float(df_environment.iloc[t]["oracle_prob"])

        prediction_error_increment = float(
            oracle_probability - chosen_probability
        )

        weight = float(weights[t])

        if policy_name in {"EXP4", "FairEXP4_DP"}:
            weighted_exp4_update(
                policy=policy,
                action=action,
                reward=reward,
                expert_advice=advice_t,
                weight=weight,
                group=group if policy_name == "FairEXP4_DP" else None,
            )
        else:
            weighted_linear_update(
                policy=policy,
                x=x_t,
                action=action,
                reward=reward,
                weight=weight,
                group=group
                if policy_name in {"FairLinUCB_DP", "FairLinTS_DP"}
                else None,
            )

        logs.append(
            {
                "t": int(t + 1),
                "group": int(group),
                "oracle_action": oracle_action,
                "action": action,
                "positive_action": int(action == 1),
                "reward": reward,
                "oracle_prob": oracle_probability,
                "chosen_prob": chosen_probability,
                "prediction_error_increment": prediction_error_increment,
                "is_correct_oracle": int(action == oracle_action),
                "update_weight": weight,
                "preprocessing": preprocessing,
                "policy": policy_name,
            }
        )

    logs_df = add_synthetic_temporal_metrics(pd.DataFrame(logs))

    return logs_df, weight_support

def synthetic_trajectory_path(
    *,
    trajectory_dir: Path,
    regime: str,
    preprocessing: str,
    policy_name: str,
    seed: int,
) -> Path:
    """
    Save the trajectory logs for a specific synthetic run to a structured directory path.
    """
    
    directory = trajectory_dir / regime / preprocessing / policy_name
    directory.mkdir(parents=True, exist_ok=True)

    return directory / f"seed_{int(seed):03d}.csv.gz"


def synthetic_weights_path(
    *,
    metadata_dir: Path,
    regime: str,
    preprocessing: str,
    seed: int,
) -> Path:
    """
    Save the preprocessing weights for a specific synthetic run to a structured directory path."""
    directory = metadata_dir / "weights" / regime / preprocessing
    directory.mkdir(parents=True, exist_ok=True)

    return directory / f"seed_{int(seed):03d}.csv"


def summarize_synthetic_checkpoints(
    logs_df: pd.DataFrame,
    *,
    checkpoints: list[int] | tuple[int, ...],
    regime: str,
    preprocessing: str,
    policy_name: str,
    seed: int,
    source_path: Path,
) -> pd.DataFrame:
    """
    Summarize one trajectory at selected checkpoints.
    """
    logs_df = normalize_metric_columns(logs_df)

    rows = []

    for checkpoint in checkpoints:
        subset = logs_df[logs_df["t"] <= int(checkpoint)].copy()

        if subset.empty:
            continue

        last = subset.iloc[-1]

        rows.append(
            {
                "regime": regime,
                "preprocessing": preprocessing,
                "policy": policy_name,
                "seed": int(seed),
                "T": int(checkpoint),
                "n_observed": int(len(subset)),
                "accuracy": float(subset["is_correct_oracle"].mean()),
                "avg_reward": float(subset["reward"].mean()),
                "average_reward": float(subset["reward"].mean()),
                "cum_reward": float(last["cum_reward"]),
                "cumulative_prediction_error": float(
                    last["cumulative_prediction_error"]
                ),
                "DP_gap": float(last["DP_gap_over_time"]),
                "TPR_gap": float(last["TPR_gap_over_time"]),
                "FPR_gap": float(last["FPR_gap_over_time"]),
                "EO_gap": float(last["EO_gap_over_time"]),
                "UtilityGap": float(last["UtilityGap_over_time"]),
                "trajectory_file": str(source_path),
            }
        )

    return pd.DataFrame(rows)


def load_existing_synthetic_checkpoint_rows(
    *,
    path: Path,
    checkpoints: list[int] | tuple[int, ...],
    regime: str,
    preprocessing: str,
    policy_name: str,
    seed: int,
) -> pd.DataFrame:
    """
    Reload a cached trajectory and rebuild checkpoint summaries.
    """
    logs_df = pd.read_csv(path, compression="gzip")
    logs_df = normalize_metric_columns(logs_df)

    return summarize_synthetic_checkpoints(
        logs_df,
        checkpoints=checkpoints,
        regime=regime,
        preprocessing=preprocessing,
        policy_name=policy_name,
        seed=seed,
        source_path=path,
    )
    
def run_online_synthetic_benchmarks(
    *,
    run_dir: Path,
    regimes: list[str] | tuple[str, ...],
    preprocessings: list[str] | tuple[str, ...],
    policies_by_regime: dict[str, list[str]],
    seeds: list[int] | tuple[int, ...],
    checkpoints: list[int] | tuple[int, ...],
    t_max: int,
    params: SyntheticPolicyParams,
    force_rerun: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the portable online synthetic CMAB benchmark.

    Outputs:
    - tables/endpoint_perseed.csv
    - tables/run_index.csv
    - trajectories/<regime>/<preprocessing>/<policy>/seed_XXX.csv.gz
    - metadata/weights/<regime>/<preprocessing>/seed_XXX.csv
    """
    run_dir = Path(run_dir)

    table_dir = run_dir / "tables"
    trajectory_dir = run_dir / "trajectories"
    metadata_dir = run_dir / "metadata"

    table_dir.mkdir(parents=True, exist_ok=True)
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    endpoint_path = table_dir / "endpoint_perseed.csv"
    run_index_path = table_dir / "run_index.csv"

    checkpoint_frames = []
    run_index_rows = []

    total_runs = sum(
        len(policies_by_regime[regime]) * len(preprocessings) * len(seeds)
        for regime in regimes
    )

    completed_count = 0
    start_time = time.time()

    for regime in regimes:
        print("Regime:", regime)

        for seed in seeds:
            environment_seed = int(seed)

            dataset = make_synthetic_cmab_dataset(
                T=int(t_max),
                d=int(params.d),
                regime=regime,
                seed=environment_seed,
            )

            potential_rewards = make_synthetic_potential_rewards(
                dataset,
                environment_seed=environment_seed,
                regime=regime,
            )

            expert_advice = None

            if regime == "adversarial_switching":
                expert_advice, _ = make_synthetic_expert_advice(
                    dataset,
                    n_experts=params.n_experts,
                    seed=environment_seed,
                )

            for preprocessing in preprocessings:
                _, weight_support = make_preprocessing_weights(
                    preprocessing=preprocessing,
                    groups=dataset["group"],
                    oracle_actions=dataset["y_opt"],
                )

                support_path = synthetic_weights_path(
                    metadata_dir=metadata_dir,
                    regime=regime,
                    preprocessing=preprocessing,
                    seed=seed,
                )

                weight_support.to_csv(support_path, index=False)

                for policy_name in policies_by_regime[regime]:
                    path = synthetic_trajectory_path(
                        trajectory_dir=trajectory_dir,
                        regime=regime,
                        preprocessing=preprocessing,
                        policy_name=policy_name,
                        seed=seed,
                    )

                    run_started = time.time()

                    if path.exists() and not force_rerun:
                        print("Loaded cached trajectory:", path)

                        checkpoint_df = load_existing_synthetic_checkpoint_rows(
                            path=path,
                            checkpoints=checkpoints,
                            regime=regime,
                            preprocessing=preprocessing,
                            policy_name=policy_name,
                            seed=seed,
                        )

                        status = "cached"

                    else:
                        print(
                            "Running:",
                            {
                                "regime": regime,
                                "preprocessing": preprocessing,
                                "policy": policy_name,
                                "seed": int(seed),
                            },
                        )

                        logs_df, _ = replay_one_synthetic_trajectory(
                            dataset=dataset,
                            potential_rewards=potential_rewards,
                            policy_name=policy_name,
                            preprocessing=preprocessing,
                            seed=seed,
                            params=params,
                            expert_advice=expert_advice,
                        )

                        logs_df["regime"] = regime
                        logs_df["seed"] = int(seed)
                        logs_df["T_max"] = int(t_max)

                        logs_df.to_csv(
                            path,
                            index=False,
                            compression="gzip",
                        )

                        checkpoint_df = summarize_synthetic_checkpoints(
                            logs_df,
                            checkpoints=checkpoints,
                            regime=regime,
                            preprocessing=preprocessing,
                            policy_name=policy_name,
                            seed=seed,
                            source_path=path,
                        )

                        status = "computed"

                    checkpoint_frames.append(checkpoint_df)

                    run_index_rows.append(
                        {
                            "regime": regime,
                            "preprocessing": preprocessing,
                            "policy": policy_name,
                            "seed": int(seed),
                            "trajectory_file": str(path),
                            "status": status,
                            "elapsed_seconds": float(time.time() - run_started),
                        }
                    )

                    completed_count += 1
                    print(f"Completed {completed_count}/{total_runs}")

                    checkpoint_all = pd.concat(
                        checkpoint_frames,
                        ignore_index=True,
                    )

                    run_index = pd.DataFrame(run_index_rows)

                    checkpoint_all.to_csv(endpoint_path, index=False)
                    run_index.to_csv(run_index_path, index=False)

    elapsed = float(time.time() - start_time)
    print(f"\nBenchmark completed in {elapsed / 60.0:.2f} minutes.")

    checkpoint_all = pd.concat(checkpoint_frames, ignore_index=True)
    run_index = pd.DataFrame(run_index_rows)

    return checkpoint_all, run_index