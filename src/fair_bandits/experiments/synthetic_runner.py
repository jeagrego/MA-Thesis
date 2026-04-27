from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..metrics.summary import summarize_metrics
from ..data.synthetic import make_synthetic_cmab_dataset
from ..policies import LinUCB, GroupAwareDPLinUCB
from ..policies.linear_ts import (
    LinearThompsonSampling,
    GroupAwareDPLinearThompsonSampling,
)

from ..policies import (
    LinUCB,
    GroupAwareDPLinUCB,
    LinearThompsonSampling,
    GroupAwareDPLinearThompsonSampling,
    EXP4,
    GroupAwareDPEXP4,
)
from ..data.synthetic import make_synthetic_cmab_dataset, make_synthetic_expert_advice


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