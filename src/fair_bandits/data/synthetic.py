from __future__ import annotations

from dataclasses import dataclass, asdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import hashlib
from typing import Any


@dataclass
class SyntheticCMABConfig:
    """
    Specifies the configuration for synthetic contextual multi-armed bandit data generation.
    """
    T: int = 500
    d: int = 10
    n_actions: int = 2
    group_prob: float = 0.5
    regime: str = "stationary_stochastic"
    reward_noise_std: float = 0.10
    theta_scale: float = 1.0
    bias_action1_group0: float = 0.45
    bias_action1_group1: float = 0.15
    bias_action0_group0: float = 0.00
    bias_action0_group1: float = 0.00
    context_mean_shift_group1: float = 0.0
    clip_logits: float = 4.0
    adversarial_block_size: int = 50
    adversarial_shock_scale: float = 1.5
    adversarial_flip_odd_blocks: bool = True


class SyntheticCMABGenerator:
    """
    Generates synthetic contextual multi-armed bandit data according to a configurable model.
    The model assumes two actions (0 and 1) and a binary sensitive group attribute (0 and 1).
    The reward probabilities are generated from a logistic model with linear terms and group-specific biases.
    The "regime" parameter controls the nature of the reward probabilities:
    - "stationary_deterministic": rewards are deterministic based on the sign of the linear terms.
    - "stationary_stochastic": rewards are stochastic with probabilities given by the logistic function.
    - "adversarial_switching": rewards are stochastic but with periodic adversarial shocks that flip the optimal action.    
    """
    def __init__(self, config: SyntheticCMABConfig, seed: int = 0):
        self.cfg = config
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        if self.cfg.n_actions != 2:
            raise ValueError("This implementation assumes n_actions=2.")
        self.base_params = self._build_base_parameters()

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    def _sample_normalized_theta(self) -> np.ndarray:
        """
        Sample a random theta vector and normalize it to have unit norm.
        """
        theta = self.rng.normal(0.0, self.cfg.theta_scale, size=self.cfg.d)
        norm = np.linalg.norm(theta)
        if norm <= 1e-12:
            return theta
        return theta / norm

    def _build_base_parameters(self) -> dict[str, np.ndarray | float]:
        """
        Sample the base parameters for the reward model.
        """
        theta0 = self._sample_normalized_theta()
        theta1 = self._sample_normalized_theta()
        return {
            "theta_action0": theta0,
            "theta_action1": theta1,
            "bias_action0_group0": float(self.cfg.bias_action0_group0),
            "bias_action0_group1": float(self.cfg.bias_action0_group1),
            "bias_action1_group0": float(self.cfg.bias_action1_group0),
            "bias_action1_group1": float(self.cfg.bias_action1_group1),
        }

    def _sample_context(self, group: int) -> np.ndarray:
        """
        Sample a context vector for a given group, applying mean shift if configured.
        """
        x = self.rng.normal(0.0, 1.0, size=self.cfg.d)
        if int(group) == 1 and self.cfg.context_mean_shift_group1 != 0.0:
            x = x + self.cfg.context_mean_shift_group1
        norm = np.linalg.norm(x)
        if norm > 1e-12:
            x = x / norm
        return x

    def _block_sign(self, t: int) -> float:
        """
        Determine the sign of the adversarial shock based on the current time step and block configuration.
        """
        block_idx = int(t // self.cfg.adversarial_block_size)
        if self.cfg.adversarial_flip_odd_blocks and (block_idx % 2 == 1):
            return -1.0
        return 1.0

    def reward_probabilities(self, x: np.ndarray, group: int, t: int) -> np.ndarray:
        """
        Compute the reward probabilities for each action given the context, group, and time step.
        """
        params = self.base_params
        g = int(group)

        lin0 = float(np.dot(x, params["theta_action0"])) + (
            params["bias_action0_group1"] if g == 1 else params["bias_action0_group0"]
        )
        lin1 = float(np.dot(x, params["theta_action1"])) + (
            params["bias_action1_group1"] if g == 1 else params["bias_action1_group0"]
        )

        regime = self.cfg.regime
        if regime == "stationary_deterministic":
            p0 = 1.0 if lin0 >= 0 else 0.0
            p1 = 1.0 if lin1 >= 0 else 0.0
            return np.array([p0, p1], dtype=np.float64)

        if regime == "stationary_stochastic":
            z = np.clip(np.array([lin0, lin1], dtype=np.float64), -self.cfg.clip_logits, self.cfg.clip_logits)
            return self._sigmoid(z)

        if regime == "adversarial_switching":
            sign = self._block_sign(t)
            shock = sign * self.cfg.adversarial_shock_scale
            z = np.array([lin0 - shock, lin1 + shock], dtype=np.float64)
            z = np.clip(z, -self.cfg.clip_logits, self.cfg.clip_logits)
            return self._sigmoid(z)

        raise ValueError(f"Unknown regime '{regime}'.")

    def sample_reward(self, chosen_action: int, true_prob: float, group: int) -> float:
        """
        Sample a reward for the chosen action based on the true reward probability and the configured regime.
        """
        if self.cfg.regime == "stationary_deterministic":
            return float(true_prob)
        return float(self.rng.binomial(1, np.clip(true_prob, 0.0, 1.0)))

    def generate(self) -> dict:
        """
        Generate the synthetic contextual multi-armed bandit.
        """
        rows = []
        X = np.zeros((self.cfg.T, self.cfg.d), dtype=np.float64)
        group = np.zeros(self.cfg.T, dtype=int)
        y_opt = np.zeros(self.cfg.T, dtype=int)

        for t in range(self.cfg.T):
            g = int(self.rng.binomial(1, self.cfg.group_prob))
            x = self._sample_context(group=g)
            probs = self.reward_probabilities(x=x, group=g, t=t)
            oracle_action = int(np.argmax(probs))
            oracle_prob = float(probs[oracle_action])

            X[t] = x
            group[t] = g
            y_opt[t] = oracle_action

            rows.append(
                {
                    "t": int(t),
                    "group": int(g),
                    "oracle_action": oracle_action,
                    "oracle_prob": oracle_prob,
                    "p_action_0": float(probs[0]),
                    "p_action_1": float(probs[1]),
                }
            )

        df = pd.DataFrame(rows)
        return {
            "df": df,
            "X": X,
            "group": group,
            "y_opt": y_opt,
            "config": asdict(self.cfg),
            "generator": self,
        }


def make_synthetic_cmab_dataset(
    *,
    T: int = 500,
    d: int = 10,
    regime: str = "stationary_stochastic",
    seed: int = 0,
    **config_overrides,
) -> dict:
    """
    Convenience function to generate a synthetic CMAB dataset with specified parameters.
    """
    cfg = SyntheticCMABConfig(T=T, d=d, regime=regime, **config_overrides)
    gen = SyntheticCMABGenerator(config=cfg, seed=seed)
    return gen.generate()

def make_imbalanced_synthetic_cmab_dataset(
    *,
    T: int = 500,
    d: int = 10,
    regime: str = "stationary_stochastic",
    seed: int = 0,
    majority_group: str = "0",
    minority_group: str = "1",
    minority_fraction: float = 0.20,
    max_attempts: int = 5,
    **config_overrides,
) -> dict:
    """
    Generate an exactly imbalanced synthetic CMAB stream.

    The base synthetic generator is called on an oversized sequence. The earliest
    required observations from each group are retained, and the selected indices
    are sorted to preserve the original temporal order.

    Supported regimes:
    - stationary_deterministic
    - stationary_stochastic
    - adversarial_switching
    """
    valid_regimes = {
        "stationary_deterministic",
        "stationary_stochastic",
        "adversarial_switching",
    }

    if regime not in valid_regimes:
        raise ValueError(
            f"Unknown synthetic regime for imbalanced generation: {regime}. "
            f"Expected one of: {sorted(valid_regimes)}."
        )

    if not 0.0 < minority_fraction < 1.0:
        raise ValueError("minority_fraction must be between 0 and 1.")

    n_minority = int(round(int(T) * float(minority_fraction)))
    n_majority = int(T) - n_minority

    source_T = max(
        3 * int(T),
        1000,
    )

    base_dataset = None
    selected_indices = None

    for attempt in range(max_attempts):
        candidate = make_synthetic_cmab_dataset(
            T=source_T,
            d=d,
            regime=regime,
            seed=int(seed) + attempt * 100_003,
            **config_overrides,
        )

        groups = np.asarray(candidate["group"]).astype(str)

        majority_indices = np.flatnonzero(groups == majority_group)[:n_majority]
        minority_indices = np.flatnonzero(groups == minority_group)[:n_minority]

        if (
            len(majority_indices) == n_majority
            and len(minority_indices) == n_minority
        ):
            base_dataset = candidate
            selected_indices = np.sort(
                np.concatenate(
                    [
                        majority_indices,
                        minority_indices,
                    ]
                )
            )
            break

        source_T *= 2

    if base_dataset is None or selected_indices is None:
        raise RuntimeError(
            "Unable to construct the requested imbalanced synthetic stream."
        )

    output = dict(base_dataset)

    output["X"] = np.asarray(base_dataset["X"])[selected_indices].copy()
    output["group"] = np.asarray(base_dataset["group"])[selected_indices].copy()
    output["y_opt"] = np.asarray(base_dataset["y_opt"])[selected_indices].copy()

    output["df"] = (
        base_dataset["df"]
        .iloc[selected_indices]
        .reset_index(drop=True)
        .copy()
    )

    output["df"]["t"] = np.arange(len(output["df"]))

    observed_groups, observed_counts = np.unique(
        np.asarray(output["group"]).astype(str),
        return_counts=True,
    )

    observed = dict(zip(observed_groups, observed_counts))

    expected = {
        majority_group: n_majority,
        minority_group: n_minority,
    }

    if observed != expected:
        raise AssertionError(
            f"Observed counts {observed} do not match expected counts {expected}."
        )

    return output

def _stable_uint32(*parts: Any) -> int:
    """
    Build a deterministic uint32 seed from arbitrary values.

    This is used to generate paired potential rewards for all policies evaluated
    on the same synthetic environment.
    """
    text = "||".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()

    return int(digest[:8], 16)


def make_synthetic_potential_rewards(
    dataset: dict,
    *,
    environment_seed: int,
    regime: str | None = None,
) -> np.ndarray:
    """
    Sample one potential reward for each round and each action.

    All policies evaluated on the same synthetic environment should use this same
    reward table. This keeps policy comparisons paired across methods.

    For the deterministic regime, the potential rewards are simply the true
    reward probabilities. For stochastic regimes, rewards are sampled once from
    the action-specific reward probabilities.
    """
    if "df" not in dataset:
        raise ValueError("dataset must contain a 'df' entry.")

    df = dataset["df"]

    required_columns = ["p_action_0", "p_action_1"]
    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Synthetic dataset is missing required columns: {missing_columns}"
        )

    if regime is None:
        regime = str(dataset.get("config", {}).get("regime", "stationary_stochastic"))

    probabilities = df[required_columns].to_numpy(dtype=float)

    if regime == "stationary_deterministic":
        return probabilities.copy()

    rng = np.random.default_rng(
        _stable_uint32(
            "potential_rewards",
            regime,
            environment_seed,
            len(df),
        )
    )

    uniforms = rng.uniform(0.0, 1.0, size=probabilities.shape)

    return (uniforms < probabilities).astype(float)

def quick_synthetic_diagnostics(dataset: dict) -> None:
    """
    Print quick diagnostics and visualizations.
    """
    df = dataset["df"].copy()

    print("Config:")
    print(dataset["config"])
    print()
    print("Group counts:")
    print(df["group"].value_counts(dropna=False).sort_index())
    print()
    print("Oracle action distribution:")
    print(df["oracle_action"].value_counts(dropna=False).sort_index())
    print()
    print("Mean oracle probability:", round(float(df["oracle_prob"].mean()), 4))
    print()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for g, gdf in df.groupby("group"):
        axes[0].hist(gdf["p_action_1"], bins=20, alpha=0.5, label=f"group={g}")
    axes[0].set_title("Distribution of P(reward | action=1)")
    axes[0].set_xlabel("Probability")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    mean_by_t = (
        df.assign(block=(df["t"] // max(1, int(dataset["config"]["adversarial_block_size"]))))
        .groupby("block", as_index=False)[["p_action_0", "p_action_1"]]
        .mean()
    )
    axes[1].plot(mean_by_t["block"], mean_by_t["p_action_0"], label="action 0")
    axes[1].plot(mean_by_t["block"], mean_by_t["p_action_1"], label="action 1")
    axes[1].set_title("Mean action probabilities by block")
    axes[1].set_xlabel("Block")
    axes[1].set_ylabel("Mean probability")
    axes[1].legend()

    plt.tight_layout()
    plt.show()
    
def make_synthetic_expert_advice(
    dataset: dict,
    *,
    n_experts: int = 6,
    seed: int = 0,
    noise_std: float = 0.25,
    bias_strength: float = 0.35,
    temperature: float = 0.75,
) -> tuple[np.ndarray, list[str]]:
    """
    Generate synthetic expert advice for EXP4.

    Output:
        advice: array of shape (T, n_experts, n_actions)
        expert_names: list of expert labels

    Expert design:
    - oracle_like: close to true reward probabilities
    - noisy_oracle: true probabilities plus stronger noise
    - group0_positive_bias: biased toward action 1 for group 0
    - group1_positive_bias: biased toward action 1 for group 1
    - conservative: biased toward action 0
    - random: weak uninformative expert

    The purpose is not to simulate real human experts perfectly, but to create
    an adversarial/expert-advice benchmark where some advice sources are useful,
    some biased, and some unreliable.
    """
    rng = np.random.default_rng(seed)

    df = dataset["df"].copy()
    T = len(df)
    n_actions = 2

    if n_experts < 2:
        raise ValueError("n_experts should be at least 2.")

    base_probs = df[["p_action_0", "p_action_1"]].to_numpy(dtype=float)
    groups = df["group"].to_numpy(dtype=int)

    advice = np.zeros((T, n_experts, n_actions), dtype=float)

    expert_names = [
        "oracle_like",
        "noisy_oracle",
        "group0_positive_bias",
        "group1_positive_bias",
        "conservative_action0",
        "random",
    ]

    while len(expert_names) < n_experts:
        expert_names.append(f"extra_noisy_{len(expert_names)}")

    expert_names = expert_names[:n_experts]

    def normalize_rows(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, 1e-8, None)
        return x / x.sum(axis=1, keepdims=True)

    for e, name in enumerate(expert_names):
        scores = base_probs.copy()

        if name == "oracle_like":
            scores = scores + rng.normal(0.0, noise_std * 0.25, size=scores.shape)

        elif name == "noisy_oracle":
            scores = scores + rng.normal(0.0, noise_std, size=scores.shape)

        elif name == "group0_positive_bias":
            scores = scores + rng.normal(0.0, noise_std, size=scores.shape)
            scores[groups == 0, 1] += bias_strength
            scores[groups == 0, 0] -= bias_strength

        elif name == "group1_positive_bias":
            scores = scores + rng.normal(0.0, noise_std, size=scores.shape)
            scores[groups == 1, 1] += bias_strength
            scores[groups == 1, 0] -= bias_strength

        elif name == "conservative_action0":
            scores = scores + rng.normal(0.0, noise_std, size=scores.shape)
            scores[:, 0] += bias_strength
            scores[:, 1] -= bias_strength

        elif name == "random":
            scores = rng.uniform(0.0, 1.0, size=scores.shape)

        else:
            scores = base_probs + rng.normal(0.0, noise_std * 1.5, size=scores.shape)

        scores = scores / max(temperature, 1e-8)
        advice[:, e, :] = normalize_rows(scores)

    return advice, expert_names