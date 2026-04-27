from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    run_mode: str = "dev"
    seed0: int = 42

    cache_dir: Path = Path("data_cache")
    results_dir: Path = Path("results")

    # Main experiment budgets
    T_group_bandit: int = 50_000
    T_contextual: int = 30_000
    n_seeds: int = 10

    # Bandit hyperparameters
    delta_fair: float = 0.05
    alpha_linucb: float = 1.5
    lambda_ridge: float = 1.0

    # Fairness hyperparameters
    dp_tau: float = 0.02
    dp_lambda: float = 2.0
    beta_smooth: float = 1.0
    min_group_count: int = 20

    # Misc
    compas_hash_features: int = 512
    n_bootstrap: int = 1000


def build_config(run_mode: str = "dev") -> ExperimentConfig:
    mode = str(run_mode).strip().lower()
    if mode not in {"dev", "full"}:
        raise ValueError("run_mode must be either 'dev' or 'full'.")

    if mode == "dev":
        cfg = ExperimentConfig(
            run_mode="dev",
            results_dir=Path("results") / "dev",
            T_group_bandit=10_000,
            T_contextual=5_000,
            n_seeds=3,
            compas_hash_features=256,
            n_bootstrap=200,
        )
    else:
        cfg = ExperimentConfig(
            run_mode="full",
            results_dir=Path("results") / "full",
            T_group_bandit=50_000,
            T_contextual=30_000,
            n_seeds=50,
            compas_hash_features=512,
            n_bootstrap=1000,
        )

    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    return cfg
