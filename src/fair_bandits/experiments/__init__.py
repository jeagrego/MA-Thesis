from .runner import (
    ExperimentBundle,
    run_bandit_on_index_sequence,
    run_multi_seed_bandit_experiment,
    sample_idx_seq,
    summarize_over_seeds,
    summarize_seed_runs,
)

from .statistics import (
    compare_methods,
    friedman_test,
    holm_correction,
    pairwise_wilcoxon,
)

from .synthetic_runner import (
    replay_bandit_on_synthetic_env,
    replay_exp4_on_synthetic_env,
    summarize_synth_logs,
    run_synth_benchmark,
    run_regime_matched_synth_benchmark,
    make_synth_temporal_ci,
    make_synth_summary_table,
    SyntheticPolicyParams,
    instantiate_synthetic_policy,
    replay_one_synthetic_trajectory,
    summarize_synthetic_checkpoints,
    load_existing_synthetic_checkpoint_rows,
    run_online_synthetic_benchmarks,
)

from .compas_exp4_race_binary import (
    ExpertPool,
    train_expert_pool,
    run_trajectory,
    run_benchmark,
    run_exp4_postprocessing_horizon_benchmark,
    evaluate_fair_exp4_with_postprocessing,
    read_csv_or_empty,
)

from .synthetic_preflight import (
    run_imbalanced_synthetic_preflight,
    run_synthetic_preflight,
)

from .adult_expert_advice import AdultExpertPool, train_adult_expert_pool

from .adult_runner import (
    ADULT_FAMILY_LABELS,
    ADULT_POLICIES_BY_FAMILY,
    AdultBanditParams,
    load_adult_family_outputs,
    run_adult_family_benchmark,
)

# Generic aliases for offline classification-derived CMAB benchmarks.
# These names are used by COMPAS and any future tabular classification benchmark.
ClassificationExpertPool = AdultExpertPool
train_classification_expert_pool = train_adult_expert_pool

CLASSIFICATION_FAMILY_LABELS = ADULT_FAMILY_LABELS
CLASSIFICATION_POLICIES_BY_FAMILY = ADULT_POLICIES_BY_FAMILY

ClassificationBanditParams = AdultBanditParams
load_classification_family_outputs = load_adult_family_outputs
run_classification_family_benchmark = run_adult_family_benchmark

__all__ = [
    "ExperimentBundle",
    "sample_idx_seq",
    "run_bandit_on_index_sequence",
    "run_multi_seed_bandit_experiment",
    "summarize_seed_runs",
    "summarize_over_seeds",
    "holm_correction",
    "friedman_test",
    "pairwise_wilcoxon",
    "compare_methods",
    "replay_bandit_on_synthetic_env",
    "summarize_synth_logs",
    "run_synth_benchmark",
    "make_synth_temporal_ci",
    "make_synth_summary_table",
    "replay_exp4_on_synthetic_env",
    "run_regime_matched_synth_benchmark",
    "ExpertPool",
    "train_expert_pool",
    "run_trajectory",
    "run_benchmark",
    "read_csv_or_empty",
    "run_exp4_postprocessing_horizon_benchmark",
    "evaluate_fair_exp4_with_postprocessing",
    "SyntheticPolicyParams",
    "instantiate_synthetic_policy",
    "replay_one_synthetic_trajectory",
    "summarize_synthetic_checkpoints",
    "load_existing_synthetic_checkpoint_rows",
    "run_online_synthetic_benchmarks",
    "run_synthetic_preflight",
    "run_imbalanced_synthetic_preflight",
    "AdultExpertPool",
    "train_adult_expert_pool",
    "ADULT_FAMILY_LABELS",
    "ADULT_POLICIES_BY_FAMILY",
    "AdultBanditParams",
    "load_adult_family_outputs",
    "run_adult_family_benchmark",
    "ClassificationExpertPool",
    "train_classification_expert_pool",
    "CLASSIFICATION_FAMILY_LABELS",
    "CLASSIFICATION_POLICIES_BY_FAMILY",
    "ClassificationBanditParams",
    "load_classification_family_outputs",
    "run_classification_family_benchmark",
]