try:
    from .adult import load_adult, prepare_adult_contextual, to_binary_income
except ModuleNotFoundError:
    load_adult = None
    prepare_adult_contextual = None
    to_binary_income = None
    
from .compas import (
    load_compas,
    clean_compas_minimal,
    binarize_race_compas,
    ensure_compas_race_binary,
    prepare_compas_contextual,
    compas_reward_from_recid,
)

from .preprocessing import (
    make_group_label_sampling_probs,
    prepare_contextual_frame,
    make_group_label_update_weights,
    make_uniform_update_weights,
    make_preprocessing_weights,
)

from .synthetic import (
    SyntheticCMABConfig,
    SyntheticCMABGenerator,
    make_synthetic_cmab_dataset,
    make_synthetic_expert_advice,
    make_synthetic_potential_rewards,
    quick_synthetic_diagnostics,
)

__all__ = [
    "load_adult",
    "prepare_adult_contextual",
    "to_binary_income",
    "load_compas",
    "clean_compas_minimal",
    "binarize_race_compas",
    "ensure_compas_race_binary",
    "prepare_compas_contextual",
    "compas_reward_from_recid",
    "make_group_label_sampling_probs",
    "prepare_contextual_frame",
    "SyntheticCMABConfig",
    "SyntheticCMABGenerator",
    "make_synthetic_cmab_dataset",
    "make_synthetic_expert_advice",
    "quick_synthetic_diagnostics",
    "make_group_label_update_weights",
    "make_uniform_update_weights",
    "make_preprocessing_weights",
    "make_synthetic_potential_rewards",
]