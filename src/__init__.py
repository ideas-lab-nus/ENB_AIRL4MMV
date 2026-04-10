"""AIRL for Mixed-Mode Ventilation Control."""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "load_and_filter_data": ".data_processing",
    "split_train_val": ".data_processing",
    "setup_scalers": ".data_processing",
    "extract_expert_trajectories": ".data_processing",
    "normalize": ".data_processing",
    "inverse_normalize": ".data_processing",
    "set_random_seed": ".data_processing",
    "Combined_CNN_LSTM": ".models",
    "MMVPolicyActorCritic": ".models",
    "GRUReward": ".models",
    "MLPReward": ".models",
    "load_dynamics_model": ".models",
    "MixedModeVentilationEnv": ".environment",
    "create_environment": ".environment",
    "run_airl_training": ".airl_training",
    "collect_trajectories": ".airl_training",
    "update_reward": ".airl_training",
    "train_ppo_irl": ".airl_training",
    "evaluate_policy_irl": ".airl_training",
    "evaluate_policy_on_validation": ".utils",
    "load_trained_models": ".utils",
    "compute_evaluation_metrics": ".utils",
    "print_training_summary": ".utils",
}

__all__ = sorted(_EXPORTS)
__version__ = "1.1.0"
__author__ = "AIRL4MMV Team"


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
