"""
Configuration settings for AIRL training.
"""
import torch
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


SUPPORTED_PUBLIC_REWARD_TYPES = (
    "temps_hold_dwell_prev_time_gru",
    "temps_hold_dwell_prev_time_comfort_zonal_gru",
    "temps_hold_dwell_prev_time_energy_zonal_gru",
    "temps_hold_dwell_prev_time_energy_comfort_zonal_gru",
)


@dataclass
class DataConfig:
    """Data processing configuration."""
    data_file: str = "l14_merged_data_with_rain.csv"
    min_trajectory_length: int = 600
    train_val_split: float = 0.8
    random_seed: int = 42


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    # Dynamics model
    dynamics_hidden_dim: int = 128
    dynamics_output_dim: int = 5
    dynamics_ac_channels: int = 13
    dynamics_nv_channels: int = 10
    
    # Policy model
    policy_hidden_dim: int = 64
    num_zones: int = 5
    
    # Reward model
    reward_hidden_dim: int = 64
    reward_type: str = "temps_hold_dwell_prev_time_gru"


@dataclass
class EnvironmentConfig:
    """Environment configuration."""
    comfort_min: float = 21.0  # °C - minimum comfortable temperature
    comfort_max: float = 27.0  # °C - maximum comfortable temperature
    heavy_wind_threshold: float = 8.0
    supply_nv: float = 0.55
    supply_ac: float = 0.20
    lc_min: float = 0.32
    lc_max: float = 0.82
    delta_supply_norm: float = 0.2


@dataclass
class TrainingConfig:
    """AIRL training configuration."""
    # AIRL iterations
    n_airl_iters: int = 5
    
    # Policy learning
    policy_lr: float = 1e-3
    ppo_epochs: int = 6
    ppo_updates: int = 10
    batch_days: int = 20
    val_days: int = 10
    
    # Reward learning
    reward_lr: float = 1e-3
    reward_weight_decay: float = 0.0
    reward_update_steps: int = 10
    
    # PPO hyperparameters
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    data: DataConfig
    model: ModelConfig
    environment: EnvironmentConfig
    training: TrainingConfig
    
    # Paths
    dynamics_model_path: str = "models/trained_model_new_full.pth"
    output_dir: str = "notebooks/outputs"
    
    # Logging
    log_interval: int = 1
    save_interval: int = 5
    
    def __post_init__(self):
        """Validate configuration."""
        assert 0 < self.data.train_val_split < 1, "Train/val split must be between 0 and 1"
        assert self.model.num_zones > 0, "Number of zones must be positive"
        assert self.training.n_airl_iters > 0, "Number of AIRL iterations must be positive"


def get_default_config() -> ExperimentConfig:
    """Get default experiment configuration."""
    return ExperimentConfig(
        data=DataConfig(),
        model=ModelConfig(),
        environment=EnvironmentConfig(),
        training=TrainingConfig()
    )


def get_quick_test_config() -> ExperimentConfig:
    """Get configuration for quick testing."""
    config = get_default_config()
    
    # Reduce iterations for quick testing
    config.training.n_airl_iters = 2
    config.training.ppo_updates = 2
    config.training.batch_days = 5
    config.training.val_days = 3
    config.training.reward_update_steps = 3
    
    return config


def get_full_training_config() -> ExperimentConfig:
    """Get configuration for full training run."""
    config = get_default_config()
    
    # Full training parameters
    config.training.n_airl_iters = 20
    config.training.ppo_updates = 20
    config.training.batch_days = 30
    config.training.val_days = 10
    config.training.reward_update_steps = 15
    
    return config


def _load_section(raw_config: Dict[str, Any], key: str, section_cls):
    values = raw_config.get(key, {})
    if values is None:
        values = {}
    return section_cls(**values)


def load_config_from_file(config_path: str) -> ExperimentConfig:
    """Load configuration from JSON file."""
    import json

    with Path(config_path).open('r', encoding='utf-8') as f:
        config_dict = json.load(f)

    defaults = get_default_config()
    return ExperimentConfig(
        data=_load_section(config_dict, "data", DataConfig),
        model=_load_section(config_dict, "model", ModelConfig),
        environment=_load_section(config_dict, "environment", EnvironmentConfig),
        training=_load_section(config_dict, "training", TrainingConfig),
        dynamics_model_path=config_dict.get("dynamics_model_path", defaults.dynamics_model_path),
        output_dir=config_dict.get("output_dir", defaults.output_dir),
        log_interval=config_dict.get("log_interval", defaults.log_interval),
        save_interval=config_dict.get("save_interval", defaults.save_interval),
    )


def save_config_to_file(config: ExperimentConfig, config_path: str):
    """Save configuration to JSON file."""
    import json
    from dataclasses import asdict
    
    config_dict = asdict(config)
    
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2)
