"""
Configuration settings for vanilla RL training.
"""
import torch
from dataclasses import dataclass
from typing import Optional

try:
    from .training_config import DataConfig, EnvironmentConfig, ModelConfig
except ImportError:
    from training_config import DataConfig, EnvironmentConfig, ModelConfig


@dataclass
class RLRewardConfig:
    """Reward function configuration for vanilla RL."""
    # Reward weights
    energy_weight: float = -1.0      # Negative to minimize energy consumption
    comfort_weight: float = -5.0     # Negative to minimize comfort violations
    switch_penalty: float = 0 #-0.5     # Penalty for window state changes (reduced)
    safety_penalty: float = 0 #-2.0     # Penalty for unsafe window operation
    
    # Comfort thresholds (Celsius)
    comfort_min: float = 21.0        # °C
    comfort_max: float = 25.0        # °C
    
    # Safety thresholds
    extreme_temp_high: float = 32.0  # °C
    extreme_temp_low: float = 18.0   # °C
    heavy_wind_threshold: float = 8.0 # m/s


@dataclass
class RLTrainingConfig:
    """Vanilla RL training configuration."""
    # PPO parameters (similar to AIRL)
    policy_lr: float = 1e-3
    ppo_epochs: int = 6
    n_updates: int = 100
    batch_days: int = 20
    val_days: int = 10
    
    # PPO hyperparameters (same as AIRL)
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2
    
    # Early stopping
    target_reward: Optional[float] = None  # Set to enable early stopping
    patience: int = 20                     # Updates without improvement
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class RLExperimentConfig:
    """Complete vanilla RL experiment configuration."""
    data: DataConfig
    model: ModelConfig
    environment: EnvironmentConfig
    reward: RLRewardConfig
    training: RLTrainingConfig
    
    # Paths (same as AIRL)
    dynamics_model_path: str = "models/trained_model_new_full.pth"
    output_dir: str = "notebooks/outputs"
    
    # Logging
    log_interval: int = 1
    save_interval: int = 10
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.reward.comfort_min < self.reward.comfort_max, "Comfort min must be less than max"
        assert self.training.n_updates > 0, "Number of updates must be positive"
        assert self.reward.energy_weight <= 0, "Energy weight should be negative (minimization)"
        assert self.reward.comfort_weight <= 0, "Comfort weight should be negative (minimization)"


def get_default_rl_config() -> RLExperimentConfig:
    """Get default vanilla RL configuration."""
    return RLExperimentConfig(
        data=DataConfig(),
        model=ModelConfig(),
        environment=EnvironmentConfig(),
        reward=RLRewardConfig(),
        training=RLTrainingConfig()
    )


def get_quick_rl_config() -> RLExperimentConfig:
    """Get configuration for quick RL testing."""
    config = get_default_rl_config()
    
    # Reduce iterations for quick testing
    config.training.n_updates = 50
    config.training.batch_days = 3
    config.training.val_days = 2
    
    # More conservative reward weights for testing
    config.reward.energy_weight = -0.5
    config.reward.comfort_weight = -2.0
    config.reward.switch_penalty = 0 # -0.1
    config.reward.safety_penalty = 0 # -1.0
    
    return config


def get_energy_focused_config() -> RLExperimentConfig:
    """Get configuration focused on energy efficiency."""
    config = get_default_rl_config()
    
    # Emphasize energy savings
    config.reward.energy_weight = -2.0
    config.reward.comfort_weight = -5.0
    config.reward.switch_penalty = -0.05  # Reduce switch penalty
    
    return config


def get_comfort_focused_config() -> RLExperimentConfig:
    """Get configuration focused on thermal comfort."""
    config = get_default_rl_config()
    
    # Emphasize comfort
    config.reward.energy_weight = -0.5
    config.reward.comfort_weight = -20.0
    config.reward.switch_penalty = -0.2  # Increase switch penalty for stability
    
    # Tighter comfort bounds
    config.reward.comfort_min = 22.0   # °C
    config.reward.comfort_max = 24.0   # °C
    
    return config


def get_balanced_config() -> RLExperimentConfig:
    """Get balanced configuration between energy and comfort."""
    config = get_default_rl_config()
    
    # Balanced weights
    config.reward.energy_weight = -1.0
    config.reward.comfort_weight = -10.0
    config.reward.switch_penalty = -0.1
    
    return config


def get_long_training_config() -> RLExperimentConfig:
    """Get configuration for extended training."""
    config = get_default_rl_config()
    
    # Extended training
    config.training.n_updates = 500
    config.training.batch_days = 30
    config.training.val_days = 10
    
    # Enable early stopping
    config.training.target_reward = -50.0  # Adjust based on your reward scale
    config.training.patience = 50
    
    return config


def compare_configs() -> None:
    """Print comparison of different configurations."""
    configs = {
        'Default': get_default_rl_config(),
        'Energy-focused': get_energy_focused_config(),
        'Comfort-focused': get_comfort_focused_config(),
        'Balanced': get_balanced_config()
    }
    
    print("RL Configuration Comparison:")
    print("=" * 60)
    
    for name, config in configs.items():
        print(f"\n{name}:")
        print(f"  Energy weight: {config.reward.energy_weight}")
        print(f"  Comfort weight: {config.reward.comfort_weight}")
        print(f"  Switch penalty: {config.reward.switch_penalty}")
        print(f"  Comfort range: [{config.reward.comfort_min:.1f}, {config.reward.comfort_max:.1f}]")
        print(f"  Training updates: {config.training.n_updates}")


if __name__ == "__main__":
    compare_configs()
