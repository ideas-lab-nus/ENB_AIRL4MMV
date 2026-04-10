"""
AIRL training utilities and algorithms.
"""
import random
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Any, Iterable, Optional, Sequence

try:
    from .data_processing import extract_expert_trajectories, inverse_normalize
    from .models import (
        GRUReward,
        GRURewardStateActionComfortEnergy,
        GRURewardStateActionOnly,
        MLPReward,
        MLPRewardStateActionOnly,
        MMVPolicyActorCritic,
    )
    from .path_utils import ensure_repo_dir
except ImportError:
    from data_processing import extract_expert_trajectories, inverse_normalize
    from models import (
        GRUReward,
        GRURewardStateActionComfortEnergy,
        GRURewardStateActionOnly,
        MLPReward,
        MLPRewardStateActionOnly,
        MMVPolicyActorCritic,
    )
    from path_utils import ensure_repo_dir

AIRL_LEGACY_CONTEXT_DIM = 7
AIRL_LOCAL_COOLING_DIM = 2


def flatten_action(a_dict: Dict[str, Any]) -> np.ndarray:
    """Flatten action dictionary into a vector."""
    return np.concatenate([
        np.array([a_dict["change"]]),
        np.array(a_dict["local_cooling"]),
        np.array(a_dict["supply_temps"])
    ])


def _num_zones_from_state_dim(state_dim: int) -> int:
    """Infer legacy AIRL zone count from a legacy 12-D state."""
    return max(int(state_dim - AIRL_LEGACY_CONTEXT_DIM), 1)


def _num_zones_from_action_dim(action_dim: int) -> int:
    """Infer zone count from flattened AIRL action dimensionality."""
    return max(int(action_dim) - 1 - AIRL_LOCAL_COOLING_DIM, 1)


def _infer_airl_num_zones(state_dim: int, action_dim: Optional[int] = None) -> int:
    """
    Infer zone count from AIRL state/action shapes.

    When policy-only clock features are appended to the state, action dimensionality
    remains the stable source of truth for zone count.
    """
    if action_dim is not None and action_dim >= (1 + AIRL_LOCAL_COOLING_DIM + 1):
        return _num_zones_from_action_dim(action_dim)
    return _num_zones_from_state_dim(state_dim)


def _split_airl_state_components(
    state: np.ndarray,
    *,
    action_dim: Optional[int] = None,
    num_zones: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Split AIRL state into stable semantic components."""
    arr = np.asarray(state, dtype=np.float32)
    state_dim = int(arr.shape[-1])

    if num_zones is None:
        num_zones = _infer_airl_num_zones(state_dim, action_dim)
    if state_dim < num_zones + AIRL_LEGACY_CONTEXT_DIM:
        raise ValueError(
            f"State dim {state_dim} is too small for {num_zones} zones and AIRL context."
        )

    return {
        'zone_temps': arr[..., :num_zones],
        'outdoor_temp': arr[..., num_zones:num_zones + 1],
        'wind_speed': arr[..., num_zones + 1:num_zones + 2],
        'wind_direction': arr[..., num_zones + 2:num_zones + 3],
        'window_hold_counter': arr[..., -4:-3],
        'switch_flag': arr[..., -3:-2],
        'window_dwell': arr[..., -2:-1],
        'prev_window_state': arr[..., -1:],
    }


def _state_scalar(value: np.ndarray) -> float:
    """Convert a 1-element state slice into a scalar float."""
    return float(np.asarray(value, dtype=np.float32).reshape(-1)[0])


def _zone_vector_from_scalar(total_value: float, num_zones: int) -> np.ndarray:
    """Fallback: evenly distribute scalar total into a per-zone vector."""
    return np.full(num_zones, float(total_value) / float(max(num_zones, 1)), dtype=np.float32)


def _zone_vector_from_info(
    info: Dict[str, Any],
    scalar_key: str,
    zone_key: str,
    num_zones: int,
) -> np.ndarray:
    """Extract per-zone feature vector from env info with robust scalar fallback."""
    total_val = float(info.get(scalar_key, 0.0))
    raw = info.get(zone_key, None)
    if raw is None:
        return _zone_vector_from_scalar(total_val, num_zones)

    arr = np.asarray(raw, dtype=np.float32).reshape(-1)
    if arr.shape[0] != num_zones:
        return _zone_vector_from_scalar(total_val, num_zones)
    return arr


def _zone_matrix_from_traj(
    traj: Dict[str, Any],
    zone_key: str,
    scalar_key: str,
    num_zones: int,
) -> np.ndarray:
    """Extract per-step per-zone feature matrix from trajectory with scalar fallback."""
    zone_vals = np.asarray(traj.get(zone_key, []), dtype=np.float32)
    if zone_vals.size == 0:
        scalar_vals = np.asarray(traj.get(scalar_key, []), dtype=np.float32).reshape(-1)
        if scalar_vals.size == 0:
            return np.zeros((0, num_zones), dtype=np.float32)
        return np.stack([_zone_vector_from_scalar(v, num_zones) for v in scalar_vals], axis=0)

    if zone_vals.ndim == 1:
        scalar_vals = np.asarray(traj.get(scalar_key, []), dtype=np.float32).reshape(-1)
        if scalar_vals.shape[0] == zone_vals.shape[0]:
            return np.stack([_zone_vector_from_scalar(v, num_zones) for v in scalar_vals], axis=0)
        zone_vals = zone_vals.reshape(-1, 1)

    if zone_vals.shape[1] != num_zones:
        scalar_vals = np.asarray(traj.get(scalar_key, []), dtype=np.float32).reshape(-1)
        if scalar_vals.shape[0] == zone_vals.shape[0]:
            return np.stack([_zone_vector_from_scalar(v, num_zones) for v in scalar_vals], axis=0)
        return np.zeros((zone_vals.shape[0], num_zones), dtype=np.float32)

    return zone_vals.astype(np.float32, copy=False)


def _scalar_matrix_from_traj(
    traj: Dict[str, Any],
    key: str,
    length: int,
    default: float = 0.0,
) -> np.ndarray:
    """Extract a scalar per-step feature vector from a trajectory with safe fallback."""
    values = np.asarray(traj.get(key, []), dtype=np.float32).reshape(-1)
    if values.shape[0] == length:
        return values.reshape(-1, 1).astype(np.float32, copy=False)

    filled = np.full(int(length), float(default), dtype=np.float32)
    if values.size > 0:
        copy_len = min(int(length), int(values.shape[0]))
        filled[:copy_len] = values[:copy_len]
    return filled.reshape(-1, 1)


def _scalar_feature_from_info(
    info: Dict[str, Any],
    key: str,
    default: float = 0.0,
) -> np.ndarray:
    """Extract a 1-step scalar feature from env info with robust numeric fallback."""
    try:
        value = float(info.get(key, default))
    except (TypeError, ValueError):
        value = float(default)
    if np.isnan(value):
        value = float(default)
    return np.asarray([[value]], dtype=np.float32)


def _resulting_window_state_from_traj(
    parts: Dict[str, np.ndarray],
    window_action: np.ndarray,
) -> np.ndarray:
    """Derive the post-action window state from prev-window state and change action."""
    prev_window = np.asarray(parts['prev_window_state'], dtype=np.float32).reshape(-1)
    change = np.asarray(window_action, dtype=np.float32).reshape(-1)
    result = np.logical_xor(prev_window >= 0.5, change >= 0.5).astype(np.float32)
    return result.reshape(-1, 1)


def _resulting_window_state_from_step(
    parts: Dict[str, np.ndarray],
    action_dict: Dict[str, Any],
    info: Dict[str, Any],
) -> np.ndarray:
    """Derive the post-action window state for a single AIRL transition."""
    if 'window_state' in info:
        try:
            value = float(info['window_state'])
        except (TypeError, ValueError):
            value = 0.0
        if np.isnan(value):
            value = 0.0
        return np.asarray([[value]], dtype=np.float32)

    prev_window = _state_scalar(parts['prev_window_state'])
    change = float(flatten_action(action_dict)[0])
    result = float(int((prev_window >= 0.5) != (change >= 0.5)))
    return np.asarray([[result]], dtype=np.float32)


def _time_encoding_for_length(length: int) -> np.ndarray:
    """Build sinusoidal per-step time encoding [sin, cos] for a trajectory length."""
    if length <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    steps = np.arange(length, dtype=np.float32)
    phase = (2.0 * np.pi * steps) / float(max(length, 1))
    return np.stack((np.sin(phase), np.cos(phase)), axis=1).astype(np.float32)


def _time_encoding_for_step(step_idx: int, horizon: int) -> np.ndarray:
    """Build sinusoidal time encoding [sin, cos] for a single step index."""
    safe_horizon = max(int(horizon), 1)
    phase = (2.0 * np.pi * float(step_idx % safe_horizon)) / float(safe_horizon)
    return np.asarray([np.sin(phase), np.cos(phase)], dtype=np.float32)


AIRL_TIME_AUGMENTED_REWARD_TYPES = (
    'temps_hold_dwell_prev_time_gru',
    'temps_hold_dwell_prev_time_rain_gru',
    'temps_hold_dwell_prev_time_rain_resulting_window_gru',
    'temps_hold_dwell_prev_time_comfort_gru',
    'temps_hold_dwell_prev_time_comfort_zonal_gru',
    'temps_hold_dwell_prev_time_energy_gru',
    'temps_hold_dwell_prev_time_energy_zonal_gru',
    'temps_hold_dwell_prev_time_energy_comfort_gru',
    'temps_hold_dwell_prev_time_energy_comfort_zonal_gru',
)


def _hold_dwell_prev_state_features(parts: Dict[str, np.ndarray]) -> np.ndarray:
    """Build the standard AIRL state features for hold/dwell/prev-window rewards."""
    return np.concatenate(
        [
            parts['zone_temps'],
            parts['outdoor_temp'],
            parts['window_hold_counter'],
            parts['window_dwell'],
            parts['prev_window_state'],
        ],
        axis=1,
    ).astype(np.float32, copy=False)


def _hold_dwell_prev_state_vector(parts: Dict[str, np.ndarray]) -> np.ndarray:
    """Build a 1-D AIRL state feature vector for hold/dwell/prev-window rewards."""
    return np.concatenate(
        [
            np.asarray(parts['zone_temps'], dtype=np.float32).reshape(-1),
            np.asarray(parts['outdoor_temp'], dtype=np.float32).reshape(-1),
            np.asarray(parts['window_hold_counter'], dtype=np.float32).reshape(-1),
            np.asarray(parts['window_dwell'], dtype=np.float32).reshape(-1),
            np.asarray(parts['prev_window_state'], dtype=np.float32).reshape(-1),
        ]
    ).astype(np.float32, copy=False)


def _time_augmented_feature_dim(reward_type: str, num_zones: int) -> int:
    """Feature dimension for time-augmented AIRL reward variants."""
    base_dim = 2  # sin/cos clock features
    if reward_type == 'temps_hold_dwell_prev_time_gru':
        return base_dim
    if reward_type in (
        'temps_hold_dwell_prev_time_rain_gru',
        'temps_hold_dwell_prev_time_comfort_gru',
        'temps_hold_dwell_prev_time_energy_gru',
    ):
        return base_dim + 1
    if reward_type == 'temps_hold_dwell_prev_time_rain_resulting_window_gru':
        return base_dim + 2
    if reward_type == 'temps_hold_dwell_prev_time_energy_comfort_gru':
        return base_dim + 2
    if reward_type in ('temps_hold_dwell_prev_time_comfort_zonal_gru', 'temps_hold_dwell_prev_time_energy_zonal_gru'):
        return base_dim + int(num_zones)
    if reward_type == 'temps_hold_dwell_prev_time_energy_comfort_zonal_gru':
        return base_dim + 2 * int(num_zones)
    raise ValueError(f"Unsupported time-augmented reward type: {reward_type}")


def _time_augmented_traj_features(
    traj: Dict[str, Any],
    reward_type: str,
    *,
    num_zones: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build state/action/feature matrices for time-augmented AIRL reward variants."""
    states = np.asarray(traj["states"], dtype=np.float32)
    actions = np.asarray(traj["actions"], dtype=np.float32)
    if num_zones is None:
        num_zones = _infer_airl_num_zones(states.shape[1], actions.shape[1])

    parts = _split_airl_state_components(states, action_dim=actions.shape[1], num_zones=num_zones)
    state_features = _hold_dwell_prev_state_features(parts)
    window_action = actions[:, 0:1]
    feature_blocks = [_time_encoding_for_length(states.shape[0])]

    if reward_type == 'temps_hold_dwell_prev_time_rain_gru':
        feature_blocks.append(
            _scalar_matrix_from_traj(traj, key='rain_features', length=states.shape[0])
        )
    elif reward_type == 'temps_hold_dwell_prev_time_rain_resulting_window_gru':
        feature_blocks.append(
            _scalar_matrix_from_traj(traj, key='rain_features', length=states.shape[0])
        )
        feature_blocks.append(_resulting_window_state_from_traj(parts, window_action))
    elif reward_type == 'temps_hold_dwell_prev_time_energy_gru':
        feature_blocks.append(np.asarray(traj["energy_features"], dtype=np.float32).reshape(-1, 1))
    elif reward_type == 'temps_hold_dwell_prev_time_comfort_gru':
        feature_blocks.append(np.asarray(traj["comfort_features"], dtype=np.float32).reshape(-1, 1))
    elif reward_type == 'temps_hold_dwell_prev_time_energy_comfort_gru':
        feature_blocks.append(np.asarray(traj["energy_features"], dtype=np.float32).reshape(-1, 1))
        feature_blocks.append(np.asarray(traj["comfort_features"], dtype=np.float32).reshape(-1, 1))
    elif reward_type == 'temps_hold_dwell_prev_time_energy_zonal_gru':
        feature_blocks.append(
            _zone_matrix_from_traj(
                traj,
                zone_key='energy_zone_features',
                scalar_key='energy_features',
                num_zones=num_zones,
            )
        )
    elif reward_type == 'temps_hold_dwell_prev_time_comfort_zonal_gru':
        feature_blocks.append(
            _zone_matrix_from_traj(
                traj,
                zone_key='comfort_zone_features',
                scalar_key='comfort_features',
                num_zones=num_zones,
            )
        )
    elif reward_type == 'temps_hold_dwell_prev_time_energy_comfort_zonal_gru':
        feature_blocks.append(
            _zone_matrix_from_traj(
                traj,
                zone_key='energy_zone_features',
                scalar_key='energy_features',
                num_zones=num_zones,
            )
        )
        feature_blocks.append(
            _zone_matrix_from_traj(
                traj,
                zone_key='comfort_zone_features',
                scalar_key='comfort_features',
                num_zones=num_zones,
            )
        )
    elif reward_type != 'temps_hold_dwell_prev_time_gru':
        raise ValueError(f"Unsupported time-augmented reward type: {reward_type}")

    features = np.concatenate(feature_blocks, axis=1).astype(np.float32, copy=False)
    return state_features, window_action, features


def _time_augmented_step_features(
    state: np.ndarray,
    action_dict: Dict[str, Any],
    env: Any,
    info: Dict[str, Any],
    reward_type: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build 1-step AIRL state/action/feature arrays for time-augmented reward variants."""
    num_zones = len(action_dict["supply_temps"])
    parts = _split_airl_state_components(state, num_zones=num_zones)
    state_features = _hold_dwell_prev_state_vector(parts)
    window_action = np.asarray([[flatten_action(action_dict)[0]]], dtype=np.float32)

    step_idx = max(int(getattr(env, 'current_step', 1)) - 1, 0)
    horizon = int(getattr(env, 'horizon', getattr(env, 'max_steps', 1)))
    feature_blocks = [_time_encoding_for_step(step_idx, horizon).reshape(1, -1)]

    if reward_type == 'temps_hold_dwell_prev_time_rain_gru':
        feature_blocks.append(_scalar_feature_from_info(info, 'rain_status'))
    elif reward_type == 'temps_hold_dwell_prev_time_rain_resulting_window_gru':
        feature_blocks.append(_scalar_feature_from_info(info, 'rain_status'))
        feature_blocks.append(_resulting_window_state_from_step(parts, action_dict, info))
    elif reward_type == 'temps_hold_dwell_prev_time_energy_gru':
        feature_blocks.append(np.asarray([[info['energy_consumption']]], dtype=np.float32))
    elif reward_type == 'temps_hold_dwell_prev_time_comfort_gru':
        feature_blocks.append(np.asarray([[info['comfort_violations']]], dtype=np.float32))
    elif reward_type == 'temps_hold_dwell_prev_time_energy_comfort_gru':
        feature_blocks.append(np.asarray([[info['energy_consumption']]], dtype=np.float32))
        feature_blocks.append(np.asarray([[info['comfort_violations']]], dtype=np.float32))
    elif reward_type == 'temps_hold_dwell_prev_time_energy_zonal_gru':
        feature_blocks.append(
            _zone_vector_from_info(
                info,
                scalar_key='energy_consumption',
                zone_key='energy_consumption_zones',
                num_zones=num_zones,
            ).reshape(1, -1)
        )
    elif reward_type == 'temps_hold_dwell_prev_time_comfort_zonal_gru':
        feature_blocks.append(
            _zone_vector_from_info(
                info,
                scalar_key='comfort_violations',
                zone_key='comfort_violations_zones',
                num_zones=num_zones,
            ).reshape(1, -1)
        )
    elif reward_type == 'temps_hold_dwell_prev_time_energy_comfort_zonal_gru':
        feature_blocks.append(
            _zone_vector_from_info(
                info,
                scalar_key='energy_consumption',
                zone_key='energy_consumption_zones',
                num_zones=num_zones,
            ).reshape(1, -1)
        )
        feature_blocks.append(
            _zone_vector_from_info(
                info,
                scalar_key='comfort_violations',
                zone_key='comfort_violations_zones',
                num_zones=num_zones,
            ).reshape(1, -1)
        )
    elif reward_type != 'temps_hold_dwell_prev_time_gru':
        raise ValueError(f"Unsupported time-augmented reward type: {reward_type}")

    features = np.concatenate(feature_blocks, axis=1).astype(np.float32, copy=False)
    return state_features, window_action, features


def collect_trajectories(env,
                         policy,
                         num_days: Optional[int] = None,
                         day_indices: Optional[Iterable[int]] = None) -> List[Dict[str, List]]:
    """
    Collect trajectories from the environment using the current policy.
    
    Args:
        env: Environment instance
        policy: Policy network
        num_days: Optional number of days to sample uniformly without replacement
        day_indices: Optional explicit iterable of day indices to simulate
        
    Returns:
        List of trajectory dictionaries
    """
    trajectories = []
    total_days = len(env.daily_data_list)

    if day_indices is not None:
        selected_days = list(day_indices)
    else:
        if num_days is None or num_days >= total_days:
            selected_days = list(range(total_days))
        else:
            selected_days = random.sample(range(total_days), num_days)
    
    with torch.no_grad():
        for day_idx in selected_days:
            s = env.reset(day_index=day_idx)
            done = False
            traj = {
                "states": [],
                "actions": [],
                "dones": [],
                "energy_features": [],
                "comfort_features": [],
                "energy_zone_features": [],
                "comfort_zone_features": [],
                "rain_features": [],
            }
            num_zones = int(getattr(env, 'num_zones', _infer_airl_num_zones(len(s))))
            
            while not done:
                s_t = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
                a_dict, _ = policy.get_action(s_t)

                lc_binary = np.array(a_dict["local_cooling"], dtype=np.float32)
                action_env = {
                    "change": a_dict["change"],
                    "supply_temps": np.array(a_dict["supply_temps"], copy=True),
                    "local_cooling": np.array(lc_binary, copy=True)
                }

                s_next, reward, done, info = env.step(action_env)

                traj["states"].append(s)
                traj["actions"].append(flatten_action(a_dict))
                traj["dones"].append(done)
                traj["energy_features"].append(info['energy_consumption'])
                traj["comfort_features"].append(info['comfort_violations'])
                traj["energy_zone_features"].append(
                    _zone_vector_from_info(info, 'energy_consumption', 'energy_consumption_zones', num_zones)
                )
                traj["comfort_zone_features"].append(
                    _zone_vector_from_info(info, 'comfort_violations', 'comfort_violations_zones', num_zones)
                )
                traj["rain_features"].append(float(info.get('rain_status', 0.0)))
                s = s_next
                
            trajectories.append(traj)
    
    return trajectories


def traj_scores(trajs: List[Dict], reward_fn: nn.Module, reward_type: str = 'full') -> torch.Tensor:
    """
    Compute per-transition discriminator logits using the reward function.
    
    Args:
        trajs: List of trajectory dictionaries
        reward_fn: Reward function
        
    Returns:
        1D tensor of logits for each time step across all trajectories
    """
    device = next(reward_fn.parameters()).device
    logits_list = []
    for t in trajs:
        traj_states = traj_actions = traj_features = feats = None
        if reward_type == 'window_only':
            # Only use window action, window-related states, energy, and comfort
            states = np.array(t["states"])  # Shape: (T, 12)
            actions = np.array(t["actions"])  # Shape: (T, 8)
            energy_feats = np.array(t["energy_features"]).reshape(-1, 1)
            comfort_feats = np.array(t["comfort_features"]).reshape(-1, 1)
            
            # Extract only relevant features:
            # - Window-related states: indices [8:12] (window_hold_counter, switch_flag, dwell, window_prev_state)
            # - Window action: index [0] (change action)
            # - Energy and comfort features
            window_states = states[:, -4:]  # (T, 4)
            window_action = actions[:, 0:1]  # (T, 1)
            features = np.concatenate([energy_feats, comfort_feats], axis=1)  # (T, 2)
            
            # Create tensors for separate normalization (GRU format)
            traj_states = torch.tensor(window_states, dtype=torch.float32).unsqueeze(0)  # (1, T, 4)
            traj_actions = torch.tensor(window_action, dtype=torch.float32).unsqueeze(0)  # (1, T, 1)  
            traj_features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # (1, T, 2)
            
            # Legacy compatibility
            feats = torch.tensor(
                np.concatenate([window_states, window_action, energy_feats, comfort_feats], axis=1),
                dtype=torch.float32
            ).unsqueeze(0)  # (1, T, 7)
        elif reward_type == 'window_only_mlp':
            # Only use window action, window-related states, energy, and comfort (MLP version)
            states = np.array(t["states"])  # Shape: (T, 12)
            actions = np.array(t["actions"])  # Shape: (T, 8)
            energy_feats = np.array(t["energy_features"]).reshape(-1, 1)
            comfort_feats = np.array(t["comfort_features"]).reshape(-1, 1)
            
            # Extract only relevant features:
            # - Window-related states: indices [8:12] (window_hold_counter, switch_flag, dwell, window_prev_state)
            # - Window action: index [0] (change action)
            # - Energy and comfort features
            window_states = states[:, -4:]  # (T, 4)
            window_action = actions[:, 0:1]  # (T, 1)
            
            traj_states = torch.tensor(window_states, dtype=torch.float32).unsqueeze(0)
            traj_actions = torch.tensor(window_action, dtype=torch.float32).unsqueeze(0)
            traj_features = torch.tensor(
                np.concatenate([energy_feats, comfort_feats], axis=1),
                dtype=torch.float32
            ).unsqueeze(0)
        elif reward_type == 'energy_comfort_only':
            # Only use energy and comfort features - separate normalization
            energy_feats = np.array(t["energy_features"]).reshape(-1, 1)
            comfort_feats = np.array(t["comfort_features"]).reshape(-1, 1)
            
            # For energy_comfort_only: no states or actions, only features
            traj_states = torch.tensor(np.zeros((len(energy_feats), 0)), dtype=torch.float32).unsqueeze(0)
            traj_actions = torch.tensor(np.zeros((len(energy_feats), 0)), dtype=torch.float32).unsqueeze(0)
            traj_features = torch.tensor(
                np.concatenate([energy_feats, comfort_feats], axis=1),
                dtype=torch.float32
            ).unsqueeze(0)  # (1, T, 2)
        elif reward_type == 'comfort_energy_window':
            # Use comfort, energy, window dwell, and window action features - separate normalization
            states = np.array(t["states"])  # Shape: (T, 12)
            actions = np.array(t["actions"])  # Shape: (T, 8)
            energy_feats = np.array(t["energy_features"]).reshape(-1, 1)
            comfort_feats = np.array(t["comfort_features"]).reshape(-1, 1)
            
            # Extract components: window dwell from states, window action from actions, energy+comfort as features
            window_dwell = states[:, 10:11]  # (T, 1) - window dwell state
            window_action = actions[:, 0:1]   # (T, 1) - change action
            features = np.concatenate([comfort_feats, energy_feats], axis=1)  # (T, 2) - comfort + energy
            
            traj_states = torch.tensor(window_dwell, dtype=torch.float32).unsqueeze(0)
            traj_actions = torch.tensor(window_action, dtype=torch.float32).unsqueeze(0)
            traj_features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        elif reward_type == 'window_conditional_comfort_energy':
            # Use all window-related states, conditional comfort, and energy - separate normalization
            states = np.array(t["states"])  # Shape: (T, 12)
            actions = np.array(t["actions"])  # Shape: (T, 8)
            energy_feats = np.array(t["energy_features"]).reshape(-1, 1)
            comfort_feats = np.array(t["comfort_features"]).reshape(-1, 1)
            
            # Extract components: window states, window action, conditional comfort + energy as features
            window_states = states[:, -4:]  # (T, 4) - window_hold_counter, switch_flag, dwell, window_prev_state
            window_action = actions[:, 0:1]  # (T, 1) - change action
            
            # Calculate conditional comfort: prev_window_state * comfort_violations
            prev_window_state = states[:, -1:]  # (T, 1) - window_prev_state
            conditional_comfort = prev_window_state * comfort_feats  # (T, 1)
            
            # Combine conditional comfort and energy as features
            features = np.concatenate([conditional_comfort, energy_feats], axis=1)  # (T, 2)
            
            traj_states = torch.tensor(window_states, dtype=torch.float32).unsqueeze(0)
            traj_actions = torch.tensor(window_action, dtype=torch.float32).unsqueeze(0)
            traj_features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        elif reward_type == 'window_conditional_energy_comfort':
            # Use all window-related states, conditional energy, and comfort - separate normalization
            states = np.array(t["states"])  # Shape: (T, 12)
            actions = np.array(t["actions"])  # Shape: (T, 8)
            energy_feats = np.array(t["energy_features"]).reshape(-1, 1)
            comfort_feats = np.array(t["comfort_features"]).reshape(-1, 1)
            
            # Extract components: window states, window action, conditional energy + comfort as features
            window_states = states[:, -4:]  # (T, 4) - window_hold_counter, switch_flag, dwell, window_prev_state
            window_action = actions[:, 0:1]  # (T, 1) - change action
            
            # Calculate conditional energy: (1 - prev_window_state) * energy_consumption
            prev_window_state = states[:, -1:]  # (T, 1) - window_prev_state
            conditional_energy = (1 - prev_window_state) * energy_feats  # (T, 1)
            
            # Combine conditional energy and comfort as features
            features = np.concatenate([conditional_energy, comfort_feats], axis=1)  # (T, 2)
            
            traj_states = torch.tensor(window_states, dtype=torch.float32).unsqueeze(0)
            traj_actions = torch.tensor(window_action, dtype=torch.float32).unsqueeze(0)
            traj_features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        elif reward_type == 'window_conditional_comfort_energy_both':
            # Use all window-related states, both conditional comfort and conditional energy - separate normalization
            states = np.array(t["states"])  # Shape: (T, 12)
            actions = np.array(t["actions"])  # Shape: (T, 8)
            energy_feats = np.array(t["energy_features"]).reshape(-1, 1)
            comfort_feats = np.array(t["comfort_features"]).reshape(-1, 1)
            
            # Extract components: window states, window action, both conditional features
            window_states = states[:, -4:]  # (T, 4) - window_hold_counter, switch_flag, dwell, window_prev_state
            window_action = actions[:, 0:1]  # (T, 1) - change action
            
            # Calculate both conditional features based on window state
            prev_window_state = states[:, -1:]  # (T, 1) - window_prev_state
            conditional_comfort = prev_window_state * comfort_feats  # (T, 1) - active when window open
            conditional_energy = (1 - prev_window_state) * energy_feats  # (T, 1) - active when window closed
            
            # Combine both conditional features
            features = np.concatenate([conditional_comfort, conditional_energy], axis=1)  # (T, 2)
            
            traj_states = torch.tensor(window_states, dtype=torch.float32).unsqueeze(0)
            traj_actions = torch.tensor(window_action, dtype=torch.float32).unsqueeze(0)
            traj_features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        elif reward_type == 'energy_comfort_only_gru':
            # GRU version: Only use energy and comfort features
            states = np.array(t["states"])  # Shape: (T, 12)
            actions = np.array(t["actions"])  # Shape: (T, 8)
            energy_feats = np.array(t["energy_features"]).reshape(-1, 1)
            comfort_feats = np.array(t["comfort_features"]).reshape(-1, 1)
            
            # For GRU, use empty state/action tensors and energy+comfort as features
            traj_states = torch.zeros((1, len(states), 0), dtype=torch.float32)  # Empty states
            traj_actions = torch.zeros((1, len(actions), 0), dtype=torch.float32)  # Empty actions
            traj_features = torch.tensor(
                np.concatenate([energy_feats, comfort_feats], axis=1),
                dtype=torch.float32
            ).unsqueeze(0)  # (1, T, 2)
            
            feats = traj_features  # For legacy compatibility
        elif reward_type == 'comfort_energy_window_gru':
            # GRU version: Use comfort, energy, window dwell, and window action features
            states = np.array(t["states"])  # Shape: (T, 12)
            actions = np.array(t["actions"])  # Shape: (T, 8)
            energy_feats = np.array(t["energy_features"]).reshape(-1, 1)
            comfort_feats = np.array(t["comfort_features"]).reshape(-1, 1)
            
            # Extract relevant features:
            window_dwell = states[:, 10:11]  # (T, 1) - window dwell time
            window_action = actions[:, 0:1]   # (T, 1) - change action
            
            # For GRU, separate into states, actions, and features
            traj_states = torch.tensor(window_dwell, dtype=torch.float32).unsqueeze(0)  # (1, T, 1)
            traj_actions = torch.tensor(window_action, dtype=torch.float32).unsqueeze(0)  # (1, T, 1)
            traj_features = torch.tensor(
                np.concatenate([comfort_feats, energy_feats], axis=1),
                dtype=torch.float32
            ).unsqueeze(0)  # (1, T, 2)
            
            feats = torch.tensor(
                np.concatenate([comfort_feats, energy_feats, window_dwell, window_action], axis=1),
                dtype=torch.float32
            ).unsqueeze(0)  # (1, T, 4) - For legacy compatibility
        elif reward_type == 'temps_comfort_energy_window_gru':
            # Extend comfort_energy_window_gru with zone temperatures and outdoor air temperature
            states = np.array(t["states"])  # Shape: (T, state_dim)
            actions = np.array(t["actions"])  # Shape: (T, action_dim)
            energy_feats = np.array(t["energy_features"]).reshape(-1, 1)
            comfort_feats = np.array(t["comfort_features"]).reshape(-1, 1)
            parts = _split_airl_state_components(states, action_dim=actions.shape[1])

            zone_temps = parts['zone_temps']
            outdoor_temp = parts['outdoor_temp']
            window_dwell = parts['window_dwell']
            window_action = actions[:, 0:1]

            state_features = np.concatenate([zone_temps, outdoor_temp, window_dwell], axis=1)

            traj_states = torch.tensor(state_features, dtype=torch.float32).unsqueeze(0)
            traj_actions = torch.tensor(window_action, dtype=torch.float32).unsqueeze(0)
            traj_features = torch.tensor(
                np.concatenate([comfort_feats, energy_feats], axis=1),
                dtype=torch.float32
            ).unsqueeze(0)

            feats = torch.tensor(
                np.concatenate([zone_temps, outdoor_temp, window_dwell, window_action, comfort_feats, energy_feats], axis=1),
                dtype=torch.float32
            ).unsqueeze(0)
        elif reward_type == 'temps_comfort_window_gru':
            # Temperature context with comfort feature only
            states = np.array(t["states"])
            actions = np.array(t["actions"])
            comfort_feats = np.array(t["comfort_features"]).reshape(-1, 1)
            parts = _split_airl_state_components(states, action_dim=actions.shape[1])

            zone_temps = parts['zone_temps']
            outdoor_temp = parts['outdoor_temp']
            window_dwell = parts['window_dwell']
            window_action = actions[:, 0:1]

            state_features = np.concatenate([zone_temps, outdoor_temp, window_dwell], axis=1)

            traj_states = torch.tensor(state_features, dtype=torch.float32).unsqueeze(0)
            traj_actions = torch.tensor(window_action, dtype=torch.float32).unsqueeze(0)
            traj_features = torch.tensor(comfort_feats, dtype=torch.float32).unsqueeze(0)

            feats = torch.tensor(
                np.concatenate([zone_temps, outdoor_temp, window_dwell, window_action, comfort_feats], axis=1),
                dtype=torch.float32
            ).unsqueeze(0)
        elif reward_type == 'temps_energy_window_gru':
            # Temperature context with energy feature only
            states = np.array(t["states"])
            actions = np.array(t["actions"])
            energy_feats = np.array(t["energy_features"]).reshape(-1, 1)
            parts = _split_airl_state_components(states, action_dim=actions.shape[1])

            zone_temps = parts['zone_temps']
            outdoor_temp = parts['outdoor_temp']
            window_dwell = parts['window_dwell']
            window_action = actions[:, 0:1]

            state_features = np.concatenate([zone_temps, outdoor_temp, window_dwell], axis=1)

            traj_states = torch.tensor(state_features, dtype=torch.float32).unsqueeze(0)
            traj_actions = torch.tensor(window_action, dtype=torch.float32).unsqueeze(0)
            traj_features = torch.tensor(energy_feats, dtype=torch.float32).unsqueeze(0)

            feats = torch.tensor(
                np.concatenate([zone_temps, outdoor_temp, window_dwell, window_action, energy_feats], axis=1),
                dtype=torch.float32
            ).unsqueeze(0)
        elif reward_type == 'temps_window_gru':
            # Temperature context with window action + previous window state feature
            states = np.array(t["states"])
            actions = np.array(t["actions"])
            parts = _split_airl_state_components(states, action_dim=actions.shape[1])

            zone_temps = parts['zone_temps']
            outdoor_temp = parts['outdoor_temp']
            window_dwell = parts['window_dwell']
            prev_window_state = parts['prev_window_state']
            window_action = actions[:, 0:1]

            state_features = np.concatenate([zone_temps, outdoor_temp, window_dwell], axis=1)

            traj_states = torch.tensor(state_features, dtype=torch.float32).unsqueeze(0)
            traj_actions = torch.tensor(window_action, dtype=torch.float32).unsqueeze(0)
            traj_features = torch.tensor(prev_window_state, dtype=torch.float32).unsqueeze(0)

            feats = torch.tensor(
                np.concatenate([zone_temps, outdoor_temp, window_dwell, window_action, prev_window_state], axis=1),
                dtype=torch.float32
            ).unsqueeze(0)
        elif reward_type == 'window_conditional_comfort_energy_gru':
            # GRU version: Use all window-related states, conditional comfort, and energy
            states = np.array(t["states"])  # Shape: (T, 12)
            actions = np.array(t["actions"])  # Shape: (T, 8)
            energy_feats = np.array(t["energy_features"]).reshape(-1, 1)
            comfort_feats = np.array(t["comfort_features"]).reshape(-1, 1)
            
            # Extract window-related features:
            window_states = states[:, -4:]  # (T, 4) - window_hold_counter, switch_flag, dwell, window_prev_state
            window_action = actions[:, 0:1]  # (T, 1) - change action
            
            # Calculate conditional comfort: prev_window_state * comfort_violations
            prev_window_state = states[:, -1:]  # (T, 1) - window_prev_state
            conditional_comfort = prev_window_state * comfort_feats  # (T, 1)
            
            # For GRU, separate into states, actions, and features
            traj_states = torch.tensor(window_states, dtype=torch.float32).unsqueeze(0)  # (1, T, 4)
            traj_actions = torch.tensor(window_action, dtype=torch.float32).unsqueeze(0)  # (1, T, 1)
            traj_features = torch.tensor(
                np.concatenate([conditional_comfort, energy_feats], axis=1),
                dtype=torch.float32
            ).unsqueeze(0)  # (1, T, 2)
            
            feats = torch.tensor(
                np.concatenate([window_states, window_action, conditional_comfort, energy_feats], axis=1),
                dtype=torch.float32
            ).unsqueeze(0)  # (1, T, 7) - For legacy compatibility
        elif reward_type == 'window_conditional_energy_comfort_gru':
            # GRU version: Use all window-related states, conditional energy, and comfort
            states = np.array(t["states"])  # Shape: (T, 12)
            actions = np.array(t["actions"])  # Shape: (T, 8)
            energy_feats = np.array(t["energy_features"]).reshape(-1, 1)
            comfort_feats = np.array(t["comfort_features"]).reshape(-1, 1)
            
            # Extract window-related features:
            window_states = states[:, -4:]  # (T, 4) - window_hold_counter, switch_flag, dwell, window_prev_state
            window_action = actions[:, 0:1]  # (T, 1) - change action
            
            # Calculate conditional energy: (1 - prev_window_state) * energy_consumption
            prev_window_state = states[:, -1:]  # (T, 1) - window_prev_state
            conditional_energy = (1 - prev_window_state) * energy_feats  # (T, 1)
            
            # For GRU, separate into states, actions, and features
            traj_states = torch.tensor(window_states, dtype=torch.float32).unsqueeze(0)  # (1, T, 4)
            traj_actions = torch.tensor(window_action, dtype=torch.float32).unsqueeze(0)  # (1, T, 1)
            traj_features = torch.tensor(
                np.concatenate([conditional_energy, comfort_feats], axis=1),
                dtype=torch.float32
            ).unsqueeze(0)  # (1, T, 2)
            
            feats = torch.tensor(
                np.concatenate([window_states, window_action, conditional_energy, comfort_feats], axis=1),
                dtype=torch.float32
            ).unsqueeze(0)  # (1, T, 7) - For legacy compatibility
        elif reward_type == 'window_conditional_comfort_energy_both_gru':
            # GRU version: Use all window-related states, both conditional comfort and conditional energy
            states = np.array(t["states"])  # Shape: (T, 12)
            actions = np.array(t["actions"])  # Shape: (T, 8)
            energy_feats = np.array(t["energy_features"]).reshape(-1, 1)
            comfort_feats = np.array(t["comfort_features"]).reshape(-1, 1)
            
            # Extract window-related features:
            window_states = states[:, -4:]  # (T, 4) - window_hold_counter, switch_flag, dwell, window_prev_state
            window_action = actions[:, 0:1]  # (T, 1) - change action
            
            # Calculate both conditional features based on window state
            prev_window_state = states[:, -1:]  # (T, 1) - window_prev_state
            conditional_comfort = prev_window_state * comfort_feats  # (T, 1) - active when window open
            conditional_energy = (1 - prev_window_state) * energy_feats  # (T, 1) - active when window closed
            
            # For GRU, separate into states, actions, and features
            traj_states = torch.tensor(window_states, dtype=torch.float32).unsqueeze(0)  # (1, T, 4)
            traj_actions = torch.tensor(window_action, dtype=torch.float32).unsqueeze(0)  # (1, T, 1)
            traj_features = torch.tensor(
                np.concatenate([conditional_comfort, conditional_energy], axis=1),
                dtype=torch.float32
            ).unsqueeze(0)  # (1, T, 2)
            
            feats = torch.tensor(
                np.concatenate([window_states, window_action, conditional_comfort, conditional_energy], axis=1),
                dtype=torch.float32
            ).unsqueeze(0)  # (1, T, 7) - For legacy compatibility
        elif reward_type == 'state_action_only_mlp':
            # Use only states and actions (no energy/comfort features) - MLP version
            states = np.array(t["states"])  # Shape: (T, 12)
            actions = np.array(t["actions"])  # Shape: (T, 8)
            
            traj_states = torch.tensor(states, dtype=torch.float32)  # (T, 12)
            traj_actions = torch.tensor(actions, dtype=torch.float32)  # (T, 8)
            # No features for state-action only
            traj_features = None
            
        elif reward_type == 'state_action_only_gru':
            # Use only states and actions (no energy/comfort features) - GRU version
            states = np.array(t["states"])  # Shape: (T, 12)
            actions = np.array(t["actions"])  # Shape: (T, 8)
            
            traj_states = torch.tensor(states, dtype=torch.float32).unsqueeze(0)  # (1, T, 12)
            traj_actions = torch.tensor(actions, dtype=torch.float32).unsqueeze(0)  # (1, T, 8)
            # No features for state-action only
            traj_features = None
        elif reward_type == 'temps_hold_dwell_prev_gru':
            # Zone/outdoor temperatures plus hold/dwell/prev-window state, and change action.
            states = np.array(t["states"])
            actions = np.array(t["actions"])
            parts = _split_airl_state_components(states, action_dim=actions.shape[1])

            zone_temps = parts['zone_temps']
            outdoor_temp = parts['outdoor_temp']
            window_hold_counter = parts['window_hold_counter']
            window_dwell = parts['window_dwell']
            prev_window_state = parts['prev_window_state']
            change_action = actions[:, 0:1]

            state_features = np.concatenate(
                [zone_temps, outdoor_temp, window_hold_counter, window_dwell, prev_window_state],
                axis=1
            )
            traj_states = torch.tensor(state_features, dtype=torch.float32).unsqueeze(0)
            traj_actions = torch.tensor(change_action, dtype=torch.float32).unsqueeze(0)
            traj_features = None
        elif reward_type in AIRL_TIME_AUGMENTED_REWARD_TYPES:
            state_features, window_action, features = _time_augmented_traj_features(t, reward_type)
            traj_states = torch.tensor(state_features, dtype=torch.float32).unsqueeze(0)
            traj_actions = torch.tensor(window_action, dtype=torch.float32).unsqueeze(0)
            traj_features = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        elif reward_type == 'temps_hold_dwell_prev_energy_gru':
            # Temperature/window context with energy feature term.
            states = np.array(t["states"])
            actions = np.array(t["actions"])
            energy_feats = np.array(t["energy_features"]).reshape(-1, 1)
            parts = _split_airl_state_components(states, action_dim=actions.shape[1])

            zone_temps = parts['zone_temps']
            outdoor_temp = parts['outdoor_temp']
            window_hold_counter = parts['window_hold_counter']
            window_dwell = parts['window_dwell']
            prev_window_state = parts['prev_window_state']
            change_action = actions[:, 0:1]

            state_features = np.concatenate(
                [zone_temps, outdoor_temp, window_hold_counter, window_dwell, prev_window_state],
                axis=1
            )
            traj_states = torch.tensor(state_features, dtype=torch.float32).unsqueeze(0)
            traj_actions = torch.tensor(change_action, dtype=torch.float32).unsqueeze(0)
            traj_features = torch.tensor(energy_feats, dtype=torch.float32).unsqueeze(0)
        elif reward_type == 'temps_hold_dwell_prev_comfort_gru':
            # Temperature/window context with comfort feature term.
            states = np.array(t["states"])
            actions = np.array(t["actions"])
            comfort_feats = np.array(t["comfort_features"]).reshape(-1, 1)
            parts = _split_airl_state_components(states, action_dim=actions.shape[1])

            zone_temps = parts['zone_temps']
            outdoor_temp = parts['outdoor_temp']
            window_hold_counter = parts['window_hold_counter']
            window_dwell = parts['window_dwell']
            prev_window_state = parts['prev_window_state']
            change_action = actions[:, 0:1]

            state_features = np.concatenate(
                [zone_temps, outdoor_temp, window_hold_counter, window_dwell, prev_window_state],
                axis=1
            )
            traj_states = torch.tensor(state_features, dtype=torch.float32).unsqueeze(0)
            traj_actions = torch.tensor(change_action, dtype=torch.float32).unsqueeze(0)
            traj_features = torch.tensor(comfort_feats, dtype=torch.float32).unsqueeze(0)
        elif reward_type == 'temps_hold_dwell_prev_energy_comfort_gru':
            # Temperature/window context with energy and comfort feature terms.
            states = np.array(t["states"])
            actions = np.array(t["actions"])
            energy_feats = np.array(t["energy_features"]).reshape(-1, 1)
            comfort_feats = np.array(t["comfort_features"]).reshape(-1, 1)
            parts = _split_airl_state_components(states, action_dim=actions.shape[1])

            zone_temps = parts['zone_temps']
            outdoor_temp = parts['outdoor_temp']
            window_hold_counter = parts['window_hold_counter']
            window_dwell = parts['window_dwell']
            prev_window_state = parts['prev_window_state']
            change_action = actions[:, 0:1]

            state_features = np.concatenate(
                [zone_temps, outdoor_temp, window_hold_counter, window_dwell, prev_window_state],
                axis=1
            )
            traj_states = torch.tensor(state_features, dtype=torch.float32).unsqueeze(0)
            traj_actions = torch.tensor(change_action, dtype=torch.float32).unsqueeze(0)
            traj_features = torch.tensor(
                np.concatenate([energy_feats, comfort_feats], axis=1),
                dtype=torch.float32
            ).unsqueeze(0)
        elif reward_type == 'temps_hold_dwell_prev_energy_zonal_gru':
            # Temperature/window context with per-zone energy feature terms.
            states = np.array(t["states"])
            actions = np.array(t["actions"])
            num_zones = _infer_airl_num_zones(states.shape[1], actions.shape[1])
            energy_zone_feats = _zone_matrix_from_traj(
                t,
                zone_key='energy_zone_features',
                scalar_key='energy_features',
                num_zones=num_zones,
            )
            parts = _split_airl_state_components(states, action_dim=actions.shape[1], num_zones=num_zones)

            zone_temps = parts['zone_temps']
            outdoor_temp = parts['outdoor_temp']
            window_hold_counter = parts['window_hold_counter']
            window_dwell = parts['window_dwell']
            prev_window_state = parts['prev_window_state']
            change_action = actions[:, 0:1]

            state_features = np.concatenate(
                [zone_temps, outdoor_temp, window_hold_counter, window_dwell, prev_window_state],
                axis=1
            )
            traj_states = torch.tensor(state_features, dtype=torch.float32).unsqueeze(0)
            traj_actions = torch.tensor(change_action, dtype=torch.float32).unsqueeze(0)
            traj_features = torch.tensor(energy_zone_feats, dtype=torch.float32).unsqueeze(0)
        elif reward_type == 'temps_hold_dwell_prev_comfort_zonal_gru':
            # Temperature/window context with per-zone comfort feature terms.
            states = np.array(t["states"])
            actions = np.array(t["actions"])
            num_zones = _infer_airl_num_zones(states.shape[1], actions.shape[1])
            comfort_zone_feats = _zone_matrix_from_traj(
                t,
                zone_key='comfort_zone_features',
                scalar_key='comfort_features',
                num_zones=num_zones,
            )
            parts = _split_airl_state_components(states, action_dim=actions.shape[1], num_zones=num_zones)

            zone_temps = parts['zone_temps']
            outdoor_temp = parts['outdoor_temp']
            window_hold_counter = parts['window_hold_counter']
            window_dwell = parts['window_dwell']
            prev_window_state = parts['prev_window_state']
            change_action = actions[:, 0:1]

            state_features = np.concatenate(
                [zone_temps, outdoor_temp, window_hold_counter, window_dwell, prev_window_state],
                axis=1
            )
            traj_states = torch.tensor(state_features, dtype=torch.float32).unsqueeze(0)
            traj_actions = torch.tensor(change_action, dtype=torch.float32).unsqueeze(0)
            traj_features = torch.tensor(comfort_zone_feats, dtype=torch.float32).unsqueeze(0)
        elif reward_type == 'temps_hold_dwell_prev_energy_comfort_zonal_gru':
            # Temperature/window context with per-zone energy and comfort features.
            states = np.array(t["states"])
            actions = np.array(t["actions"])
            num_zones = _infer_airl_num_zones(states.shape[1], actions.shape[1])
            energy_zone_feats = _zone_matrix_from_traj(
                t,
                zone_key='energy_zone_features',
                scalar_key='energy_features',
                num_zones=num_zones,
            )
            comfort_zone_feats = _zone_matrix_from_traj(
                t,
                zone_key='comfort_zone_features',
                scalar_key='comfort_features',
                num_zones=num_zones,
            )
            parts = _split_airl_state_components(states, action_dim=actions.shape[1], num_zones=num_zones)

            zone_temps = parts['zone_temps']
            outdoor_temp = parts['outdoor_temp']
            window_hold_counter = parts['window_hold_counter']
            window_dwell = parts['window_dwell']
            prev_window_state = parts['prev_window_state']
            change_action = actions[:, 0:1]

            state_features = np.concatenate(
                [zone_temps, outdoor_temp, window_hold_counter, window_dwell, prev_window_state],
                axis=1
            )
            traj_states = torch.tensor(state_features, dtype=torch.float32).unsqueeze(0)
            traj_actions = torch.tensor(change_action, dtype=torch.float32).unsqueeze(0)
            traj_features = torch.tensor(
                np.concatenate([energy_zone_feats, comfort_zone_feats], axis=1),
                dtype=torch.float32
            ).unsqueeze(0)
        elif reward_type == 'temps_hold_prev_gru':
            # Zone/outdoor temperatures plus hold/prev-window state, and change action.
            states = np.array(t["states"])
            actions = np.array(t["actions"])
            parts = _split_airl_state_components(states, action_dim=actions.shape[1])

            zone_temps = parts['zone_temps']
            outdoor_temp = parts['outdoor_temp']
            window_hold_counter = parts['window_hold_counter']
            prev_window_state = parts['prev_window_state']
            change_action = actions[:, 0:1]

            state_features = np.concatenate(
                [zone_temps, outdoor_temp, window_hold_counter, prev_window_state],
                axis=1
            )
            traj_states = torch.tensor(state_features, dtype=torch.float32).unsqueeze(0)
            traj_actions = torch.tensor(change_action, dtype=torch.float32).unsqueeze(0)
            traj_features = None
        elif reward_type == 'state_action_comfort_energy_gru':
            # Use full state/action trajectories with comfort and energy features
            states = np.array(t["states"])  # Shape: (T, state_dim)
            actions = np.array(t["actions"])  # Shape: (T, action_dim)
            energy_feats = np.array(t["energy_features"]).reshape(-1, 1)
            comfort_feats = np.array(t["comfort_features"]).reshape(-1, 1)

            traj_states = torch.tensor(states, dtype=torch.float32).unsqueeze(0)  # (1, T, state_dim)
            traj_actions = torch.tensor(actions, dtype=torch.float32).unsqueeze(0)  # (1, T, action_dim)
            traj_features = torch.tensor(
                np.concatenate([comfort_feats, energy_feats], axis=1),
                dtype=torch.float32
            ).unsqueeze(0)  # (1, T, 2)
        else:  # 'full' - original behavior (GRU)
            # Combine states, actions, energy features, and comfort features
            states = np.array(t["states"])  # Shape: (T, 12)
            actions = np.array(t["actions"])  # Shape: (T, 8)
            energy_feats = np.array(t["energy_features"]).reshape(-1, 1)
            comfort_feats = np.array(t["comfort_features"]).reshape(-1, 1)
            
            # For legacy GRU, use all data as features
            traj_states = torch.tensor(states, dtype=torch.float32).unsqueeze(0)  # (1, T, 12)
            traj_actions = torch.tensor(actions, dtype=torch.float32).unsqueeze(0)  # (1, T, 8)
            traj_features = torch.tensor(
                np.concatenate([energy_feats, comfort_feats], axis=1),
                dtype=torch.float32
            ).unsqueeze(0)  # (1, T, 2)
            
            feats = torch.tensor(
                np.concatenate([states, actions, energy_feats, comfort_feats], axis=1),
                dtype=torch.float32
            ).unsqueeze(0)  # (1, T, d) - For legacy compatibility
            
        # Move tensors to the reward function device
        if traj_states is not None:
            traj_states = traj_states.to(device)
        if traj_actions is not None:
            traj_actions = traj_actions.to(device)
        if traj_features is not None:
            traj_features = traj_features.to(device)
        if feats is not None:
            feats = feats.to(device)

        # Compute per-transition logits based on reward function type
        if reward_type in ['energy_comfort_only', 'comfort_energy_window', 'window_conditional_comfort_energy',
                           'window_conditional_energy_comfort', 'window_conditional_comfort_energy_both', 'window_only_mlp',
                           'temps_hold_dwell_prev_energy_gru', 'temps_hold_dwell_prev_comfort_gru',
                           'temps_hold_dwell_prev_energy_comfort_gru',
                           'temps_hold_dwell_prev_energy_zonal_gru', 'temps_hold_dwell_prev_comfort_zonal_gru',
                           'temps_hold_dwell_prev_energy_comfort_zonal_gru'] or reward_type in AIRL_TIME_AUGMENTED_REWARD_TYPES:
            logits = reward_fn(traj_states, traj_actions, traj_features)
        elif reward_type in [
            'state_action_only_mlp',
            'state_action_only_gru',
            'temps_hold_dwell_prev_gru',
            'temps_hold_prev_gru',
        ]:
            logits = reward_fn(traj_states, traj_actions)
        elif 'gru' in reward_type or reward_type in ('window_only', 'full'):
            logits = reward_fn(traj_states, traj_actions, traj_features)
        else:
            logits = reward_fn(feats)

        logits_list.append(logits.reshape(-1))

    if logits_list:
        return torch.cat(logits_list)
    return torch.empty(0, device=device)


def update_reward(reward_fn: nn.Module, 
                 expert_trajs: List[Dict], 
                 policy_trajs: List[Dict], 
                 lr: float = 1e-4, 
                 weight_decay: float = 1e-4, 
                 num_steps: int = 10,
                 reward_type: str = 'full',
                 grad_clip_max_norm: float = 2.0) -> None:
    """
    Update reward function parameters using AIRL objective.
    
    Args:
        reward_fn: Reward function to update
        expert_trajs: Expert trajectory data
        policy_trajs: Policy trajectory data
        lr: Learning rate
        weight_decay: Weight decay
        num_steps: Number of update steps
    """
    optimizer = torch.optim.Adam(reward_fn.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    device = next(reward_fn.parameters()).device

    for step in range(num_steps):
        logits_exp = traj_scores(expert_trajs, reward_fn, reward_type)
        logits_pol = traj_scores(policy_trajs, reward_fn, reward_type)

        if logits_exp.numel() == 0 or logits_pol.numel() == 0:
            raise RuntimeError("Encountered empty logits when updating reward function. "
                               "Check trajectory collection and reward_type preprocessing.")

        # Binary cross-entropy on logits (expert=1, policy=0)
        target_exp = torch.ones_like(logits_exp, device=device)
        target_pol = torch.zeros_like(logits_pol, device=device)

        loss_exp = criterion(logits_exp, target_exp)
        loss_pol = criterion(logits_pol, target_pol)
        loss = loss_exp + loss_pol

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # Add gradient clipping for reward function training as well
        grad_norm = torch.nn.utils.clip_grad_norm_(reward_fn.parameters(), max_norm=grad_clip_max_norm)
        if torch.isnan(grad_norm) or grad_norm > grad_clip_max_norm * 5.0:
            print(f"Warning: Large reward gradient norm detected: {grad_norm.item():.4f}")
            
        optimizer.step()

        with torch.no_grad():
            d_exp = torch.sigmoid(logits_exp).mean().item()
            d_pol = torch.sigmoid(logits_pol).mean().item()

        print(
            f"Reward Update Step {step + 1}, "
            f"Loss: {loss.item():.4f} "
            f"(exp={loss_exp.item():.4f}, pol={loss_pol.item():.4f}), "
            f"D(exp)={d_exp:.3f}, D(pol)={d_pol:.3f}"
        )


def discriminator_mean_probs(
    reward_fn: nn.Module,
    expert_trajs: List[Dict],
    policy_trajs: List[Dict],
    reward_type: str,
) -> Tuple[float, float]:
    """Return mean discriminator probabilities for expert and policy trajectories."""
    with torch.no_grad():
        logits_exp = traj_scores(expert_trajs, reward_fn, reward_type)
        logits_pol = traj_scores(policy_trajs, reward_fn, reward_type)

    if logits_exp.numel() == 0 or logits_pol.numel() == 0:
        raise RuntimeError(
            "Encountered empty logits when evaluating discriminator separation. "
            "Check trajectory collection and reward_type preprocessing."
        )

    d_exp = float(torch.sigmoid(logits_exp).mean().item())
    d_pol = float(torch.sigmoid(logits_pol).mean().item())
    return d_exp, d_pol


def compute_gae(rewards: np.ndarray, 
               values: np.ndarray, 
               dones: np.ndarray, 
               gamma: float = 0.99, 
               lam: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Generalized Advantage Estimation.
    
    Args:
        rewards: Reward sequence
        values: Value estimates
        dones: Done flags
        gamma: Discount factor
        lam: GAE lambda parameter
        
    Returns:
        Tuple of (returns, advantages)
    """
    advantages = np.zeros_like(rewards, dtype=np.float32)
    returns = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0.0
            next_done = 1.0
        else:
            next_value = values[t + 1]
            next_done = dones[t + 1]

        delta = rewards[t] + gamma * next_value * (1 - next_done) - values[t]
        gae = delta + gamma * lam * gae * (1 - next_done)
        advantages[t] = gae
        returns[t] = advantages[t] + values[t]

    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return returns, advantages


def evaluate_logp(policy: MMVPolicyActorCritic, 
                 states: torch.Tensor, 
                 w_change: torch.Tensor, 
                 lc: torch.Tensor, 
                 sup: torch.Tensor,
                 window_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Evaluate log probabilities under current policy.
    
    Args:
        policy: Policy network
        states: State tensor (B, state_dim)
        w_change: Window change actions (B,)
        lc: Local cooling actions (B, 2)
        sup: Supply temperature actions (B, num_zones)
        window_state: Resulting window mode after action (B,) where 0=AC, 1=NV
        
    Returns:
        Tuple of (log_probs, values, entropy)
    """
    out = policy(states)
    window_state = window_state.to(states.device)
    w_change = w_change.to(states.device)

    # Window change probabilities
    w_dist = torch.distributions.Bernoulli(logits=out["change_logit"])
    logp_w = w_dist.log_prob(w_change)

    # Masks for different modes
    mask_nv = (window_state > 0.5)   # Natural ventilation
    mask_ac = (window_state <= 0.5)  # Air conditioning

    # Local cooling probabilities (only for NV mode)
    lc_dist = torch.distributions.Bernoulli(logits=out["lc_logits_nv"])
    
    # Supply temperature probabilities (only for AC mode)
    sup_mu = out["supply_mu_ac"]
    sup_std = policy.log_std_ac.exp()
    sup_dist = torch.distributions.Normal(sup_mu, sup_std)
    
    lc_logp = lc_dist.log_prob(lc).sum(-1)
    sup_logp = sup_dist.log_prob(sup).sum(-1)

    mask_nv_f = mask_nv.to(logp_w.dtype)
    mask_ac_f = mask_ac.to(logp_w.dtype)

    logp = logp_w + mask_nv_f * lc_logp + mask_ac_f * sup_logp

    entropy_w = w_dist.entropy()
    entropy_lc = lc_dist.entropy().sum(-1)
    entropy_sup = sup_dist.entropy().sum(-1)
    entropy = entropy_w + mask_nv_f * entropy_lc + mask_ac_f * entropy_sup
    value = out["value"]

    return logp, value, entropy


def train_ppo_irl(env, 
                 validation_env, 
                 policy: MMVPolicyActorCritic, 
                 reward_fn: nn.Module, 
                 optimizer: torch.optim.Optimizer,
                 outer_iter: int,
                 batch_days: int = 5, 
                 epochs: int = 10, 
                 updates: int = 500,
                 gamma: float = 0.99, 
                 lam: float = 0.95, 
                 clip_eps: float = 0.2, 
                 val_days: int = 5,
                 reward_type: str = 'full',
                 grad_clip_max_norm: float = 2.0,
                 entropy_coeff: float = 0.005) -> None:
    """
    Train policy using PPO with learned reward function.
    
    Args:
        env: Training environment
        validation_env: Validation environment
        policy: Policy network
        reward_fn: Learned reward function
        optimizer: Policy optimizer
        outer_iter: Outer iteration number (for saving)
        batch_days: Number of days per batch
        epochs: PPO epochs per update
        updates: Number of PPO updates
        gamma: Discount factor
        lam: GAE lambda
        clip_eps: PPO clipping parameter
        val_days: Validation days
    """
    for update in range(updates):
        # Initialize rollout buffer
        buffer = {k: [] for k in ("states", "w_act", "window_state", "lc_act", "s_act", "logp_old", "rewards", "values", "dones")}

        # Collect rollouts
        import random
        num_train_days = len(env.daily_data_list)
        if batch_days >= num_train_days:
            day_indices = list(range(num_train_days))
        else:
            day_indices = random.sample(range(num_train_days), batch_days)

        for d in day_indices:
            s = env.reset(day_index=d)
            done, ep_ret = False, 0.0

            while not done:
                s_t = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
                a_dict, logp_old = policy.get_action(s_t)

                lc_binary = np.array(a_dict["local_cooling"], dtype=np.float32)
                action_env = {
                    "change": a_dict["change"],
                    "supply_temps": np.array(a_dict["supply_temps"], copy=True),
                    "local_cooling": np.array(lc_binary, copy=True)
                }

                next_s_env, reward, done, info = env.step(action_env)
                # Keep reward semantics consistent with collected trajectories:
                # reward input is (state_t, action_t, transition_features_t->t+1).
                next_s = s

                # Compute reward using learned reward function based on reward type
                if reward_type == 'window_only':
                    # Only use window-related states, window action, energy, and comfort
                    window_states = next_s[-4:]  # window_hold_counter, switch_flag, dwell, window_prev_state
                    window_action = [flatten_action(a_dict)[0]]  # change action
                    
                    # Create tensors for separate normalization (GRU format)
                    states_t = torch.tensor(np.array([window_states]), dtype=torch.float32).unsqueeze(0)  # (1, 1, 4)
                    actions_t = torch.tensor(np.array([window_action]), dtype=torch.float32).unsqueeze(0)  # (1, 1, 1)
                    features_t = torch.tensor(
                        np.array([[info['energy_consumption'], info['comfort_violations']]]),
                        dtype=torch.float32
                    ).unsqueeze(0)  # (1, 1, 2)
                    
                    # Legacy compatibility
                    reward_input = torch.tensor(
                        np.concatenate([
                            window_states,
                            window_action,
                            [info['energy_consumption']],
                            [info['comfort_violations']]
                        ]),
                        dtype=torch.float32
                    ).unsqueeze(0).unsqueeze(0)  # (1, 1, 7)
                elif reward_type == 'window_only_mlp':
                    # Only use window-related states, window action, energy, and comfort (MLP version)
                    window_states = next_s[-4:]  # window_hold_counter, switch_flag, dwell, window_prev_state
                    window_action = [flatten_action(a_dict)[0]]  # change action
                    reward_input = torch.tensor(
                        np.concatenate([
                            window_states,
                            window_action,
                            [info['energy_consumption']],
                            [info['comfort_violations']]
                        ]),
                        dtype=torch.float32
                    ).unsqueeze(0).unsqueeze(0)  # (1, 1, 7)
                elif reward_type == 'energy_comfort_only':
                    # Only use energy and comfort features - separate normalization
                    states_t = torch.tensor(np.zeros((1, 0)), dtype=torch.float32).unsqueeze(0)  # Empty states
                    actions_t = torch.tensor(np.zeros((1, 0)), dtype=torch.float32).unsqueeze(0)  # Empty actions
                    features_t = torch.tensor(
                        np.array([[info['energy_consumption'], info['comfort_violations']]]),
                        dtype=torch.float32
                    ).unsqueeze(0)  # (1, 1, 2)
                elif reward_type == 'comfort_energy_window':
                    # Use comfort, energy, window dwell, and window action features - separate normalization
                    states_t = torch.tensor(np.array([[next_s[10]]]), dtype=torch.float32).unsqueeze(0)  # Window dwell state
                    actions_t = torch.tensor(np.array([[flatten_action(a_dict)[0]]]), dtype=torch.float32).unsqueeze(0)  # Window action
                    features_t = torch.tensor(
                        np.array([[info['comfort_violations'], info['energy_consumption']]]),
                        dtype=torch.float32
                    ).unsqueeze(0)  # (1, 1, 2)
                elif reward_type == 'window_conditional_comfort_energy':
                    # Use all window-related states, conditional comfort, and energy - separate normalization
                    window_states = next_s[-4:]  # window_hold_counter, switch_flag, dwell, window_prev_state
                    window_action = flatten_action(a_dict)[0]  # change action
                    
                    # Calculate conditional comfort: prev_window_state * comfort_violations
                    prev_window_state = next_s[-1]  # window_prev_state
                    conditional_comfort = prev_window_state * info['comfort_violations']
                    
                    states_t = torch.tensor(np.array([window_states]), dtype=torch.float32).unsqueeze(0)  # Window states
                    actions_t = torch.tensor(np.array([[window_action]]), dtype=torch.float32).unsqueeze(0)  # Window action
                    features_t = torch.tensor(
                        np.array([[conditional_comfort, info['energy_consumption']]]),
                        dtype=torch.float32
                    ).unsqueeze(0)  # (1, 1, 2)
                elif reward_type == 'window_conditional_energy_comfort':
                    # Use all window-related states, conditional energy, and comfort - separate normalization
                    window_states = next_s[-4:]  # window_hold_counter, switch_flag, dwell, window_prev_state
                    window_action = flatten_action(a_dict)[0]  # change action
                    
                    # Calculate conditional energy: (1 - prev_window_state) * energy_consumption
                    prev_window_state = next_s[-1]  # window_prev_state
                    conditional_energy = (1 - prev_window_state) * info['energy_consumption']
                    
                    states_t = torch.tensor(np.array([window_states]), dtype=torch.float32).unsqueeze(0)  # Window states
                    actions_t = torch.tensor(np.array([[window_action]]), dtype=torch.float32).unsqueeze(0)  # Window action
                    features_t = torch.tensor(
                        np.array([[conditional_energy, info['comfort_violations']]]),
                        dtype=torch.float32
                    ).unsqueeze(0)  # (1, 1, 2)
                elif reward_type == 'window_conditional_comfort_energy_both':
                    # Use all window-related states, both conditional comfort and conditional energy - separate normalization
                    window_states = next_s[-4:]  # window_hold_counter, switch_flag, dwell, window_prev_state
                    window_action = flatten_action(a_dict)[0]  # change action
                    
                    # Calculate both conditional features based on window state
                    prev_window_state = next_s[-1]  # window_prev_state
                    conditional_comfort = prev_window_state * info['comfort_violations']  # active when window open
                    conditional_energy = (1 - prev_window_state) * info['energy_consumption']  # active when window closed
                    
                    states_t = torch.tensor(np.array([window_states]), dtype=torch.float32).unsqueeze(0)  # Window states
                    actions_t = torch.tensor(np.array([[window_action]]), dtype=torch.float32).unsqueeze(0)  # Window action
                    features_t = torch.tensor(
                        np.array([[conditional_comfort, conditional_energy]]),
                        dtype=torch.float32
                    ).unsqueeze(0)  # (1, 1, 2)
                elif reward_type == 'energy_comfort_only_gru':
                    # GRU version: Only use energy and comfort features
                    states_t = torch.zeros((1, 1, 0), dtype=torch.float32)  # Empty states
                    actions_t = torch.zeros((1, 1, 0), dtype=torch.float32)  # Empty actions
                    features_t = torch.tensor(
                        np.array([[info['energy_consumption'], info['comfort_violations']]]),
                        dtype=torch.float32
                    ).unsqueeze(0)  # (1, 1, 2)
                    
                    reward_input = torch.tensor(
                        np.concatenate([
                            [info['energy_consumption']],
                            [info['comfort_violations']]
                        ]),
                        dtype=torch.float32
                    ).unsqueeze(0).unsqueeze(0)  # (1, 1, 2)
                elif reward_type == 'comfort_energy_window_gru':
                    # GRU version: Use comfort, energy, window dwell, and window action features
                    window_dwell = [next_s[10]]  # window dwell time
                    window_action = [flatten_action(a_dict)[0]]  # change action
                    
                    states_t = torch.tensor(np.array([window_dwell]), dtype=torch.float32).unsqueeze(0)  # (1, 1, 1)
                    actions_t = torch.tensor(np.array([window_action]), dtype=torch.float32).unsqueeze(0)  # (1, 1, 1)
                    features_t = torch.tensor(
                        np.array([[info['comfort_violations'], info['energy_consumption']]]),
                        dtype=torch.float32
                    ).unsqueeze(0)  # (1, 1, 2)
                    
                    reward_input = torch.tensor(
                        np.concatenate([
                            [info['comfort_violations']],
                            [info['energy_consumption']],
                            window_dwell,
                            window_action
                        ]),
                        dtype=torch.float32
                    ).unsqueeze(0).unsqueeze(0)  # (1, 1, 4)
                elif reward_type == 'window_conditional_comfort_energy_gru':
                    # GRU version: Use all window-related states, conditional comfort, and energy
                    window_states = next_s[-4:]  # window_hold_counter, switch_flag, dwell, window_prev_state
                    window_action = flatten_action(a_dict)[0]  # change action
                    
                    # Calculate conditional comfort: prev_window_state * comfort_violations
                    prev_window_state = next_s[-1]  # window_prev_state
                    conditional_comfort = prev_window_state * info['comfort_violations']
                    
                    states_t = torch.tensor(np.array([window_states]), dtype=torch.float32).unsqueeze(0)  # (1, 1, 4)
                    actions_t = torch.tensor(np.array([[window_action]]), dtype=torch.float32).unsqueeze(0)  # (1, 1, 1)
                    features_t = torch.tensor(
                        np.array([[conditional_comfort, info['energy_consumption']]]),
                        dtype=torch.float32
                    ).unsqueeze(0)  # (1, 1, 2)
                    
                    reward_input = torch.tensor(
                        np.concatenate([
                            window_states,
                            [window_action],
                            [conditional_comfort],
                            [info['energy_consumption']]
                        ]),
                        dtype=torch.float32
                    ).unsqueeze(0).unsqueeze(0)  # (1, 1, 7)
                elif reward_type == 'window_conditional_energy_comfort_gru':
                    # GRU version: Use all window-related states, conditional energy, and comfort
                    window_states = next_s[-4:]  # window_hold_counter, switch_flag, dwell, window_prev_state
                    window_action = flatten_action(a_dict)[0]  # change action
                    
                    # Calculate conditional energy: (1 - prev_window_state) * energy_consumption
                    prev_window_state = next_s[-1]  # window_prev_state
                    conditional_energy = (1 - prev_window_state) * info['energy_consumption']
                    
                    states_t = torch.tensor(np.array([window_states]), dtype=torch.float32).unsqueeze(0)  # (1, 1, 4)
                    actions_t = torch.tensor(np.array([[window_action]]), dtype=torch.float32).unsqueeze(0)  # (1, 1, 1)
                    features_t = torch.tensor(
                        np.array([[conditional_energy, info['comfort_violations']]]),
                        dtype=torch.float32
                    ).unsqueeze(0)  # (1, 1, 2)
                    
                    reward_input = torch.tensor(
                        np.concatenate([
                            window_states,
                            [window_action],
                            [conditional_energy],
                            [info['comfort_violations']]
                        ]),
                        dtype=torch.float32
                    ).unsqueeze(0).unsqueeze(0)  # (1, 1, 7)
                elif reward_type == 'window_conditional_comfort_energy_both_gru':
                    # GRU version: Use all window-related states, both conditional comfort and conditional energy
                    window_states = next_s[-4:]  # window_hold_counter, switch_flag, dwell, window_prev_state
                    window_action = flatten_action(a_dict)[0]  # change action
                    
                    # Calculate both conditional features based on window state
                    prev_window_state = next_s[-1]  # window_prev_state
                    conditional_comfort = prev_window_state * info['comfort_violations']  # active when window open
                    conditional_energy = (1 - prev_window_state) * info['energy_consumption']  # active when window closed
                    
                    states_t = torch.tensor(np.array([window_states]), dtype=torch.float32).unsqueeze(0)  # (1, 1, 4)
                    actions_t = torch.tensor(np.array([[window_action]]), dtype=torch.float32).unsqueeze(0)  # (1, 1, 1)
                    features_t = torch.tensor(
                        np.array([[conditional_comfort, conditional_energy]]),
                        dtype=torch.float32
                    ).unsqueeze(0)  # (1, 1, 2)
                    
                    reward_input = torch.tensor(
                        np.concatenate([
                            window_states,
                            [window_action],
                            [conditional_comfort],
                            [conditional_energy]
                        ]),
                        dtype=torch.float32
                    ).unsqueeze(0).unsqueeze(0)  # (1, 1, 7)
                elif reward_type == 'state_action_only_mlp':
                    # Use only states and actions (no energy/comfort features) - MLP version
                    states_t = torch.tensor(np.array([next_s]), dtype=torch.float32)  # (1, 12)
                    actions_t = torch.tensor(np.array([flatten_action(a_dict)]), dtype=torch.float32)  # (1, 8)
                    features_t = None
                elif reward_type == 'state_action_only_gru':
                    # Use only states and actions (no energy/comfort features) - GRU version
                    states_t = torch.tensor(np.array([next_s]), dtype=torch.float32).unsqueeze(0)  # (1, 1, 12)
                    actions_t = torch.tensor(np.array([flatten_action(a_dict)]), dtype=torch.float32).unsqueeze(0)  # (1, 1, 8)
                    features_t = None
                elif reward_type == 'temps_hold_dwell_prev_gru':
                    parts = _split_airl_state_components(next_s, num_zones=len(a_dict["supply_temps"]))
                    zone_temps = parts['zone_temps']
                    outdoor_temp = _state_scalar(parts['outdoor_temp'])
                    window_hold_counter = _state_scalar(parts['window_hold_counter'])
                    window_dwell = _state_scalar(parts['window_dwell'])
                    prev_window_state = _state_scalar(parts['prev_window_state'])

                    state_features = np.concatenate(
                        [zone_temps, [outdoor_temp], [window_hold_counter], [window_dwell], [prev_window_state]]
                    )
                    states_t = torch.tensor(state_features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    actions_t = torch.tensor(np.array([[flatten_action(a_dict)[0]]]), dtype=torch.float32).unsqueeze(0)
                    features_t = None
                elif reward_type in AIRL_TIME_AUGMENTED_REWARD_TYPES:
                    state_features, window_action, feature_values = _time_augmented_step_features(
                        next_s,
                        a_dict,
                        env,
                        info,
                        reward_type,
                    )
                    states_t = torch.tensor(state_features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    actions_t = torch.tensor(window_action, dtype=torch.float32).unsqueeze(0)
                    features_t = torch.tensor(feature_values, dtype=torch.float32).unsqueeze(0)
                elif reward_type == 'temps_hold_dwell_prev_energy_gru':
                    parts = _split_airl_state_components(next_s, num_zones=len(a_dict["supply_temps"]))
                    zone_temps = parts['zone_temps']
                    outdoor_temp = _state_scalar(parts['outdoor_temp'])
                    window_hold_counter = _state_scalar(parts['window_hold_counter'])
                    window_dwell = _state_scalar(parts['window_dwell'])
                    prev_window_state = _state_scalar(parts['prev_window_state'])

                    state_features = np.concatenate(
                        [zone_temps, [outdoor_temp], [window_hold_counter], [window_dwell], [prev_window_state]]
                    )
                    states_t = torch.tensor(state_features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    actions_t = torch.tensor(np.array([[flatten_action(a_dict)[0]]]), dtype=torch.float32).unsqueeze(0)
                    features_t = torch.tensor(
                        np.array([[info['energy_consumption']]]),
                        dtype=torch.float32
                    ).unsqueeze(0)
                elif reward_type == 'temps_hold_dwell_prev_comfort_gru':
                    parts = _split_airl_state_components(next_s, num_zones=len(a_dict["supply_temps"]))
                    zone_temps = parts['zone_temps']
                    outdoor_temp = _state_scalar(parts['outdoor_temp'])
                    window_hold_counter = _state_scalar(parts['window_hold_counter'])
                    window_dwell = _state_scalar(parts['window_dwell'])
                    prev_window_state = _state_scalar(parts['prev_window_state'])

                    state_features = np.concatenate(
                        [zone_temps, [outdoor_temp], [window_hold_counter], [window_dwell], [prev_window_state]]
                    )
                    states_t = torch.tensor(state_features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    actions_t = torch.tensor(np.array([[flatten_action(a_dict)[0]]]), dtype=torch.float32).unsqueeze(0)
                    features_t = torch.tensor(
                        np.array([[info['comfort_violations']]]),
                        dtype=torch.float32
                    ).unsqueeze(0)
                elif reward_type == 'temps_hold_dwell_prev_energy_comfort_gru':
                    parts = _split_airl_state_components(next_s, num_zones=len(a_dict["supply_temps"]))
                    zone_temps = parts['zone_temps']
                    outdoor_temp = _state_scalar(parts['outdoor_temp'])
                    window_hold_counter = _state_scalar(parts['window_hold_counter'])
                    window_dwell = _state_scalar(parts['window_dwell'])
                    prev_window_state = _state_scalar(parts['prev_window_state'])

                    state_features = np.concatenate(
                        [zone_temps, [outdoor_temp], [window_hold_counter], [window_dwell], [prev_window_state]]
                    )
                    states_t = torch.tensor(state_features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    actions_t = torch.tensor(np.array([[flatten_action(a_dict)[0]]]), dtype=torch.float32).unsqueeze(0)
                    features_t = torch.tensor(
                        np.array([[info['energy_consumption'], info['comfort_violations']]]),
                        dtype=torch.float32
                    ).unsqueeze(0)
                elif reward_type == 'temps_hold_dwell_prev_energy_zonal_gru':
                    num_zones = len(a_dict["supply_temps"])
                    parts = _split_airl_state_components(next_s, num_zones=num_zones)
                    zone_temps = parts['zone_temps']
                    outdoor_temp = _state_scalar(parts['outdoor_temp'])
                    window_hold_counter = _state_scalar(parts['window_hold_counter'])
                    window_dwell = _state_scalar(parts['window_dwell'])
                    prev_window_state = _state_scalar(parts['prev_window_state'])
                    energy_zone = _zone_vector_from_info(
                        info,
                        scalar_key='energy_consumption',
                        zone_key='energy_consumption_zones',
                        num_zones=num_zones,
                    )

                    state_features = np.concatenate(
                        [zone_temps, [outdoor_temp], [window_hold_counter], [window_dwell], [prev_window_state]]
                    )
                    states_t = torch.tensor(state_features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    actions_t = torch.tensor(np.array([[flatten_action(a_dict)[0]]]), dtype=torch.float32).unsqueeze(0)
                    features_t = torch.tensor(np.array([energy_zone]), dtype=torch.float32).unsqueeze(0)
                elif reward_type == 'temps_hold_dwell_prev_comfort_zonal_gru':
                    num_zones = len(a_dict["supply_temps"])
                    parts = _split_airl_state_components(next_s, num_zones=num_zones)
                    zone_temps = parts['zone_temps']
                    outdoor_temp = _state_scalar(parts['outdoor_temp'])
                    window_hold_counter = _state_scalar(parts['window_hold_counter'])
                    window_dwell = _state_scalar(parts['window_dwell'])
                    prev_window_state = _state_scalar(parts['prev_window_state'])
                    comfort_zone = _zone_vector_from_info(
                        info,
                        scalar_key='comfort_violations',
                        zone_key='comfort_violations_zones',
                        num_zones=num_zones,
                    )

                    state_features = np.concatenate(
                        [zone_temps, [outdoor_temp], [window_hold_counter], [window_dwell], [prev_window_state]]
                    )
                    states_t = torch.tensor(state_features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    actions_t = torch.tensor(np.array([[flatten_action(a_dict)[0]]]), dtype=torch.float32).unsqueeze(0)
                    features_t = torch.tensor(np.array([comfort_zone]), dtype=torch.float32).unsqueeze(0)
                elif reward_type == 'temps_hold_dwell_prev_energy_comfort_zonal_gru':
                    num_zones = len(a_dict["supply_temps"])
                    parts = _split_airl_state_components(next_s, num_zones=num_zones)
                    zone_temps = parts['zone_temps']
                    outdoor_temp = _state_scalar(parts['outdoor_temp'])
                    window_hold_counter = _state_scalar(parts['window_hold_counter'])
                    window_dwell = _state_scalar(parts['window_dwell'])
                    prev_window_state = _state_scalar(parts['prev_window_state'])
                    energy_zone = _zone_vector_from_info(
                        info,
                        scalar_key='energy_consumption',
                        zone_key='energy_consumption_zones',
                        num_zones=num_zones,
                    )
                    comfort_zone = _zone_vector_from_info(
                        info,
                        scalar_key='comfort_violations',
                        zone_key='comfort_violations_zones',
                        num_zones=num_zones,
                    )

                    state_features = np.concatenate(
                        [zone_temps, [outdoor_temp], [window_hold_counter], [window_dwell], [prev_window_state]]
                    )
                    states_t = torch.tensor(state_features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    actions_t = torch.tensor(np.array([[flatten_action(a_dict)[0]]]), dtype=torch.float32).unsqueeze(0)
                    features_t = torch.tensor(
                        np.array([np.concatenate([energy_zone, comfort_zone], axis=0)]),
                        dtype=torch.float32
                    ).unsqueeze(0)
                elif reward_type == 'temps_hold_prev_gru':
                    parts = _split_airl_state_components(next_s, num_zones=len(a_dict["supply_temps"]))
                    zone_temps = parts['zone_temps']
                    outdoor_temp = _state_scalar(parts['outdoor_temp'])
                    window_hold_counter = _state_scalar(parts['window_hold_counter'])
                    prev_window_state = _state_scalar(parts['prev_window_state'])

                    state_features = np.concatenate(
                        [zone_temps, [outdoor_temp], [window_hold_counter], [prev_window_state]]
                    )
                    states_t = torch.tensor(state_features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    actions_t = torch.tensor(np.array([[flatten_action(a_dict)[0]]]), dtype=torch.float32).unsqueeze(0)
                    features_t = None
                elif reward_type == 'temps_comfort_energy_window_gru':
                    parts = _split_airl_state_components(next_s, num_zones=len(a_dict["supply_temps"]))
                    zone_temps = parts['zone_temps']
                    outdoor_temp = _state_scalar(parts['outdoor_temp'])
                    window_dwell = _state_scalar(parts['window_dwell'])

                    state_features = np.concatenate([zone_temps, [outdoor_temp], [window_dwell]])

                    states_t = torch.tensor(state_features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, zone_dim+2)
                    actions_t = torch.tensor(np.array([[flatten_action(a_dict)[0]]]), dtype=torch.float32).unsqueeze(0)  # (1, 1, 1)
                    features_t = torch.tensor(
                        np.array([[info['comfort_violations'], info['energy_consumption']]]),
                        dtype=torch.float32
                    ).unsqueeze(0)  # (1, 1, 2)
                elif reward_type == 'temps_comfort_window_gru':
                    parts = _split_airl_state_components(next_s, num_zones=len(a_dict["supply_temps"]))
                    zone_temps = parts['zone_temps']
                    outdoor_temp = _state_scalar(parts['outdoor_temp'])
                    window_dwell = _state_scalar(parts['window_dwell'])

                    state_features = np.concatenate([zone_temps, [outdoor_temp], [window_dwell]])

                    states_t = torch.tensor(state_features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    actions_t = torch.tensor(np.array([[flatten_action(a_dict)[0]]]), dtype=torch.float32).unsqueeze(0)
                    features_t = torch.tensor(
                        np.array([[info['comfort_violations']]]),
                        dtype=torch.float32
                    ).unsqueeze(0)
                elif reward_type == 'temps_energy_window_gru':
                    parts = _split_airl_state_components(next_s, num_zones=len(a_dict["supply_temps"]))
                    zone_temps = parts['zone_temps']
                    outdoor_temp = _state_scalar(parts['outdoor_temp'])
                    window_dwell = _state_scalar(parts['window_dwell'])

                    state_features = np.concatenate([zone_temps, [outdoor_temp], [window_dwell]])

                    states_t = torch.tensor(state_features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    actions_t = torch.tensor(np.array([[flatten_action(a_dict)[0]]]), dtype=torch.float32).unsqueeze(0)
                    features_t = torch.tensor(
                        np.array([[info['energy_consumption']]]),
                        dtype=torch.float32
                    ).unsqueeze(0)
                elif reward_type == 'temps_window_gru':
                    parts = _split_airl_state_components(next_s, num_zones=len(a_dict["supply_temps"]))
                    zone_temps = parts['zone_temps']
                    outdoor_temp = _state_scalar(parts['outdoor_temp'])
                    window_dwell = _state_scalar(parts['window_dwell'])
                    prev_window_state = _state_scalar(parts['prev_window_state'])

                    state_features = np.concatenate([zone_temps, [outdoor_temp], [window_dwell]])

                    states_t = torch.tensor(state_features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    actions_t = torch.tensor(np.array([[flatten_action(a_dict)[0]]]), dtype=torch.float32).unsqueeze(0)
                    features_t = torch.tensor(
                        np.array([[prev_window_state]], dtype=np.float32),
                        dtype=torch.float32
                    ).unsqueeze(0)
                elif reward_type == 'state_action_comfort_energy_gru':
                    # Use full states/actions with comfort and energy features
                    states_t = torch.tensor(np.array([next_s]), dtype=torch.float32).unsqueeze(0)  # (1, 1, 12)
                    actions_t = torch.tensor(np.array([flatten_action(a_dict)]), dtype=torch.float32).unsqueeze(0)  # (1, 1, 8)
                    features_t = torch.tensor(
                        np.array([[info['comfort_violations'], info['energy_consumption']]]),
                        dtype=torch.float32
                    ).unsqueeze(0)  # (1, 1, 2)
                else:  # 'full' - original behavior (GRU)
                    states_t = torch.tensor(np.array([next_s]), dtype=torch.float32).unsqueeze(0)  # (1, 1, 12)
                    actions_t = torch.tensor(np.array([flatten_action(a_dict)]), dtype=torch.float32).unsqueeze(0)  # (1, 1, 8)
                    features_t = torch.tensor(
                        np.array([[info['energy_consumption'], info['comfort_violations']]]),
                        dtype=torch.float32
                    ).unsqueeze(0)  # (1, 1, 2)
                    
                    reward_input = torch.tensor(
                        np.concatenate([
                            next_s, 
                            flatten_action(a_dict),
                            [info['energy_consumption']],
                            [info['comfort_violations']]
                        ]),
                        dtype=torch.float32
                    ).unsqueeze(0).unsqueeze(0)  # (1, 1, d) for GRU input
                
                # Compute reward based on reward function type
                if reward_type in ['energy_comfort_only', 'comfort_energy_window', 'window_conditional_comfort_energy', 'window_conditional_energy_comfort', 'window_conditional_comfort_energy_both', 'window_only_mlp', 'temps_hold_dwell_prev_energy_gru', 'temps_hold_dwell_prev_comfort_gru', 'temps_hold_dwell_prev_energy_comfort_gru', 'temps_hold_dwell_prev_energy_zonal_gru', 'temps_hold_dwell_prev_comfort_zonal_gru', 'temps_hold_dwell_prev_energy_comfort_zonal_gru'] or reward_type in AIRL_TIME_AUGMENTED_REWARD_TYPES:
                    # Use separate normalization for MLP reward functions
                    r_t = reward_fn(states_t, actions_t, features_t).item()
                elif reward_type in [
                    'state_action_only_mlp',
                    'state_action_only_gru',
                    'temps_hold_dwell_prev_gru',
                    'temps_hold_prev_gru',
                ]:
                    # Use only states and actions (no features)
                    r_t = reward_fn(states_t, actions_t).item()
                elif 'gru' in reward_type or reward_type in ('window_only', 'full'):
                    # GRU reward functions, window_only, and window_only_mlp use separate normalization
                    r_t = reward_fn(states_t, actions_t, features_t).item()
                else:
                    # Legacy: Use concatenated input for original GRU reward functions
                    r_t = reward_fn(reward_input).item()

                ep_ret += r_t

                # Store transition
                buffer["states"].append(s)
                buffer["w_act"].append(a_dict["change"])
                buffer["lc_act"].append(a_dict["local_cooling"])
                buffer["s_act"].append(a_dict["supply_temps"])
                buffer["window_state"].append(info.get("window_state", env.window_prev_state))
                buffer["logp_old"].append(logp_old.item())
                buffer["values"].append(0.0)  # Placeholder, filled later
                buffer["rewards"].append(r_t)
                buffer["dones"].append(done)

                s = next_s_env

        # Convert buffer to tensors
        states_t = torch.tensor(np.array(buffer["states"]), dtype=torch.float32)
        w_t = torch.tensor(buffer["w_act"], dtype=torch.float32)
        window_mode_t = torch.tensor(buffer["window_state"], dtype=torch.float32)
        lc_t = torch.tensor(np.array(buffer["lc_act"]), dtype=torch.float32)
        s_t = torch.tensor(np.array(buffer["s_act"]), dtype=torch.float32)

        # Compute value estimates
        with torch.no_grad():
            _, v_pred, _ = evaluate_logp(policy, states_t, w_t, lc_t, s_t, window_mode_t)
        buffer["values"] = v_pred.squeeze().cpu().numpy()

        # Compute advantages and returns
        returns, adv = compute_gae(
            np.array(buffer["rewards"]),
            buffer["values"],
            np.array(buffer["dones"]),
            gamma, lam
        )

        # PPO update tensors
        returns_t = torch.tensor(returns, dtype=torch.float32)
        adv_t = torch.tensor(adv, dtype=torch.float32)
        logp_old_t = torch.tensor(buffer["logp_old"], dtype=torch.float32)

        # PPO updates
        for _ in range(epochs):
            logp_new, value_pred, entropy = evaluate_logp(policy, states_t, w_t, lc_t, s_t, window_mode_t)

            # PPO loss with entropy regularization
            ratio = torch.exp(logp_new - logp_old_t)
            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_t
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.MSELoss()(value_pred.squeeze(), returns_t)
            entropy_loss = -entropy.mean()  # Negative because we want to maximize entropy

            loss = policy_loss + 0.5 * value_loss + entropy_coeff * entropy_loss  # Configurable entropy regularization
            optimizer.zero_grad()
            loss.backward()
            
            # Check for NaN gradients and gradient norms
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=grad_clip_max_norm)
            if torch.isnan(grad_norm) or grad_norm > grad_clip_max_norm * 5.0:
                print(f"Warning: Large gradient norm detected: {grad_norm.item():.4f}")
                
            optimizer.step()
            
            # Clamp log std parameters for stability
            with torch.no_grad():
                policy.log_std_ac.data.clamp_(-4.0, 2.0)

        # Validation and checkpointing
        val_ret = evaluate_policy_irl(policy, validation_env, reward_fn, num_days=val_days, reward_type=reward_type)
        print(f"Update {update:3d} | Val return {val_ret:8.2f}")




def evaluate_policy_metrics(policy: MMVPolicyActorCritic,
                            env,
                            reward_fn: nn.Module,
                            num_days: int = 5,
                            reward_type: str = 'full') -> Dict[str, float]:
    policy_was_training = policy.training
    reward_was_training = reward_fn.training

    policy.eval()
    reward_fn.eval()

    learned_returns: List[float] = []
    total_energy = 0.0
    total_comfort = 0.0
    total_window_open = 0.0
    total_switches = 0
    total_steps = 0
    comfort_minutes_exceed = 0

    w_energy = float(getattr(env, 'w_energy', 1.0))
    w_comfort = float(getattr(env, 'w_comfort', 1.0))
    w_window = float(getattr(env, 'w_window', 1.0))

    scalers = getattr(env, 'scalers', None)
    zone_cols = []
    num_zones = getattr(env, 'num_zones', 0)
    if hasattr(env, 'columns'):
        zone_cols = list(env.columns.get('zone_cols', []))
        if not num_zones:
            num_zones = len(zone_cols)
    reference_temp = 30.0

    def _ensure_sequence(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if t is None:
            return None
        while t.dim() > 3 and t.size(0) == 1:
            t = t.squeeze(0)
        if t.dim() == 1:
            t = t.unsqueeze(0).unsqueeze(0)
        elif t.dim() == 2:
            t = t.unsqueeze(0)
        return t

    with torch.no_grad():
        max_days = min(num_days, len(env.daily_data_list))
        for d in range(max_days):
            s = env.reset(day_index=d)
            done = False
            ret = 0.0
            prev_window = int(env.window_prev_state)

            while not done:
                a_dict, _ = policy.get_action(torch.tensor(s, dtype=torch.float32).unsqueeze(0))
                s_next_env, _, done, info = env.step(a_dict)
                # Keep reward semantics consistent with collected trajectories:
                # reward input is (state_t, action_t, transition_features_t->t+1).
                s_next = s

                if reward_type == 'window_only':
                    window_states = s_next[-4:]
                    window_action = [flatten_action(a_dict)[0]]

                    eval_states_t = torch.tensor(np.array([window_states]), dtype=torch.float32).unsqueeze(0)
                    eval_actions_t = torch.tensor(np.array([window_action]), dtype=torch.float32).unsqueeze(0)
                    eval_features_t = torch.tensor(
                        np.array([[info['energy_consumption'], info['comfort_violations']]]),
                        dtype=torch.float32
                    ).unsqueeze(0)

                    feat_t = np.concatenate([
                        window_states,
                        window_action,
                        [info['energy_consumption']],
                        [info['comfort_violations']]
                    ])
                    feat_t = torch.tensor(feat_t, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                elif reward_type == 'window_only_mlp':
                    window_states = s_next[-4:]
                    window_action = [flatten_action(a_dict)[0]]
                    eval_states_t = torch.tensor(np.array([window_states]), dtype=torch.float32).unsqueeze(0)
                    eval_actions_t = torch.tensor(np.array([window_action]), dtype=torch.float32).unsqueeze(0)
                    eval_features_t = torch.tensor(
                        np.array([[info['energy_consumption'], info['comfort_violations']]]),
                        dtype=torch.float32
                    ).unsqueeze(0)

                    feat_t = np.concatenate([
                        window_states,
                        window_action,
                        [info['energy_consumption']],
                        [info['comfort_violations']]
                    ])
                    feat_t = torch.tensor(feat_t, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                elif reward_type == 'energy_comfort_only':
                    eval_states_t = torch.tensor(np.zeros((1, 0)), dtype=torch.float32).unsqueeze(0)
                    eval_actions_t = torch.tensor(np.zeros((1, 0)), dtype=torch.float32).unsqueeze(0)
                    eval_features_t = torch.tensor(
                        np.array([[info['energy_consumption'], info['comfort_violations']]]),
                        dtype=torch.float32
                    ).unsqueeze(0)
                    feat_t = eval_features_t
                elif reward_type == 'energy_comfort_only_gru':
                    eval_states_t = torch.tensor(np.zeros((1, 0)), dtype=torch.float32).unsqueeze(0)
                    eval_actions_t = torch.tensor(np.zeros((1, 0)), dtype=torch.float32).unsqueeze(0)
                    eval_features_t = torch.tensor(
                        np.array([[info['energy_consumption'], info['comfort_violations']]]),
                        dtype=torch.float32
                    ).unsqueeze(0)
                    feat_t = eval_features_t
                elif reward_type == 'comfort_energy_window':
                    window_dwell = s_next[10:11]
                    window_action = flatten_action(a_dict)[0]

                    eval_states_t = torch.tensor(np.array([[window_dwell]]), dtype=torch.float32).unsqueeze(0)
                    eval_actions_t = torch.tensor(np.array([[window_action]]), dtype=torch.float32).unsqueeze(0)
                    eval_features_t = torch.tensor(
                        np.array([[info['comfort_violations'], info['energy_consumption']]]),
                        dtype=torch.float32
                    ).unsqueeze(0)

                    feat_t = np.concatenate([
                        window_dwell,
                        [window_action],
                        [info['comfort_violations']],
                        [info['energy_consumption']]
                    ])
                    feat_t = torch.tensor(feat_t, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                elif reward_type == 'comfort_energy_window_gru':
                    window_dwell = s_next[10:11]
                    window_action = flatten_action(a_dict)[0]

                    eval_states_t = torch.tensor(np.array([[window_dwell]]), dtype=torch.float32).unsqueeze(0)
                    eval_actions_t = torch.tensor(np.array([[window_action]]), dtype=torch.float32).unsqueeze(0)
                    eval_features_t = torch.tensor(
                        np.array([[info['comfort_violations'], info['energy_consumption']]]),
                        dtype=torch.float32
                    ).unsqueeze(0)

                    feat_t = np.concatenate([
                        window_dwell,
                        [window_action],
                        [info['comfort_violations']],
                        [info['energy_consumption']]
                    ])
                    feat_t = torch.tensor(feat_t, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                elif reward_type == 'window_conditional_comfort_energy':
                    window_states = s_next[-4:]
                    window_action = flatten_action(a_dict)[0]
                    prev_window_state = s_next[-1]
                    conditional_comfort = prev_window_state * info['comfort_violations']

                    eval_states_t = torch.tensor(np.array([window_states]), dtype=torch.float32).unsqueeze(0)
                    eval_actions_t = torch.tensor(np.array([[window_action]]), dtype=torch.float32).unsqueeze(0)
                    eval_features_t = torch.tensor(
                        np.array([[conditional_comfort, info['energy_consumption']]]),
                        dtype=torch.float32
                    ).unsqueeze(0)

                    feat_t = np.concatenate([
                        window_states,
                        [window_action],
                        [conditional_comfort],
                        [info['energy_consumption']]
                    ])
                    feat_t = torch.tensor(feat_t, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                elif reward_type == 'window_conditional_comfort_energy_gru':
                    window_states = s_next[-4:]
                    window_action = flatten_action(a_dict)[0]
                    prev_window_state = s_next[-1]
                    conditional_comfort = prev_window_state * info['comfort_violations']

                    eval_states_t = torch.tensor(np.array([window_states]), dtype=torch.float32).unsqueeze(0)
                    eval_actions_t = torch.tensor(np.array([[window_action]]), dtype=torch.float32).unsqueeze(0)
                    eval_features_t = torch.tensor(
                        np.array([[conditional_comfort, info['energy_consumption']]]),
                        dtype=torch.float32
                    ).unsqueeze(0)

                    feat_t = np.concatenate([
                        window_states,
                        [window_action],
                        [conditional_comfort],
                        [info['energy_consumption']]
                    ])
                    feat_t = torch.tensor(feat_t, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                elif reward_type == 'window_conditional_energy_comfort':
                    window_states = s_next[-4:]
                    window_action = flatten_action(a_dict)[0]
                    prev_window_state = s_next[-1]
                    conditional_energy = (1 - prev_window_state) * info['energy_consumption']

                    eval_states_t = torch.tensor(np.array([window_states]), dtype=torch.float32).unsqueeze(0)
                    eval_actions_t = torch.tensor(np.array([[window_action]]), dtype=torch.float32).unsqueeze(0)
                    eval_features_t = torch.tensor(
                        np.array([[conditional_energy, info['comfort_violations']]]),
                        dtype=torch.float32
                    ).unsqueeze(0)

                    feat_t = np.concatenate([
                        window_states,
                        [window_action],
                        [conditional_energy],
                        [info['comfort_violations']]
                    ])
                    feat_t = torch.tensor(feat_t, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                elif reward_type == 'window_conditional_energy_comfort_gru':
                    window_states = s_next[-4:]
                    window_action = flatten_action(a_dict)[0]
                    prev_window_state = s_next[-1]
                    conditional_energy = (1 - prev_window_state) * info['energy_consumption']

                    eval_states_t = torch.tensor(np.array([window_states]), dtype=torch.float32).unsqueeze(0)
                    eval_actions_t = torch.tensor(np.array([[window_action]]), dtype=torch.float32).unsqueeze(0)
                    eval_features_t = torch.tensor(
                        np.array([[conditional_energy, info['comfort_violations']]]),
                        dtype=torch.float32
                    ).unsqueeze(0)

                    feat_t = np.concatenate([
                        window_states,
                        [window_action],
                        [conditional_energy],
                        [info['comfort_violations']]
                    ])
                    feat_t = torch.tensor(feat_t, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                elif reward_type == 'window_conditional_comfort_energy_both':
                    window_states = s_next[-4:]
                    window_action = flatten_action(a_dict)[0]
                    prev_window_state = s_next[-1]
                    conditional_comfort = prev_window_state * info['comfort_violations']
                    conditional_energy = (1 - prev_window_state) * info['energy_consumption']

                    eval_states_t = torch.tensor(np.array([window_states]), dtype=torch.float32).unsqueeze(0)
                    eval_actions_t = torch.tensor(np.array([[window_action]]), dtype=torch.float32).unsqueeze(0)
                    eval_features_t = torch.tensor(
                        np.array([[conditional_comfort, conditional_energy]]),
                        dtype=torch.float32
                    ).unsqueeze(0)

                    feat_t = np.concatenate([
                        window_states,
                        [window_action],
                        [conditional_comfort],
                        [conditional_energy]
                    ])
                    feat_t = torch.tensor(feat_t, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                elif reward_type == 'window_conditional_comfort_energy_both_gru':
                    window_states = s_next[-4:]
                    window_action = flatten_action(a_dict)[0]
                    prev_window_state = s_next[-1]
                    conditional_comfort = prev_window_state * info['comfort_violations']
                    conditional_energy = (1 - prev_window_state) * info['energy_consumption']

                    eval_states_t = torch.tensor(np.array([window_states]), dtype=torch.float32).unsqueeze(0)
                    eval_actions_t = torch.tensor(np.array([[window_action]]), dtype=torch.float32).unsqueeze(0)
                    eval_features_t = torch.tensor(
                        np.array([[conditional_comfort, conditional_energy]]),
                        dtype=torch.float32
                    ).unsqueeze(0)

                    feat_t = np.concatenate([
                        window_states,
                        [window_action],
                        [conditional_comfort],
                        [conditional_energy]
                    ])
                    feat_t = torch.tensor(feat_t, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                elif reward_type == 'state_action_only_mlp':
                    eval_states_t = torch.tensor(np.array([s_next]), dtype=torch.float32)
                    eval_actions_t = torch.tensor(np.array([flatten_action(a_dict)]), dtype=torch.float32)
                    eval_features_t = None
                    feat_t = None
                elif reward_type == 'state_action_only_gru':
                    eval_states_t = torch.tensor(np.array([s_next]), dtype=torch.float32).unsqueeze(0)
                    eval_actions_t = torch.tensor(np.array([flatten_action(a_dict)]), dtype=torch.float32).unsqueeze(0)
                    eval_features_t = None
                    feat_t = None
                elif reward_type == 'temps_hold_dwell_prev_gru':
                    parts = _split_airl_state_components(s_next, num_zones=len(a_dict["supply_temps"]))
                    zone_temps = parts['zone_temps']
                    outdoor_temp = _state_scalar(parts['outdoor_temp'])
                    window_hold_counter = _state_scalar(parts['window_hold_counter'])
                    window_dwell = _state_scalar(parts['window_dwell'])
                    prev_window_state = _state_scalar(parts['prev_window_state'])

                    state_features = np.concatenate(
                        [zone_temps, [outdoor_temp], [window_hold_counter], [window_dwell], [prev_window_state]]
                    )
                    eval_states_t = torch.tensor(state_features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    eval_actions_t = torch.tensor(np.array([[flatten_action(a_dict)[0]]]), dtype=torch.float32).unsqueeze(0)
                    eval_features_t = None
                    feat_t = None
                elif reward_type in AIRL_TIME_AUGMENTED_REWARD_TYPES:
                    state_features, window_action, feature_values = _time_augmented_step_features(
                        s_next,
                        a_dict,
                        env,
                        info,
                        reward_type,
                    )
                    eval_states_t = torch.tensor(state_features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    eval_actions_t = torch.tensor(window_action, dtype=torch.float32).unsqueeze(0)
                    eval_features_t = torch.tensor(feature_values, dtype=torch.float32).unsqueeze(0)
                    feat_t = None
                elif reward_type == 'temps_hold_dwell_prev_energy_gru':
                    parts = _split_airl_state_components(s_next, num_zones=len(a_dict["supply_temps"]))
                    zone_temps = parts['zone_temps']
                    outdoor_temp = _state_scalar(parts['outdoor_temp'])
                    window_hold_counter = _state_scalar(parts['window_hold_counter'])
                    window_dwell = _state_scalar(parts['window_dwell'])
                    prev_window_state = _state_scalar(parts['prev_window_state'])

                    state_features = np.concatenate(
                        [zone_temps, [outdoor_temp], [window_hold_counter], [window_dwell], [prev_window_state]]
                    )
                    eval_states_t = torch.tensor(state_features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    eval_actions_t = torch.tensor(np.array([[flatten_action(a_dict)[0]]]), dtype=torch.float32).unsqueeze(0)
                    eval_features_t = torch.tensor(
                        np.array([[info['energy_consumption']]]),
                        dtype=torch.float32
                    ).unsqueeze(0)
                    feat_t = None
                elif reward_type == 'temps_hold_dwell_prev_comfort_gru':
                    parts = _split_airl_state_components(s_next, num_zones=len(a_dict["supply_temps"]))
                    zone_temps = parts['zone_temps']
                    outdoor_temp = _state_scalar(parts['outdoor_temp'])
                    window_hold_counter = _state_scalar(parts['window_hold_counter'])
                    window_dwell = _state_scalar(parts['window_dwell'])
                    prev_window_state = _state_scalar(parts['prev_window_state'])

                    state_features = np.concatenate(
                        [zone_temps, [outdoor_temp], [window_hold_counter], [window_dwell], [prev_window_state]]
                    )
                    eval_states_t = torch.tensor(state_features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    eval_actions_t = torch.tensor(np.array([[flatten_action(a_dict)[0]]]), dtype=torch.float32).unsqueeze(0)
                    eval_features_t = torch.tensor(
                        np.array([[info['comfort_violations']]]),
                        dtype=torch.float32
                    ).unsqueeze(0)
                    feat_t = None
                elif reward_type == 'temps_hold_dwell_prev_energy_comfort_gru':
                    parts = _split_airl_state_components(s_next, num_zones=len(a_dict["supply_temps"]))
                    zone_temps = parts['zone_temps']
                    outdoor_temp = _state_scalar(parts['outdoor_temp'])
                    window_hold_counter = _state_scalar(parts['window_hold_counter'])
                    window_dwell = _state_scalar(parts['window_dwell'])
                    prev_window_state = _state_scalar(parts['prev_window_state'])

                    state_features = np.concatenate(
                        [zone_temps, [outdoor_temp], [window_hold_counter], [window_dwell], [prev_window_state]]
                    )
                    eval_states_t = torch.tensor(state_features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    eval_actions_t = torch.tensor(np.array([[flatten_action(a_dict)[0]]]), dtype=torch.float32).unsqueeze(0)
                    eval_features_t = torch.tensor(
                        np.array([[info['energy_consumption'], info['comfort_violations']]]),
                        dtype=torch.float32
                    ).unsqueeze(0)
                    feat_t = None
                elif reward_type == 'temps_hold_dwell_prev_energy_zonal_gru':
                    num_zones = len(a_dict["supply_temps"])
                    parts = _split_airl_state_components(s_next, num_zones=num_zones)
                    zone_temps = parts['zone_temps']
                    outdoor_temp = _state_scalar(parts['outdoor_temp'])
                    window_hold_counter = _state_scalar(parts['window_hold_counter'])
                    window_dwell = _state_scalar(parts['window_dwell'])
                    prev_window_state = _state_scalar(parts['prev_window_state'])
                    energy_zone = _zone_vector_from_info(
                        info,
                        scalar_key='energy_consumption',
                        zone_key='energy_consumption_zones',
                        num_zones=num_zones,
                    )

                    state_features = np.concatenate(
                        [zone_temps, [outdoor_temp], [window_hold_counter], [window_dwell], [prev_window_state]]
                    )
                    eval_states_t = torch.tensor(state_features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    eval_actions_t = torch.tensor(np.array([[flatten_action(a_dict)[0]]]), dtype=torch.float32).unsqueeze(0)
                    eval_features_t = torch.tensor(np.array([energy_zone]), dtype=torch.float32).unsqueeze(0)
                    feat_t = None
                elif reward_type == 'temps_hold_dwell_prev_comfort_zonal_gru':
                    num_zones = len(a_dict["supply_temps"])
                    parts = _split_airl_state_components(s_next, num_zones=num_zones)
                    zone_temps = parts['zone_temps']
                    outdoor_temp = _state_scalar(parts['outdoor_temp'])
                    window_hold_counter = _state_scalar(parts['window_hold_counter'])
                    window_dwell = _state_scalar(parts['window_dwell'])
                    prev_window_state = _state_scalar(parts['prev_window_state'])
                    comfort_zone = _zone_vector_from_info(
                        info,
                        scalar_key='comfort_violations',
                        zone_key='comfort_violations_zones',
                        num_zones=num_zones,
                    )

                    state_features = np.concatenate(
                        [zone_temps, [outdoor_temp], [window_hold_counter], [window_dwell], [prev_window_state]]
                    )
                    eval_states_t = torch.tensor(state_features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    eval_actions_t = torch.tensor(np.array([[flatten_action(a_dict)[0]]]), dtype=torch.float32).unsqueeze(0)
                    eval_features_t = torch.tensor(np.array([comfort_zone]), dtype=torch.float32).unsqueeze(0)
                    feat_t = None
                elif reward_type == 'temps_hold_dwell_prev_energy_comfort_zonal_gru':
                    num_zones = len(a_dict["supply_temps"])
                    parts = _split_airl_state_components(s_next, num_zones=num_zones)
                    zone_temps = parts['zone_temps']
                    outdoor_temp = _state_scalar(parts['outdoor_temp'])
                    window_hold_counter = _state_scalar(parts['window_hold_counter'])
                    window_dwell = _state_scalar(parts['window_dwell'])
                    prev_window_state = _state_scalar(parts['prev_window_state'])
                    energy_zone = _zone_vector_from_info(
                        info,
                        scalar_key='energy_consumption',
                        zone_key='energy_consumption_zones',
                        num_zones=num_zones,
                    )
                    comfort_zone = _zone_vector_from_info(
                        info,
                        scalar_key='comfort_violations',
                        zone_key='comfort_violations_zones',
                        num_zones=num_zones,
                    )

                    state_features = np.concatenate(
                        [zone_temps, [outdoor_temp], [window_hold_counter], [window_dwell], [prev_window_state]]
                    )
                    eval_states_t = torch.tensor(state_features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    eval_actions_t = torch.tensor(np.array([[flatten_action(a_dict)[0]]]), dtype=torch.float32).unsqueeze(0)
                    eval_features_t = torch.tensor(
                        np.array([np.concatenate([energy_zone, comfort_zone], axis=0)]),
                        dtype=torch.float32
                    ).unsqueeze(0)
                    feat_t = None
                elif reward_type == 'temps_hold_prev_gru':
                    parts = _split_airl_state_components(s_next, num_zones=len(a_dict["supply_temps"]))
                    zone_temps = parts['zone_temps']
                    outdoor_temp = _state_scalar(parts['outdoor_temp'])
                    window_hold_counter = _state_scalar(parts['window_hold_counter'])
                    prev_window_state = _state_scalar(parts['prev_window_state'])

                    state_features = np.concatenate(
                        [zone_temps, [outdoor_temp], [window_hold_counter], [prev_window_state]]
                    )
                    eval_states_t = torch.tensor(state_features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    eval_actions_t = torch.tensor(np.array([[flatten_action(a_dict)[0]]]), dtype=torch.float32).unsqueeze(0)
                    eval_features_t = None
                    feat_t = None
                elif reward_type == 'temps_comfort_energy_window_gru':
                    parts = _split_airl_state_components(s_next, num_zones=len(a_dict["supply_temps"]))
                    zone_temps = parts['zone_temps']
                    outdoor_temp = _state_scalar(parts['outdoor_temp'])
                    window_dwell = _state_scalar(parts['window_dwell'])

                    state_features = np.concatenate([zone_temps, [outdoor_temp], [window_dwell]])

                    eval_states_t = torch.tensor(state_features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    eval_actions_t = torch.tensor(np.array([[flatten_action(a_dict)[0]]]), dtype=torch.float32).unsqueeze(0)
                    eval_features_t = torch.tensor(
                        np.array([[info['comfort_violations'], info['energy_consumption']]]),
                        dtype=torch.float32
                    ).unsqueeze(0)
                    feat_t = None
                elif reward_type == 'temps_comfort_window_gru':
                    parts = _split_airl_state_components(s_next, num_zones=len(a_dict["supply_temps"]))
                    zone_temps = parts['zone_temps']
                    outdoor_temp = _state_scalar(parts['outdoor_temp'])
                    window_dwell = _state_scalar(parts['window_dwell'])

                    state_features = np.concatenate([zone_temps, [outdoor_temp], [window_dwell]])

                    eval_states_t = torch.tensor(state_features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    eval_actions_t = torch.tensor(np.array([[flatten_action(a_dict)[0]]]), dtype=torch.float32).unsqueeze(0)
                    eval_features_t = torch.tensor(
                        np.array([[info['comfort_violations']]]),
                        dtype=torch.float32
                    ).unsqueeze(0)
                    feat_t = None
                elif reward_type == 'temps_energy_window_gru':
                    parts = _split_airl_state_components(s_next, num_zones=len(a_dict["supply_temps"]))
                    zone_temps = parts['zone_temps']
                    outdoor_temp = _state_scalar(parts['outdoor_temp'])
                    window_dwell = _state_scalar(parts['window_dwell'])

                    state_features = np.concatenate([zone_temps, [outdoor_temp], [window_dwell]])

                    eval_states_t = torch.tensor(state_features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    eval_actions_t = torch.tensor(np.array([[flatten_action(a_dict)[0]]]), dtype=torch.float32).unsqueeze(0)
                    eval_features_t = torch.tensor(
                        np.array([[info['energy_consumption']]]),
                        dtype=torch.float32
                    ).unsqueeze(0)
                    feat_t = None
                elif reward_type == 'temps_window_gru':
                    parts = _split_airl_state_components(s_next, num_zones=len(a_dict["supply_temps"]))
                    zone_temps = parts['zone_temps']
                    outdoor_temp = _state_scalar(parts['outdoor_temp'])
                    window_dwell = _state_scalar(parts['window_dwell'])
                    prev_window_state = _state_scalar(parts['prev_window_state'])

                    state_features = np.concatenate([zone_temps, [outdoor_temp], [window_dwell]])

                    eval_states_t = torch.tensor(state_features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    eval_actions_t = torch.tensor(np.array([[flatten_action(a_dict)[0]]]), dtype=torch.float32).unsqueeze(0)
                    eval_features_t = torch.tensor(
                        np.array([[prev_window_state]], dtype=np.float32),
                        dtype=torch.float32
                    ).unsqueeze(0)
                    feat_t = None
                elif reward_type == 'state_action_comfort_energy_gru':
                    eval_states_t = torch.tensor(np.array([s_next]), dtype=torch.float32).unsqueeze(0)
                    eval_actions_t = torch.tensor(np.array([flatten_action(a_dict)]), dtype=torch.float32).unsqueeze(0)
                    eval_features_t = torch.tensor(
                        np.array([[info['comfort_violations'], info['energy_consumption']]]),
                        dtype=torch.float32
                    ).unsqueeze(0)
                    feat_t = None
                else:  # 'full'
                    eval_states_t = torch.tensor(np.array([s_next]), dtype=torch.float32).unsqueeze(0)
                    eval_actions_t = torch.tensor(np.array([flatten_action(a_dict)]), dtype=torch.float32).unsqueeze(0)
                    eval_features_t = torch.tensor(
                        np.array([[info['energy_consumption'], info['comfort_violations']]]),
                        dtype=torch.float32
                    ).unsqueeze(0)

                    feat_t = np.concatenate([
                        s_next,
                        flatten_action(a_dict),
                        [info['energy_consumption']],
                        [info['comfort_violations']]
                    ])
                    feat_t = torch.tensor(feat_t, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

                if reward_type in [
                    'energy_comfort_only',
                    'energy_comfort_only_gru',
                    'comfort_energy_window',
                    'comfort_energy_window_gru',
                    'window_conditional_comfort_energy',
                    'window_conditional_comfort_energy_gru',
                    'window_conditional_energy_comfort',
                    'window_conditional_energy_comfort_gru',
                    'window_conditional_comfort_energy_both',
                    'window_conditional_comfort_energy_both_gru',
                    'window_only_mlp',
                    'temps_hold_dwell_prev_energy_gru',
                    'temps_hold_dwell_prev_comfort_gru',
                    'temps_hold_dwell_prev_energy_comfort_gru',
                    'temps_hold_dwell_prev_energy_zonal_gru',
                    'temps_hold_dwell_prev_comfort_zonal_gru',
                    'temps_hold_dwell_prev_energy_comfort_zonal_gru',
                ] or reward_type in AIRL_TIME_AUGMENTED_REWARD_TYPES:
                    eval_states_t = _ensure_sequence(eval_states_t)
                    eval_actions_t = _ensure_sequence(eval_actions_t)
                    eval_features_t = _ensure_sequence(eval_features_t)
                    step_reward = reward_fn(eval_states_t, eval_actions_t, eval_features_t).item()
                elif reward_type in [
                    'state_action_only_mlp',
                    'state_action_only_gru',
                    'temps_hold_dwell_prev_gru',
                    'temps_hold_prev_gru',
                ]:
                    step_reward = reward_fn(eval_states_t, eval_actions_t).item()
                elif 'gru' in reward_type or reward_type in ('window_only', 'full'):
                    eval_states_t = _ensure_sequence(eval_states_t)
                    eval_actions_t = _ensure_sequence(eval_actions_t)
                    eval_features_t = _ensure_sequence(eval_features_t)
                    step_reward = reward_fn(eval_states_t, eval_actions_t, eval_features_t).item()
                else:
                    step_reward = reward_fn(feat_t).item()

                ret += step_reward

                energy = info.get('energy_consumption', 0.0)
                comfort = info.get('comfort_violations', 0.0)
                window_state = int(info.get('window_state', env.window_prev_state))

                total_energy += energy
                total_comfort += comfort
                total_window_open += window_state
                if window_state != prev_window:
                    total_switches += 1

                if scalers is not None and zone_cols:
                    zone_slice = s_next_env[:num_zones]
                    temps_c = []
                    for idx in range(min(num_zones, len(zone_cols))):
                        temps_c.append(
                            inverse_normalize(float(zone_slice[idx]), zone_cols[idx], scalers)
                        )
                    if temps_c:
                        avg_temp = float(sum(temps_c) / len(temps_c))
                        if avg_temp > reference_temp:
                            comfort_minutes_exceed += 1

                prev_window = window_state
                total_steps += 1
                s = s_next_env

            learned_returns.append(ret)

    if policy_was_training:
        policy.train()
    if reward_was_training:
        reward_fn.train()

    avg_learned_return = float(np.mean(learned_returns)) if learned_returns else 0.0
    avg_energy_per_minute = total_energy / max(total_steps, 1)
    avg_energy_per_hour = avg_energy_per_minute * 60.0
    avg_comfort_deg = total_comfort / max(total_steps, 1)
    avg_comfort_pct = (comfort_minutes_exceed / max(total_steps, 1)) * 100.0
    avg_open_ratio = total_window_open / max(total_steps, 1)
    avg_switches = total_switches / max(len(learned_returns), 1)
    avg_cost = -(w_energy * avg_energy_per_minute + w_comfort * avg_comfort_deg + w_window * avg_switches)

    return {
        'avg_return': avg_cost,
        'avg_learned_return': avg_learned_return,
        'avg_energy_per_hour': avg_energy_per_hour,
        'avg_comfort_pct': avg_comfort_pct,
        'avg_window_open_ratio': avg_open_ratio,
        'avg_window_switches': avg_switches,
        'avg_energy_per_minute': avg_energy_per_minute,
        'avg_comfort_deg': avg_comfort_deg,
    }


def evaluate_policy_irl(policy: MMVPolicyActorCritic, 
                       env, 
                       reward_fn: nn.Module, 
                       num_days: int = 5,
                       reward_type: str = 'full') -> float:
    metrics = evaluate_policy_metrics(
        policy,
        env,
        reward_fn,
        num_days=num_days,
        reward_type=reward_type,
    )
    return metrics['avg_return']


def summarize_expert_metrics(
    expert_trajs: List[Dict],
    reward_fn: nn.Module,
    reward_type: str = 'full',
    w_energy: float = 1.0,
    w_comfort: float = 1.0,
    w_window: float = 1.0,
    scalers: Optional[Dict[str, Dict[str, float]]] = None,
    zone_cols: Optional[Sequence[str]] = None,
    num_zones: Optional[int] = None,
) -> Dict[str, float]:
    reward_was_training = reward_fn.training
    reward_fn.eval()

    learned_returns: List[float] = []
    total_energy = 0.0
    total_comfort = 0.0
    total_window_open = 0.0
    total_switches = 0
    total_steps = 0
    comfort_minutes_exceed = 0

    zone_cols = list(zone_cols) if zone_cols is not None else []
    if num_zones is None:
        num_zones = len(zone_cols)
    reference_temp = 30.0

    with torch.no_grad():
        for traj in expert_trajs:
            states = np.asarray(traj.get('states', []), dtype=np.float32)
            actions = np.asarray(traj.get('actions', []), dtype=np.float32)
            energy = np.asarray(traj.get('energy_features', []), dtype=np.float32)
            comfort = np.asarray(traj.get('comfort_features', []), dtype=np.float32)

            if states.size == 0 or actions.size == 0:
                continue

            logits = traj_scores([traj], reward_fn, reward_type)
            learned_returns.append(float(logits.sum().item()))

            total_energy += float(energy.sum())
            total_comfort += float(comfort.sum())
            total_steps += len(actions)

            prev_window = int(round(states[0, -1]))
            for t_idx, change_val in enumerate(actions[:, 0]):
                change = int(round(float(change_val)))
                window_state = prev_window ^ change
                total_window_open += window_state
                if window_state != prev_window:
                    total_switches += 1
                prev_window = window_state

                if scalers is not None and zone_cols:
                    if t_idx < states.shape[0]:
                        zone_slice = states[t_idx, :num_zones]
                        temps_c = []
                        for idx in range(min(num_zones, len(zone_cols))):
                            temps_c.append(
                                inverse_normalize(float(zone_slice[idx]), zone_cols[idx], scalers)
                            )
                        if temps_c:
                            avg_temp = float(sum(temps_c) / len(temps_c))
                            if avg_temp > reference_temp:
                                comfort_minutes_exceed += 1

    if reward_was_training:
        reward_fn.train()

    episodes = len(learned_returns)
    avg_learned_return = float(np.mean(learned_returns)) if learned_returns else 0.0
    avg_energy_per_minute = total_energy / max(total_steps, 1)
    avg_energy_per_hour = avg_energy_per_minute * 60.0
    avg_comfort_deg = total_comfort / max(total_steps, 1)
    avg_comfort_pct = (comfort_minutes_exceed / max(total_steps, 1)) * 100.0
    avg_open_ratio = total_window_open / max(total_steps, 1)
    avg_switches = total_switches / max(episodes, 1)
    avg_cost = -(w_energy * avg_energy_per_minute + w_comfort * avg_comfort_deg + w_window * avg_switches)

    return {
        'avg_return': avg_cost,
        'avg_learned_return': avg_learned_return,
        'avg_energy_per_hour': avg_energy_per_hour,
        'avg_comfort_pct': avg_comfort_pct,
        'avg_window_open_ratio': avg_open_ratio,
        'avg_window_switches': avg_switches,
        'avg_energy_per_minute': avg_energy_per_minute,
        'avg_comfort_deg': avg_comfort_deg,
    }


def run_airl_training(env,
                     validation_env,
                     expert_trajs: List[Dict],
                     state_dim: int,
                     action_dim: int,
                     device: torch.device,
                     n_iters: int = 10,
                     policy_lr: float = 1e-4,
                     reward_lr: float = 1e-3,
                     reward_update_steps: int = 10,
                     ppo_updates: int = 10,
                     reward_type: str = 'full',
                     policy_weight_decay: float = 1e-5,
                     grad_clip_max_norm: float = 2.0,
                     entropy_coeff: float = 0.005,
                     initial_reward_update_steps: Optional[int] = None,
                     initial_reward_min_margin: float = 0.0,
                     initial_reward_max_attempts: int = 1,
                     collect_days: Optional[int] = None,
                     ppo_batch_days: Optional[int] = None,
                     ppo_epochs: Optional[int] = None,
                     validation_days: Optional[int] = None,
                     output_dir: str = "notebooks/outputs") -> Tuple[MMVPolicyActorCritic, nn.Module]:
    """
    Run complete AIRL training loop.
    
    Args:
        env: Training environment
        validation_env: Validation environment
        expert_trajs: Expert demonstration trajectories
        state_dim: State space dimension
        action_dim: Action space dimension
        device: PyTorch device
        n_iters: Number of AIRL iterations
        policy_lr: Policy learning rate
        reward_lr: Reward learning rate
        initial_reward_update_steps: Optional reward update steps for iteration 1 warm-up
        initial_reward_min_margin: Required margin for initial discriminator, i.e. D(exp) > D(pol) + margin
        initial_reward_max_attempts: Max warm-up attempts on iteration 1 to satisfy initial margin
        collect_days: Number of training days to roll out per AIRL iteration (random sample)
        ppo_batch_days: Number of days per PPO batch (random sample)
        ppo_epochs: PPO epochs per update
        validation_days: Number of validation days per evaluation
        
    Returns:
        Tuple of (trained_policy, trained_reward_fn)
    """
    eval_env = validation_env if validation_env is not None else env
    num_zones = _num_zones_from_action_dim(action_dim)

    # Initialize policy and reward function
    policy = MMVPolicyActorCritic(state_dim=state_dim, num_zones=num_zones)
    optimizer = torch.optim.Adam(
        policy.parameters(), 
        lr=policy_lr,
        weight_decay=policy_weight_decay,
        eps=1e-8,
        betas=(0.9, 0.999)
    )
    
    # Initialize reward function based on reward type
    if reward_type == 'window_only':
        # 4 window states + 1 window action + 2 features (energy, comfort) = 7
        reward_fn = GRUReward(in_dim=7, hid=64).to(device)
        print(f"Using window-only reward function (input_dim=7)")
        
        # Initialize separate normalization from expert trajectories
        print("Initializing separate normalization from expert data...")
        expert_states = []
        expert_actions = []
        expert_features = []
        
        for traj in expert_trajs:
            states = np.array(traj["states"])  # Shape: (T, 12)
            actions = np.array(traj["actions"])  # Shape: (T, 8)
            energy_feats = np.array(traj["energy_features"]).reshape(-1, 1)
            comfort_feats = np.array(traj["comfort_features"]).reshape(-1, 1)
            
            # Extract window components
            window_states = states[:, -4:]  # (T, 4) - window_hold_counter, switch_flag, dwell, window_prev_state
            window_action = actions[:, 0:1]  # (T, 1) - change action
            features = np.concatenate([energy_feats, comfort_feats], axis=1)  # (T, 2)
            
            expert_states.append(window_states)
            expert_actions.append(window_action)
            expert_features.append(features)
        
        # Combine all expert data for normalization
        all_states = np.concatenate(expert_states, axis=0)
        all_actions = np.concatenate(expert_actions, axis=0)
        all_features = np.concatenate(expert_features, axis=0)
        
        states_tensor = torch.tensor(all_states, dtype=torch.float32).to(device)
        actions_tensor = torch.tensor(all_actions, dtype=torch.float32).to(device)
        features_tensor = torch.tensor(all_features, dtype=torch.float32).to(device)
        
        reward_fn.initialize_normalization(states_tensor, actions_tensor, features_tensor)
        print(f"Separate normalization initialized - States: {reward_fn.normalizer.state_mean.shape}, Actions: {reward_fn.normalizer.action_mean.shape}, Features: {reward_fn.normalizer.feature_mean.shape}")
    elif reward_type == 'window_only_mlp':
        # 4 window states + 1 window action + 2 features (energy, comfort) = 7 (MLP version)
        reward_fn = MLPReward(input_dim=7, hidden_dims=[64, 64]).to(device)
        print(f"Using window-only MLP reward function (input_dim=7)")
        
        # Initialize separate normalization from expert trajectories
        print("Initializing separate normalization from expert data...")
        expert_states = []
        expert_actions = []
        expert_features = []
        
        for traj in expert_trajs:
            states = np.array(traj["states"])  # Shape: (T, 12)
            actions = np.array(traj["actions"])  # Shape: (T, 8)
            energy_feats = np.array(traj["energy_features"]).reshape(-1, 1)
            comfort_feats = np.array(traj["comfort_features"]).reshape(-1, 1)
            
            # Extract window components
            window_states = states[:, -4:]  # (T, 4) - window_hold_counter, switch_flag, dwell, window_prev_state
            window_action = actions[:, 0:1]  # (T, 1) - change action
            features = np.concatenate([energy_feats, comfort_feats], axis=1)  # (T, 2)
            
            expert_states.append(window_states)
            expert_actions.append(window_action)
            expert_features.append(features)
        
        # Combine all expert data for normalization
        all_states = np.concatenate(expert_states, axis=0)
        all_actions = np.concatenate(expert_actions, axis=0)
        all_features = np.concatenate(expert_features, axis=0)
        
        states_tensor = torch.tensor(all_states, dtype=torch.float32).to(device)
        actions_tensor = torch.tensor(all_actions, dtype=torch.float32).to(device)
        features_tensor = torch.tensor(all_features, dtype=torch.float32).to(device)
        
        reward_fn.initialize_normalization(states_tensor, actions_tensor, features_tensor)
        print(f"Separate normalization initialized - States: {reward_fn.normalizer.state_mean.shape}, Actions: {reward_fn.normalizer.action_mean.shape}, Features: {reward_fn.normalizer.feature_mean.shape}")
    elif reward_type == 'energy_comfort_only':
        # Only energy and comfort features = 2
        reward_fn = MLPReward(input_dim=2, hidden_dims=[32, 32]).to(device)
        print(f"Using energy-comfort-only reward function (input_dim=2)")
        
        # Initialize separate normalization from expert trajectories
        print("Initializing separate normalization from expert data...")
        expert_states = []
        expert_actions = []
        expert_features = []
        
        for traj in expert_trajs:
            energy_feats = np.array(traj["energy_features"]).reshape(-1, 1)
            comfort_feats = np.array(traj["comfort_features"]).reshape(-1, 1)
            
            # For energy_comfort_only: no states or actions, only features
            states = np.zeros((len(energy_feats), 0))  # Empty states
            actions = np.zeros((len(energy_feats), 0))  # Empty actions
            features = np.concatenate([energy_feats, comfort_feats], axis=1)
            
            expert_states.append(states)
            expert_actions.append(actions)
            expert_features.append(features)
        
        # Combine all expert data for normalization
        all_states = np.concatenate(expert_states, axis=0) if expert_states[0].shape[1] > 0 else np.zeros((0, 0))
        all_actions = np.concatenate(expert_actions, axis=0) if expert_actions[0].shape[1] > 0 else np.zeros((0, 0))
        all_features = np.concatenate(expert_features, axis=0)
        
        states_tensor = torch.tensor(all_states, dtype=torch.float32).to(device)
        actions_tensor = torch.tensor(all_actions, dtype=torch.float32).to(device) 
        features_tensor = torch.tensor(all_features, dtype=torch.float32).to(device)
        
        reward_fn.initialize_normalization(states_tensor, actions_tensor, features_tensor)
        print(f"Separate normalization initialized - Features: mean={reward_fn.normalizer.feature_mean}, std={reward_fn.normalizer.feature_std}")
        
    elif reward_type == 'comfort_energy_window':
        # Comfort, energy, window dwell, and window action features = 4
        reward_fn = MLPReward(input_dim=4, hidden_dims=[64, 64]).to(device)
        print(f"Using comfort-energy-window reward function (input_dim=4)")
        
        # Initialize separate normalization from expert trajectories
        print("Initializing separate normalization from expert data...")
        expert_states = []
        expert_actions = []
        expert_features = []
        
        for traj in expert_trajs:
            states = np.array(traj["states"])  # Shape: (T, 12)
            actions = np.array(traj["actions"])  # Shape: (T, 8)
            energy_feats = np.array(traj["energy_features"]).reshape(-1, 1)
            comfort_feats = np.array(traj["comfort_features"]).reshape(-1, 1)
            
            # Extract components: window dwell from states, window action from actions, energy+comfort as features
            window_dwell = states[:, 10:11]  # (T, 1) - window dwell state
            window_action = actions[:, 0:1]   # (T, 1) - change action
            features = np.concatenate([comfort_feats, energy_feats], axis=1)  # (T, 2) - comfort + energy
            
            expert_states.append(window_dwell)
            expert_actions.append(window_action)
            expert_features.append(features)
        
        # Combine all expert data for normalization
        all_states = np.concatenate(expert_states, axis=0)
        all_actions = np.concatenate(expert_actions, axis=0)
        all_features = np.concatenate(expert_features, axis=0)
        
        states_tensor = torch.tensor(all_states, dtype=torch.float32).to(device)
        actions_tensor = torch.tensor(all_actions, dtype=torch.float32).to(device)
        features_tensor = torch.tensor(all_features, dtype=torch.float32).to(device)
        
        reward_fn.initialize_normalization(states_tensor, actions_tensor, features_tensor)
        print(f"Separate normalization initialized - States: {reward_fn.normalizer.state_mean.shape}, Actions: {reward_fn.normalizer.action_mean.shape}, Features: {reward_fn.normalizer.feature_mean.shape}")
        
    elif reward_type == 'window_conditional_comfort_energy':
        # Window states + window action + conditional comfort + energy = 7
        reward_fn = MLPReward(input_dim=7, hidden_dims=[64, 64]).to(device)
        print(f"Using window-conditional-comfort-energy reward function (input_dim=7)")
        
        # Initialize separate normalization from expert trajectories
        print("Initializing separate normalization from expert data...")
        expert_states = []
        expert_actions = []
        expert_features = []
        
        for traj in expert_trajs:
            states = np.array(traj["states"])  # Shape: (T, 12)
            actions = np.array(traj["actions"])  # Shape: (T, 8)
            energy_feats = np.array(traj["energy_features"]).reshape(-1, 1)
            comfort_feats = np.array(traj["comfort_features"]).reshape(-1, 1)
            
            # Extract components: window states, window action, conditional comfort + energy as features
            window_states = states[:, -4:]  # (T, 4) - window_hold_counter, switch_flag, dwell, window_prev_state
            window_action = actions[:, 0:1]  # (T, 1) - change action
            
            # Calculate conditional comfort: prev_window_state * comfort_violations
            prev_window_state = states[:, -1:]  # (T, 1) - window_prev_state
            conditional_comfort = prev_window_state * comfort_feats  # (T, 1)
            
            # Combine conditional comfort and energy as features
            features = np.concatenate([conditional_comfort, energy_feats], axis=1)  # (T, 2)
            
            expert_states.append(window_states)
            expert_actions.append(window_action)
            expert_features.append(features)
        
        # Combine all expert data for normalization
        all_states = np.concatenate(expert_states, axis=0)
        all_actions = np.concatenate(expert_actions, axis=0)
        all_features = np.concatenate(expert_features, axis=0)
        
        states_tensor = torch.tensor(all_states, dtype=torch.float32).to(device)
        actions_tensor = torch.tensor(all_actions, dtype=torch.float32).to(device)
        features_tensor = torch.tensor(all_features, dtype=torch.float32).to(device)
        
        reward_fn.initialize_normalization(states_tensor, actions_tensor, features_tensor)
        print(f"Separate normalization initialized - States: {reward_fn.normalizer.state_mean.shape}, Actions: {reward_fn.normalizer.action_mean.shape}, Features: {reward_fn.normalizer.feature_mean.shape}")
        
    elif reward_type == 'window_conditional_energy_comfort':
        # Window states + window action + conditional energy + comfort = 7
        reward_fn = MLPReward(input_dim=7, hidden_dims=[64, 64]).to(device)
        print(f"Using window-conditional-energy-comfort reward function (input_dim=7)")
        
        # Initialize separate normalization from expert trajectories
        print("Initializing separate normalization from expert data...")
        expert_states = []
        expert_actions = []
        expert_features = []
        
        for traj in expert_trajs:
            states = np.array(traj["states"])  # Shape: (T, 12)
            actions = np.array(traj["actions"])  # Shape: (T, 8)
            energy_feats = np.array(traj["energy_features"]).reshape(-1, 1)
            comfort_feats = np.array(traj["comfort_features"]).reshape(-1, 1)
            
            # Extract components: window states, window action, conditional energy + comfort as features
            window_states = states[:, -4:]  # (T, 4) - window_hold_counter, switch_flag, dwell, window_prev_state
            window_action = actions[:, 0:1]  # (T, 1) - change action
            
            # Calculate conditional energy: (1 - prev_window_state) * energy_consumption
            prev_window_state = states[:, -1:]  # (T, 1) - window_prev_state
            conditional_energy = (1 - prev_window_state) * energy_feats  # (T, 1)
            
            # Combine conditional energy and comfort as features
            features = np.concatenate([conditional_energy, comfort_feats], axis=1)  # (T, 2)
            
            expert_states.append(window_states)
            expert_actions.append(window_action)
            expert_features.append(features)
        
        # Combine all expert data for normalization
        all_states = np.concatenate(expert_states, axis=0)
        all_actions = np.concatenate(expert_actions, axis=0)
        all_features = np.concatenate(expert_features, axis=0)
        
        states_tensor = torch.tensor(all_states, dtype=torch.float32).to(device)
        actions_tensor = torch.tensor(all_actions, dtype=torch.float32).to(device)
        features_tensor = torch.tensor(all_features, dtype=torch.float32).to(device)
        
        reward_fn.initialize_normalization(states_tensor, actions_tensor, features_tensor)
        print(f"Separate normalization initialized - States: {reward_fn.normalizer.state_mean.shape}, Actions: {reward_fn.normalizer.action_mean.shape}, Features: {reward_fn.normalizer.feature_mean.shape}")
        
    elif reward_type == 'window_conditional_comfort_energy_both':
        # Window states + window action + conditional comfort + conditional energy = 7
        reward_fn = MLPReward(input_dim=7, hidden_dims=[64, 64]).to(device)
        print(f"Using window-conditional-comfort-energy-both reward function (input_dim=7)")
        
        # Initialize separate normalization from expert trajectories
        print("Initializing separate normalization from expert data...")
        expert_states = []
        expert_actions = []
        expert_features = []
        
        for traj in expert_trajs:
            states = np.array(traj["states"])  # Shape: (T, 12)
            actions = np.array(traj["actions"])  # Shape: (T, 8)
            energy_feats = np.array(traj["energy_features"]).reshape(-1, 1)
            comfort_feats = np.array(traj["comfort_features"]).reshape(-1, 1)
            
            # Extract components: window states, window action, both conditional features
            window_states = states[:, -4:]  # (T, 4) - window_hold_counter, switch_flag, dwell, window_prev_state
            window_action = actions[:, 0:1]  # (T, 1) - change action
            
            # Calculate both conditional features based on window state
            prev_window_state = states[:, -1:]  # (T, 1) - window_prev_state
            conditional_comfort = prev_window_state * comfort_feats  # (T, 1) - active when window open
            conditional_energy = (1 - prev_window_state) * energy_feats  # (T, 1) - active when window closed
            
            # Combine both conditional features
            features = np.concatenate([conditional_comfort, conditional_energy], axis=1)  # (T, 2)
            
            expert_states.append(window_states)
            expert_actions.append(window_action)
            expert_features.append(features)
        
        # Combine all expert data for normalization
        all_states = np.concatenate(expert_states, axis=0)
        all_actions = np.concatenate(expert_actions, axis=0)
        all_features = np.concatenate(expert_features, axis=0)
        
        states_tensor = torch.tensor(all_states, dtype=torch.float32).to(device)
        actions_tensor = torch.tensor(all_actions, dtype=torch.float32).to(device)
        features_tensor = torch.tensor(all_features, dtype=torch.float32).to(device)
        
        reward_fn.initialize_normalization(states_tensor, actions_tensor, features_tensor)
        print(f"Separate normalization initialized - States: {reward_fn.normalizer.state_mean.shape}, Actions: {reward_fn.normalizer.action_mean.shape}, Features: {reward_fn.normalizer.feature_mean.shape}")
        
    elif reward_type == 'energy_comfort_only_gru':
        # GRU version: Only energy and comfort features = 2
        reward_fn = GRUReward(in_dim=2, hid=64).to(device)
        print(f"Using energy-comfort-only GRU reward function (input_dim=2)")
        
        # Initialize separate normalization from expert trajectories
        print("Initializing separate normalization from expert data...")
        expert_states = []
        expert_actions = []
        expert_features = []
        
        for traj in expert_trajs:
            energy_feats = np.array(traj["energy_features"]).reshape(-1, 1)
            comfort_feats = np.array(traj["comfort_features"]).reshape(-1, 1)
            
            # For energy_comfort_only_gru: no states or actions, only features
            states = np.zeros((len(energy_feats), 0))  # Empty states
            actions = np.zeros((len(energy_feats), 0))  # Empty actions
            features = np.concatenate([energy_feats, comfort_feats], axis=1)
            
            expert_states.append(states)
            expert_actions.append(actions)
            expert_features.append(features)
        
        # Combine all expert data for normalization
        all_states = np.concatenate(expert_states, axis=0) if expert_states[0].shape[1] > 0 else np.zeros((0, 0))
        all_actions = np.concatenate(expert_actions, axis=0) if expert_actions[0].shape[1] > 0 else np.zeros((0, 0))
        all_features = np.concatenate(expert_features, axis=0)
        
        states_tensor = torch.tensor(all_states, dtype=torch.float32).to(device)
        actions_tensor = torch.tensor(all_actions, dtype=torch.float32).to(device) 
        features_tensor = torch.tensor(all_features, dtype=torch.float32).to(device)
        
        reward_fn.initialize_normalization(states_tensor, actions_tensor, features_tensor)
        print(f"Separate normalization initialized - Features: mean={reward_fn.normalizer.feature_mean}, std={reward_fn.normalizer.feature_std}")
        
    elif reward_type == 'comfort_energy_window_gru':
        # GRU version: Comfort, energy, window dwell, and window action features = 4
        reward_fn = GRUReward(in_dim=4, hid=64).to(device)
        print(f"Using comfort-energy-window GRU reward function (input_dim=4)")
        
        # Initialize separate normalization from expert trajectories
        print("Initializing separate normalization from expert data...")
        expert_states = []
        expert_actions = []
        expert_features = []
        
        for traj in expert_trajs:
            states = np.array(traj["states"])  # Shape: (T, 12)
            actions = np.array(traj["actions"])  # Shape: (T, 8)
            energy_feats = np.array(traj["energy_features"]).reshape(-1, 1)
            comfort_feats = np.array(traj["comfort_features"]).reshape(-1, 1)
            
            # Extract components: window dwell from states, window action from actions, energy+comfort as features
            window_dwell = states[:, 10:11]  # (T, 1) - window dwell state
            window_action = actions[:, 0:1]   # (T, 1) - change action
            features = np.concatenate([comfort_feats, energy_feats], axis=1)  # (T, 2) - comfort + energy
            
            expert_states.append(window_dwell)
            expert_actions.append(window_action)
            expert_features.append(features)
        
        # Combine all expert data for normalization
        all_states = np.concatenate(expert_states, axis=0)
        all_actions = np.concatenate(expert_actions, axis=0)
        all_features = np.concatenate(expert_features, axis=0)
        
        states_tensor = torch.tensor(all_states, dtype=torch.float32).to(device)
        actions_tensor = torch.tensor(all_actions, dtype=torch.float32).to(device)
        features_tensor = torch.tensor(all_features, dtype=torch.float32).to(device)
        
        reward_fn.initialize_normalization(states_tensor, actions_tensor, features_tensor)
        print(f"Separate normalization initialized - States: {reward_fn.normalizer.state_mean.shape}, Actions: {reward_fn.normalizer.action_mean.shape}, Features: {reward_fn.normalizer.feature_mean.shape}")
    elif reward_type == 'temps_comfort_energy_window_gru':
        if not expert_trajs:
            raise ValueError("Expert trajectories are required to initialize temps_comfort_energy_window_gru.")

        state_dim_full = expert_trajs[0]['states'].shape[1]
        action_dim_full = expert_trajs[0]['actions'].shape[1]
        num_zones = _infer_airl_num_zones(state_dim_full, action_dim_full)
        if num_zones <= 0:
            raise ValueError("Invalid zone dimension computed for temps_comfort_energy_window_gru.")

        state_feature_dim = num_zones + 2  # zone temps + outdoor temp + dwell
        action_feature_dim = 1
        feature_dim = 2  # comfort + energy
        in_dim = state_feature_dim + action_feature_dim + feature_dim
        reward_fn = GRUReward(in_dim=in_dim, hid=64).to(device)
        print(f"Using temps-comfort-energy-window GRU reward function (input_dim={in_dim})")

        print("Initializing separate normalization from expert data...")
        expert_states = []
        expert_actions = []
        expert_features = []

        for traj in expert_trajs:
            states = np.array(traj["states"], dtype=np.float32)
            actions = np.array(traj["actions"], dtype=np.float32)
            energy_feats = np.array(traj["energy_features"], dtype=np.float32).reshape(-1, 1)
            comfort_feats = np.array(traj["comfort_features"], dtype=np.float32).reshape(-1, 1)

            if states.shape[1] != state_dim_full:
                raise ValueError("Inconsistent state dimensionality detected in expert trajectories.")

            parts = _split_airl_state_components(states, action_dim=actions.shape[1], num_zones=num_zones)
            zone_temps = parts['zone_temps']
            outdoor_temp = parts['outdoor_temp']
            window_dwell = parts['window_dwell']
            window_action = actions[:, 0:1]
            features = np.concatenate([comfort_feats, energy_feats], axis=1)
            state_features = np.concatenate([zone_temps, outdoor_temp, window_dwell], axis=1)

            expert_states.append(state_features)
            expert_actions.append(window_action)
            expert_features.append(features)

        all_states = np.concatenate(expert_states, axis=0)
        all_actions = np.concatenate(expert_actions, axis=0)
        all_features = np.concatenate(expert_features, axis=0)

        states_tensor = torch.tensor(all_states, dtype=torch.float32, device=device)
        actions_tensor = torch.tensor(all_actions, dtype=torch.float32, device=device)
        features_tensor = torch.tensor(all_features, dtype=torch.float32, device=device)

        reward_fn.initialize_normalization(states_tensor, actions_tensor, features_tensor)
        print("Separate normalization initialized for temps-comfort-energy-window GRU reward")
    elif reward_type == 'temps_comfort_window_gru':
        if not expert_trajs:
            raise ValueError("Expert trajectories are required to initialize temps_comfort_window_gru.")

        state_dim_full = expert_trajs[0]['states'].shape[1]
        action_dim_full = expert_trajs[0]['actions'].shape[1]
        num_zones = _infer_airl_num_zones(state_dim_full, action_dim_full)
        if num_zones <= 0:
            raise ValueError("Invalid zone dimension computed for temps_comfort_window_gru.")

        state_feature_dim = num_zones + 2  # zone temps + outdoor temp + dwell
        action_feature_dim = 1
        feature_dim = 1  # comfort only
        in_dim = state_feature_dim + action_feature_dim + feature_dim
        reward_fn = GRUReward(in_dim=in_dim, hid=64).to(device)
        print(f"Using temps-comfort-window GRU reward function (input_dim={in_dim})")

        print("Initializing separate normalization from expert data...")
        expert_states = []
        expert_actions = []
        expert_features = []

        for traj in expert_trajs:
            states = np.array(traj["states"], dtype=np.float32)
            actions = np.array(traj["actions"], dtype=np.float32)
            comfort_feats = np.array(traj["comfort_features"], dtype=np.float32).reshape(-1, 1)

            if states.shape[1] != state_dim_full:
                raise ValueError("Inconsistent state dimensionality detected in expert trajectories.")

            parts = _split_airl_state_components(states, action_dim=actions.shape[1], num_zones=num_zones)
            zone_temps = parts['zone_temps']
            outdoor_temp = parts['outdoor_temp']
            window_dwell = parts['window_dwell']
            window_action = actions[:, 0:1]
            state_features = np.concatenate([zone_temps, outdoor_temp, window_dwell], axis=1)

            expert_states.append(state_features)
            expert_actions.append(window_action)
            expert_features.append(comfort_feats)

        all_states = np.concatenate(expert_states, axis=0)
        all_actions = np.concatenate(expert_actions, axis=0)
        all_features = np.concatenate(expert_features, axis=0)

        states_tensor = torch.tensor(all_states, dtype=torch.float32, device=device)
        actions_tensor = torch.tensor(all_actions, dtype=torch.float32, device=device)
        features_tensor = torch.tensor(all_features, dtype=torch.float32, device=device)

        reward_fn.initialize_normalization(states_tensor, actions_tensor, features_tensor)
        print("Separate normalization initialized for temps-comfort-window GRU reward")
    elif reward_type == 'temps_energy_window_gru':
        if not expert_trajs:
            raise ValueError("Expert trajectories are required to initialize temps_energy_window_gru.")

        state_dim_full = expert_trajs[0]['states'].shape[1]
        action_dim_full = expert_trajs[0]['actions'].shape[1]
        num_zones = _infer_airl_num_zones(state_dim_full, action_dim_full)
        if num_zones <= 0:
            raise ValueError("Invalid zone dimension computed for temps_energy_window_gru.")

        state_feature_dim = num_zones + 2  # zone temps + outdoor temp + dwell
        action_feature_dim = 1
        feature_dim = 1  # energy only
        in_dim = state_feature_dim + action_feature_dim + feature_dim
        reward_fn = GRUReward(in_dim=in_dim, hid=64).to(device)
        print(f"Using temps-energy-window GRU reward function (input_dim={in_dim})")

        print("Initializing separate normalization from expert data...")
        expert_states = []
        expert_actions = []
        expert_features = []

        for traj in expert_trajs:
            states = np.array(traj["states"], dtype=np.float32)
            actions = np.array(traj["actions"], dtype=np.float32)
            energy_feats = np.array(traj["energy_features"], dtype=np.float32).reshape(-1, 1)

            if states.shape[1] != state_dim_full:
                raise ValueError("Inconsistent state dimensionality detected in expert trajectories.")

            parts = _split_airl_state_components(states, action_dim=actions.shape[1], num_zones=num_zones)
            zone_temps = parts['zone_temps']
            outdoor_temp = parts['outdoor_temp']
            window_dwell = parts['window_dwell']
            window_action = actions[:, 0:1]
            state_features = np.concatenate([zone_temps, outdoor_temp, window_dwell], axis=1)

            expert_states.append(state_features)
            expert_actions.append(window_action)
            expert_features.append(energy_feats)

        all_states = np.concatenate(expert_states, axis=0)
        all_actions = np.concatenate(expert_actions, axis=0)
        all_features = np.concatenate(expert_features, axis=0)

        states_tensor = torch.tensor(all_states, dtype=torch.float32, device=device)
        actions_tensor = torch.tensor(all_actions, dtype=torch.float32, device=device)
        features_tensor = torch.tensor(all_features, dtype=torch.float32, device=device)

        reward_fn.initialize_normalization(states_tensor, actions_tensor, features_tensor)
        print("Separate normalization initialized for temps-energy-window GRU reward")
    elif reward_type == 'temps_window_gru':
        if not expert_trajs:
            raise ValueError("Expert trajectories are required to initialize temps_window_gru.")

        state_dim_full = expert_trajs[0]['states'].shape[1]
        action_dim_full = expert_trajs[0]['actions'].shape[1]
        num_zones = _infer_airl_num_zones(state_dim_full, action_dim_full)
        if num_zones <= 0:
            raise ValueError("Invalid zone dimension computed for temps_window_gru.")

        state_feature_dim = num_zones + 2  # zone temps + outdoor temp + dwell
        action_feature_dim = 1
        feature_dim = 1  # previous window state
        in_dim = state_feature_dim + action_feature_dim + feature_dim
        reward_fn = GRUReward(in_dim=in_dim, hid=64).to(device)
        print(f"Using temps-window GRU reward function (input_dim={in_dim})")

        print("Initializing separate normalization from expert data...")
        expert_states = []
        expert_actions = []
        expert_features = []

        for traj in expert_trajs:
            states = np.array(traj["states"], dtype=np.float32)
            actions = np.array(traj["actions"], dtype=np.float32)

            if states.shape[1] != state_dim_full:
                raise ValueError("Inconsistent state dimensionality detected in expert trajectories.")

            parts = _split_airl_state_components(states, action_dim=actions.shape[1], num_zones=num_zones)
            zone_temps = parts['zone_temps']
            outdoor_temp = parts['outdoor_temp']
            window_dwell = parts['window_dwell']
            prev_window_state = parts['prev_window_state']
            window_action = actions[:, 0:1]
            state_features = np.concatenate([zone_temps, outdoor_temp, window_dwell], axis=1)

            expert_states.append(state_features)
            expert_actions.append(window_action)
            expert_features.append(prev_window_state)

        all_states = np.concatenate(expert_states, axis=0)
        all_actions = np.concatenate(expert_actions, axis=0)
        all_features = np.concatenate(expert_features, axis=0)

        states_tensor = torch.tensor(all_states, dtype=torch.float32, device=device)
        actions_tensor = torch.tensor(all_actions, dtype=torch.float32, device=device)
        features_tensor = torch.tensor(all_features, dtype=torch.float32, device=device)

        reward_fn.initialize_normalization(states_tensor, actions_tensor, features_tensor)
        print("Separate normalization initialized for temps-window GRU reward")

    elif reward_type == 'window_conditional_comfort_energy_gru':
        # GRU version: Window states + window action + conditional comfort + energy = 7
        reward_fn = GRUReward(in_dim=7, hid=64).to(device)
        print(f"Using window-conditional-comfort-energy GRU reward function (input_dim=7)")
        
        # Initialize separate normalization from expert trajectories
        print("Initializing separate normalization from expert data...")
        expert_states = []
        expert_actions = []
        expert_features = []
        
        for traj in expert_trajs:
            states = np.array(traj["states"])  # Shape: (T, 12)
            actions = np.array(traj["actions"])  # Shape: (T, 8)
            energy_feats = np.array(traj["energy_features"]).reshape(-1, 1)
            comfort_feats = np.array(traj["comfort_features"]).reshape(-1, 1)
            
            # Extract components: window states, window action, conditional comfort + energy as features
            window_states = states[:, -4:]  # (T, 4) - window_hold_counter, switch_flag, dwell, window_prev_state
            window_action = actions[:, 0:1]  # (T, 1) - change action
            
            # Calculate conditional comfort: prev_window_state * comfort_violations
            prev_window_state = states[:, -1:]  # (T, 1) - window_prev_state
            conditional_comfort = prev_window_state * comfort_feats  # (T, 1)
            
            # Combine conditional comfort and energy as features
            features = np.concatenate([conditional_comfort, energy_feats], axis=1)  # (T, 2)
            
            expert_states.append(window_states)
            expert_actions.append(window_action)
            expert_features.append(features)
        
        # Combine all expert data for normalization
        all_states = np.concatenate(expert_states, axis=0)
        all_actions = np.concatenate(expert_actions, axis=0)
        all_features = np.concatenate(expert_features, axis=0)
        
        states_tensor = torch.tensor(all_states, dtype=torch.float32).to(device)
        actions_tensor = torch.tensor(all_actions, dtype=torch.float32).to(device)
        features_tensor = torch.tensor(all_features, dtype=torch.float32).to(device)
        
        reward_fn.initialize_normalization(states_tensor, actions_tensor, features_tensor)
        print(f"Separate normalization initialized - States: {reward_fn.normalizer.state_mean.shape}, Actions: {reward_fn.normalizer.action_mean.shape}, Features: {reward_fn.normalizer.feature_mean.shape}")
        
    elif reward_type == 'window_conditional_energy_comfort_gru':
        # GRU version: Window states + window action + conditional energy + comfort = 7
        reward_fn = GRUReward(in_dim=7, hid=64).to(device)
        print(f"Using window-conditional-energy-comfort GRU reward function (input_dim=7)")
        
        # Initialize separate normalization from expert trajectories
        print("Initializing separate normalization from expert data...")
        expert_states = []
        expert_actions = []
        expert_features = []
        
        for traj in expert_trajs:
            states = np.array(traj["states"])  # Shape: (T, 12)
            actions = np.array(traj["actions"])  # Shape: (T, 8)
            energy_feats = np.array(traj["energy_features"]).reshape(-1, 1)
            comfort_feats = np.array(traj["comfort_features"]).reshape(-1, 1)
            
            # Extract components: window states, window action, conditional energy + comfort as features
            window_states = states[:, -4:]  # (T, 4) - window_hold_counter, switch_flag, dwell, window_prev_state
            window_action = actions[:, 0:1]  # (T, 1) - change action
            
            # Calculate conditional energy: (1 - prev_window_state) * energy_consumption
            prev_window_state = states[:, -1:]  # (T, 1) - window_prev_state
            conditional_energy = (1 - prev_window_state) * energy_feats  # (T, 1)
            
            # Combine conditional energy and comfort as features
            features = np.concatenate([conditional_energy, comfort_feats], axis=1)  # (T, 2)
            
            expert_states.append(window_states)
            expert_actions.append(window_action)
            expert_features.append(features)
        
        # Combine all expert data for normalization
        all_states = np.concatenate(expert_states, axis=0)
        all_actions = np.concatenate(expert_actions, axis=0)
        all_features = np.concatenate(expert_features, axis=0)
        
        states_tensor = torch.tensor(all_states, dtype=torch.float32).to(device)
        actions_tensor = torch.tensor(all_actions, dtype=torch.float32).to(device)
        features_tensor = torch.tensor(all_features, dtype=torch.float32).to(device)
        
        reward_fn.initialize_normalization(states_tensor, actions_tensor, features_tensor)
        print(f"Separate normalization initialized - States: {reward_fn.normalizer.state_mean.shape}, Actions: {reward_fn.normalizer.action_mean.shape}, Features: {reward_fn.normalizer.feature_mean.shape}")
        
    elif reward_type == 'window_conditional_comfort_energy_both_gru':
        # GRU version: Window states + window action + conditional comfort + conditional energy = 7
        reward_fn = GRUReward(in_dim=7, hid=64).to(device)
        print(f"Using window-conditional-comfort-energy-both GRU reward function (input_dim=7)")
        
        # Initialize separate normalization from expert trajectories
        print("Initializing separate normalization from expert data...")
        expert_states = []
        expert_actions = []
        expert_features = []
        
        for traj in expert_trajs:
            states = np.array(traj["states"])  # Shape: (T, 12)
            actions = np.array(traj["actions"])  # Shape: (T, 8)
            energy_feats = np.array(traj["energy_features"]).reshape(-1, 1)
            comfort_feats = np.array(traj["comfort_features"]).reshape(-1, 1)
            
            # Extract components: window states, window action, both conditional features
            window_states = states[:, -4:]  # (T, 4) - window_hold_counter, switch_flag, dwell, window_prev_state
            window_action = actions[:, 0:1]  # (T, 1) - change action
            
            # Calculate both conditional features based on window state
            prev_window_state = states[:, -1:]  # (T, 1) - window_prev_state
            conditional_comfort = prev_window_state * comfort_feats  # (T, 1) - active when window open
            conditional_energy = (1 - prev_window_state) * energy_feats  # (T, 1) - active when window closed
            
            # Combine both conditional features
            features = np.concatenate([conditional_comfort, conditional_energy], axis=1)  # (T, 2)
            
            expert_states.append(window_states)
            expert_actions.append(window_action)
            expert_features.append(features)
        
        # Combine all expert data for normalization
        all_states = np.concatenate(expert_states, axis=0)
        all_actions = np.concatenate(expert_actions, axis=0)
        all_features = np.concatenate(expert_features, axis=0)
        
        states_tensor = torch.tensor(all_states, dtype=torch.float32).to(device)
        actions_tensor = torch.tensor(all_actions, dtype=torch.float32).to(device)
        features_tensor = torch.tensor(all_features, dtype=torch.float32).to(device)
        
        reward_fn.initialize_normalization(states_tensor, actions_tensor, features_tensor)
        print(f"Separate normalization initialized - States: {reward_fn.normalizer.state_mean.shape}, Actions: {reward_fn.normalizer.action_mean.shape}, Features: {reward_fn.normalizer.feature_mean.shape}")
        
    elif reward_type == 'temps_hold_dwell_prev_gru':
        if not expert_trajs:
            raise ValueError("Expert trajectories are required to initialize temps_hold_dwell_prev_gru.")

        state_dim_full = expert_trajs[0]['states'].shape[1]
        action_dim_full = expert_trajs[0]['actions'].shape[1]
        num_zones = _infer_airl_num_zones(state_dim_full, action_dim_full)
        if num_zones <= 0:
            raise ValueError("Invalid zone dimension computed for temps_hold_dwell_prev_gru.")

        state_feature_dim = num_zones + 4  # zone temps + outdoor temp + hold counter + dwell + prev_window_state
        reward_fn = GRURewardStateActionOnly(state_dim=state_feature_dim, action_dim=1, hid=64).to(device)
        print(f"Using temps-hold-dwell-prev GRU reward function (state_dim={state_feature_dim}, action_dim=1)")

        print("Initializing state/action normalization from expert data...")
        expert_states = []
        expert_actions = []

        for traj in expert_trajs:
            states = np.array(traj["states"], dtype=np.float32)
            actions = np.array(traj["actions"], dtype=np.float32)

            if states.shape[1] != state_dim_full:
                raise ValueError("Inconsistent state dimensionality detected in expert trajectories.")

            parts = _split_airl_state_components(states, action_dim=actions.shape[1], num_zones=num_zones)
            zone_temps = parts['zone_temps']
            outdoor_temp = parts['outdoor_temp']
            window_hold_counter = parts['window_hold_counter']
            window_dwell = parts['window_dwell']
            prev_window_state = parts['prev_window_state']
            change_action = actions[:, 0:1]

            state_features = np.concatenate(
                [zone_temps, outdoor_temp, window_hold_counter, window_dwell, prev_window_state],
                axis=1
            )
            expert_states.append(state_features)
            expert_actions.append(change_action)

        all_states = np.concatenate(expert_states, axis=0)
        all_actions = np.concatenate(expert_actions, axis=0)

        states_tensor = torch.tensor(all_states, dtype=torch.float32).unsqueeze(0).to(device)
        actions_tensor = torch.tensor(all_actions, dtype=torch.float32).unsqueeze(0).to(device)
        reward_fn.initialize_normalization(states_tensor, actions_tensor)
        print("State/action normalization initialized for temps-hold-dwell-prev GRU reward")

    elif reward_type in AIRL_TIME_AUGMENTED_REWARD_TYPES:
        if not expert_trajs:
            raise ValueError(f"Expert trajectories are required to initialize {reward_type}.")

        state_dim_full = expert_trajs[0]['states'].shape[1]
        action_dim_full = expert_trajs[0]['actions'].shape[1]
        num_zones = _infer_airl_num_zones(state_dim_full, action_dim_full)
        if num_zones <= 0:
            raise ValueError(f"Invalid zone dimension computed for {reward_type}.")

        state_feature_dim = num_zones + 4  # zone temps + outdoor temp + hold counter + dwell + prev_window_state
        action_feature_dim = 1
        feature_dim = _time_augmented_feature_dim(reward_type, num_zones)
        in_dim = state_feature_dim + action_feature_dim + feature_dim
        reward_fn = GRUReward(in_dim=in_dim, hid=64).to(device)
        print(f"Using {reward_type.replace('_', '-')} reward function (input_dim={in_dim})")

        print("Initializing separate normalization from expert data...")
        expert_states = []
        expert_actions = []
        expert_features = []

        for traj in expert_trajs:
            states = np.array(traj["states"], dtype=np.float32)
            if states.shape[1] != state_dim_full:
                raise ValueError("Inconsistent state dimensionality detected in expert trajectories.")

            state_features, window_action, feature_values = _time_augmented_traj_features(
                traj,
                reward_type,
                num_zones=num_zones,
            )
            expert_states.append(state_features)
            expert_actions.append(window_action)
            expert_features.append(feature_values)

        all_states = np.concatenate(expert_states, axis=0)
        all_actions = np.concatenate(expert_actions, axis=0)
        all_features = np.concatenate(expert_features, axis=0)

        states_tensor = torch.tensor(all_states, dtype=torch.float32).to(device)
        actions_tensor = torch.tensor(all_actions, dtype=torch.float32).to(device)
        features_tensor = torch.tensor(all_features, dtype=torch.float32).to(device)
        reward_fn.initialize_normalization(states_tensor, actions_tensor, features_tensor)
        print(f"Separate normalization initialized for {reward_type.replace('_', '-')} reward")

    elif reward_type == 'temps_hold_dwell_prev_energy_gru':
        if not expert_trajs:
            raise ValueError("Expert trajectories are required to initialize temps_hold_dwell_prev_energy_gru.")

        state_dim_full = expert_trajs[0]['states'].shape[1]
        action_dim_full = expert_trajs[0]['actions'].shape[1]
        num_zones = _infer_airl_num_zones(state_dim_full, action_dim_full)
        if num_zones <= 0:
            raise ValueError("Invalid zone dimension computed for temps_hold_dwell_prev_energy_gru.")

        state_feature_dim = num_zones + 4  # zone temps + outdoor temp + hold counter + dwell + prev_window_state
        action_feature_dim = 1
        feature_dim = 1  # energy only
        in_dim = state_feature_dim + action_feature_dim + feature_dim
        reward_fn = GRUReward(in_dim=in_dim, hid=64).to(device)
        print(f"Using temps-hold-dwell-prev-energy GRU reward function (input_dim={in_dim})")

        print("Initializing separate normalization from expert data...")
        expert_states = []
        expert_actions = []
        expert_features = []

        for traj in expert_trajs:
            states = np.array(traj["states"], dtype=np.float32)
            actions = np.array(traj["actions"], dtype=np.float32)
            energy_feats = np.array(traj["energy_features"], dtype=np.float32).reshape(-1, 1)

            if states.shape[1] != state_dim_full:
                raise ValueError("Inconsistent state dimensionality detected in expert trajectories.")

            parts = _split_airl_state_components(states, action_dim=actions.shape[1], num_zones=num_zones)
            zone_temps = parts['zone_temps']
            outdoor_temp = parts['outdoor_temp']
            window_hold_counter = parts['window_hold_counter']
            window_dwell = parts['window_dwell']
            prev_window_state = parts['prev_window_state']
            change_action = actions[:, 0:1]

            state_features = np.concatenate(
                [zone_temps, outdoor_temp, window_hold_counter, window_dwell, prev_window_state],
                axis=1
            )
            expert_states.append(state_features)
            expert_actions.append(change_action)
            expert_features.append(energy_feats)

        all_states = np.concatenate(expert_states, axis=0)
        all_actions = np.concatenate(expert_actions, axis=0)
        all_features = np.concatenate(expert_features, axis=0)

        states_tensor = torch.tensor(all_states, dtype=torch.float32).to(device)
        actions_tensor = torch.tensor(all_actions, dtype=torch.float32).to(device)
        features_tensor = torch.tensor(all_features, dtype=torch.float32).to(device)
        reward_fn.initialize_normalization(states_tensor, actions_tensor, features_tensor)
        print("Separate normalization initialized for temps-hold-dwell-prev-energy GRU reward")

    elif reward_type == 'temps_hold_dwell_prev_comfort_gru':
        if not expert_trajs:
            raise ValueError("Expert trajectories are required to initialize temps_hold_dwell_prev_comfort_gru.")

        state_dim_full = expert_trajs[0]['states'].shape[1]
        action_dim_full = expert_trajs[0]['actions'].shape[1]
        num_zones = _infer_airl_num_zones(state_dim_full, action_dim_full)
        if num_zones <= 0:
            raise ValueError("Invalid zone dimension computed for temps_hold_dwell_prev_comfort_gru.")

        state_feature_dim = num_zones + 4  # zone temps + outdoor temp + hold counter + dwell + prev_window_state
        action_feature_dim = 1
        feature_dim = 1  # comfort only
        in_dim = state_feature_dim + action_feature_dim + feature_dim
        reward_fn = GRUReward(in_dim=in_dim, hid=64).to(device)
        print(f"Using temps-hold-dwell-prev-comfort GRU reward function (input_dim={in_dim})")

        print("Initializing separate normalization from expert data...")
        expert_states = []
        expert_actions = []
        expert_features = []

        for traj in expert_trajs:
            states = np.array(traj["states"], dtype=np.float32)
            actions = np.array(traj["actions"], dtype=np.float32)
            comfort_feats = np.array(traj["comfort_features"], dtype=np.float32).reshape(-1, 1)

            if states.shape[1] != state_dim_full:
                raise ValueError("Inconsistent state dimensionality detected in expert trajectories.")

            parts = _split_airl_state_components(states, action_dim=actions.shape[1], num_zones=num_zones)
            zone_temps = parts['zone_temps']
            outdoor_temp = parts['outdoor_temp']
            window_hold_counter = parts['window_hold_counter']
            window_dwell = parts['window_dwell']
            prev_window_state = parts['prev_window_state']
            change_action = actions[:, 0:1]

            state_features = np.concatenate(
                [zone_temps, outdoor_temp, window_hold_counter, window_dwell, prev_window_state],
                axis=1
            )
            expert_states.append(state_features)
            expert_actions.append(change_action)
            expert_features.append(comfort_feats)

        all_states = np.concatenate(expert_states, axis=0)
        all_actions = np.concatenate(expert_actions, axis=0)
        all_features = np.concatenate(expert_features, axis=0)

        states_tensor = torch.tensor(all_states, dtype=torch.float32).to(device)
        actions_tensor = torch.tensor(all_actions, dtype=torch.float32).to(device)
        features_tensor = torch.tensor(all_features, dtype=torch.float32).to(device)
        reward_fn.initialize_normalization(states_tensor, actions_tensor, features_tensor)
        print("Separate normalization initialized for temps-hold-dwell-prev-comfort GRU reward")

    elif reward_type == 'temps_hold_dwell_prev_energy_comfort_gru':
        if not expert_trajs:
            raise ValueError("Expert trajectories are required to initialize temps_hold_dwell_prev_energy_comfort_gru.")

        state_dim_full = expert_trajs[0]['states'].shape[1]
        action_dim_full = expert_trajs[0]['actions'].shape[1]
        num_zones = _infer_airl_num_zones(state_dim_full, action_dim_full)
        if num_zones <= 0:
            raise ValueError("Invalid zone dimension computed for temps_hold_dwell_prev_energy_comfort_gru.")

        state_feature_dim = num_zones + 4  # zone temps + outdoor temp + hold counter + dwell + prev_window_state
        action_feature_dim = 1
        feature_dim = 2  # energy + comfort
        in_dim = state_feature_dim + action_feature_dim + feature_dim
        reward_fn = GRUReward(in_dim=in_dim, hid=64).to(device)
        print(f"Using temps-hold-dwell-prev-energy-comfort GRU reward function (input_dim={in_dim})")

        print("Initializing separate normalization from expert data...")
        expert_states = []
        expert_actions = []
        expert_features = []

        for traj in expert_trajs:
            states = np.array(traj["states"], dtype=np.float32)
            actions = np.array(traj["actions"], dtype=np.float32)
            energy_feats = np.array(traj["energy_features"], dtype=np.float32).reshape(-1, 1)
            comfort_feats = np.array(traj["comfort_features"], dtype=np.float32).reshape(-1, 1)

            if states.shape[1] != state_dim_full:
                raise ValueError("Inconsistent state dimensionality detected in expert trajectories.")

            parts = _split_airl_state_components(states, action_dim=actions.shape[1], num_zones=num_zones)
            zone_temps = parts['zone_temps']
            outdoor_temp = parts['outdoor_temp']
            window_hold_counter = parts['window_hold_counter']
            window_dwell = parts['window_dwell']
            prev_window_state = parts['prev_window_state']
            change_action = actions[:, 0:1]

            state_features = np.concatenate(
                [zone_temps, outdoor_temp, window_hold_counter, window_dwell, prev_window_state],
                axis=1
            )
            features = np.concatenate([energy_feats, comfort_feats], axis=1)
            expert_states.append(state_features)
            expert_actions.append(change_action)
            expert_features.append(features)

        all_states = np.concatenate(expert_states, axis=0)
        all_actions = np.concatenate(expert_actions, axis=0)
        all_features = np.concatenate(expert_features, axis=0)

        states_tensor = torch.tensor(all_states, dtype=torch.float32).to(device)
        actions_tensor = torch.tensor(all_actions, dtype=torch.float32).to(device)
        features_tensor = torch.tensor(all_features, dtype=torch.float32).to(device)
        reward_fn.initialize_normalization(states_tensor, actions_tensor, features_tensor)
        print("Separate normalization initialized for temps-hold-dwell-prev-energy-comfort GRU reward")

    elif reward_type == 'temps_hold_dwell_prev_energy_zonal_gru':
        if not expert_trajs:
            raise ValueError("Expert trajectories are required to initialize temps_hold_dwell_prev_energy_zonal_gru.")

        state_dim_full = expert_trajs[0]['states'].shape[1]
        action_dim_full = expert_trajs[0]['actions'].shape[1]
        num_zones = _infer_airl_num_zones(state_dim_full, action_dim_full)
        if num_zones <= 0:
            raise ValueError("Invalid zone dimension computed for temps_hold_dwell_prev_energy_zonal_gru.")

        state_feature_dim = num_zones + 4  # zone temps + outdoor temp + hold counter + dwell + prev_window_state
        action_feature_dim = 1
        feature_dim = num_zones  # per-zone energy vector
        in_dim = state_feature_dim + action_feature_dim + feature_dim
        reward_fn = GRUReward(in_dim=in_dim, hid=64).to(device)
        print(f"Using temps-hold-dwell-prev-energy-zonal GRU reward function (input_dim={in_dim})")

        print("Initializing separate normalization from expert data...")
        expert_states = []
        expert_actions = []
        expert_features = []

        for traj in expert_trajs:
            states = np.array(traj["states"], dtype=np.float32)
            actions = np.array(traj["actions"], dtype=np.float32)
            energy_zone_feats = _zone_matrix_from_traj(
                traj,
                zone_key='energy_zone_features',
                scalar_key='energy_features',
                num_zones=num_zones,
            )

            if states.shape[1] != state_dim_full:
                raise ValueError("Inconsistent state dimensionality detected in expert trajectories.")

            parts = _split_airl_state_components(states, action_dim=actions.shape[1], num_zones=num_zones)
            zone_temps = parts['zone_temps']
            outdoor_temp = parts['outdoor_temp']
            window_hold_counter = parts['window_hold_counter']
            window_dwell = parts['window_dwell']
            prev_window_state = parts['prev_window_state']
            change_action = actions[:, 0:1]

            state_features = np.concatenate(
                [zone_temps, outdoor_temp, window_hold_counter, window_dwell, prev_window_state],
                axis=1
            )
            expert_states.append(state_features)
            expert_actions.append(change_action)
            expert_features.append(energy_zone_feats)

        all_states = np.concatenate(expert_states, axis=0)
        all_actions = np.concatenate(expert_actions, axis=0)
        all_features = np.concatenate(expert_features, axis=0)

        states_tensor = torch.tensor(all_states, dtype=torch.float32).to(device)
        actions_tensor = torch.tensor(all_actions, dtype=torch.float32).to(device)
        features_tensor = torch.tensor(all_features, dtype=torch.float32).to(device)
        reward_fn.initialize_normalization(states_tensor, actions_tensor, features_tensor)
        print("Separate normalization initialized for temps-hold-dwell-prev-energy-zonal GRU reward")

    elif reward_type == 'temps_hold_dwell_prev_comfort_zonal_gru':
        if not expert_trajs:
            raise ValueError("Expert trajectories are required to initialize temps_hold_dwell_prev_comfort_zonal_gru.")

        state_dim_full = expert_trajs[0]['states'].shape[1]
        action_dim_full = expert_trajs[0]['actions'].shape[1]
        num_zones = _infer_airl_num_zones(state_dim_full, action_dim_full)
        if num_zones <= 0:
            raise ValueError("Invalid zone dimension computed for temps_hold_dwell_prev_comfort_zonal_gru.")

        state_feature_dim = num_zones + 4  # zone temps + outdoor temp + hold counter + dwell + prev_window_state
        action_feature_dim = 1
        feature_dim = num_zones  # per-zone comfort vector
        in_dim = state_feature_dim + action_feature_dim + feature_dim
        reward_fn = GRUReward(in_dim=in_dim, hid=64).to(device)
        print(f"Using temps-hold-dwell-prev-comfort-zonal GRU reward function (input_dim={in_dim})")

        print("Initializing separate normalization from expert data...")
        expert_states = []
        expert_actions = []
        expert_features = []

        for traj in expert_trajs:
            states = np.array(traj["states"], dtype=np.float32)
            actions = np.array(traj["actions"], dtype=np.float32)
            comfort_zone_feats = _zone_matrix_from_traj(
                traj,
                zone_key='comfort_zone_features',
                scalar_key='comfort_features',
                num_zones=num_zones,
            )

            if states.shape[1] != state_dim_full:
                raise ValueError("Inconsistent state dimensionality detected in expert trajectories.")

            parts = _split_airl_state_components(states, action_dim=actions.shape[1], num_zones=num_zones)
            zone_temps = parts['zone_temps']
            outdoor_temp = parts['outdoor_temp']
            window_hold_counter = parts['window_hold_counter']
            window_dwell = parts['window_dwell']
            prev_window_state = parts['prev_window_state']
            change_action = actions[:, 0:1]

            state_features = np.concatenate(
                [zone_temps, outdoor_temp, window_hold_counter, window_dwell, prev_window_state],
                axis=1
            )
            expert_states.append(state_features)
            expert_actions.append(change_action)
            expert_features.append(comfort_zone_feats)

        all_states = np.concatenate(expert_states, axis=0)
        all_actions = np.concatenate(expert_actions, axis=0)
        all_features = np.concatenate(expert_features, axis=0)

        states_tensor = torch.tensor(all_states, dtype=torch.float32).to(device)
        actions_tensor = torch.tensor(all_actions, dtype=torch.float32).to(device)
        features_tensor = torch.tensor(all_features, dtype=torch.float32).to(device)
        reward_fn.initialize_normalization(states_tensor, actions_tensor, features_tensor)
        print("Separate normalization initialized for temps-hold-dwell-prev-comfort-zonal GRU reward")

    elif reward_type == 'temps_hold_dwell_prev_energy_comfort_zonal_gru':
        if not expert_trajs:
            raise ValueError("Expert trajectories are required to initialize temps_hold_dwell_prev_energy_comfort_zonal_gru.")

        state_dim_full = expert_trajs[0]['states'].shape[1]
        action_dim_full = expert_trajs[0]['actions'].shape[1]
        num_zones = _infer_airl_num_zones(state_dim_full, action_dim_full)
        if num_zones <= 0:
            raise ValueError("Invalid zone dimension computed for temps_hold_dwell_prev_energy_comfort_zonal_gru.")

        state_feature_dim = num_zones + 4  # zone temps + outdoor temp + hold counter + dwell + prev_window_state
        action_feature_dim = 1
        feature_dim = 2 * num_zones  # per-zone energy + per-zone comfort
        in_dim = state_feature_dim + action_feature_dim + feature_dim
        reward_fn = GRUReward(in_dim=in_dim, hid=64).to(device)
        print(f"Using temps-hold-dwell-prev-energy-comfort-zonal GRU reward function (input_dim={in_dim})")

        print("Initializing separate normalization from expert data...")
        expert_states = []
        expert_actions = []
        expert_features = []

        for traj in expert_trajs:
            states = np.array(traj["states"], dtype=np.float32)
            actions = np.array(traj["actions"], dtype=np.float32)
            energy_zone_feats = _zone_matrix_from_traj(
                traj,
                zone_key='energy_zone_features',
                scalar_key='energy_features',
                num_zones=num_zones,
            )
            comfort_zone_feats = _zone_matrix_from_traj(
                traj,
                zone_key='comfort_zone_features',
                scalar_key='comfort_features',
                num_zones=num_zones,
            )

            if states.shape[1] != state_dim_full:
                raise ValueError("Inconsistent state dimensionality detected in expert trajectories.")

            parts = _split_airl_state_components(states, action_dim=actions.shape[1], num_zones=num_zones)
            zone_temps = parts['zone_temps']
            outdoor_temp = parts['outdoor_temp']
            window_hold_counter = parts['window_hold_counter']
            window_dwell = parts['window_dwell']
            prev_window_state = parts['prev_window_state']
            change_action = actions[:, 0:1]

            state_features = np.concatenate(
                [zone_temps, outdoor_temp, window_hold_counter, window_dwell, prev_window_state],
                axis=1
            )
            features = np.concatenate([energy_zone_feats, comfort_zone_feats], axis=1)
            expert_states.append(state_features)
            expert_actions.append(change_action)
            expert_features.append(features)

        all_states = np.concatenate(expert_states, axis=0)
        all_actions = np.concatenate(expert_actions, axis=0)
        all_features = np.concatenate(expert_features, axis=0)

        states_tensor = torch.tensor(all_states, dtype=torch.float32).to(device)
        actions_tensor = torch.tensor(all_actions, dtype=torch.float32).to(device)
        features_tensor = torch.tensor(all_features, dtype=torch.float32).to(device)
        reward_fn.initialize_normalization(states_tensor, actions_tensor, features_tensor)
        print("Separate normalization initialized for temps-hold-dwell-prev-energy-comfort-zonal GRU reward")

    elif reward_type == 'temps_hold_prev_gru':
        if not expert_trajs:
            raise ValueError("Expert trajectories are required to initialize temps_hold_prev_gru.")

        state_dim_full = expert_trajs[0]['states'].shape[1]
        action_dim_full = expert_trajs[0]['actions'].shape[1]
        num_zones = _infer_airl_num_zones(state_dim_full, action_dim_full)
        if num_zones <= 0:
            raise ValueError("Invalid zone dimension computed for temps_hold_prev_gru.")

        state_feature_dim = num_zones + 3  # zone temps + outdoor temp + hold counter + prev_window_state
        reward_fn = GRURewardStateActionOnly(state_dim=state_feature_dim, action_dim=1, hid=64).to(device)
        print(f"Using temps-hold-prev GRU reward function (state_dim={state_feature_dim}, action_dim=1)")

        print("Initializing state/action normalization from expert data...")
        expert_states = []
        expert_actions = []

        for traj in expert_trajs:
            states = np.array(traj["states"], dtype=np.float32)
            actions = np.array(traj["actions"], dtype=np.float32)

            if states.shape[1] != state_dim_full:
                raise ValueError("Inconsistent state dimensionality detected in expert trajectories.")

            parts = _split_airl_state_components(states, action_dim=actions.shape[1], num_zones=num_zones)
            zone_temps = parts['zone_temps']
            outdoor_temp = parts['outdoor_temp']
            window_hold_counter = parts['window_hold_counter']
            prev_window_state = parts['prev_window_state']
            change_action = actions[:, 0:1]

            state_features = np.concatenate(
                [zone_temps, outdoor_temp, window_hold_counter, prev_window_state],
                axis=1
            )
            expert_states.append(state_features)
            expert_actions.append(change_action)

        all_states = np.concatenate(expert_states, axis=0)
        all_actions = np.concatenate(expert_actions, axis=0)

        states_tensor = torch.tensor(all_states, dtype=torch.float32).unsqueeze(0).to(device)
        actions_tensor = torch.tensor(all_actions, dtype=torch.float32).unsqueeze(0).to(device)
        reward_fn.initialize_normalization(states_tensor, actions_tensor)
        print("State/action normalization initialized for temps-hold-prev GRU reward")

    elif reward_type == 'state_action_only_mlp':
        # Only states and actions (no energy/comfort features) - MLP version
        reward_fn = MLPRewardStateActionOnly(state_dim=state_dim, action_dim=action_dim, hidden_dims=[64, 64]).to(device)
        print(f"Using state-action-only MLP reward function (state_dim={state_dim}, action_dim={action_dim})")

        # Initialize normalization from expert trajectories to mirror other reward heads
        print("Initializing state/action normalization from expert data...")
        expert_states = []
        expert_actions = []

        for traj in expert_trajs:
            states = np.array(traj["states"], dtype=np.float32)
            actions = np.array(traj["actions"], dtype=np.float32)
            if states.size == 0 or actions.size == 0:
                continue
            expert_states.append(states)
            expert_actions.append(actions)

        if expert_states and expert_actions:
            all_states = np.concatenate(expert_states, axis=0)
            all_actions = np.concatenate(expert_actions, axis=0)
            states_tensor = torch.tensor(all_states, dtype=torch.float32).to(device)
            actions_tensor = torch.tensor(all_actions, dtype=torch.float32).to(device)
            reward_fn.initialize_normalization(states_tensor, actions_tensor)
            print("State/action normalization initialized for MLP reward")
        else:
            print("Warning: no expert data available to initialize normalization for state-action-only MLP")

    elif reward_type == 'state_action_only_gru':
        # Only states and actions (no energy/comfort features) - GRU version
        reward_fn = GRURewardStateActionOnly(state_dim=state_dim, action_dim=action_dim, hid=64).to(device)
        print(f"Using state-action-only GRU reward function (state_dim={state_dim}, action_dim={action_dim})")

        print("Initializing state/action normalization from expert data...")
        expert_states = []
        expert_actions = []

        for traj in expert_trajs:
            states = np.array(traj["states"], dtype=np.float32)
            actions = np.array(traj["actions"], dtype=np.float32)
            if states.size == 0 or actions.size == 0:
                continue
            expert_states.append(states)
            expert_actions.append(actions)

        if expert_states and expert_actions:
            all_states = np.concatenate(expert_states, axis=0)
            all_actions = np.concatenate(expert_actions, axis=0)
            states_tensor = torch.tensor(all_states, dtype=torch.float32).unsqueeze(0).to(device)
            actions_tensor = torch.tensor(all_actions, dtype=torch.float32).unsqueeze(0).to(device)
            reward_fn.initialize_normalization(states_tensor, actions_tensor)
            print("State/action normalization initialized for GRU reward")
        else:
            print("Warning: no expert data available to initialize normalization for state-action-only GRU")

    elif reward_type == 'state_action_comfort_energy_gru':
        # States, actions, plus comfort & energy feature channels
        reward_fn = GRURewardStateActionComfortEnergy(
            state_dim=state_dim,
            action_dim=action_dim,
            feature_dim=2,
            hid=64,
        ).to(device)
        print("Using state-action + comfort/energy GRU reward function")

        print("Initializing separate normalization from expert data...")
        expert_states = []
        expert_actions = []
        expert_features = []

        for traj in expert_trajs:
            states = np.array(traj["states"], dtype=np.float32)
            actions = np.array(traj["actions"], dtype=np.float32)
            energy_feats = np.array(traj["energy_features"], dtype=np.float32).reshape(-1, 1)
            comfort_feats = np.array(traj["comfort_features"], dtype=np.float32).reshape(-1, 1)
            if states.size == 0 or actions.size == 0:
                continue

            features = np.concatenate([comfort_feats, energy_feats], axis=1)
            expert_states.append(states)
            expert_actions.append(actions)
            expert_features.append(features)

        if expert_states and expert_actions and expert_features:
            all_states = np.concatenate(expert_states, axis=0)
            all_actions = np.concatenate(expert_actions, axis=0)
            all_features = np.concatenate(expert_features, axis=0)

            states_tensor = torch.tensor(all_states, dtype=torch.float32).to(device)
            actions_tensor = torch.tensor(all_actions, dtype=torch.float32).to(device)
            features_tensor = torch.tensor(all_features, dtype=torch.float32).to(device)
            reward_fn.initialize_normalization(states_tensor, actions_tensor, features_tensor)
            print("Separate normalization initialized for state-action comfort/energy GRU reward")
        else:
            print("Warning: insufficient expert data to initialize state-action comfort/energy GRU reward")

    else:  # 'full'
        # Add 2 extra dimensions for energy and comfort features
        reward_fn = GRUReward(in_dim=state_dim + action_dim + 2, hid=64).to(device)
        print(f"Using full reward function (input_dim={state_dim + action_dim + 2})")
        
        # Initialize separate normalization from expert trajectories
        print("Initializing separate normalization from expert data...")
        expert_states = []
        expert_actions = []
        expert_features = []
        
        for traj in expert_trajs:
            states = np.array(traj["states"])  # Shape: (T, 12)
            actions = np.array(traj["actions"])  # Shape: (T, 8)
            energy_feats = np.array(traj["energy_features"]).reshape(-1, 1)
            comfort_feats = np.array(traj["comfort_features"]).reshape(-1, 1)
            
            # For full reward, use all states, all actions, energy+comfort as features
            features = np.concatenate([energy_feats, comfort_feats], axis=1)  # (T, 2)
            
            expert_states.append(states)
            expert_actions.append(actions)
            expert_features.append(features)
        
        # Combine all expert data for normalization
        all_states = np.concatenate(expert_states, axis=0)
        all_actions = np.concatenate(expert_actions, axis=0)
        all_features = np.concatenate(expert_features, axis=0)
        
        states_tensor = torch.tensor(all_states, dtype=torch.float32).to(device)
        actions_tensor = torch.tensor(all_actions, dtype=torch.float32).to(device)
        features_tensor = torch.tensor(all_features, dtype=torch.float32).to(device)
        
        reward_fn.initialize_normalization(states_tensor, actions_tensor, features_tensor)
        print(f"Separate normalization initialized - States: {reward_fn.normalizer.state_mean.shape}, Actions: {reward_fn.normalizer.action_mean.shape}, Features: {reward_fn.normalizer.feature_mean.shape}")
    
    if collect_days is None:
        collect_days = min(20, len(env.daily_data_list))
    if ppo_batch_days is None:
        ppo_batch_days = min(20, len(env.daily_data_list))
    if ppo_epochs is None:
        ppo_epochs = 6
    if validation_days is None:
        validation_days = min(10, len(eval_env.daily_data_list))

    output_dir_path = ensure_repo_dir(output_dir)

    print(f"Starting AIRL training for {n_iters} iterations...")

    iteration_metrics: List[Dict[str, float]] = []

    for outer_iter in range(n_iters):
        print(f"\n=== AIRL Iteration {outer_iter + 1}/{n_iters} ===")
        
        # 1) Generate trajectories with current policy
        policy_trajs = collect_trajectories(env, policy, num_days=collect_days)
        
        # 2) Update reward function
        steps_this_iter = int(reward_update_steps)
        if outer_iter == 0 and initial_reward_update_steps is not None:
            steps_this_iter = int(initial_reward_update_steps)
        steps_this_iter = max(1, steps_this_iter)

        update_reward(
            reward_fn,
            expert_trajs,
            policy_trajs,
            lr=reward_lr,
            weight_decay=0,
            num_steps=steps_this_iter,
            reward_type=reward_type,
            grad_clip_max_norm=grad_clip_max_norm,
        )

        do_initial_warmup_alignment = (
            outer_iter == 0 and (
                initial_reward_update_steps is not None
                or initial_reward_min_margin > 0.0
                or initial_reward_max_attempts > 1
            )
        )
        if do_initial_warmup_alignment:
            target_margin = float(initial_reward_min_margin)
            max_attempts = max(1, int(initial_reward_max_attempts))
            attempt_idx = 1
            d_exp, d_pol = discriminator_mean_probs(reward_fn, expert_trajs, policy_trajs, reward_type)
            print(
                f"Initial discriminator check attempt {attempt_idx}/{max_attempts}: "
                f"D(exp)={d_exp:.3f}, D(pol)={d_pol:.3f}, margin={d_exp - d_pol:.3f}"
            )

            while attempt_idx < max_attempts and d_exp <= d_pol + target_margin:
                attempt_idx += 1
                print(
                    f"Initial discriminator under target (need D(exp) > D(pol)+{target_margin:.3f}). "
                    f"Running extra warm-up attempt {attempt_idx}/{max_attempts}..."
                )
                update_reward(
                    reward_fn,
                    expert_trajs,
                    policy_trajs,
                    lr=reward_lr,
                    weight_decay=0,
                    num_steps=steps_this_iter,
                    reward_type=reward_type,
                    grad_clip_max_norm=grad_clip_max_norm,
                )
                d_exp, d_pol = discriminator_mean_probs(reward_fn, expert_trajs, policy_trajs, reward_type)
                print(
                    f"Initial discriminator check attempt {attempt_idx}/{max_attempts}: "
                    f"D(exp)={d_exp:.3f}, D(pol)={d_pol:.3f}, margin={d_exp - d_pol:.3f}"
                )

            if d_exp <= d_pol + target_margin:
                print(
                    f"Warning: initial discriminator target not met after {max_attempts} attempts "
                    f"(D(exp)={d_exp:.3f}, D(pol)={d_pol:.3f}, required margin={target_margin:.3f})."
                )
            else:
                print(
                    f"Initial discriminator target met after {attempt_idx} attempt(s) "
                    f"(D(exp)={d_exp:.3f}, D(pol)={d_pol:.3f})."
                )

        # 3) Update policy using PPO with learned reward
        train_ppo_irl(
            env, validation_env, policy, reward_fn, optimizer, outer_iter,
            batch_days=ppo_batch_days, epochs=ppo_epochs, updates=ppo_updates,
            gamma=0.99, lam=0.95, clip_eps=0.2, val_days=validation_days,
            reward_type=reward_type, grad_clip_max_norm=grad_clip_max_norm,
            entropy_coeff=entropy_coeff
        )

        metrics = evaluate_policy_metrics(
            policy,
            eval_env,
            reward_fn,
            num_days=validation_days,
            reward_type=reward_type,
        )
        iteration_metrics.append(metrics)
        print(
            f"Iteration {outer_iter + 1} metrics | "
            f"cost {metrics['avg_return']:.3f} | "
            f"learned {metrics['avg_learned_return']:.3f} | "
            f"energy/h {metrics['avg_energy_per_hour']:.4f} | "
            f"comfort % {metrics['avg_comfort_pct']:.2f} | "
            f"window switches/day {metrics['avg_window_switches']:.2f} | "
            f"window open ratio {metrics['avg_window_open_ratio']:.3f}"
        )

        policy_ckpt_path = output_dir_path / f"AIRL_policy_{reward_type}_{outer_iter}.pth"
        reward_ckpt_path = output_dir_path / f"AIRL_reward_{reward_type}_{outer_iter}.pth"
        torch.save(policy.state_dict(), policy_ckpt_path)
        torch.save(reward_fn.state_dict(), reward_ckpt_path)
        print(f"Saved iteration checkpoints: {policy_ckpt_path}, {reward_ckpt_path}")
    
    policy.training_metrics = iteration_metrics
    summary_env = eval_env
    summary_w_energy = float(getattr(summary_env, 'w_energy', getattr(env, 'w_energy', 1.0)))
    summary_w_comfort = float(getattr(summary_env, 'w_comfort', getattr(env, 'w_comfort', 1.0)))
    summary_w_window = float(getattr(summary_env, 'w_window', getattr(env, 'w_window', 1.0)))
    summary_scalers = getattr(summary_env, 'scalers', getattr(env, 'scalers', None))
    summary_zone_cols: Sequence[str] = []
    if hasattr(summary_env, 'columns'):
        summary_zone_cols = list(summary_env.columns.get('zone_cols', []))
    summary_num_zones = getattr(summary_env, 'num_zones', len(summary_zone_cols))

    expert_summary_source = expert_trajs
    if validation_env is not None:
        try:
            expert_summary_source = extract_expert_trajectories(
                validation_env.daily_data_list,
                validation_env.scalers,
                window_col='Z1 Windows Open Close Status',
                comfort_max=getattr(validation_env, 'comfort_max', getattr(env, 'comfort_max', 27.0)),
                comfort_min=getattr(validation_env, 'comfort_min', getattr(env, 'comfort_min', 21.0)),
            )
        except Exception as exc:
            print(f"Warning: failed to extract validation expert trajectories ({exc}); using training demos instead.")
    policy.expert_metrics = summarize_expert_metrics(
        expert_summary_source,
        reward_fn,
        reward_type,
        w_energy=summary_w_energy,
        w_comfort=summary_w_comfort,
        w_window=summary_w_window,
        scalers=summary_scalers,
        zone_cols=summary_zone_cols,
        num_zones=summary_num_zones,
    )

    print("\nAIRL training completed!")
    return policy, reward_fn
