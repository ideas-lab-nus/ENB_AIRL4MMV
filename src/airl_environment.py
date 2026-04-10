"""
Hybrid-mode environment wrapper for AIRL training.

Bridges legacy AIRL action dictionaries (window change + supply temps)
onto the discrete hybrid action space used by the PPO environment, while
keeping rewards at zero so the AIRL reward function can be learned.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np

try:
    from .data_processing import inverse_normalize, normalize
    from .environment import MixedModeVentilationEnv
    from .rl_environment import (
        MODE_MECH,
        MODE_MIXED,
        MODE_NATURAL,
        MixedModeVentilationRLEnv,
        idx_of_nearest,
    )
except ImportError:
    from data_processing import inverse_normalize, normalize
    from environment import MixedModeVentilationEnv
    from rl_environment import (
        MODE_MECH,
        MODE_MIXED,
        MODE_NATURAL,
        MixedModeVentilationRLEnv,
        idx_of_nearest,
    )


class MixedModeVentilationAIRLEnv(MixedModeVentilationRLEnv):
    """AIRL-compatible wrapper over the hybrid PPO environment."""

    def __init__(self, *args: Any, policy_time_features: bool = False, **kwargs: Any) -> None:
        # Keep reward weights (used internally for normalization) but AIRL ignores shaped reward.
        self.policy_time_features = bool(policy_time_features)
        super().__init__(*args, **kwargs)
        self._airl_force_local_on = False

    def _policy_time_encoding(self) -> np.ndarray:
        """Compact clock features appended to AIRL observations for the policy."""
        horizon = max(int(getattr(self, 'horizon', getattr(self, 'max_steps', 1))), 1)
        step = min(int(getattr(self, 'current_step', 0)), horizon - 1)
        phase = (2.0 * np.pi * float(step)) / float(horizon)
        return np.asarray([np.sin(phase), np.cos(phase)], dtype=np.float32)

    def _legacy_state(self) -> np.ndarray:
        """Return the legacy 12-D state vector expected by AIRL reward pipelines."""
        state = MixedModeVentilationEnv._get_state(self)
        if not self.policy_time_features:
            return state

        prefix = state[:self.num_zones + 3]
        suffix = state[-4:]
        return np.concatenate([prefix, self._policy_time_encoding(), suffix]).astype(np.float32)

    def reset(self, day_index: Optional[int] = None) -> np.ndarray:
        super().reset(day_index=day_index)
        return self._legacy_state()

    def step(
        self,
        action: Union[
            Tuple[int, Optional[Sequence[int]]],
            Dict[str, Any],
        ],
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one step allowing either hybrid (mode, fcu_idx) tuples or legacy AIRL dictionaries.
        Returns zero reward to allow AIRL to provide learned rewards.
        """
        hybrid_action = self._convert_action(action)
        _, _reward, done, info = super().step(hybrid_action)

        # Align info fields expected by AIRL training code
        info.setdefault('energy_consumption', info.get('energy', 0.0))
        info.setdefault('comfort_violations', info.get('comfort', 0.0))
        info.setdefault(
            'energy_consumption_zones',
            np.full(self.num_zones, float(info.get('energy_consumption', 0.0)) / max(self.num_zones, 1), dtype=np.float32),
        )
        info.setdefault(
            'comfort_violations_zones',
            np.full(self.num_zones, float(info.get('comfort_violations', 0.0)) / max(self.num_zones, 1), dtype=np.float32),
        )

        # Override shaped reward with zero for AIRL objective
        reward = 0.0
        return self._legacy_state(), reward, done, info

    def _convert_action(
        self,
        action: Union[
            Tuple[int, Optional[Sequence[int]]],
            Dict[str, Any],
        ],
    ) -> Tuple[int, Optional[Sequence[int]]]:
        """
        Convert legacy AIRL action dicts into the hybrid (mode, fcu_idx) tuple.
        """
        if isinstance(action, tuple):
            return action
        if not isinstance(action, dict):
            raise TypeError(f"Unsupported action type: {type(action)}")

        change = int(action.get('change', 0))
        target_window = (self.prev_window ^ change) & 1

        if target_window == 0:
            mode = MODE_MECH
            supply = np.asarray(action.get('supply_temps', self.prev_supply_temps), dtype=np.float32)
            fcu_idx: Sequence[int] = self._map_supply_to_indices(supply)
            self._airl_force_local_on = True
        else:
            # Legacy AIRL policy had no explicit mixed-mode output; default to natural ventilation.
            lc = np.asarray(action.get('local_cooling', self.prev_lc_temps), dtype=np.float32)
            mixed_flag = bool(np.mean(lc) > 0.5)
            mode = MODE_MIXED if mixed_flag else MODE_NATURAL
            fcu_idx = None
            self._airl_force_local_on = False

        return mode, fcu_idx

    def _map_supply_to_indices(self, supply_norm: Iterable[float]) -> Sequence[int]:
        """
        Map normalized supply temperature proposals to the discrete index set used by the hybrid env.
        """
        indices = []
        for zone, value in enumerate(supply_norm):
            col = self.columns['fcu_supply_cols'][zone]
            supply_c = inverse_normalize(float(value), col, self.scalers)
            indices.append(idx_of_nearest(supply_c))
        return indices

    def _prepare_local_targets(self, doas: int) -> Tuple[np.ndarray, np.ndarray]:
        lc_cols = self.columns['lc_cols']
        num_lc = len(lc_cols)
        if num_lc == 0:
            return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32)

        idx = min(self.current_step, self.max_steps - 1)
        row = self.current_day_data.iloc[idx]

        force_local = getattr(self, '_airl_force_local_on', False)
        if doas or force_local:
            target_c = 21.0
            lc_on = np.ones(num_lc, dtype=np.float32)
        else:
            target_c = float(row['OutdoorTemperatureWindow'])
            lc_on = np.zeros(num_lc, dtype=np.float32)

        temps = []
        for i, col in enumerate(lc_cols):
            prev_c = inverse_normalize(
                self.prev_lc_temps[i],
                col,
                self.scalers,
            ) if self.prev_lc_temps is not None else target_c
            lo = prev_c - 3.0
            hi = prev_c + 3.0
            clamped_c = float(np.clip(target_c, lo, hi))
            temps.append(normalize(clamped_c, col, self.scalers))

        return np.asarray(temps, dtype=np.float32), lc_on.astype(np.float32)


def create_airl_environment(
    daily_data_list,
    dynamics_model,
    scalers,
    **kwargs: Any,
) -> MixedModeVentilationAIRLEnv:
    """Factory mirroring rl_environment.create_rl_environment signature."""
    return MixedModeVentilationAIRLEnv(
        daily_data_list=daily_data_list,
        lstm_model=dynamics_model,
        scalers=scalers,
        **kwargs,
    )
