"""Hybrid-mode PPO environment with hierarchical discrete actions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch

try:
    from .data_processing import inverse_normalize, normalize
    from .environment import MixedModeVentilationEnv
except ImportError:
    from data_processing import inverse_normalize, normalize
    from environment import MixedModeVentilationEnv


# ====== Hybrid action constants ======
DT_MIN = 1
DAY_STEPS = 11 * 60 + 30  # 7:30 ~ 19:00 => 690 steps

MODE_DIM = 3
MODE_NATURAL = 0
MODE_MIXED = 1
MODE_MECH = 2

FCU_ZONES = 5
TEMP_MIN, TEMP_MAX, TEMP_STEP = 12.0, 23.0, 0.5
TEMP_LEVELS = [TEMP_MIN + k * TEMP_STEP for k in range(int((TEMP_MAX - TEMP_MIN) / TEMP_STEP) + 1)]
N_TEMP = len(TEMP_LEVELS)

MAX_RISE_PER_STEP = 2.0   # ℃ increase limit per step (warmer supply air)
MAX_DROP_PER_STEP = 6.0   # ℃ decrease limit per step (cooler supply air)
MAX_RISE_IDX = int(MAX_RISE_PER_STEP / TEMP_STEP)
MAX_DROP_IDX = int(MAX_DROP_PER_STEP / TEMP_STEP)

T_SET = 29.0
DEFAULT_W_E = 1.0
DEFAULT_W_C = 3.0
DEFAULT_W_W = 0.2


def _ensure_numpy(array: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
    if isinstance(array, np.ndarray):
        return array.astype(np.float32, copy=False)
    return np.asarray(array, dtype=np.float32)


class RunningNorm:
    """Track running mean/variance for online normalization."""

    def __init__(self, eps: float = 1e-4):
        self.count = eps
        self.mean = 0.0
        self.var = 1.0

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        batch_mean = x.mean(axis=0)
        batch_var = ((x - batch_mean) ** 2).mean(axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        mixed = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        new_var = mixed / tot_count

        self.mean = new_mean
        self.var = np.maximum(new_var, 1e-6)
        self.count = tot_count

    def normalize(self, x: np.ndarray, update: bool = True) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if update:
            self.update(x.reshape(1, -1))
        return ((x - self.mean) / np.sqrt(self.var + 1e-8)).astype(np.float32)

    def __call__(self, x: np.ndarray, update: bool = True) -> np.ndarray:
        return self.normalize(x, update=update)


class AdaptiveMaxScaler:
    """Maintain a slowly adapting max for positive quantities."""

    def __init__(self, init_max: float = 1.0, decay: float = 0.995):
        self.value = max(init_max, 1e-6)
        self.decay = decay

    def normalize(self, x: float) -> float:
        self.value = max(self.value * self.decay, x, 1e-6)
        return float(np.clip(x / self.value, 0.0, 1.0))


def time_encoding(step: int, total: int = DAY_STEPS) -> np.ndarray:
    phase = 2.0 * math.pi * (step % total) / float(max(total, 1))
    return np.array([
        math.sin(phase),
        math.cos(phase),
        math.sin(2 * phase),
        math.cos(2 * phase),
    ], dtype=np.float32)


def one_hot(index: int, dim: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    if 0 <= index < dim:
        vec[index] = 1.0
    return vec


def idx_of_nearest(temp_celsius: float) -> int:
    diffs = [abs(temp_celsius - level) for level in TEMP_LEVELS]
    return int(np.argmin(diffs))


def clamp_index(idx: int, lo: int, hi: int) -> int:
    return int(np.clip(idx, lo, hi))


@dataclass
class HybridStepInfo:
    energy: float
    comfort: float
    window_switch: float
    mode: int
    window: int
    doas: int
    fcu_indices: Optional[List[int]]


class HybridEnergyEstimator:
    """Lightweight energy estimator combining FCU and DOAS loads."""

    def __init__(self,
                 scalers: Dict[str, Dict[str, float]],
                 zone_cols: Sequence[str],
                 supply_cols: Sequence[str],
                 doas_kw: float = 4.0,
                 fcu_kw_per_deg: float = 0.25):
        self.scalers = scalers
        self.zone_cols = list(zone_cols)
        self.supply_cols = list(supply_cols)
        self.doas_kw = doas_kw
        self.fcu_kw_per_deg = fcu_kw_per_deg

    def estimate(self,
                 window: int,
                 doas: int,
                 fcu_indices: Optional[Sequence[int]],
                 zone_temps_norm: np.ndarray) -> float:
        energy_total, _ = self.estimate_with_breakdown(window, doas, fcu_indices, zone_temps_norm)
        return energy_total

    def estimate_with_breakdown(self,
                                window: int,
                                doas: int,
                                fcu_indices: Optional[Sequence[int]],
                                zone_temps_norm: np.ndarray) -> Tuple[float, np.ndarray]:
        zone_c = [
            inverse_normalize(zone_temps_norm[i], self.zone_cols[i], self.scalers)
            for i in range(len(self.zone_cols))
        ]

        energy_kw = 0.0
        zone_energy_kw = np.zeros(len(self.zone_cols), dtype=np.float32)

        # FCU cooling load only when windows closed (mechanical)
        if window == 0 and fcu_indices is not None:
            for z, (idx, zone_temp) in enumerate(zip(fcu_indices, zone_c)):
                setpoint = TEMP_LEVELS[idx]
                delta = max(0.0, zone_temp - setpoint)
                zone_kw = self.fcu_kw_per_deg * delta
                energy_kw += zone_kw
                zone_energy_kw[z] += float(zone_kw)

        # DOAS fan/coil penalty when active (mixed + mechanical)
        if doas:
            energy_kw += self.doas_kw
            if len(zone_energy_kw) > 0:
                zone_energy_kw += float(self.doas_kw) / float(len(zone_energy_kw))

        # Convert to kWh per minute
        return energy_kw / 60.0, zone_energy_kw / 60.0


class MixedModeVentilationRLEnv(MixedModeVentilationEnv):
    """Hybrid hierarchical control environment tailored for H-PPO."""

    def __init__(self,
                 daily_data_list: List[pd.DataFrame],
                 lstm_model: torch.nn.Module,
                 scalers: Dict[str, Dict[str, float]],
                 num_zones: int = FCU_ZONES,
                 comfort_min: float = 21.0,
                 comfort_max: float = 30.0,
                 heavy_wind_threshold: float = 8.0,
                 energy_estimator: Optional[HybridEnergyEstimator] = None,
                 weight_energy: float = DEFAULT_W_E,
                 weight_comfort: float = DEFAULT_W_C,
                 weight_window: float = DEFAULT_W_W):
        super().__init__(daily_data_list, lstm_model, scalers, num_zones,
                         comfort_min, comfort_max, heavy_wind_threshold)

        self.state_norm = RunningNorm()
        self.energy_scaler = AdaptiveMaxScaler(init_max=1.0)
        self.energy_estimator = energy_estimator

        self.w_energy = weight_energy
        self.w_comfort = weight_comfort
        self.w_window = weight_window

        self.total_energy = 0.0
        self.total_violation = 0.0
        self.total_switch = 0.0

        self.prev_mode = MODE_MECH
        self.prev_window = 0
        self.prev_fcu_idx: List[int] = [idx_of_nearest(T_SET)] * self.num_zones
        self.horizon = DAY_STEPS

    def clone(self, daily_data: Optional[List[pd.DataFrame]] = None) -> "MixedModeVentilationRLEnv":
        """Create a fresh copy sharing the same dynamics/scalers but new internal state."""
        doas_kw = getattr(self.energy_estimator, 'doas_kw', 4.0) if self.energy_estimator else 4.0
        fcu_kw = getattr(self.energy_estimator, 'fcu_kw_per_deg', 0.25) if self.energy_estimator else 0.25
        estimator = None
        if self.energy_estimator is not None:
            estimator = HybridEnergyEstimator(
                self.scalers,
                self.columns['zone_cols'],
                self.columns['fcu_supply_cols'][:self.num_zones],
                doas_kw,
                fcu_kw,
            )

        return MixedModeVentilationRLEnv(
            daily_data_list=daily_data if daily_data is not None else self.daily_data_list,
            lstm_model=self.lstm_model,
            scalers=self.scalers,
            num_zones=self.num_zones,
            comfort_min=self.comfort_min,
            comfort_max=self.comfort_max,
            heavy_wind_threshold=self.heavy_wind_threshold,
            energy_estimator=estimator,
            weight_energy=self.w_energy,
            weight_comfort=self.w_comfort,
            weight_window=self.w_window,
        )

    # ------------------------------------------------------------------
    # Reset & observation helpers
    # ------------------------------------------------------------------
    def reset(self, day_index: Optional[int] = None) -> np.ndarray:
        super().reset(day_index)
        self.horizon = min(self.max_steps, DAY_STEPS)
        self.current_step = 0

        # Initialize FCU setpoints from historical supply temps
        initial_supply = [
            inverse_normalize(self.prev_supply_temps[z], self.columns['fcu_supply_cols'][z], self.scalers)
            for z in range(self.num_zones)
        ]
        self.prev_fcu_idx = [idx_of_nearest(t) for t in initial_supply]

        self.prev_mode = MODE_MECH
        self.prev_window = 0

        self.total_energy = 0.0
        self.total_violation = 0.0
        self.total_switch = 0.0

        obs = self._build_obs()
        return self.state_norm(obs, update=True)

    def _build_obs(self) -> np.ndarray:
        # Use current row or last available for observation features
        idx = min(self.current_step, self.max_steps - 1)
        row = self.current_day_data.iloc[idx]

        zone_c = [
            inverse_normalize(self.zone_temps[i], self.columns['zone_cols'][i], self.scalers)
            for i in range(self.num_zones)
        ]
        tout = float(row['OutdoorTemperatureWindow'])
        horizon = max(int(self.horizon), 1)
        progress = min(float(self.current_step) / float(horizon), 1.0)
        time_progress = np.array([progress, 1.0 - progress], dtype=np.float32)
        obs = np.concatenate([
            np.asarray(zone_c, dtype=np.float32),
            np.array([tout], dtype=np.float32),
            time_encoding(self.current_step),
            time_progress,
            one_hot(self.prev_mode, MODE_DIM),
            np.array([TEMP_LEVELS[idx] for idx in self.prev_fcu_idx], dtype=np.float32)
        ]).astype(np.float32)
        return obs

    # ------------------------------------------------------------------
    # Core step logic
    # ------------------------------------------------------------------
    def step(self, action: Union[Tuple[int, Optional[Sequence[int]]], Dict[str, Any]]) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if isinstance(action, dict):
            mode = int(action.get('mode', MODE_MECH))
            fcu_idx = action.get('fcu_idx')
        else:
            mode, fcu_idx = action

        if mode not in (MODE_NATURAL, MODE_MIXED, MODE_MECH):
            raise ValueError(f"Invalid mode {mode}")

        window = 1 if mode in (MODE_NATURAL, MODE_MIXED) else 0
        doas = 1 if mode in (MODE_MIXED, MODE_MECH) else 0

        # Maintain legacy window state tracking for AIRL compatibility
        self.dwell = 1.0 / max(1, self.current_step - self.last_switch_time)
        if window != getattr(self, 'window_prev_state', 0):
            self.switch_flag = 1
            self.window_hold_counter = 0.0
            self.last_switch_time = self.current_step
        else:
            self.switch_flag = 0
            self.window_hold_counter += 1.0 / 720.0

        prev_zone_norm = self.zone_temps.copy()
        prev_window = self.prev_window

        effective_idx: Optional[List[int]] = None
        if mode == MODE_MECH:
            if fcu_idx is None:
                raise ValueError("Mechanical mode requires FCU indices")
            if len(fcu_idx) != self.num_zones:
                raise ValueError(f"Expected {self.num_zones} FCU indices, got {len(fcu_idx)}")
            effective_idx = []
            for j, idx in enumerate(fcu_idx):
                prev_idx = self.prev_fcu_idx[j]
                lo = max(0, prev_idx - MAX_DROP_IDX)
                hi = min(N_TEMP - 1, prev_idx + MAX_RISE_IDX)
                effective_idx.append(clamp_index(int(idx), lo, hi))
        else:
            effective_idx = None

        supply_norm = self._prepare_supply_targets(effective_idx)
        local_norm, lc_on = self._prepare_local_targets(doas)

        row = self.current_day_data.iloc[min(self.current_step, self.max_steps - 1)]
        outdoor_norm = normalize(row['OutdoorTemperatureWindow'], 'OutdoorTemperatureWindow', self.scalers)
        wind_speed_norm = normalize(row['Wind Speed'], 'Wind Speed', self.scalers)
        wind_dir_norm = normalize(row['Wind Direction'], 'Wind Direction', self.scalers)

        action_dict = {
            'supply_temps': supply_norm,
            'local_cooling': local_norm,
            'lc_on': lc_on,
        }

        next_zone_norm = self._predict_next_temperatures(
            action_dict,
            outdoor_norm,
            wind_speed_norm,
            wind_dir_norm,
            window
        )

        # Energy estimate uses previous zone temps for load computation
        self.zone_temps = prev_zone_norm
        energy_action = {
            'supply_temps': supply_norm,
            'local_cooling': local_norm,
            'lc_on': lc_on,
        }
        if self.energy_estimator is not None:
            energy, energy_zone = self.energy_estimator.estimate_with_breakdown(
                window, doas, effective_idx, prev_zone_norm
            )
        else:
            energy, energy_zone = self._compute_energy_consumption(
                energy_action, window, return_zone_breakdown=True
            )
        self.zone_temps = next_zone_norm

        violation_by_zone = self._compute_violation_per_zone(next_zone_norm)
        violation = float(np.sum(violation_by_zone))
        delta_w = 1.0 if window != prev_window else 0.0

        avg_violation = violation / (float(self.num_zones) + 1e-6)
        reward = -(
            self.w_energy * self.energy_scaler.normalize(energy) +
            self.w_comfort * avg_violation +
            self.w_window * delta_w
        )

        self.total_energy += energy
        self.total_violation += violation
        self.total_switch += delta_w

        self.prev_mode = mode
        self.prev_window = window
        self.window_prev_state = window
        if effective_idx is not None:
            self.prev_fcu_idx = effective_idx

        self.prev_supply_temps = supply_norm.copy()
        self.prev_lc_temps = local_norm.copy()

        self.current_step += 1
        done = self.current_step >= self.horizon

        info = HybridStepInfo(
            energy=energy,
            comfort=violation,
            window_switch=delta_w,
            mode=mode,
            window=window,
            doas=doas,
            fcu_indices=effective_idx.copy() if effective_idx is not None else None
        ).__dict__
        info['energy_consumption_zones'] = np.asarray(energy_zone, dtype=np.float32)
        info['comfort_violations_zones'] = np.asarray(violation_by_zone, dtype=np.float32)

        obs = self._build_obs()
        return self.state_norm(obs, update=True), reward, done, info

    # ------------------------------------------------------------------
    # Helper routines
    # ------------------------------------------------------------------
    def _prepare_supply_targets(self, effective_idx: Optional[Sequence[int]]) -> np.ndarray:
        if effective_idx is None:
            # FCUs idle: set to current zone temperatures (keeps dynamics consistent)
            zone_c = [
                inverse_normalize(self.zone_temps[i], self.columns['zone_cols'][i], self.scalers)
                for i in range(self.num_zones)
            ]
            temps_c = zone_c
        else:
            temps_c = [TEMP_LEVELS[idx] for idx in effective_idx]

        temps_c = self._clamp_supply_changes(temps_c)
        supply_norm = [
            normalize(temps_c[i], self.columns['fcu_supply_cols'][i], self.scalers)
            for i in range(self.num_zones)
        ]
        return np.asarray(supply_norm, dtype=np.float32)

    def _clamp_supply_changes(self, proposed_c: Sequence[float]) -> List[float]:
        clamped = []
        for i, temp_c in enumerate(proposed_c):
            prev_c = inverse_normalize(
                self.prev_supply_temps[i],
                self.columns['fcu_supply_cols'][i],
                self.scalers,
            )
            lo = prev_c - MAX_DROP_PER_STEP
            hi = prev_c + MAX_RISE_PER_STEP
            clamped.append(float(np.clip(temp_c, lo, hi)))
        return clamped

    def _prepare_local_targets(self, doas: int) -> Tuple[np.ndarray, np.ndarray]:
        lc_cols = self.columns['lc_cols']
        num_lc = len(lc_cols)
        if num_lc == 0:
            return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32)

        idx = min(self.current_step, self.max_steps - 1)
        row = self.current_day_data.iloc[idx]

        if doas:
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

    def _compute_violation(self, zone_temps_norm: np.ndarray) -> float:
        return float(np.sum(self._compute_violation_per_zone(zone_temps_norm)))

    def _compute_violation_per_zone(self, zone_temps_norm: np.ndarray) -> np.ndarray:
        zone_violations = np.zeros(self.num_zones, dtype=np.float32)
        for i in range(self.num_zones):
            temp_c = inverse_normalize(zone_temps_norm[i], self.columns['zone_cols'][i], self.scalers)
            over = max(0.0, temp_c - T_SET)
            zone_violations[i] = float(over)
        return zone_violations

    # ------------------------------------------------------------------
    # Episode reporting
    # ------------------------------------------------------------------
    def get_episode_metrics(self) -> Dict[str, float]:
        return {
            'total_energy_consumption': self.total_energy,
            'total_comfort_violation': self.total_violation,
            'total_window_switch': self.total_switch,
            'episode_length': float(self.current_step)
        }


def create_rl_environment(
    daily_data_list: List[pd.DataFrame],
    dynamics_model: torch.nn.Module,
    scalers: Dict[str, Dict[str, float]],
    **kwargs: Any,
) -> MixedModeVentilationRLEnv:
    return MixedModeVentilationRLEnv(
        daily_data_list=daily_data_list,
        lstm_model=dynamics_model,
        scalers=scalers,
        **kwargs,
    )
