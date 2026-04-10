"""
Mixed-Mode Ventilation Environment for AIRL training.
"""
import numpy as np
import torch
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional, Sequence

try:
    from .data_processing import (
        extract_rain_status,
        get_column_definitions,
        inverse_normalize,
        normalize,
    )
except ImportError:
    from data_processing import extract_rain_status, get_column_definitions, inverse_normalize, normalize


# Environment constants
SUPPLY_NV = 0.55   # Natural ventilation mode (FCU off)
SUPPLY_AC = 0.2   # Air conditioning mode (low SAT)
LC_MIN, LC_MAX = 0.32, 0.82   # Local cooling bounds
DELTA_LC_NORM = 0.08  # Maximum change per timestep for local cooling supply temps

SUPPLY_MAX_DROP_C = 6.0  # ℃ decrease limit per step (cooler supply air)
SUPPLY_MAX_RISE_C = 2.0  # ℃ increase limit per step (warmer supply air)

LC_MAX_DROP_C = 6.0  # ℃ decrease limit per step for local cooling targets
LC_MAX_RISE_C = 6.0  # ℃ increase limit per step for local cooling targets


def clamp_ramp(new_sat: np.ndarray, old_sat: np.ndarray, delta: float) -> np.ndarray:
    """Apply rate limiting to supply air temperature changes."""
    return np.clip(new_sat, old_sat - delta, old_sat + delta)


class MixedModeVentilationEnv:
    """
    Environment simulation for mixed-mode ventilation control.
    
    Simulates building thermal dynamics using a pre-trained neural network model
    and provides reward signals for energy efficiency and thermal comfort.
    """
    
    def __init__(self, 
                 daily_data_list: List[pd.DataFrame], 
                 lstm_model: torch.nn.Module, 
                 scalers: Dict[str, Dict[str, float]],
                 num_zones: int = 5,
                 comfort_min: float = 21.0, 
                 comfort_max: float = 30.0,
                 heavy_wind_threshold: float = 8):
        """
        Initialize the environment.
        
        Args:
            daily_data_list: List of daily building data DataFrames
            lstm_model: Pre-trained dynamics model
            scalers: Feature normalization scalers
            num_zones: Number of thermal zones
            comfort_min: Minimum comfort temperature (°C)
            comfort_max: Maximum comfort temperature (°C)
            heavy_wind_threshold: Wind speed threshold for safety
        """
        self.daily_data_list = daily_data_list
        self.lstm_model = lstm_model
        self.scalers = scalers
        self.num_zones = num_zones
        
        # Comfort and safety parameters
        self.comfort_min = comfort_min
        self.comfort_max = comfort_max
        self.heavy_wind_threshold = heavy_wind_threshold
        
        # Get column definitions
        self.columns = get_column_definitions()
        
        # Internal state variables
        self.reset_state_variables()

    def reset_state_variables(self):
        """Reset all internal state variables."""
        self.current_day_data = None
        self.current_step = 0
        self.max_steps = 0
        self.zone_temps = None
        self.current_outdoor = None
        self.lstm_hidden = None
        self.window_hold_counter = 0
        self.window_prev_state = 0
        self.switch_flag = 0
        self.dwell = 0
        self.last_switch_time = -10000
        self.prev_supply_temps = None
        self.prev_lc_temps = None
        self.safety_violation = False

    def reset(self, day_index: Optional[int] = None) -> np.ndarray:
        """
        Reset environment for a new episode.
        
        Args:
            day_index: Specific day to use, or None for random selection
            
        Returns:
            Initial state vector
        """
        if day_index is None:
            day_index = np.random.choice(len(self.daily_data_list))
            
        self.current_day_data = self.daily_data_list[day_index].reset_index(drop=True)
        self.max_steps = len(self.current_day_data)
        self.current_step = 0
        
        # Initialize from first row of data
        row = self.current_day_data.iloc[self.current_step]
        
        # Initialize previous supply temperatures
        self.prev_supply_temps = np.array([
            normalize(row[col], col, self.scalers) 
            for col in self.columns['fcu_supply_cols']
        ], dtype=np.float32)
        
        self.prev_lc_temps = np.array([
            normalize(row[col], col, self.scalers) 
            for col in self.columns['lc_cols']
        ], dtype=np.float32)
        
        # Initialize zone temperatures
        self.zone_temps = np.array([
            normalize(row[col], col, self.scalers) 
            for col in self.columns['zone_cols']
        ], dtype=np.float32)
        
        # Reset other state variables
        self.window_hold_counter = 0
        self.switch_flag = 0
        self.dwell = 0
        self.window_prev_state = 0
        self.last_switch_time = -10000
        
        return self._get_state()

    def _clamp_supply_temperatures(self, proposed_c: Sequence[float]) -> np.ndarray:
        """Clamp FCU supply targets in Celsius before re-normalizing."""
        clamped = []
        for idx, temp_c in enumerate(proposed_c):
            prev_c = inverse_normalize(
                self.prev_supply_temps[idx],
                self.columns['fcu_supply_cols'][idx],
                self.scalers,
            )
            lo = prev_c - SUPPLY_MAX_DROP_C
            hi = prev_c + SUPPLY_MAX_RISE_C
            clamped_c = float(np.clip(temp_c, lo, hi))
            clamped.append(
                normalize(clamped_c, self.columns['fcu_supply_cols'][idx], self.scalers)
            )
        return np.asarray(clamped, dtype=np.float32)

    def _clamp_local_cooling_temperatures(self, proposed_norm: Sequence[float]) -> np.ndarray:
        """Clamp local cooling targets in Celsius before re-normalizing."""
        clamped = []
        for idx, temp_norm in enumerate(proposed_norm):
            col_name = self.columns['lc_cols'][idx]
            prev_c = inverse_normalize(
                self.prev_lc_temps[idx],
                col_name,
                self.scalers,
            )
            target_c = inverse_normalize(
                temp_norm,
                col_name,
                self.scalers,
            )
            lo = prev_c - LC_MAX_DROP_C
            hi = prev_c + LC_MAX_RISE_C
            clamped_c = float(np.clip(target_c, lo, hi))
            clamped.append(
                normalize(clamped_c, col_name, self.scalers)
            )
        return np.asarray(clamped, dtype=np.float32)

    def step(self, action: Dict[str, Any]) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Dictionary containing:
                - 'change': Binary signal to flip window state
                - 'supply_temps': FCU supply air temperatures
                - 'local_cooling': Binary on/off flags for localized cooling units
                
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        # Extract action components
        change = int(action['change'])
        windows = self.window_prev_state ^ change  # XOR flip

        # Get current weather conditions
        current_row = self.current_day_data.iloc[self.current_step]
        outdoor_temp = normalize(current_row['OutdoorTemperatureWindow'], 'OutdoorTemperatureWindow', self.scalers)
        wind_speed = normalize(current_row['Wind Speed'], 'Wind Speed', self.scalers)
        wind_direction = normalize(current_row['Wind Direction'], 'Wind Direction', self.scalers)
        
        # Update dwell time
        self.dwell = 1 / max(1, self.current_step - self.last_switch_time)

        # Process supply temperature actions based on mode
        proposed_supply_temps = np.asarray(action['supply_temps'], dtype=np.float32).copy()
        proposed_supply_c = np.zeros(self.num_zones, dtype=np.float32)

        # Interpret local cooling action as binary on/off decisions
        lc_action = np.array(action.get('local_cooling', 1), dtype=np.float32).flatten()
        if lc_action.size == 0:
            lc_action = np.ones(len(self.columns['lc_cols']), dtype=np.float32)
        if lc_action.size == 1 and len(self.columns['lc_cols']) > 1:
            lc_action = np.repeat(lc_action, len(self.columns['lc_cols']))
        lc_action = np.clip(lc_action, 0.0, 1.0)

        if windows == 0:
            lc_action = np.ones_like(lc_action)

        action['lc_on'] = lc_action.astype(np.float32)

        proposed_supply_temps_lc = np.zeros(len(self.columns['lc_cols']), dtype=np.float32)

        if windows == 0:  # AC mode
            for i, col in enumerate(self.columns['fcu_supply_cols']):
                proposed_supply_c[i] = inverse_normalize(
                    proposed_supply_temps[i], col, self.scalers
                )
            for i, col in enumerate(self.columns['lc_cols']):
                conditioned_temp = normalize(21.0, col, self.scalers)
                proposed_supply_temps_lc[i] = conditioned_temp
        else:  # NV mode
            # Set FCU supply temps to zone temps (natural ventilation)
            for z in range(self.num_zones):
                proposed_supply_c[z] = inverse_normalize(
                    self.zone_temps[z], self.columns['zone_cols'][z], self.scalers
                )
            for i, col in enumerate(self.columns['lc_cols']):
                if lc_action[i] >= 0.5:
                    conditioned_temp = normalize(21.0, col, self.scalers)
                    proposed_supply_temps_lc[i] = conditioned_temp
                else:
                    oat_norm = normalize(current_row['OutdoorTemperatureWindow'], col, self.scalers)
                    proposed_supply_temps_lc[i] = oat_norm
        action['supply_temps'] = self._clamp_supply_temperatures(proposed_supply_c)

        # Apply rate limiting with previous window state deltas
        action['local_cooling'] = self._clamp_local_cooling_temperatures(
            proposed_supply_temps_lc
        ).astype(np.float32)
        
        # Update window state tracking
        if windows != self.window_prev_state:
            self.switch_flag = 1
            self.window_hold_counter = 0
            self.last_switch_time = self.current_step
        else:
            self.window_hold_counter += 1/720
            self.switch_flag = 0
        
        # Predict next zone temperatures using dynamics model
        next_zone_temps = self._predict_next_temperatures(action, outdoor_temp, wind_speed, wind_direction, windows)
        
        # Compute energy consumption and comfort metrics for AIRL
        energy_consumption = self._compute_energy_consumption(action, windows)
        comfort_violations = self._compute_comfort_violations(self.zone_temps)
        
        # Update state
        self.zone_temps = next_zone_temps
        self.current_step += 1
        self.prev_supply_temps = action['supply_temps']
        self.prev_lc_temps = action['local_cooling']
        self.window_prev_state = windows
        
        # Check if episode is done
        done = self.current_step >= self.max_steps
        next_state = self._get_state()
        
        # Reward is zero for base environment (AIRL learns its own reward)
        reward = 0.0
        
        # Info contains energy and comfort features for AIRL reward network
        info = {
            'energy_consumption': energy_consumption,  # kWh per minute
            'comfort_violations': comfort_violations,  # degrees Celsius violation
            'rain_status': extract_rain_status(current_row),
            'window_state': windows,
            'switch_flag': self.switch_flag,
            'current_step': self.current_step,
            'zone_temperatures': self.zone_temps.copy(),
            'outdoor_temperature': current_row['OutdoorTemperatureWindow'],
            'lc_on': lc_action.copy()
        }
        
        return next_state, reward, done, info

    def _predict_next_temperatures(self, action: Dict[str, Any], outdoor_temp: float, 
                                 wind_speed: float, wind_direction: float, windows: int) -> np.ndarray:
        """Use the dynamics model to predict next zone temperatures."""
        # Prepare inputs for dynamics model
        indoor_temps = torch.tensor(self.zone_temps, dtype=torch.float32).unsqueeze(0).unsqueeze(1)
        local_cooling = torch.tensor(action['local_cooling'], dtype=torch.float32).unsqueeze(0).unsqueeze(1)
        supply_temps = torch.tensor(action['supply_temps'], dtype=torch.float32).unsqueeze(0).unsqueeze(1)
        out_temp = torch.tensor(outdoor_temp, dtype=torch.float32).unsqueeze(0).unsqueeze(1).unsqueeze(2)
        
        # AC branch input: indoor + outdoor + local cooling + supply temps
        ac_input = torch.cat([indoor_temps, out_temp, local_cooling, supply_temps], dim=-1)
        
        # NV branch input: indoor + outdoor conditions + local cooling
        outdoor_conditions = torch.tensor([out_temp, wind_speed, wind_direction], dtype=torch.float32).unsqueeze(0).unsqueeze(1)
        nv_input = torch.cat([indoor_temps, outdoor_conditions, local_cooling], dim=-1)
        
        # Mode selection signal
        window_signal = torch.tensor([windows], dtype=torch.float32)
        
        # # Debug model inputs
        # print(f"Model inputs debug (step {self.current_step}):")
        # print(f"  windows: {windows}")
        # print(f"  indoor_temps shape: {indoor_temps.shape}, values: {indoor_temps.squeeze()}")
        # print(f"  supply_temps shape: {supply_temps.shape}, values: {supply_temps.squeeze()}")
        # print(f"  local_cooling shape: {local_cooling.shape}, values: {local_cooling.squeeze()}")
        # print(f"  outdoor_temp: {outdoor_temp}")
        # print(f"  ac_input shape: {ac_input.shape}")
        # print(f"  nv_input shape: {nv_input.shape}")
        
        # Predict using dynamics model
        with torch.no_grad():
            model_output = self.lstm_model(x_ac=ac_input, x_nv=nv_input, j=window_signal)
            next_zone_temps = model_output.squeeze(0).detach().numpy()[:self.num_zones]
            # print(f"  model_output shape: {model_output.shape}")
            # print(f"  predicted next_zone_temps: {next_zone_temps}")
            
            # # Convert to Celsius for debugging
            # next_temps_celsius = [inverse_normalize(temp, self.columns['zone_cols'][i], self.scalers) 
            #                     for i, temp in enumerate(next_zone_temps)]
            # print(f"  predicted temps in Celsius: {next_temps_celsius}")
        
        return next_zone_temps

    def _compute_energy_consumption(self, action: Dict[str, Any], windows: int, return_zone_breakdown: bool = False):
        """
        Compute energy consumption for current action with proper denormalization.
        
        Args:
            action: Current action dictionary
            windows: Current window state (0=AC, 1=NV)
            
        Returns:
            Energy consumption value in kWh per minute.
            If return_zone_breakdown=True, also returns a per-zone energy vector.
        """
        # Constants for energy calculation
        SPECIFIC_HEAT_AIR = 1005  # J/(kg·K) - specific heat capacity of air
        AIR_DENSITY = 1.225       # kg/m³ - air density at standard conditions
        FCU_AIRFLOW_CPH = 800     # CFH - air changes per hour for FCU
        LOCAL_COOLING_CPH = 1000  # CFH - air changes per hour for localized cooling
        
        energy_consumption = 0.0
        zone_energy = np.zeros(self.num_zones, dtype=np.float32)
        
        if windows == 0:  # AC mode - FCU systems active
            for z in range(self.num_zones):
                # DENORMALIZE temperatures to actual °C values
                zone_temp_C = inverse_normalize(
                    self.zone_temps[z], self.columns['zone_cols'][z], self.scalers
                )
                supply_temp_C = inverse_normalize(
                    action['supply_temps'][z], self.columns['fcu_supply_cols'][z], self.scalers
                )
                
                # Calculate temperature difference (cooling load)
                delta_t = max(0.0, zone_temp_C - supply_temp_C)
                
                # Convert to energy consumption (simplified thermodynamic model)
                if delta_t > 0.5:  # Only count significant cooling loads
                    volume_flow_rate = FCU_AIRFLOW_CPH / 3600  # Convert CFH to m³/s
                    mass_flow_rate = volume_flow_rate * AIR_DENSITY  # kg/s
                    power_watts = mass_flow_rate * SPECIFIC_HEAT_AIR * delta_t  # Watts
                    energy_kwh_per_minute = power_watts / 1000 / 60  # Convert to kWh per minute
                    energy_consumption += energy_kwh_per_minute
                    zone_energy[z] += float(energy_kwh_per_minute)

        # Local cooling contributes whenever it is on (mixed/mechanical windows)
        lc_on_flags = action.get('lc_on', np.ones_like(action['local_cooling']))
        for i, lc_temp_norm in enumerate(action['local_cooling']):
            if i < len(lc_on_flags) and lc_on_flags[i] < 0.5:
                continue  # Local cooling turned off

            # DENORMALIZE local cooling temperature
            lc_temp_C = inverse_normalize(
                lc_temp_norm, self.columns['lc_cols'][i], self.scalers
            )

            # Calculate average zone temperature for reference
            avg_zone_temp_C = np.mean([
                inverse_normalize(zt, self.columns['zone_cols'][j], self.scalers)
                for j, zt in enumerate(self.zone_temps)
            ])
            
            # Calculate cooling energy for local cooling unit
            delta_t = max(0.0, avg_zone_temp_C - lc_temp_C)
            if delta_t > 0.5:  # Only count significant cooling
                volume_flow_rate = LOCAL_COOLING_CPH / 3600  # Convert CFH to m³/s
                mass_flow_rate = volume_flow_rate * AIR_DENSITY  # kg/s
                power_watts = mass_flow_rate * SPECIFIC_HEAT_AIR * delta_t  # Watts
                energy_kwh_per_minute = power_watts / 1000 / 60  # Convert to kWh per minute
                energy_consumption += energy_kwh_per_minute
                if self.num_zones > 0:
                    # Local cooling is not directly zone-mapped in this model.
                    # Spread the local contribution across zones so per-zone sums
                    # remain consistent with the scalar total.
                    zone_energy += float(energy_kwh_per_minute) / float(self.num_zones)
        
        if return_zone_breakdown:
            return energy_consumption, zone_energy
        return energy_consumption

    def _compute_comfort_violations(self, zone_temps: np.ndarray, return_zone_breakdown: bool = False):
        """
        Compute comfort violations for current zone temperatures using real temperatures.
        Each zone uses its own scaler to denormalize before checking against comfort bounds.
        The comfort_min and comfort_max parameters are in Celsius and used directly.
        
        Args:
            zone_temps: Current zone temperatures (normalized)
            
        Returns:
            Comfort violation value (in degrees Celsius).
            If return_zone_breakdown=True, also returns a per-zone violation vector.
        """
        comfort_violations = 0.0
        zone_violations = np.zeros(self.num_zones, dtype=np.float32)
        
        for z, temp_norm in enumerate(zone_temps):
            # Denormalize current zone temperature to real Celsius
            temp_celsius = inverse_normalize(
                temp_norm, self.columns['zone_cols'][z], self.scalers
            )
            
            # Use comfort bounds directly (already in Celsius)
            comfort_min_celsius = self.comfort_min
            comfort_max_celsius = self.comfort_max
            
            # Check if temperature is outside comfort range (in real temperatures)
            if temp_celsius > comfort_max_celsius:  # Too warm
                violation = temp_celsius - comfort_max_celsius
                comfort_violations += violation
                zone_violations[z] = float(violation)
            elif temp_celsius < comfort_min_celsius:  # Too cold
                violation = comfort_min_celsius - temp_celsius
                comfort_violations += violation
                zone_violations[z] = float(violation)
        
        if return_zone_breakdown:
            return comfort_violations, zone_violations
        return comfort_violations

    def _get_state(self) -> np.ndarray:
        """
        Build state vector from current conditions.
        
        Returns:
            State vector: [zone_temps, outdoor_temp, wind_speed, wind_direction, 
                          window_hold_counter, switch_flag, dwell, prev_window_state]
        """
        if self.current_step < self.max_steps:
            row = self.current_day_data.iloc[self.current_step]
        else:
            row = self.current_day_data.iloc[-1]  # Use last available data
        
        state = np.concatenate((
            self.zone_temps,
            [
                normalize(row['OutdoorTemperatureWindow'], 'OutdoorTemperatureWindow', self.scalers),
                normalize(row['Wind Speed'], 'Wind Speed', self.scalers),
                normalize(row['Wind Direction'], 'Wind Direction', self.scalers),
                self.window_hold_counter,
                self.switch_flag,
                self.dwell,
                self.window_prev_state
            ]
        )).astype(np.float32)
        
        return state

    def get_current_day_info(self) -> Dict[str, Any]:
        """Get information about the current day being simulated."""
        if self.current_day_data is not None:
            return {
                'day_start': self.current_day_data['date'].iloc[0],
                'day_length': len(self.current_day_data),
                'current_step': self.current_step
            }
        return {}


def create_environment(daily_data_list: List[pd.DataFrame], 
                      dynamics_model: torch.nn.Module,
                      scalers: Dict[str, Dict[str, float]],
                      **kwargs) -> MixedModeVentilationEnv:
    """
    Factory function to create environment with default parameters.
    
    Args:
        daily_data_list: List of daily building data
        dynamics_model: Pre-trained dynamics model
        scalers: Feature normalization scalers
        **kwargs: Additional environment parameters
        
    Returns:
        Configured environment instance
    """
    return MixedModeVentilationEnv(
        daily_data_list=daily_data_list,
        lstm_model=dynamics_model,
        scalers=scalers,
        **kwargs
    )
