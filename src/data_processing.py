"""
Data processing utilities for building sensor data.
"""
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Sequence, Tuple, Optional
import random
import torch

try:
    from .path_utils import resolve_repo_path
except ImportError:
    from path_utils import resolve_repo_path


# Configuration constants
COLUMN_DEFINITIONS = {
    'zone_cols': ['Zone 1 Temperature', 'Zone 2 Temperature', 'Zone 3 Temperature', 'Zone 4 Temperature', 'Zone 5 Temperature'],
    'ac_cols': ['OutdoorTemperatureWindow', 'PFCU-01 Supply Air Temp', 'PFCU-02 Supply Air Temp',
                'FCU-01 Supply Air Temp', 'FCU-02 Supply Air Temp - 1 min', 'FCU-03 Supply Air Temp','FCU-04 Supply Air Temp','FCU-05 Supply Air Temp'],
    'nv_cols': ['Wind Speed', 'Wind Direction', 'OutdoorTemperatureWindow', 'PFCU-01 Supply Air Temp', 'PFCU-02 Supply Air Temp'],
    'fcu_supply_cols': ['FCU-01 Supply Air Temp', 'FCU-02 Supply Air Temp - 1 min', 'FCU-03 Supply Air Temp', 'FCU-04 Supply Air Temp', 'FCU-05 Supply Air Temp'],
    'lc_cols': ['PFCU-01 Supply Air Temp', 'PFCU-02 Supply Air Temp'],
    'ac_action_cols': ['PFCU-01 Supply Air Temp', 'PFCU-02 Supply Air Temp', 'FCU-01 Supply Air Temp', 'FCU-02 Supply Air Temp - 1 min', 'FCU-03 Supply Air Temp','FCU-04 Supply Air Temp','FCU-05 Supply Air Temp'],
    'disturbance_cols': ['Wind Speed', 'Wind Direction', 'OutdoorTemperatureWindow']
}

COLUMNS_TO_CHECK = ['Zone 1 Temperature', 'Zone 2 Temperature', 'Zone 3 Temperature', 'Zone 4 Temperature', 'Zone 5 Temperature', 
                    'OutdoorTemperatureWindow',  'Wind Speed', 'Wind Direction', 'PFCU-01 Supply Air Temp', 'PFCU-02 Supply Air Temp',  
                    'FCU-01 Supply Air Temp', 'FCU-02 Supply Air Temp - 1 min', 'FCU-03 Supply Air Temp','FCU-04 Supply Air Temp','FCU-05 Supply Air Temp']
RAIN_COLUMN_CANDIDATES = ('rain_status', 'Rain Status', 'Rain', 'rain')


def extract_rain_status(row: pd.Series, default: float = 0.0) -> float:
    """Return a numeric rain indicator from a dataframe row if present."""
    for column in RAIN_COLUMN_CANDIDATES:
        if column not in row.index:
            continue
        try:
            value = float(row[column])
        except (TypeError, ValueError):
            continue
        if np.isnan(value):
            continue
        return value
    return float(default)


def set_random_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_and_filter_data(file_path: str,
                         min_length: int = 600,
                         filter_rainy_days: bool = False) -> List[pd.DataFrame]:
    """
    Load building sensor data and filter it into continuous daily segments.
    
    Args:
        file_path: Path to the CSV data file
        min_length: Minimum number of timesteps for a valid day
        filter_rainy_days: If True, drop any daily dataframe that contains rainy periods
        
    Returns:
        List of DataFrames, each representing a continuous day
    """
    data_path = resolve_repo_path(file_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find data file: {data_path}")

    # Load data with the publication dataset's explicit timestamp format.
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%y %H:%M')
    
    # Filter the dataframe
    filtered_df = df.copy()
    
    # Filter for working hours (7:30-19:00) and weekdays only
    filtered_df = filtered_df[
        (filtered_df['date'].dt.time >= pd.to_datetime('07:30').time()) & 
        (filtered_df['date'].dt.time <= pd.to_datetime('19:00').time())
    ]
    filtered_df = filtered_df[filtered_df['date'].dt.weekday < 5]
    
    # Remove specific dates (holidays)
    filtered_df = filtered_df[filtered_df['date'].dt.date != pd.to_datetime('2024-10-31').date()]
    
    # Sort by date
    filtered_df = filtered_df.sort_values(by='date')

    if filtered_df.empty:
        return []
    
    # Split into continuous segments
    list_of_dfs = []
    current_df = [filtered_df.iloc[0]]
    
    for i in range(1, len(filtered_df)):
        # Check if current timestep is continuous (1 minute apart)
        if (filtered_df.iloc[i]['date'] - filtered_df.iloc[i-1]['date']).seconds == 60:
            current_df.append(filtered_df.iloc[i])
        else:
            # Save current segment and start new one
            list_of_dfs.append(pd.DataFrame(current_df))
            current_df = [filtered_df.iloc[i]]
    
    # Add the last dataframe
    list_of_dfs.append(pd.DataFrame(current_df))
    
    # Filter by minimum length
    list_of_dfs = [df for df in list_of_dfs if len(df) >= min_length]
    
    # Reset indices
    list_of_dfs = [df.reset_index(drop=True) for df in list_of_dfs]
    
    # Filter out dataframes with NaN values in critical columns
    list_of_dfs = [
        df.dropna(subset=COLUMNS_TO_CHECK) 
        for df in list_of_dfs 
        if not df[COLUMNS_TO_CHECK].isna().any().any()
    ]
    
    if filter_rainy_days:
        temp_column = None
        for candidate in ('OutdoorTemperatureWindow', 'Outdoor Temperature', 'OutdoorTempAve'):
            if candidate in df.columns:
                temp_column = candidate
                break

        window_column = None
        for candidate in ('Z1 Windows Open Close Status', 'Z1 Window Open Close Status', 'Zone 1 Windows Open Close Status'):
            if candidate in df.columns:
                window_column = candidate
                break

        if temp_column is None or window_column is None:
            missing = []
            if temp_column is None:
                missing.append("outdoor temperature column")
            if window_column is None:
                missing.append("Z1 window status column")
            print(f"Warning: filter_rainy_days=True but missing {', '.join(missing)}; skipping rainy day filtering.")
        else:
            before_count = len(list_of_dfs)

            def is_rainy(day_df: pd.DataFrame) -> bool:
                temps = pd.to_numeric(day_df[temp_column], errors='coerce')
                windows = pd.to_numeric(day_df[window_column], errors='coerce')
                combo = (temps < 29.0) & (windows == 0)
                consecutive = 0
                for flag in combo.fillna(False):
                    if flag:
                        consecutive += 1
                        if consecutive >= 5:
                            return True
                    else:
                        consecutive = 0
                return False

            list_of_dfs = [df for df in list_of_dfs if not is_rainy(df)]
            removed = before_count - len(list_of_dfs)
            if removed > 0:
                print(f"Filtered out {removed} rainy day(s) using '{temp_column}' and '{window_column}'.")
   
    print(f"Number of continuous dataframes after filtering: {len(list_of_dfs)}")
    return list_of_dfs


def split_train_val(dataframes: List[pd.DataFrame],
                    split_ratio: float = 0.8,
                    shuffle: bool = False,
                    random_state: Optional[int] = None) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    """
    Split dataframes into training and validation sets.

    Args:
        dataframes: Daily dataframe segments to split.
        split_ratio: Fraction of data assigned to the training set.
        shuffle: If True, randomize dataframe ordering before splitting.
        random_state: Optional seed to make shuffling reproducible.
    """
    ordered_dfs = list(dataframes)
    if shuffle:
        rng = random.Random(random_state)
        rng.shuffle(ordered_dfs)

    split_index = int(len(ordered_dfs) * split_ratio)
    train_dfs = ordered_dfs[:split_index]
    val_dfs = ordered_dfs[split_index:]
    
    print(f"Number of training dataframes: {len(train_dfs)}")
    print(f"Number of validation dataframes: {len(val_dfs)}")
    
    return train_dfs, val_dfs


def setup_scalers(train_dfs: List[pd.DataFrame], val_dfs: List[pd.DataFrame]) -> Dict[str, Dict[str, float]]:
    """
    Set up normalization scalers for all features.
    
    Args:
        train_dfs: Training dataframes
        val_dfs: Validation dataframes
        
    Returns:
        Dictionary of scalers with min/max values for each feature
    """
    if not train_dfs and not val_dfs:
        raise ValueError("Cannot fit scalers with no training or validation dataframes.")

    # Combine all data for fitting scalers
    all_dfs = pd.concat(train_dfs + val_dfs)
    
    # Create custom scalers dictionary
    scalers = {}
    for col in COLUMNS_TO_CHECK:
        col_values = all_dfs[col].dropna().values
        min_val, max_val = col_values.min(), col_values.max()
        scalers[col] = {'min': min_val, 'max': max_val}
    
    return scalers


def normalize(value: float, feature_name: str, scalers: Dict[str, Dict[str, float]]) -> float:
    """Normalize a value using the feature's scaler."""
    scaler = scalers[feature_name]
    return (value - scaler['min']) / (scaler['max'] - scaler['min'])


def inverse_normalize(value: float, feature_name: str, scalers: Dict[str, Dict[str, float]]) -> float:
    """Inverse normalize a value using the feature's scaler."""
    scaler = scalers[feature_name]
    return value * (scaler['max'] - scaler['min']) + scaler['min']


def _compute_energy_consumption_expert(zone_temps_norm, supply_act, lc_act, windows_state, 
                                      zone_cols, fcu_supply_cols, lc_cols, scalers,
                                      return_zone_breakdown: bool = False):
    """
    Compute energy consumption for expert trajectories matching environment.py logic.
    
    Args:
        zone_temps_norm: Normalized zone temperatures 
        supply_act: FCU supply temperature actions (normalized)
        lc_act: Local cooling actions (normalized)
        windows_state: Window state (0=AC, 1=NV)
        zone_cols: Zone temperature column names
        fcu_supply_cols: FCU supply temperature column names
        lc_cols: Local cooling column names
        scalers: Feature scalers
        
    Returns:
        Energy consumption in kWh per minute.
        If return_zone_breakdown=True, returns a tuple of:
        (total_energy, zone_energy_vector[kWh/min] with one value per zone)
    """
    # Constants matching environment.py
    SPECIFIC_HEAT_AIR = 1005  # J/(kg·K)
    AIR_DENSITY = 1.225       # kg/m³
    FCU_AIRFLOW_CPH = 800     # CFH
    LOCAL_COOLING_CPH = 1000  # CFH
    
    energy_consumption = 0.0
    zone_energy = np.zeros(len(zone_temps_norm), dtype=np.float32)
    
    if windows_state == 0:  # AC mode - FCU systems active
        for z in range(len(zone_temps_norm)):
            if z < len(supply_act):  # Make sure we have supply temp for this zone
                # Denormalize temperatures to actual °C values
                zone_temp_C = inverse_normalize(zone_temps_norm[z], zone_cols[z], scalers)
                supply_temp_C = inverse_normalize(supply_act[z], fcu_supply_cols[z], scalers)
                
                # Calculate temperature difference (cooling load)
                delta_t = max(0.0, zone_temp_C - supply_temp_C)
                
                # Convert to energy consumption (same as environment.py)
                if delta_t > 0.5:  # Only count significant cooling loads
                    volume_flow_rate = FCU_AIRFLOW_CPH / 3600  # m³/s
                    mass_flow_rate = volume_flow_rate * AIR_DENSITY  # kg/s
                    power_watts = mass_flow_rate * SPECIFIC_HEAT_AIR * delta_t  # Watts
                    energy_kwh_per_minute = power_watts / 1000 / 60  # kWh per minute
                    energy_consumption += energy_kwh_per_minute
                    zone_energy[z] += float(energy_kwh_per_minute)

    # Local cooling contribution (aligned with environment.py behavior where
    # local cooling can contribute whenever it is active, not only in NV mode).
    # Expert trajectories do not carry explicit lc_on flags, so we infer
    # activation using the same threshold used historically in this pipeline.
    for i, lc_temp_norm in enumerate(lc_act):
        if lc_temp_norm > 0.1:
            # Denormalize local cooling temperature
            lc_temp_C = inverse_normalize(lc_temp_norm, lc_cols[i], scalers)
            
            # Calculate average zone temperature for reference
            avg_zone_temp_C = np.mean([
                inverse_normalize(zt, zone_cols[j], scalers)
                for j, zt in enumerate(zone_temps_norm)
            ])
            
            # Calculate cooling energy for local cooling unit
            delta_t = max(0.0, avg_zone_temp_C - lc_temp_C)
            if delta_t > 0.5:  # Only count significant cooling
                volume_flow_rate = LOCAL_COOLING_CPH / 3600  # m³/s
                mass_flow_rate = volume_flow_rate * AIR_DENSITY  # kg/s
                power_watts = mass_flow_rate * SPECIFIC_HEAT_AIR * delta_t  # Watts
                energy_kwh_per_minute = power_watts / 1000 / 60  # kWh per minute
                energy_consumption += energy_kwh_per_minute
                if len(zone_energy) > 0:
                    # Local cooling units are not mapped one-to-one to all FCU zones.
                    # Distribute local load evenly to preserve total-energy consistency.
                    zone_energy += float(energy_kwh_per_minute) / float(len(zone_energy))
    
    if return_zone_breakdown:
        return energy_consumption, zone_energy
    return energy_consumption


def extract_expert_trajectories(
    dfs: List[pd.DataFrame], 
    scalers: Dict[str, Dict[str, float]],
    window_col: str = 'Z1 Windows Open Close Status',
    comfort_max: float = 25.0,
    comfort_min: float = 21.0
) -> List[Dict[str, np.ndarray]]:
    """
    Extract expert trajectories from historical building operation data.
    Energy features now match environment.py calculation for AIRL consistency.
    
    Args:
        dfs: List of daily dataframes
        scalers: Normalization scalers
        window_col: Column name for window status
        comfort_max: Maximum comfort temperature threshold (°C)
        comfort_min: Minimum comfort temperature threshold (°C)
        
    Returns:
        List of trajectory dictionaries with states, actions, and features
        Features: [energy_consumption (kWh/min), comfort_violations (°C), violation_flag, toggle_penalty]
        Additional trajectory keys include scalar energy, comfort, and rain side-features.
    """
    zone_cols = COLUMN_DEFINITIONS['zone_cols']
    ac_action_cols = COLUMN_DEFINITIONS['ac_action_cols']
    disturbance_cols = COLUMN_DEFINITIONS['disturbance_cols']
    
    expert_trajectories = []
    
    for day_df in dfs:
        features = []
        states = []
        actions = []
        energy_zone_features = []
        comfort_zone_features = []
        rain_features = []
        # Track legacy AIRL window state so expert tuples match policy semantics:
        # (state_t before applying action_t, action_t, transition features_t).
        last_window_state = 0.0
        window_toggle_counter = 0.0
        switch_flag_state = 0.0
        dwell_state = 0.0
        last_switch_time = -10000
        
        for i, row in day_df.iterrows():
            # Extract and normalize state values
            zone_temps = np.array([normalize(row[col], col, scalers) for col in zone_cols], dtype=np.float32)
            outside_temp = normalize(row[disturbance_cols[2]], disturbance_cols[2], scalers)
            wind_speed = normalize(row[disturbance_cols[0]], disturbance_cols[0], scalers)
            wind_direction = normalize(row[disturbance_cols[1]], disturbance_cols[1], scalers)
            
            # Extract action values
            window_act = row[window_col]
            if np.isnan(window_act):
                window_act = last_window_state
            else:
                window_act = float(window_act)
            
            ac_act = np.array([normalize(row[col], col, scalers) for col in ac_action_cols], dtype=np.float32)
            switch_flag_action = int(np.any(window_act != last_window_state))
            
            # Separate FCU supply temps and local cooling temps from actions
            lc_act = ac_act[:2]  # First 2: PFCU-01, PFCU-02
            fcu_supply_act = ac_act[2:]  # Rest: FCU supply temps
            
            # Get column definitions
            fcu_supply_cols = COLUMN_DEFINITIONS['fcu_supply_cols']
            lc_cols = COLUMN_DEFINITIONS['lc_cols']
            
            # Compute energy consumption using the same method as environment.py
            energy_use, energy_zone_use = _compute_energy_consumption_expert(
                zone_temps, fcu_supply_act, lc_act, int(window_act), 
                zone_cols, fcu_supply_cols, lc_cols, scalers,
                return_zone_breakdown=True
            )
            
            comfort_penalty = 0.0
            comfort_zone_penalty = np.zeros(len(zone_temps), dtype=np.float32)
            for z, temp_norm in enumerate(zone_temps):
                # Denormalize zone temperature to real Celsius
                temp_celsius = inverse_normalize(temp_norm, zone_cols[z], scalers)
                
                # Compare against Celsius comfort bounds
                if temp_celsius > comfort_max: 
                    violation = temp_celsius - comfort_max
                    comfort_penalty += violation
                    comfort_zone_penalty[z] = float(violation)
                elif temp_celsius < comfort_min:
                    violation = comfort_min - temp_celsius
                    comfort_penalty += violation
                    comfort_zone_penalty[z] = float(violation)
            
            violation_flag = 0.0
            if np.any(window_act > 0):
                # Denormalize outside temperature for comparison
                outside_temp_celsius = inverse_normalize(outside_temp, disturbance_cols[2], scalers)
                if outside_temp_celsius > comfort_max and wind_speed > 0.6:
                    violation_flag = 1.0
            
            # Build state and action vectors
            state = np.concatenate((
                zone_temps, 
                [
                    outside_temp,
                    wind_speed,
                    wind_direction,
                    window_toggle_counter,
                    switch_flag_state,
                    dwell_state,
                    last_window_state
                ]
            ), axis=0)
            
            action = np.concatenate(([switch_flag_action], ac_act), axis=0)
            
            features.append([energy_use, comfort_penalty, violation_flag, 0.0])  # toggle_penalty placeholder
            states.append(state)
            actions.append(action)
            energy_zone_features.append(np.asarray(energy_zone_use, dtype=np.float32))
            comfort_zone_features.append(np.asarray(comfort_zone_penalty, dtype=np.float32))
            rain_features.append(extract_rain_status(row))

            # Advance legacy window trackers for next state's pre-action context.
            next_dwell = 1 / max(1, i - last_switch_time)
            if switch_flag_action == 1:
                window_toggle_counter = 0.0
                last_switch_time = i
            else:
                window_toggle_counter += 1 / 720
            switch_flag_state = float(switch_flag_action)
            dwell_state = float(next_dwell)
            last_window_state = float(window_act)
        
        # Convert to arrays
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.float32)
        features = np.array(features, dtype=np.float32)
        energy_zone_features = np.asarray(energy_zone_features, dtype=np.float32)
        comfort_zone_features = np.asarray(comfort_zone_features, dtype=np.float32)
        rain_features = np.asarray(rain_features, dtype=np.float32)
        
        # Extract energy and comfort features for AIRL
        energy_features = features[:, 0]  # Energy consumption
        comfort_features = features[:, 1]  # Comfort violations
        
        expert_trajectories.append({
            'states': states,
            'actions': actions,
            'features': features,
            'energy_features': energy_features,
            'comfort_features': comfort_features,
            'energy_zone_features': energy_zone_features,
            'comfort_zone_features': comfort_zone_features,
            'rain_features': rain_features,
            'violation_features': features[:, 2],
            'dones': np.zeros(len(states), dtype=bool)  # Add dones for AIRL compatibility
        })
    
    return expert_trajectories


def get_column_definitions() -> Dict[str, List[str]]:
    """Get column definitions for different data types."""
    return COLUMN_DEFINITIONS.copy()


# ---------------------------------------------------------------------------
# Helper utilities for hybrid control discretisation
# ---------------------------------------------------------------------------
def denormalize_vector(values: Sequence[float], columns: Sequence[str], scalers: Dict[str, Dict[str, float]]) -> np.ndarray:
    """Denormalize a vector given matching column names."""
    if len(values) != len(columns):
        raise ValueError(f"Length mismatch: {len(values)} values vs {len(columns)} columns")
    return np.array([
        inverse_normalize(val, col, scalers) for val, col in zip(values, columns)
    ], dtype=np.float32)


def discretize_temperatures(temps_celsius: Sequence[float], temp_levels: Sequence[float]) -> np.ndarray:
    """Map continuous temperatures to nearest discrete index."""
    levels = np.asarray(temp_levels, dtype=np.float32)
    temps = np.asarray(temps_celsius, dtype=np.float32)
    indices = np.abs(temps[..., None] - levels).argmin(axis=-1)
    return indices.astype(np.int64)


def clamp_discrete_transition(previous: Sequence[int], proposed: Sequence[int], max_jump: int) -> np.ndarray:
    """Clamp discrete actions to satisfy per-step index jump."""
    prev = np.asarray(previous, dtype=np.int64)
    prop = np.asarray(proposed, dtype=np.int64)
    if prev.shape != prop.shape:
        raise ValueError("Previous and proposed arrays must share the same shape")
    lo = np.maximum(0, prev - max_jump)
    hi = prev + max_jump
    return np.clip(prop, lo, hi)
