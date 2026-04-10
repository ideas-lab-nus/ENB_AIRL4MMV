"""
Utility functions for AIRL training and evaluation.
"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional

try:
    from .data_processing import get_column_definitions, inverse_normalize
    from .path_utils import resolve_repo_path
except ImportError:
    from data_processing import get_column_definitions, inverse_normalize
    from path_utils import resolve_repo_path


def setup_plotting_style():
    """Set up consistent plotting style."""
    colors = sns.color_palette("Set2")
    plt.rcParams['figure.figsize'] = (12, 6)
    return colors


def load_trained_models(policy_path: str, 
                       reward_path: str, 
                       state_dim: int, 
                       device: torch.device) -> Tuple[Any, Any]:
    """
    Load trained policy and reward models.
    
    Args:
        policy_path: Path to policy model
        reward_path: Path to reward model
        state_dim: State dimension
        device: PyTorch device
        
    Returns:
        Tuple of (policy, reward_fn)
    """
    try:
        from .models import GRUReward, MMVPolicyActorCritic
    except ImportError:
        from models import GRUReward, MMVPolicyActorCritic

    policy_path = resolve_repo_path(policy_path)
    reward_path = resolve_repo_path(reward_path)
    
    # Load policy
    policy = MMVPolicyActorCritic(state_dim=state_dim, num_zones=5).to(device)
    policy.load_state_dict(torch.load(policy_path, map_location=device))
    policy.eval()
    
    # Load reward function
    reward_fn = GRUReward(in_dim=state_dim + 8, hid=64).to(device)  # 8 = action_dim
    reward_fn.load_state_dict(torch.load(reward_path, map_location=device))
    
    return policy, reward_fn


def evaluate_policy_on_validation(policy, 
                                 validation_env, 
                                 scalers: Dict[str, Dict[str, float]],
                                 num_days: Optional[int] = None,
                                 show_plots: bool = True) -> Dict[str, List]:
    """
    Evaluate policy on validation data and create visualizations.
    
    Args:
        policy: Trained policy
        validation_env: Validation environment
        scalers: Feature normalization scalers
        num_days: Number of days to evaluate (None for all)
        show_plots: Whether to display plots
        
    Returns:
        Dictionary with evaluation results
    """
    colors = setup_plotting_style()
    
    if num_days is None:
        num_days = len(validation_env.daily_data_list)
    
    results = {
        'states': [],
        'zone_temperatures': [],
        'supply_actions': [],
        'window_actions': [],
        'local_cooling_actions': [],
        'historical_data': [],
        'energy_consumption': [],
        'comfort_violations': [],
    }
    
    for val_day in range(min(num_days, len(validation_env.daily_data_list))):
        # Run policy on validation day
        day_results = run_policy_on_day(policy, validation_env, val_day, scalers)
        
        # Store results
        if 'states' in day_results:
            results['states'].append(day_results['states'])
            results['zone_temperatures'].append(day_results['states'])

        for key in ('supply_actions', 'window_actions', 'local_cooling_actions', 'historical_data',
                    'energy_consumption', 'comfort_violations'):
            if key in day_results:
                results[key].append(day_results[key])
        
        if show_plots:
            # Plot zone temperatures
            plot_zone_temperatures(day_results, val_day, colors)
            
            # Plot supply air temperatures
            plot_supply_temperatures(day_results, val_day, colors)
            
            # Plot window status
            plot_window_status(day_results, val_day)
    
    return results


def run_policy_on_day(policy, 
                     env, 
                     day_index: int, 
                     scalers: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    """
    Run policy on a single day and collect results.
    
    Args:
        policy: Policy to evaluate
        env: Environment
        day_index: Day index to evaluate
        scalers: Feature normalization scalers
        
    Returns:
        Dictionary with day results
    """
    columns = get_column_definitions()
    zone_cols = columns['zone_cols']
    
    state = env.reset(day_index=day_index)
    done = False
    
    # Storage for results
    states = []
    sat_actions = []
    win_actions = []
    lc_actions = []
    energy_consumption = []
    comfort_violations = []
    
    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action, _ = policy.get_action(state_tensor)
        
        # Extract window action
        change = int(action['change'])
        windows = int(state[-1]) ^ change  # XOR flip
        
        # Step environment
        next_state, reward, done, info = env.step(action)
        
        # Store results
        states.append(state)
        lc_actions.append(action['local_cooling'])
        sat_actions.append(action['supply_temps'])
        win_actions.append(windows)
        energy_consumption.append(float(info.get('energy_consumption', info.get('energy', 0.0))))
        comfort_violations.append(float(info.get('comfort_violations', info.get('comfort', 0.0))))
        
        state = next_state
    
    # Convert to DataFrames for easier processing
    base_state_columns = zone_cols + ['OutdoorTemperatureWindow', 'Wind Speed', 'Wind Direction']
    suffix_columns = ['Window Hold Counter', 'Window Switch Flag', 'Dwell', 'prev_w']
    state_width = len(states[0]) if states else len(base_state_columns) + len(suffix_columns)
    extra_feature_count = max(state_width - len(base_state_columns) - len(suffix_columns), 0)
    if extra_feature_count == 2:
        extra_feature_columns = ['ClockSin', 'ClockCos']
    else:
        extra_feature_columns = [f'ExtraFeature{i + 1}' for i in range(extra_feature_count)]
    state_columns = base_state_columns + extra_feature_columns + suffix_columns
    if len(state_columns) != state_width:
        raise ValueError(f"State width mismatch: expected {len(state_columns)} columns, got {state_width}")
    states_df = pd.DataFrame(states, columns=state_columns)
    
    # FCU supply temperatures (excluding last column which might be duplicated)
    sat_actions_array = np.array(sat_actions)
    if sat_actions_array.shape[1] > 4:
        sat_actions_array = sat_actions_array[:, :4]  # Take first 4 columns
    
    sat_columns = ['FCU-01 Supply Air Temp', 'FCU-02 Supply Air Temp - 1 min', 
                  'FCU-03 Supply Air Temp', 'FCU-05 Supply Air Temp']
    sat_actions_df = pd.DataFrame(sat_actions_array, columns=sat_columns[:sat_actions_array.shape[1]])
    
    win_actions_df = pd.DataFrame(win_actions, columns=['Windows'])
    local_cooling_df = pd.DataFrame(lc_actions, columns=['PFCU-01 Supply Air Temp', 'PFCU-02 Supply Air Temp'])
    
    # Inverse normalize states (excluding last 4 columns which are internal state)
    for col in states_df.columns[:-4]:
        if col in scalers:
            states_df[col] = states_df[col].apply(lambda x: inverse_normalize(x, col, scalers))
    
    # Inverse normalize actions
    for col in sat_actions_df.columns:
        if col in scalers:
            sat_actions_df[col] = sat_actions_df[col].apply(lambda x: inverse_normalize(x, col, scalers))
    
    for col in local_cooling_df.columns:
        if col in scalers:
            local_cooling_df[col] = local_cooling_df[col].apply(lambda x: inverse_normalize(x, col, scalers))
    
    return {
        'states': states_df,
        'supply_actions': sat_actions_df,
        'window_actions': win_actions_df,
        'local_cooling_actions': local_cooling_df,
        'historical_data': env.current_day_data,
        'energy_consumption': pd.Series(energy_consumption, name='energy_consumption'),
        'comfort_violations': pd.Series(comfort_violations, name='comfort_violations'),
        'day_info': env.get_current_day_info()
    }


def plot_zone_temperatures(day_results: Dict[str, Any], val_day: int, colors):
    """Plot zone temperature comparison."""
    states_df = day_results['states']
    historical_data = day_results['historical_data']
    
    zone_cols = ['Zone 1 Temperature', 'Zone 2 Temperature', 'Zone 3 Temperature', 
                'Zone 4 Temperature', 'Zone 5 Temperature']
    
    # Create time axis from 7:30 to 19:00
    start_hour = 7.5  # 7:30 AM
    end_hour = 19.0   # 7:00 PM
    num_steps = len(states_df)
    time_hours = np.linspace(start_hour, end_hour, num_steps)
    
    # Convert to time labels (e.g., "08:30", "15:45")
    time_labels = []
    for t in time_hours:
        hour = int(t)
        minute = int((t - hour) * 60)
        time_labels.append(f"{hour:02d}:{minute:02d}")
    
    plt.figure(figsize=(12, 6))
    for i, col in enumerate(zone_cols):
        plt.plot(time_hours, states_df[col], label=f'Policy {col}', color=colors[i])
        plt.plot(time_hours[:len(historical_data[col].iloc[:len(states_df)])], 
                historical_data[col].iloc[:len(states_df)], 
                label=f'Historical {col}', color=colors[i], linestyle='--')
    
    plt.plot(time_hours, states_df['OutdoorTemperatureWindow'], label='Outdoor Temperature', color='orange')
    plt.plot(time_hours[:len(historical_data['OutdoorTemperatureWindow'].iloc[:len(states_df)])], 
            historical_data['OutdoorTemperatureWindow'].iloc[:len(states_df)], 
            label='Historical Outdoor Temperature', color='magenta', linestyle=':')
    
    plt.xlabel('Time of Day')
    plt.ylabel('Temperature (°C)')
    
    # Set x-axis ticks to show times at regular intervals
    tick_positions = np.linspace(start_hour, end_hour, 8)  # Show 8 time points
    tick_labels = [f"{int(t):02d}:{int((t - int(t)) * 60):02d}" for t in tick_positions]
    plt.xticks(tick_positions, tick_labels, rotation=45)
    
    # Use actual date from day_info if available, otherwise fall back to day number
    if 'day_info' in day_results and 'day_start' in day_results['day_info']:
        date_str = str(day_results['day_info']['day_start']).split()[0]  # Extract date part
        plt.title(f'Zone Temperature Comparison - {date_str}')
    else:
        plt.title(f'Zone Temperature Comparison - Validation Day {val_day}')
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_supply_temperatures(day_results: Dict[str, Any], val_day: int, colors):
    """Plot supply air temperature comparison."""
    sat_actions_df = day_results['supply_actions']
    local_cooling_df = day_results['local_cooling_actions']
    historical_data = day_results['historical_data']
    
    # Create time axis from 7:30 to 19:00
    start_hour = 7.5  # 7:30 AM
    end_hour = 19.0   # 7:00 PM
    num_steps = len(sat_actions_df)
    time_hours = np.linspace(start_hour, end_hour, num_steps)
    
    # FCU supply temperatures
    plt.figure(figsize=(12, 6))
    for i, col in enumerate(sat_actions_df.columns):
        plt.plot(time_hours, sat_actions_df[col], label=f'Policy {col}', color=colors[i])
        if col in historical_data.columns:
            plt.plot(time_hours[:len(historical_data[col].iloc[:len(sat_actions_df)])], 
                    historical_data[col].iloc[:len(sat_actions_df)], 
                    label=f'Historical {col}', color=colors[i], linestyle='--')
    
    plt.xlabel('Time of Day')
    plt.ylabel('Supply Temperature (°C)')
    
    # Set x-axis ticks to show times at regular intervals
    tick_positions = np.linspace(start_hour, end_hour, 8)  # Show 8 time points
    tick_labels = [f"{int(t):02d}:{int((t - int(t)) * 60):02d}" for t in tick_positions]
    plt.xticks(tick_positions, tick_labels, rotation=45)
    
    # Use actual date from day_info if available, otherwise fall back to day number
    if 'day_info' in day_results and 'day_start' in day_results['day_info']:
        date_str = str(day_results['day_info']['day_start']).split()[0]  # Extract date part
        plt.title(f'FCU Supply Air Temperature Comparison - {date_str}')
    else:
        plt.title(f'FCU Supply Air Temperature Comparison - Day {val_day}')
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # PFCU supply temperatures
    plt.figure(figsize=(12, 6))
    pfcu_cols = ['PFCU-01 Supply Air Temp', 'PFCU-02 Supply Air Temp']
    
    # Create time axis for local cooling data
    num_steps_lc = len(local_cooling_df)
    time_hours_lc = np.linspace(start_hour, end_hour, num_steps_lc)
    
    for j, col in enumerate(pfcu_cols):
        plt.plot(time_hours_lc[:len(historical_data[col].iloc[:len(local_cooling_df)])], 
                historical_data[col].iloc[:len(local_cooling_df)], 
                label=f'Historical {col}', color=colors[j+4], linestyle='--')
        plt.plot(time_hours_lc, local_cooling_df[col], label=f'Policy {col}', color=colors[j+4])
    
    plt.xlabel('Time of Day')
    plt.ylabel('Supply Temperature (°C)')
    
    # Set x-axis ticks to show times at regular intervals
    tick_positions = np.linspace(start_hour, end_hour, 8)  # Show 8 time points
    tick_labels = [f"{int(t):02d}:{int((t - int(t)) * 60):02d}" for t in tick_positions]
    plt.xticks(tick_positions, tick_labels, rotation=45)
    
    # Use actual date from day_info if available, otherwise fall back to day number
    if 'day_info' in day_results and 'day_start' in day_results['day_info']:
        date_str = str(day_results['day_info']['day_start']).split()[0]  # Extract date part
        plt.title(f'PFCU Supply Air Temperature Comparison - {date_str}')
    else:
        plt.title(f'PFCU Supply Air Temperature Comparison - Day {val_day}')
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_window_status(day_results: Dict[str, Any], val_day: int):
    """Plot window status comparison."""
    win_actions_df = day_results['window_actions']
    historical_data = day_results['historical_data']
    
    # Create time axis from 7:30 to 19:00
    start_hour = 7.5  # 7:30 AM
    end_hour = 19.0   # 7:00 PM
    num_steps = len(win_actions_df)
    time_hours = np.linspace(start_hour, end_hour, num_steps)
    
    plt.figure(figsize=(12, 6))
    plt.plot(time_hours[:len(historical_data['Z1 Windows Open Close Status'].iloc[:len(win_actions_df)])], 
            historical_data['Z1 Windows Open Close Status'].iloc[:len(win_actions_df)], 
            label='Historical Window Status', color='blue', linestyle='--')
    plt.plot(time_hours, win_actions_df['Windows'], label='Policy Window Status', color='red')
    
    plt.xlabel('Time of Day')
    plt.ylabel('Window Status')
    
    # Set x-axis ticks to show times at regular intervals
    tick_positions = np.linspace(start_hour, end_hour, 8)  # Show 8 time points
    tick_labels = [f"{int(t):02d}:{int((t - int(t)) * 60):02d}" for t in tick_positions]
    plt.xticks(tick_positions, tick_labels, rotation=45)
    
    # Use actual date from day_info if available, otherwise fall back to day number
    if 'day_info' in day_results and 'day_start' in day_results['day_info']:
        date_str = str(day_results['day_info']['day_start']).split()[0]  # Extract date part
        plt.title(f'Window Status Comparison - {date_str}')
    else:
        plt.title(f'Window Status Comparison - Validation Day {val_day}')
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def compute_evaluation_metrics(results: Dict[str, List]) -> Dict[str, float]:
    """
    Compute aggregate evaluation metrics from validation results.
    
    Args:
        results: Results from evaluate_policy_on_validation
        
    Returns:
        Dictionary of evaluation metrics
    """
    def _flatten_numeric_batches(items: List[Any]) -> np.ndarray:
        values = []
        for item in items:
            if item is None:
                continue
            if isinstance(item, (pd.Series, pd.DataFrame)):
                array = item.to_numpy(dtype=np.float32)
            else:
                array = np.asarray(item, dtype=np.float32)
            if array.size:
                values.append(array.reshape(-1))
        if not values:
            return np.asarray([], dtype=np.float32)
        return np.concatenate(values, axis=0)

    zone_cols = get_column_definitions()['zone_cols']

    energy_values = _flatten_numeric_batches(results.get('energy_consumption', []))
    comfort_values = _flatten_numeric_batches(results.get('comfort_violations', []))

    window_switch_counts = []
    for item in results.get('window_actions', []):
        if isinstance(item, pd.DataFrame) and 'Windows' in item.columns:
            series = item['Windows'].to_numpy(dtype=np.float32)
        else:
            series = np.asarray(item, dtype=np.float32).reshape(-1)
        if len(series) > 1:
            window_switch_counts.append(float(np.count_nonzero(np.diff(series.astype(int)) != 0)))
        elif len(series) == 1:
            window_switch_counts.append(0.0)

    squared_errors = []
    state_batches = results.get('states') or results.get('zone_temperatures', [])
    for states_df, historical_df in zip(state_batches, results.get('historical_data', [])):
        if not isinstance(states_df, pd.DataFrame) or not isinstance(historical_df, pd.DataFrame):
            continue

        valid_zone_cols = [col for col in zone_cols if col in states_df.columns and col in historical_df.columns]
        if not valid_zone_cols:
            continue

        common_len = min(len(states_df), len(historical_df))
        if common_len <= 0:
            continue

        state_values = states_df.loc[:common_len - 1, valid_zone_cols].to_numpy(dtype=np.float32)
        historical_values = historical_df.loc[:common_len - 1, valid_zone_cols].to_numpy(dtype=np.float32)
        squared_errors.append(np.square(state_values - historical_values).reshape(-1))

    if squared_errors:
        rmse = float(np.sqrt(np.mean(np.concatenate(squared_errors, axis=0))))
    else:
        rmse = 0.0

    metrics = {
        'avg_energy_consumption': float(np.mean(energy_values)) if energy_values.size else 0.0,
        'total_energy_consumption': float(np.sum(energy_values)) if energy_values.size else 0.0,
        'comfort_violations': float(np.sum(comfort_values)) if comfort_values.size else 0.0,
        'avg_comfort_violations': float(np.mean(comfort_values)) if comfort_values.size else 0.0,
        'window_switches_per_day': float(np.mean(window_switch_counts)) if window_switch_counts else 0.0,
        'temperature_rmse': rmse,
        'num_days_evaluated': float(len(results.get('historical_data', []))),
    }
    
    return metrics


def save_evaluation_results(results: Dict[str, Any], 
                          metrics: Dict[str, float], 
                          output_path: str):
    """Save evaluation results to file."""
    import pickle
    
    evaluation_data = {
        'results': results,
        'metrics': metrics,
        'timestamp': pd.Timestamp.now()
    }

    output_path = resolve_repo_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open('wb') as f:
        pickle.dump(evaluation_data, f)
    
    print(f"Evaluation results saved to {output_path}")


def print_training_summary(policy, reward_fn, expert_trajs: List[Dict]):
    """Print summary of training setup."""
    print("=== AIRL Training Summary ===")
    print(f"Policy parameters: {sum(p.numel() for p in policy.parameters()):,}")
    print(f"Reward function parameters: {sum(p.numel() for p in reward_fn.parameters()):,}")
    print(f"Expert trajectories: {len(expert_trajs)}")
    print(f"Total expert timesteps: {sum(len(traj['states']) for traj in expert_trajs):,}")
    print("============================\n")
