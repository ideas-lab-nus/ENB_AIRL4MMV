"""
Neural network models for AIRL-based building control.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List, Any

try:
    from .path_utils import resolve_repo_path
except ImportError:
    from path_utils import resolve_repo_path


class CNN_LSTM_Branch(nn.Module):
    """
    A single branch that processes sequential input data through CNN and LSTM layers.
    Used as building block for the combined dynamics model.
    """
    def __init__(self, input_channels: int, hidden_size: int = 64):
        super(CNN_LSTM_Branch, self).__init__()
        # CNN layers for feature extraction
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(input_size=32, hidden_size=hidden_size, batch_first=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CNN-LSTM branch.
        
        Args:
            x: Input tensor of shape (batch_size, time_steps, input_channels)
            
        Returns:
            Hidden representation tensor of shape (batch_size, hidden_size)
        """
        # Transpose for CNN: (B, C, T)
        x = x.transpose(1, 2)
        
        # CNN feature extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # Transpose back to (B, T, C) for LSTM
        x = x.transpose(1, 2)
        
        # LSTM processing
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Return last hidden state
        last_hidden = h_n[-1]  # shape: (B, hidden_size)
        return last_hidden


class Combined_CNN_LSTM(nn.Module):
    """
    Combined dynamics model with dual branches for AC and NV modes.
    Uses a binary gate to switch between air conditioning and natural ventilation branches.
    """
    def __init__(self, hidden_size: int = 64, output_size: int = 5, ac_channels: int = 13, nv_channels: int = 10):
        super(Combined_CNN_LSTM, self).__init__()
        
        # Air conditioning branch (13 channels: zones + outdoor + local cooling + supply temps)
        self.branch_AC = CNN_LSTM_Branch(input_channels=ac_channels, hidden_size=hidden_size)
        
        # Natural ventilation branch (10 channels: zones + outdoor conditions + local cooling)
        self.branch_NV = CNN_LSTM_Branch(input_channels=nv_channels, hidden_size=hidden_size)
        
        # Shared output layer for zone temperature prediction
        self.output_layer = nn.Linear(hidden_size, output_size)
    
    def forward(self, x_ac: torch.Tensor, x_nv: torch.Tensor, j: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with mode switching.
        
        Args:
            x_ac: AC mode inputs (batch_size, time_steps, ac_channels)
            x_nv: NV mode inputs (batch_size, time_steps, nv_channels)
            j: Mode selection tensor (batch_size,) - 0=AC, 1=NV
            
        Returns:
            Predicted zone temperatures (batch_size, output_size)
        """
        batch_size = x_ac.size(0)
        outputs = []
        
        for i in range(batch_size):
            if j[i] == 0:
                # Use AC branch
                hidden_vec = self.branch_AC(x_ac[i].unsqueeze(0))
            else:
                # Use NV branch
                hidden_vec = self.branch_NV(x_nv[i].unsqueeze(0))
            
            out = self.output_layer(hidden_vec)
            outputs.append(out)
        
        outputs = torch.cat(outputs, dim=0)
        return outputs


class CNN_LSTM_Branch_Gated(nn.Module):
    """CNN-LSTM branch variant used by the gated rain/solar checkpoint."""

    def __init__(self, input_channels: int, hidden_size: int = 64):
        super(CNN_LSTM_Branch_Gated, self).__init__()
        self.cnn_layers = nn.ModuleList([
            nn.Conv1d(in_channels=input_channels, out_channels=16, kernel_size=3, padding=1),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
        ])
        self.lstm = nn.LSTM(input_size=32, hidden_size=hidden_size, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        for layer in self.cnn_layers:
            x = F.relu(layer(x))
        x = x.transpose(1, 2)
        _lstm_out, (h_n, _c_n) = self.lstm(x)
        return h_n[-1]


class Combined_CNN_LSTM_Gated(nn.Module):
    """
    Gated dual-branch dynamics model.

    The checkpoint blends AC/NV hidden states with a small learned gate driven by
    the mode signal `j`, instead of a hard branch switch.
    """

    def __init__(self, hidden_size: int = 64, output_size: int = 5,
                 ac_channels: int = 14, nv_channels: int = 11):
        super(Combined_CNN_LSTM_Gated, self).__init__()
        self.ac_channels = ac_channels
        self.nv_channels = nv_channels
        self.branch_AC = CNN_LSTM_Branch_Gated(input_channels=ac_channels, hidden_size=hidden_size)
        self.branch_NV = CNN_LSTM_Branch_Gated(input_channels=nv_channels, hidden_size=hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.gate = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x_ac: torch.Tensor, x_nv: torch.Tensor, j: torch.Tensor) -> torch.Tensor:
        hidden_ac = self.branch_AC(x_ac)
        hidden_nv = self.branch_NV(x_nv)
        gate_in = j.float().view(-1, 1)
        gate_val = torch.sigmoid(self.gate(gate_in))
        hidden = gate_val * hidden_nv + (1.0 - gate_val) * hidden_ac
        return self.output_layer(hidden)


# ============================================================================
# LSTM-Only Models (without CNN layers or branches)
# ============================================================================

class LSTM_Direct(nn.Module):
    """
    Simple LSTM with direct prediction (no encoder-decoder, no branches).
    This is a simpler alternative to Combined_CNN_LSTM.
    Note: This model expects combined input (not separate AC/NV branches).
    """
    def __init__(self, input_dim: int = 15, hidden_dim: int = 128,
                 output_dim: int = 5, num_layers: int = 1, dropout: float = 0.1):
        super(LSTM_Direct, self).__init__()

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Direct output layer (no decoder)
        self.fc = nn.Linear(hidden_dim, output_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through simple LSTM.

        Args:
            x: Input tensor (batch_size, time_steps, input_dim)

        Returns:
            Predicted zone temperatures (batch_size, output_dim)
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Take last hidden state
        last_hidden = h_n[-1]  # (batch_size, hidden_dim)

        # Apply dropout
        last_hidden = self.dropout(last_hidden)

        # Direct prediction
        output = self.fc(last_hidden)
        return output


class LSTM_Direct_Branched(nn.Module):
    """
    Wrapper for LSTM_Direct that accepts branched AC/NV inputs like Combined_CNN_LSTM.
    This allows drop-in replacement with minimal code changes.

    The LSTM was trained with combined features (zones + controls = 15 features).
    AC branch: 13 channels (5 zones + 8 controls including supply temps)
    NV branch: 10 channels (5 zones + 5 controls, no supply temps)

    This wrapper uses 15 features: 5 zones + 10 control features (max of AC/NV controls).
    """
    def __init__(self, hidden_size: int = 128, output_size: int = 5,
                 ac_channels: int = 13, nv_channels: int = 10,
                 num_layers: int = 1, dropout: float = 0.1):
        super(LSTM_Direct_Branched, self).__init__()

        # LSTM expects 15 features (5 zones + 10 control features)
        self.input_dim = 15
        self.ac_channels = ac_channels
        self.nv_channels = nv_channels

        # Single LSTM model with 15 input features
        self.lstm_model = LSTM_Direct(
            input_dim=15,  # 5 zones + 10 control features
            hidden_dim=hidden_size,
            output_dim=output_size,
            num_layers=num_layers,
            dropout=dropout
        )

    def forward(self, x_ac: torch.Tensor, x_nv: torch.Tensor, j: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with mode-based input selection.

        Args:
            x_ac: AC mode inputs (batch_size, time_steps, 13)
                  [5 zone temps, 8 control features including supply temps]
            x_nv: NV mode inputs (batch_size, time_steps, 10)
                  [5 zone temps, 5 control features without supply temps]
            j: Mode selection tensor (batch_size,) - 0=AC, 1=NV

        Returns:
            Predicted zone temperatures (batch_size, output_size)
        """
        batch_size = x_ac.size(0)
        outputs = []

        for i in range(batch_size):
            if j[i] == 0:
                # Use AC input: 5 zones + 8 controls
                # Pad to 15 features (5 zones + 10 controls) by adding 2 zeros
                ac_input = x_ac[i].unsqueeze(0)
                padding = torch.zeros(1, ac_input.size(1), 2, device=ac_input.device)
                input_data = torch.cat([ac_input, padding], dim=2)
            else:
                # Use NV input: 5 zones + 5 controls
                # Pad to 15 features (5 zones + 10 controls) by adding 5 zeros
                nv_input = x_nv[i].unsqueeze(0)
                padding = torch.zeros(1, nv_input.size(1), 5, device=nv_input.device)
                input_data = torch.cat([nv_input, padding], dim=2)

            # Forward through LSTM
            out = self.lstm_model(input_data)
            outputs.append(out)

        outputs = torch.cat(outputs, dim=0)
        return outputs


class SeparateNormalizer(nn.Module):
    """Normalizer that handles states, actions, and features separately."""
    def __init__(self):
        super(SeparateNormalizer, self).__init__()
        self.normalization_initialized = False
        
        # Separate normalization parameters - will be initialized based on expert data
        self.register_buffer('state_mean', torch.tensor([]))
        self.register_buffer('state_std', torch.tensor([]))
        self.register_buffer('action_mean', torch.tensor([]))
        self.register_buffer('action_std', torch.tensor([]))
        self.register_buffer('feature_mean', torch.tensor([]))
        self.register_buffer('feature_std', torch.tensor([]))
        
    def initialize_normalization(self, states: torch.Tensor, actions: torch.Tensor, features: torch.Tensor):
        """
        Initialize normalization parameters separately for states, actions, and features.
        
        Args:
            states: State data tensor of shape (..., state_dim)
            actions: Action data tensor of shape (..., action_dim)  
            features: Feature data tensor of shape (..., feature_dim)
        """
        # Flatten each component for statistics computation
        states_flat = states.view(-1, states.shape[-1])
        actions_flat = actions.view(-1, actions.shape[-1])
        features_flat = features.view(-1, features.shape[-1])
        
        # Compute separate statistics
        self.state_mean.data = states_flat.mean(dim=0)
        self.state_std.data = states_flat.std(dim=0) + 1e-8
        
        self.action_mean.data = actions_flat.mean(dim=0)
        self.action_std.data = actions_flat.std(dim=0) + 1e-8
        
        self.feature_mean.data = features_flat.mean(dim=0)
        self.feature_std.data = features_flat.std(dim=0) + 1e-8
        
        self.normalization_initialized = True
        
    def normalize(self, states: torch.Tensor, actions: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Normalize states, actions, and features separately, then concatenate.
        
        Args:
            states: State tensor (..., state_dim)
            actions: Action tensor (..., action_dim)
            features: Feature tensor (..., feature_dim)
            
        Returns:
            Normalized and concatenated tensor
        """
        if not self.normalization_initialized:
            # If not initialized, just concatenate without normalization
            return torch.cat([states, actions, features], dim=-1)
            
        # Normalize each component separately
        norm_states = (states - self.state_mean) / self.state_std
        norm_actions = (actions - self.action_mean) / self.action_std
        norm_features = (features - self.feature_mean) / self.feature_std
        
        # Concatenate normalized components
        return torch.cat([norm_states, norm_actions, norm_features], dim=-1)


class StateActionNormalizer(nn.Module):
    """Normalize state and action tensors independently, then concatenate."""

    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.register_buffer('state_mean', torch.zeros(state_dim))
        self.register_buffer('state_std', torch.ones(state_dim))
        self.register_buffer('action_mean', torch.zeros(action_dim))
        self.register_buffer('action_std', torch.ones(action_dim))
        self.normalization_initialized = False

    def initialize_normalization(self, states: torch.Tensor, actions: torch.Tensor) -> None:
        if states.numel() > 0:
            states_flat = states.view(-1, states.shape[-1])
            self.state_mean.data = states_flat.mean(dim=0)
            self.state_std.data = states_flat.std(dim=0) + 1e-8
        if actions.numel() > 0:
            actions_flat = actions.view(-1, actions.shape[-1])
            self.action_mean.data = actions_flat.mean(dim=0)
            self.action_std.data = actions_flat.std(dim=0) + 1e-8
        self.normalization_initialized = True

    def normalize(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if not self.normalization_initialized:
            return torch.cat([states, actions], dim=-1)
        norm_states = (states - self.state_mean) / self.state_std
        norm_actions = (actions - self.action_mean) / self.action_std
        return torch.cat([norm_states, norm_actions], dim=-1)


class MLPReward(nn.Module):
    """MLP-based reward/discriminator head for AIRL that outputs logits."""
    def __init__(self, input_dim: int = 19, hidden_dims: List[int] = [64, 64]):
        super(MLPReward, self).__init__()
        self.input_dim = input_dim
        self.normalizer = SeparateNormalizer()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        
        # Output layer (1 scalar logit)
        layers.append(nn.Linear(prev_dim, 1))
        self.model = nn.Sequential(*layers)

    def initialize_normalization(self, states: torch.Tensor, actions: torch.Tensor, features: torch.Tensor):
        """
        Initialize separate normalization for states, actions, and features.
        
        Args:
            states: State data tensor
            actions: Action data tensor
            features: Feature data tensor
        """
        self.normalizer.initialize_normalization(states, actions, features)

    def forward(self, states: torch.Tensor, actions: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute reward with separate normalization.
        
        Args:
            states: State tensor (..., state_dim)
            actions: Action tensor (..., action_dim)
            features: Feature tensor (..., feature_dim)
            
        Returns:
            Logit scores with the same leading dimensions as the inputs
        """
        # Apply separate normalization and concatenate
        x = self.normalizer.normalize(states, actions, features)
        
        logits = self.model(x)
        return logits.squeeze(-1)


class GRUReward(nn.Module):
    """GRU-based reward/discriminator for sequential inputs that outputs logits."""
    def __init__(self, in_dim: int = 19, hid: int = 32):
        super().__init__()
        self.input_dim = in_dim
        self.normalizer = SeparateNormalizer()
        self.gru = nn.GRU(in_dim, hid, batch_first=True)
        self.head = nn.Linear(hid, 1)

    def initialize_normalization(self, states: torch.Tensor, actions: torch.Tensor, features: torch.Tensor):
        """
        Initialize separate normalization for states, actions, and features.
        
        Args:
            states: State data tensor
            actions: Action data tensor
            features: Feature data tensor
        """
        self.normalizer.initialize_normalization(states, actions, features)

    def forward(self, states: torch.Tensor, actions: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GRU reward model with separate normalization.
        
        Args:
            states: State tensor (B, T, state_dim)
            actions: Action tensor (B, T, action_dim)
            features: Feature tensor (B, T, feature_dim)
            
        Returns:
            Logit sequence of shape (B, T)
        """
        # Apply separate normalization and concatenate
        phi_seq = self.normalizer.normalize(states, actions, features)
        
        h, _ = self.gru(phi_seq)
        logits = self.head(h)
        return logits.squeeze(-1)


class MLPRewardStateActionOnly(nn.Module):
    """State-action-only MLP discriminator head that outputs logits."""
    def __init__(self, state_dim: int = 12, action_dim: int = 8, hidden_dims: List[int] = [64, 64]):
        super(MLPRewardStateActionOnly, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        input_dim = state_dim + action_dim
        self.normalizer = StateActionNormalizer(state_dim, action_dim)

        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        
        # Output layer (1 scalar logit)
        layers.append(nn.Linear(prev_dim, 1))
        self.model = nn.Sequential(*layers)

    def initialize_normalization(self, states: torch.Tensor, actions: torch.Tensor) -> None:
        self.normalizer.initialize_normalization(states, actions)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute reward using only states and actions.
        
        Args:
            states: State tensor (..., state_dim)
            actions: Action tensor (..., action_dim)
            
        Returns:
            Logit scores with the same leading dimensions as the inputs
        """
        # Concatenate normalized states and actions
        x = self.normalizer.normalize(states, actions)

        logits = self.model(x)
        return logits.squeeze(-1)


class GRURewardStateActionOnly(nn.Module):
    """GRU-based discriminator that only consumes state and action sequences."""
    def __init__(self, state_dim: int = 12, action_dim: int = 8, hid: int = 32):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        input_dim = state_dim + action_dim
        self.normalizer = StateActionNormalizer(state_dim, action_dim)

        self.gru = nn.GRU(input_dim, hid, batch_first=True)
        self.head = nn.Linear(hid, 1)

    def initialize_normalization(self, states: torch.Tensor, actions: torch.Tensor) -> None:
        self.normalizer.initialize_normalization(states, actions)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GRU reward model using only states and actions.
        
        Args:
            states: State tensor (B, T, state_dim)
            actions: Action tensor (B, T, action_dim)
            
        Returns:
            Logit sequence of shape (B, T)
        """
        # Concatenate normalized states and actions
        x = self.normalizer.normalize(states, actions)

        h, _ = self.gru(x)
        logits = self.head(h)
        return logits.squeeze(-1)


class GRURewardStateActionComfortEnergy(nn.Module):
    """
    GRU-based discriminator that consumes state/action sequences along with
    comfort and energy feature channels.
    """

    def __init__(
        self,
        state_dim: int = 12,
        action_dim: int = 8,
        feature_dim: int = 2,
        hid: int = 32,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.feature_dim = feature_dim
        self.normalizer = SeparateNormalizer()
        self.gru = nn.GRU(state_dim + action_dim + feature_dim, hid, batch_first=True)
        self.head = nn.Linear(hid, 1)

    def initialize_normalization(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        features: torch.Tensor,
    ) -> None:
        self.normalizer.initialize_normalization(states, actions, features)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through GRU reward model using states, actions,
        and comfort/energy features.

        Args:
            states: State tensor (B, T, state_dim)
            actions: Action tensor (B, T, action_dim)
            features: Feature tensor (B, T, feature_dim)

        Returns:
            Logit sequence of shape (B, T)
        """
        phi_seq = self.normalizer.normalize(states, actions, features)
        h, _ = self.gru(phi_seq)
        logits = self.head(h)
        return logits.squeeze(-1)


class MMVPolicyActorCritic(nn.Module):
    """
    Actor-Critic policy network for Mixed-Mode Ventilation control.
    Supports hierarchical action space with window control and HVAC setpoints.
    """
    def __init__(self, state_dim: int, num_zones: int = 5):
        super().__init__()
        hidden = 64
        self.num_zones = num_zones

        # Shared trunk
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)

        # Action heads
        self.change_head = nn.Linear(hidden, 1)            # Bernoulli for window change
        self.supply_head_ac = nn.Linear(hidden, num_zones) # AC supply temperatures
        self.lc_head_nv = nn.Linear(hidden, 2)             # Local cooling logits for NV mode

        # Learnable standard deviations for Gaussian actions
        self.log_std_ac = nn.Parameter(torch.full((num_zones,), -0.5))

        # Value head
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, s: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass to compute policy outputs.
        
        Args:
            s: State tensor (batch_size, state_dim)
            
        Returns:
            Dictionary with policy outputs
        """
        # Check for NaN inputs and use LeakyReLU for stability
        if torch.isnan(s).any():
            print("NaN detected in state input:", s)
        
        x = nn.LeakyReLU(0.1)(self.fc1(s))
        if torch.isnan(x).any():
            print("NaN detected after first layer:", x)
            
        x = nn.LeakyReLU(0.1)(self.fc2(x))
        if torch.isnan(x).any():
            print("NaN detected after second layer:", x)
        
        return dict(
            change_logit=self.change_head(x).squeeze(-1),     # (B,)
            supply_mu_ac=self.supply_head_ac(x),              # (B, num_zones)
            lc_logits_nv=self.lc_head_nv(x),                  # (B, 2)
            value=self.value_head(x).squeeze(-1)              # (B,)
        )

    @torch.no_grad()
    def get_action(self, state: torch.Tensor) -> Tuple[Dict, torch.Tensor]:
        """
        Sample actions from the policy.
        
        Args:
            state: State tensor (batch_size, state_dim)
            
        Returns:
            Tuple of (action_dict, log_prob)
        """
        out = self(state)
        
        # Get previous window state from last element of state vector
        prev_w = state[:, -1]  # shape (B,)

        # Sample window change decision
        ch_dist = torch.distributions.Bernoulli(logits=out["change_logit"])
        change = ch_dist.sample()  # 0 = stay, 1 = flip
        w_action = (1 - change) * prev_w + change * (1 - prev_w)

        logp_total = ch_dist.log_prob(change)
        
        actions = []
        for b in range(state.size(0)):
            if w_action[b] == 1:  # Natural ventilation mode
                # Sample local cooling binary decisions
                lc_logits = out["lc_logits_nv"][b]
                lc_dist = torch.distributions.Bernoulli(logits=lc_logits)
                lc = lc_dist.sample()
                logp_total[b] += lc_dist.log_prob(lc).sum()

                # FCU supply temps determined by zone temps (natural ventilation)
                supply = state[b, :self.num_zones]

            else:  # Air conditioning mode
                # Local cooling must be active when windows closed
                lc = torch.ones_like(out["lc_logits_nv"][b])

                # Sample FCU supply temperatures
                sup_mu = out["supply_mu_ac"][b]
                sup_std = self.log_std_ac.exp()
                sup_dist = torch.distributions.Normal(sup_mu, sup_std)
                supply = sup_dist.sample().clamp(0.10, 0.40)  # Low temperature range
                logp_total[b] += sup_dist.log_prob(supply).sum()
            
            actions.append(dict(
                change=int(change[b].item()),
                local_cooling=lc.cpu().numpy(),
                supply_temps=supply.cpu().numpy()
            ))

        return actions[0] if len(actions) == 1 else actions, logp_total

    def act(self, state: torch.Tensor) -> Tuple[Dict[str, Any], torch.Tensor, torch.Tensor]:
        """
        Action method for compatibility with PPO trainers.
        
        Args:
            state: State tensor (batch_size, state_dim)
            
        Returns:
            Tuple of (action_dict, log_prob, value)
        """
        action_result, log_prob = self.get_action(state)
        
        # Handle case where get_action returns a list for batch_size > 1
        if isinstance(action_result, list):
            action_dict = action_result[0]  # Take first action for single state
        else:
            action_dict = action_result
        
        # Get value estimate
        with torch.no_grad():
            out = self(state)
            value = out["value"]
        
        return action_dict, log_prob, value

    def evaluate_actions(self, states: torch.Tensor, actions: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO training.
        
        Args:
            states: State tensor (batch_size, state_dim)
            actions: List of action dictionaries
            
        Returns:
            Tuple of (log_probs, values, entropy)
        """
        out = self(states)
        
        # Extract action components
        changes = torch.tensor([a["change"] for a in actions], dtype=torch.float32, device=states.device)
        lc_actions = torch.tensor([a["local_cooling"] for a in actions], dtype=torch.float32, device=states.device)
        supply_actions = torch.tensor([a["supply_temps"] for a in actions], dtype=torch.float32, device=states.device)
        
        # Compute log probabilities
        ch_dist = torch.distributions.Bernoulli(logits=out["change_logit"])
        logp_changes = ch_dist.log_prob(changes)
        
        # Local cooling log probs (for NV mode)
        lc_dist = torch.distributions.Bernoulli(logits=out["lc_logits_nv"])
        logp_lc = lc_dist.log_prob(lc_actions).sum(-1)
        
        # Supply temperature log probs (for AC mode)
        sup_std = self.log_std_ac.exp()
        sup_dist = torch.distributions.Normal(out["supply_mu_ac"], sup_std)
        logp_supply = sup_dist.log_prob(supply_actions).sum(-1)
        
        # Combine log probabilities
        log_probs = logp_changes + logp_lc + logp_supply
        
        # Compute entropy
        entropy = ch_dist.entropy() + lc_dist.entropy().sum(-1) + sup_dist.entropy().sum(-1)
        
        return log_probs, out["value"], entropy


def load_dynamics_model(model_path: str, device: torch.device,
                       hidden_dim: int = 128, output_dim: int = 5,
                       ac_dim: int = 13, nv_dim: int = 10) -> nn.Module:
    """
    Load a pre-trained dynamics model, automatically detecting the architecture.

    Supports legacy `Combined_CNN_LSTM` checkpoints as well as the newer
    simple LSTM checkpoints (saved from `trained_lstm_direct.pth`).

    Args:
        model_path: Path to saved model state dict
        device: PyTorch device
        hidden_dim: Hidden dimension size
        output_dim: Output dimension (number of zones)
        ac_dim: AC input channels
        nv_dim: NV input channels

    Returns:
        Loaded dynamics model on the requested device.
    """
    model_path = resolve_repo_path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Could not find dynamics checkpoint: {model_path}")

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except Exception:
        # Fallback for older PyTorch versions or non-weights-only files
        checkpoint = torch.load(model_path, map_location=device)

    # Some checkpoints might wrap the state dict in another dict
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break

    if not isinstance(checkpoint, (dict, nn.Module)):
        raise RuntimeError(f"Unexpected checkpoint type at {model_path}: {type(checkpoint)}")

    # Weighted-only loading returns dict/OrderedDict; ensure we have raw state dict
    model_state_dict = checkpoint.state_dict() if isinstance(checkpoint, nn.Module) else checkpoint
    state_keys = set(model_state_dict.keys())

    # Detect architecture by inspecting parameter names
    if "gate.0.weight" in state_keys and any(k.startswith("branch_AC.cnn_layers") for k in state_keys):
        inferred_ac_dim = int(model_state_dict["branch_AC.cnn_layers.0.weight"].shape[1])
        inferred_nv_dim = int(model_state_dict["branch_NV.cnn_layers.0.weight"].shape[1])
        inferred_hidden = int(model_state_dict["output_layer.weight"].shape[1])
        inferred_output = int(model_state_dict["output_layer.weight"].shape[0])
        model = Combined_CNN_LSTM_Gated(
            hidden_size=inferred_hidden,
            output_size=inferred_output,
            ac_channels=inferred_ac_dim,
            nv_channels=inferred_nv_dim,
        ).to(device)
        model.load_state_dict(model_state_dict)
    elif any(k.startswith("branch_AC") for k in state_keys):
        model = Combined_CNN_LSTM(
            hidden_size=hidden_dim,
            output_size=output_dim,
            ac_channels=ac_dim,
            nv_channels=nv_dim
        ).to(device)
        model.load_state_dict(model_state_dict)
    elif any(k.startswith("lstm.") for k in state_keys) and "fc.weight" in state_keys:
        model = LSTM_Direct_Branched(
            hidden_size=hidden_dim,
            output_size=output_dim,
            ac_channels=ac_dim,
            nv_channels=nv_dim
        ).to(device)
        model.lstm_model.load_state_dict(model_state_dict)
    else:
        raise RuntimeError(
            "Unable to determine dynamics model architecture from checkpoint keys: "
            f"{sorted(list(state_keys))[:5]}..."
        )

    model.eval()
    return model


def load_lstm_dynamics_model(model_path: str, device: torch.device,
                             hidden_dim: int = 128, output_dim: int = 5,
                             ac_dim: int = 13, nv_dim: int = 10,
                             num_layers: int = 1, dropout: float = 0.1) -> LSTM_Direct_Branched:
    """
    Load a pre-trained LSTM dynamics model with branched interface.
    Uses LSTM_Direct_Branched wrapper to match Combined_CNN_LSTM interface.

    Args:
        model_path: Path to saved model state dict
        device: PyTorch device
        hidden_dim: Hidden dimension size
        output_dim: Output dimension (number of zones)
        ac_dim: AC input channels
        nv_dim: NV input channels
        num_layers: Number of LSTM layers
        dropout: Dropout rate

    Returns:
        Loaded LSTM dynamics model with branched interface
    """
    model_path = resolve_repo_path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Could not find LSTM dynamics checkpoint: {model_path}")

    # Create branched wrapper model
    model = LSTM_Direct_Branched(
        hidden_size=hidden_dim,
        output_size=output_dim,
        ac_channels=ac_dim,
        nv_channels=nv_dim,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)

    # Load the trained LSTM weights into the inner lstm_model
    try:
        model_state_dict = torch.load(model_path, map_location=device, weights_only=True)
    except Exception:
        # Fallback for older PyTorch versions
        model_state_dict = torch.load(model_path, map_location=device)

    model.lstm_model.load_state_dict(model_state_dict)
    model.eval()

    return model
