"""Hybrid PPO training utilities for hierarchical mode control."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

try:
    from .rl_environment import (
        FCU_ZONES,
        MAX_DROP_IDX,
        MAX_RISE_IDX,
        MODE_DIM,
        MODE_MECH,
        N_TEMP,
        TEMP_LEVELS,
        MixedModeVentilationRLEnv,
    )
except ImportError:
    from rl_environment import (
        FCU_ZONES,
        MAX_DROP_IDX,
        MAX_RISE_IDX,
        MODE_DIM,
        MODE_MECH,
        N_TEMP,
        TEMP_LEVELS,
        MixedModeVentilationRLEnv,
    )


# ---------------------------------------------------------------------------
# Masking helpers
# ---------------------------------------------------------------------------

def make_range_mask(lo: torch.Tensor, hi: torch.Tensor, size: int) -> torch.Tensor:
    """Create inclusive [lo, hi] masks for each sample."""
    indices = torch.arange(size, device=lo.device).unsqueeze(0)  # (1, size)
    lo = lo.unsqueeze(-1)  # (B, 1)
    hi = hi.unsqueeze(-1)  # (B, 1)
    mask = (indices >= lo) & (indices <= hi)  # (B, size)
    return mask.to(dtype=lo.dtype)


def masked_categorical(logits: torch.Tensor, mask: torch.Tensor) -> Categorical:
    safe_mask = torch.clamp(mask, min=1e-9)
    masked_logits = logits + torch.log(safe_mask)
    return Categorical(logits=masked_logits)


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    logits = logits - logits.max(dim=dim, keepdim=True)[0]
    exp_logits = torch.exp(logits) * mask
    return exp_logits / (exp_logits.sum(dim=dim, keepdim=True) + 1e-9)


# ---------------------------------------------------------------------------
# Policy network
# ---------------------------------------------------------------------------


class Trunk(nn.Module):
    """Simple MLP trunk shared by policy and value heads."""

    def __init__(self, obs_dim: int, hidden: int = 256, depth: int = 3):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = obs_dim
        for d in range(depth):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.LayerNorm(hidden))
            layers.append(nn.Tanh())
            in_dim = hidden
        self.model = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.model(obs)


class HybridPPOPolicy(nn.Module):
    """Hierarchical policy with mode head and masked FCU heads."""

    def __init__(self, obs_dim: int, hidden: int = 256):
        super().__init__()
        self.trunk = Trunk(obs_dim, hidden=hidden, depth=3)
        self.head_mode = nn.Linear(hidden, MODE_DIM)
        self.head_fcu = nn.ModuleList([nn.Linear(hidden, N_TEMP) for _ in range(FCU_ZONES)])
        self.v_head = nn.Linear(hidden, 1)

    def forward(
        self,
        obs: torch.Tensor,
        prev_fcu_idx: torch.Tensor,
        prev_mode: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
        h = self.trunk(obs)
        logits_mode = self.head_mode(h)
        logits_fcu = [head(h) for head in self.head_fcu]

        lo = torch.clamp(prev_fcu_idx - MAX_DROP_IDX, min=0)
        hi = torch.clamp(prev_fcu_idx + MAX_RISE_IDX, max=N_TEMP - 1)
        masks_fcu = [
            make_range_mask(lo[:, j], hi[:, j], N_TEMP)
            for j in range(FCU_ZONES)
        ]

        value = self.v_head(h).squeeze(-1)
        return logits_mode, logits_fcu, masks_fcu, value


# ---------------------------------------------------------------------------
# Rollout buffer
# ---------------------------------------------------------------------------


@dataclass
class RolloutBatch:
    obs: torch.Tensor
    prev_mode: torch.Tensor
    prev_fcu_idx: torch.Tensor
    mode_action: torch.Tensor
    fcu_action: torch.Tensor
    old_logp_mode: torch.Tensor
    old_logp_fcu: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class RolloutBuffer:
    def __init__(self):
        self.obs: List[np.ndarray] = []
        self.prev_mode: List[int] = []
        self.prev_fcu_idx: List[np.ndarray] = []
        self.mode_action: List[int] = []
        self.fcu_action: List[np.ndarray] = []
        self.logp_mode: List[float] = []
        self.logp_fcu: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.dones: List[bool] = []
        self.advantages: Optional[np.ndarray] = None
        self.returns: Optional[np.ndarray] = None
        self.info: List[Dict[str, Any]] = []

    @property
    def size(self) -> int:
        return len(self.obs)

    def add(
        self,
        obs: np.ndarray,
        prev_mode: int,
        prev_fcu_idx: Sequence[int],
        mode_action: int,
        fcu_action: Sequence[int],
        logp_mode: float,
        logp_fcu: Sequence[float],
        reward: float,
        value: float,
        done: bool,
        info: Dict[str, Any],
    ) -> None:
        self.obs.append(obs.astype(np.float32))
        self.prev_mode.append(int(prev_mode))
        self.prev_fcu_idx.append(np.asarray(prev_fcu_idx, dtype=np.int64))
        self.mode_action.append(int(mode_action))
        self.fcu_action.append(np.asarray(fcu_action, dtype=np.int64))
        self.logp_mode.append(float(logp_mode))
        self.logp_fcu.append(np.asarray(logp_fcu, dtype=np.float32))
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.dones.append(bool(done))
        self.info.append(info)

    def to_tensors(self, device: torch.device) -> RolloutBatch:
        if self.advantages is None or self.returns is None:
            raise RuntimeError("Advantages/returns have not been computed")

        obs = torch.tensor(np.asarray(self.obs), dtype=torch.float32, device=device)
        prev_mode = torch.tensor(self.prev_mode, dtype=torch.long, device=device)
        prev_fcu_idx = torch.tensor(np.asarray(self.prev_fcu_idx), dtype=torch.long, device=device)
        mode_action = torch.tensor(self.mode_action, dtype=torch.long, device=device)
        fcu_action = torch.tensor(np.asarray(self.fcu_action), dtype=torch.long, device=device)
        old_logp_mode = torch.tensor(self.logp_mode, dtype=torch.float32, device=device)
        old_logp_fcu = torch.tensor(np.asarray(self.logp_fcu), dtype=torch.float32, device=device)
        advantages = torch.tensor(self.advantages, dtype=torch.float32, device=device)
        returns = torch.tensor(self.returns, dtype=torch.float32, device=device)

        return RolloutBatch(
            obs=obs,
            prev_mode=prev_mode,
            prev_fcu_idx=prev_fcu_idx,
            mode_action=mode_action,
            fcu_action=fcu_action,
            old_logp_mode=old_logp_mode,
            old_logp_fcu=old_logp_fcu,
            advantages=advantages,
            returns=returns,
        )

    def iter_batches(self, batch_size: int, device: torch.device) -> Iterable[RolloutBatch]:
        dataset = self.to_tensors(device)
        indices = torch.randperm(self.size, device=device)
        for start in range(0, self.size, batch_size):
            end = min(start + batch_size, self.size)
            batch_idx = indices[start:end]
            yield RolloutBatch(
                obs=dataset.obs[batch_idx],
                prev_mode=dataset.prev_mode[batch_idx],
                prev_fcu_idx=dataset.prev_fcu_idx[batch_idx],
                mode_action=dataset.mode_action[batch_idx],
                fcu_action=dataset.fcu_action[batch_idx],
                old_logp_mode=dataset.old_logp_mode[batch_idx],
                old_logp_fcu=dataset.old_logp_fcu[batch_idx],
                advantages=dataset.advantages[batch_idx],
                returns=dataset.returns[batch_idx],
            )


# ---------------------------------------------------------------------------
# GAE computation
# ---------------------------------------------------------------------------


def compute_gae(buffer: RolloutBuffer, gamma: float = 0.99, lam: float = 0.95) -> None:
    size = buffer.size
    advantages = np.zeros(size, dtype=np.float32)
    returns = np.zeros(size, dtype=np.float32)

    next_value = 0.0
    gae = 0.0

    for t in reversed(range(size)):
        reward = buffer.rewards[t]
        value = buffer.values[t]
        done = float(buffer.dones[t])

        delta = reward + gamma * next_value * (1.0 - done) - value
        gae = delta + gamma * lam * (1.0 - done) * gae

        advantages[t] = gae
        returns[t] = gae + value

        next_value = value
        if done > 0.5:
            next_value = 0.0
            gae = 0.0

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    buffer.advantages = advantages
    buffer.returns = returns


# ---------------------------------------------------------------------------
# Rollout collection
# ---------------------------------------------------------------------------


@dataclass
class RolloutStats:
    returns: List[float]
    energy: List[float]
    violation: List[float]
    switches: List[float]


def _sample_day_index(env: MixedModeVentilationRLEnv) -> int:
    return random.randrange(len(env.daily_data_list))


def collect_rollouts(
    env: MixedModeVentilationRLEnv,
    policy: HybridPPOPolicy,
    n_days: int,
    device: torch.device,
) -> Tuple[RolloutBuffer, RolloutStats]:
    buffer = RolloutBuffer()
    policy.eval()

    episode_returns: List[float] = []
    episode_energy: List[float] = []
    episode_violation: List[float] = []
    episode_switches: List[float] = []

    for _ in range(n_days):
        day_idx = _sample_day_index(env)
        obs = env.reset(day_idx)
        done = False
        ep_ret = 0.0
        ep_energy = 0.0
        ep_violation = 0.0
        ep_switch = 0.0

        while not done:
            prev_mode_val = int(env.prev_mode)
            prev_fcu_idx_list = list(env.prev_fcu_idx)

            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            prev_fcu = torch.tensor([prev_fcu_idx_list], dtype=torch.long, device=device)
            prev_mode = torch.tensor([prev_mode_val], dtype=torch.long, device=device)

            with torch.no_grad():
                logits_mode, logits_fcu, masks_fcu, value = policy(obs_tensor, prev_fcu, prev_mode)

            dist_mode = Categorical(logits=logits_mode.squeeze(0))
            mode_action = dist_mode.sample()
            logp_mode = dist_mode.log_prob(mode_action)

            fcu_indices = np.full(FCU_ZONES, -1, dtype=np.int64)
            logp_fcu = np.zeros(FCU_ZONES, dtype=np.float32)

            mode_int = int(mode_action.item())
            if mode_int == MODE_MECH:
                for j in range(FCU_ZONES):
                    dist_j = masked_categorical(logits_fcu[j].squeeze(0), masks_fcu[j].squeeze(0))
                    idx = dist_j.sample()
                    fcu_indices[j] = int(idx.item())
                    logp_fcu[j] = float(dist_j.log_prob(idx).item())

            env_action = (mode_int, fcu_indices.tolist() if mode_int == MODE_MECH else None)
            next_obs, reward, done, info = env.step(env_action)

            buffer.add(
                obs=obs,
                prev_mode=prev_mode_val,
                prev_fcu_idx=prev_fcu_idx_list,
                mode_action=mode_int,
                fcu_action=fcu_indices,
                logp_mode=float(logp_mode.item()),
                logp_fcu=logp_fcu,
                reward=reward,
                value=float(value.item()),
                done=done,
                info=info,
            )

            obs = next_obs
            ep_ret += reward
            ep_energy += info['energy']
            ep_violation += info['comfort']
            ep_switch += info['window_switch']

        episode_returns.append(ep_ret)
        episode_energy.append(ep_energy)
        episode_violation.append(ep_violation)
        episode_switches.append(ep_switch)

    policy.train()
    stats = RolloutStats(episode_returns, episode_energy, episode_violation, episode_switches)
    return buffer, stats


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------


@dataclass
class UpdateStats:
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float


def hppo_update(
    policy: HybridPPOPolicy,
    optimizer: torch.optim.Optimizer,
    buffer: RolloutBuffer,
    device: torch.device,
    clip_eps: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 1.0,
    epochs: int = 10,
    batch_size: int = 4096,
) -> UpdateStats:
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_kl = 0.0
    total_clip = 0.0
    total_batches = 0

    for _ in range(epochs):
        for batch in buffer.iter_batches(batch_size, device):
            logits_mode, logits_fcu, masks_fcu, values = policy(batch.obs, batch.prev_fcu_idx, batch.prev_mode)

            dist_mode = Categorical(logits=logits_mode)
            logp_mode = dist_mode.log_prob(batch.mode_action)
            entropy_mode = dist_mode.entropy()

            mech_mask = (batch.mode_action == MODE_MECH)
            logp_fcu = torch.zeros_like(logp_mode)
            entropy_fcu = torch.zeros_like(logp_mode)

            if mech_mask.any():
                for j in range(FCU_ZONES):
                    logits_subset = logits_fcu[j][mech_mask]
                    mask_subset = masks_fcu[j][mech_mask]

                    dist_j = masked_categorical(logits_subset, mask_subset)
                    idx_subset = batch.fcu_action[mech_mask, j]

                    log_prob_j = dist_j.log_prob(idx_subset)
                    entropy_j = dist_j.entropy()

                    logp_fcu[mech_mask] += log_prob_j
                    entropy_fcu[mech_mask] += entropy_j

            old_logp_total = batch.old_logp_mode + batch.old_logp_fcu.sum(dim=1)
            logp_total = logp_mode + logp_fcu

            ratio = torch.exp(logp_total - old_logp_total)
            clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)

            policy_loss = -torch.mean(torch.min(ratio * batch.advantages, clipped_ratio * batch.advantages))
            value_loss = F.mse_loss(values, batch.returns)

            entropy = (entropy_mode + entropy_fcu).mean()

            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = torch.mean(old_logp_total - logp_total).item()
                clip_frac = torch.mean((torch.abs(ratio - 1.0) > clip_eps).float()).item()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
            total_kl += approx_kl
            total_clip += clip_frac
            total_batches += 1

    if total_batches == 0:
        return UpdateStats(0.0, 0.0, 0.0, 0.0, 0.0)

    return UpdateStats(
        policy_loss=total_policy_loss / total_batches,
        value_loss=total_value_loss / total_batches,
        entropy=total_entropy / total_batches,
        approx_kl=total_kl / total_batches,
        clip_fraction=total_clip / total_batches,
    )


# ---------------------------------------------------------------------------
# Training loop and evaluation
# ---------------------------------------------------------------------------


@dataclass
class TrainingHistory:
    train_return: List[float]
    train_energy: List[float]
    train_violation: List[float]
    train_switch: List[float]
    val_return: List[float]
    policy_loss: List[float]
    value_loss: List[float]
    entropy: List[float]
    approx_kl: List[float]


def train_hppo(
    env: MixedModeVentilationRLEnv,
    policy: HybridPPOPolicy,
    optimizer: torch.optim.Optimizer,
    total_updates: int = 500,
    days_per_update: int = 8,
    gamma: float = 0.99,
    lam: float = 0.95,
    clip_eps: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 1.0,
    epochs: int = 10,
    batch_size: int = 4096,
    eval_env: Optional[MixedModeVentilationRLEnv] = None,
    eval_days: Optional[Sequence[int]] = None,
    device: Optional[torch.device] = None,
) -> TrainingHistory:
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    policy.to(device)

    history = TrainingHistory([], [], [], [], [], [], [], [], [])

    for update in range(total_updates):
        buffer, stats = collect_rollouts(env, policy, days_per_update, device)
        compute_gae(buffer, gamma=gamma, lam=lam)
        update_stats = hppo_update(
            policy,
            optimizer,
            buffer,
            device=device,
            clip_eps=clip_eps,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            epochs=epochs,
            batch_size=batch_size,
        )

        avg_return = float(np.mean(stats.returns)) if stats.returns else 0.0
        avg_energy = float(np.mean(stats.energy)) if stats.energy else 0.0
        avg_violation = float(np.mean(stats.violation)) if stats.violation else 0.0
        avg_switch = float(np.mean(stats.switches)) if stats.switches else 0.0

        history.train_return.append(avg_return)
        history.train_energy.append(avg_energy)
        history.train_violation.append(avg_violation)
        history.train_switch.append(avg_switch)
        history.policy_loss.append(update_stats.policy_loss)
        history.value_loss.append(update_stats.value_loss)
        history.entropy.append(update_stats.entropy)
        history.approx_kl.append(update_stats.approx_kl)

        if eval_env is not None:
            fresh_eval_env = eval_env.clone()
            val_ret = evaluate(fresh_eval_env, policy, eval_days, device=device)
        else:
            val_ret = float('nan')
        history.val_return.append(val_ret)

        print(
            f"Update {update:03d} | Return {avg_return:7.2f} | Val {val_ret:7.2f} | "
            f"Energy {avg_energy:6.3f} | Comfort {avg_violation:6.3f} | Switch {avg_switch:5.2f} | "
            f"Policy {update_stats.policy_loss:6.4f} | Value {update_stats.value_loss:6.4f} | "
            f"Entropy {update_stats.entropy:6.4f} | KL {update_stats.approx_kl:6.4f}"
        )

    return history


def evaluate(
    env: MixedModeVentilationRLEnv,
    policy: HybridPPOPolicy,
    eval_days: Optional[Sequence[int]],
    device: Optional[torch.device] = None,
) -> float:
    if eval_days is None:
        eval_days = list(range(len(env.daily_data_list)))

    device = device or next(policy.parameters()).device
    policy.eval()

    returns = []
    for day in eval_days:
        obs = env.reset(day)
        done = False
        total = 0.0

        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            prev_fcu = torch.tensor([env.prev_fcu_idx], dtype=torch.long, device=device)
            prev_mode = torch.tensor([env.prev_mode], dtype=torch.long, device=device)

            with torch.no_grad():
                logits_mode, logits_fcu, masks_fcu, _ = policy(obs_tensor, prev_fcu, prev_mode)

            mode = torch.argmax(logits_mode[0]).item()
            fcu_idx = None
            if mode == MODE_MECH:
                fcu_idx = []
                for j in range(FCU_ZONES):
                    probs = masked_softmax(logits_fcu[j][0], masks_fcu[j][0])
                    fcu_idx.append(int(torch.argmax(probs).item()))

            obs, reward, done, _ = env.step((mode, fcu_idx))
            total += reward

        returns.append(total)

    policy.train()
    return float(np.mean(returns)) if returns else 0.0


# ---------------------------------------------------------------------------
# Convenience entry points
# ---------------------------------------------------------------------------


def run_hppo_training(
    env: MixedModeVentilationRLEnv,
    validation_env: Optional[MixedModeVentilationRLEnv] = None,
    total_updates: int = 200,
    days_per_update: int = 8,
    policy_lr: float = 3e-4,
    hidden_dim: int = 256,
    device: Optional[torch.device] = None,
    ppo_epochs: int = 10,
    batch_size: int = 4096,
    clip_eps: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 1.0,
) -> Tuple[HybridPPOPolicy, TrainingHistory]:
    initial_obs = env.reset(0)
    obs_dim = initial_obs.shape[0]

    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    policy = HybridPPOPolicy(obs_dim=obs_dim, hidden=hidden_dim).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=policy_lr)

    history = train_hppo(
        env=env,
        policy=policy,
        optimizer=optimizer,
        total_updates=total_updates,
        days_per_update=days_per_update,
        clip_eps=clip_eps,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        epochs=ppo_epochs,
        batch_size=batch_size,
        eval_env=validation_env,
        eval_days=list(range(len(validation_env.daily_data_list))) if validation_env else None,
        device=device,
    )

    return policy, history


def aggregate_rollout_info(buffer: RolloutBuffer) -> Dict[str, float]:
    energy = [info['energy'] for info in buffer.info]
    comfort = [info['comfort'] for info in buffer.info]
    switches = [info['window_switch'] for info in buffer.info]
    return {
        'avg_energy': float(np.mean(energy)) if energy else 0.0,
        'avg_comfort_violation': float(np.mean(comfort)) if comfort else 0.0,
        'avg_window_switch': float(np.mean(switches)) if switches else 0.0,
    }
