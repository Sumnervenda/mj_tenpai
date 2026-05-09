"""PPO Agent —— GAE + Clipped Objective 的日麻强化学习实现。

架构:
  RolloutBuffer: 存储自对弈轨迹
  compute_gae:  计算 Generalized Advantage Estimation
  PPOAgent.update: 执行 PPO clipped objective 更新（支持 AMP 混合精度）
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.amp import GradScaler, autocast

from models import MahjongPolicyValueNet
from .selfplay_env import RolloutStep, GameTrajectory


@dataclass
class RolloutBuffer:
    """PPO 轨迹缓冲区。

    将自对弈轨迹展平为 (state, action, reward, log_prob, value, mask, done) 序列。
    """
    states: List[np.ndarray] = field(default_factory=list)
    masks: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    player_indices: List[int] = field(default_factory=list)
    game_ids: List[int] = field(default_factory=list)
    sl_log_probs: List[float] = field(default_factory=list)

    def add_step(self, step: RolloutStep) -> None:
        self.states.append(step.state)
        self.masks.append(step.mask)
        self.actions.append(step.action)
        self.log_probs.append(step.log_prob)
        self.rewards.append(step.reward)
        self.values.append(step.value)
        self.dones.append(step.done)
        self.player_indices.append(step.player_idx)
        self.game_ids.append(step.game_id)
        self.sl_log_probs.append(step.sl_log_prob)

    def extend_trajectory(self, traj: GameTrajectory) -> None:
        for step in traj.steps:
            self.add_step(step)

    def clear(self) -> None:
        self.states.clear()
        self.masks.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
        self.player_indices.clear()
        self.game_ids.clear()
        self.sl_log_probs.clear()

    def __len__(self) -> int:
        return len(self.states)

    def to_tensors(self, device: str = 'cpu'
                   ) -> Tuple[torch.Tensor, ...]:
        """将所有数据转换为张量。"""
        return (
            torch.tensor(np.stack(self.states), dtype=torch.float32).to(device),
            torch.tensor(np.stack(self.masks), dtype=torch.float32).to(device),
            torch.tensor(self.actions, dtype=torch.long).to(device),
            torch.tensor(self.log_probs, dtype=torch.float32).to(device),
            torch.tensor(self.rewards, dtype=torch.float32).to(device),
            torch.tensor(self.values, dtype=torch.float32).to(device),
            torch.tensor(self.dones, dtype=torch.float32).to(device),
            torch.tensor(self.sl_log_probs, dtype=torch.float32).to(device),
        )


def compute_gae(rewards: torch.Tensor,
                values: torch.Tensor,
                dones: torch.Tensor,
                gamma: float = 0.99,
                gae_lambda: float = 0.95
                ) -> Tuple[torch.Tensor, torch.Tensor]:
    """计算 Generalized Advantage Estimation。

    Args:
        rewards: (T,) 即时奖励序列
        values: (T,) 状态价值预测序列
        dones: (T,) 终局标记序列 (1=terminal)
        gamma: 折扣因子
        gae_lambda: GAE 平滑参数

    Returns:
        returns: (T,) 折扣回报
        advantages: (T,) GAE 优势
    """
    T = len(rewards)
    advantages = torch.zeros(T, device=rewards.device)
    returns = torch.zeros(T, device=rewards.device)

    gae = 0.0
    next_value = 0.0

    for t in reversed(range(T)):
        if t == T - 1:
            delta = rewards[t] - values[t]
        else:
            mask = 1.0 - dones[t]
            delta = (rewards[t] + gamma * values[t + 1] * mask
                      - values[t])

        gae = delta + gamma * gae_lambda * (1.0 - dones[t]) * gae
        advantages[t] = gae
        returns[t] = advantages[t] + values[t]

    return returns, advantages


def compute_gae_grouped(rewards: torch.Tensor,
                        values: torch.Tensor,
                        dones: torch.Tensor,
                        game_ids: list,
                        player_indices: list,
                        gamma: float = 0.99,
                        gae_lambda: float = 0.95
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
    """按 (game_id, player_idx) 分组计算 GAE，避免跨玩家 value 混合。

    日麻自对弈中四家交错行动（P0, P1, P2, P3, P0...），
    不同玩家的 value 不应互相关联。此函数为每个玩家独立计算 GAE。

    Args:
        rewards: (T,) 即时奖励
        values: (T,) 状态价值
        dones: (T,) 终局标记
        game_ids: [int] 每个 step 所属 game_id
        player_indices: [int] 每个 step 所属 player_idx
        gamma: 折扣因子
        gae_lambda: GAE λ

    Returns:
        returns: (T,) 折扣回报
        advantages: (T,) GAE 优势
    """
    returns = torch.zeros_like(rewards)
    advantages = torch.zeros_like(rewards)

    groups: dict = {}
    for i, (gid, pid) in enumerate(zip(game_ids, player_indices)):
        key = (gid, pid)
        if key not in groups:
            groups[key] = []
        groups[key].append(i)

    for indices in groups.values():
        idx_tensor = torch.tensor(indices, device=rewards.device)
        sub_ret, sub_adv = compute_gae(
            rewards[idx_tensor], values[idx_tensor], dones[idx_tensor],
            gamma=gamma, gae_lambda=gae_lambda,
        )
        returns[idx_tensor] = sub_ret
        advantages[idx_tensor] = sub_adv

    return returns, advantages


class PPOAgent:
    """PPO 训练 Agent。

    Args:
        model: 策略-价值双头网络
        device: 训练设备
        lr: 学习率
        clip_epsilon: PPO 裁剪范围
        gamma: 折扣因子
        gae_lambda: GAE λ 参数
        entropy_coef: 熵正则化系数
        value_loss_coef: 价值损失系数
        max_grad_norm: 梯度裁剪阈值
        use_amp: 是否使用自动混合精度
    """

    def __init__(self,
                 model: MahjongPolicyValueNet,
                 device: str = 'cuda',
                 lr: float = 3e-4,
                 clip_epsilon: float = 0.2,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 entropy_coef: float = 0.01,
                 value_loss_coef: float = 0.5,
                 max_grad_norm: float = 1.0,
                 use_amp: bool = True,
                 kl_coef: float = 0.01):
        self.model = model.to(device)
        self.device = device
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp and device == 'cuda'
        self.kl_coef = kl_coef

        self.optimizer = AdamW(model.parameters(), lr=lr)
        self.scaler = GradScaler('cuda', enabled=self.use_amp)
        self.buffer = RolloutBuffer()

    def collect_trajectories(self, trajectories: List[GameTrajectory]) -> None:
        """将轨迹收集到缓冲区。"""
        for traj in trajectories:
            self.buffer.extend_trajectory(traj)

    def clear_buffer(self) -> None:
        self.buffer.clear()

    def update(self, ppo_epochs: int = 10,
               mini_batch_size: int = 512) -> dict:
        """执行 PPO 更新（支持 AMP 混合精度）。

        Args:
            ppo_epochs: PPO 更新轮数
            mini_batch_size: 小批量大小

        Returns:
            训练指标字典
        """
        if len(self.buffer) == 0:
            return {'policy_loss': 0.0, 'value_loss': 0.0,
                    'entropy': 0.0, 'total_loss': 0.0}

        self.model.train()  # PPO 更新需要 BatchNorm 在 train 模式

        states, masks, actions, old_log_probs, rewards, values, dones, sl_log_probs = \
            self.buffer.to_tensors(self.device)

        # GAE — 按 (game_id, player_idx) 分组，避免跨玩家 value 混合
        returns, advantages = compute_gae_grouped(
            rewards, values, dones,
            self.buffer.game_ids, self.buffer.player_indices,
            gamma=self.gamma, gae_lambda=self.gae_lambda,
        )

        # 优势标准化
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_steps = len(states)
        metrics = {'policy_loss': 0.0, 'value_loss': 0.0,
                    'entropy': 0.0, 'total_loss': 0.0}
        num_updates = 0

        for _ in range(ppo_epochs):
            # 随机打乱
            indices = torch.randperm(total_steps, device=self.device)
            for start in range(0, total_steps, mini_batch_size):
                batch_idx = indices[start:start + mini_batch_size]

                s_batch = states[batch_idx]
                m_batch = masks[batch_idx]
                a_batch = actions[batch_idx]
                old_lp_batch = old_log_probs[batch_idx]
                sl_lp_batch = sl_log_probs[batch_idx]
                adv_batch = advantages[batch_idx]
                ret_batch = returns[batch_idx]

                self.optimizer.zero_grad(set_to_none=True)

                with autocast('cuda', enabled=self.use_amp):
                    new_log_probs, new_values, entropy = \
                        self.model.evaluate_actions(s_batch, m_batch, a_batch)

                    # Policy loss (clipped)
                    ratio = torch.exp(new_log_probs - old_lp_batch)
                    surr1 = ratio * adv_batch
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon,
                                        1.0 + self.clip_epsilon) * adv_batch
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # Value loss
                    value_loss = F.mse_loss(new_values.squeeze(-1), ret_batch)

                    # Entropy bonus
                    entropy_mean = entropy.mean()

                    # KL 散度正则化：防止策略偏离 SL 预训练太远
                    kl_loss = (new_log_probs - sl_lp_batch).mean()

                    # Total loss
                    total_loss = (policy_loss +
                                  self.value_loss_coef * value_loss -
                                  self.entropy_coef * entropy_mean +
                                  self.kl_coef * kl_loss)

                if self.use_amp:
                    self.scaler.scale(total_loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(),
                                              self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(),
                                              self.max_grad_norm)
                    self.optimizer.step()

                metrics['policy_loss'] += policy_loss.item()
                metrics['value_loss'] += value_loss.item()
                metrics['entropy'] += entropy_mean.item()
                metrics['total_loss'] += total_loss.item()
                num_updates += 1

        for k in metrics:
            metrics[k] /= max(num_updates, 1)

        return metrics

    def save_checkpoint(self, path: str, epoch: int,
                         metadata: Optional[dict] = None) -> None:
        """保存 checkpoint。"""
        from models import save_checkpoint
        save_checkpoint(self.model, self.optimizer, epoch, path,
                        metadata=metadata)

    def load_checkpoint(self, path: str) -> Tuple[int, dict]:
        """加载 checkpoint。"""
        from models import load_checkpoint
        return load_checkpoint(self.model, path,
                               optimizer=self.optimizer,
                               device=self.device)
# 中文注释：PPO 强化学习智能体，负责采样动作、缓存轨迹并执行策略更新。
