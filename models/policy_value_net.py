"""日麻策略-价值双头网络。

完整架构：
  354 维 state → FeatureEncoder 拆分
    → 空间 (B,10,34) → ResNet1D → (B,128)
    → 元数据 (B,14) → MLP → (B,128)
    → concat (B,256) → 策略头 (B,77) + 价值头 (B,1)
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .feature_encoder import StateFeatureEncoder
from .resnet1d import ResNet1DBackbone


class MetadataMLP(nn.Module):
    """元数据 MLP：将 14 维标量元数据映射到与 CNN 输出相同维度。"""

    def __init__(self, input_dim: int = 14, output_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MahjongPolicyValueNet(nn.Module):
    """日麻 AI 完整模型：1D-CNN 残差骨干 + 策略头 + 价值头。

    Args:
        spatial_channels: 空间输入通道数 (10)
        base_channels: 残差骨干通道数 (128)
        num_res_blocks: 残差块数量 (6)
        metadata_dim: 元数据维度 (14)
        action_dim: 动作空间大小 (77)
    """

    def __init__(self,
                 spatial_channels: int = 10,
                 base_channels: int = 128,
                 num_res_blocks: int = 6,
                 metadata_dim: int = 14,
                 action_dim: int = 77):
        super().__init__()

        # 骨干网络
        self.backbone = ResNet1DBackbone(
            in_channels=spatial_channels,
            base_channels=base_channels,
            num_blocks=num_res_blocks,
        )

        # 元数据编码
        self.metadata_mlp = MetadataMLP(
            input_dim=metadata_dim,
            output_dim=base_channels,
        )

        # 融合维度
        fusion_dim = base_channels * 2  # 128 + 128 = 256

        # 双头
        self.policy_head = nn.Linear(fusion_dim, action_dim)
        # Value head 使用 Tanh 将输出限制在 [-1, 1]。
        # 因此 reward 需按分数/10000 归一化，且终端排名奖励控制在 ±1 附近，
        # 否则 value 网络无法拟合真实 return，GAE advantage 会被污染。
        self.value_head = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

        # 保存配置
        self.spatial_channels = spatial_channels
        self.base_channels = base_channels
        self.num_res_blocks = num_res_blocks
        self.metadata_dim = metadata_dim
        self.action_dim = action_dim

    def forward(self, state_tensor: torch.Tensor,
                action_mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播。

        Args:
            state_tensor: (B, 354) 或 (354,) 的游戏状态特征
            action_mask: (B, 77) 合法动作掩码（1=合法，0=非法）。
                         传入后 policy_logits 中非法动作设为当前 dtype 的最小值。

        Returns:
            policy_logits: (B, 77) 未归一化策略对数概率
            value: (B, 1) 状态价值，范围 [-1, 1]
        """
        # 确保 batch 维度
        single_input = state_tensor.dim() == 1
        if single_input:
            state_tensor = state_tensor.unsqueeze(0)
            if action_mask is not None:
                action_mask = action_mask.unsqueeze(0)

        # 特征拆分
        spatial, metadata = StateFeatureEncoder.encode(state_tensor)

        # 双路前向
        cnn_features = self.backbone(spatial)         # (B, C)
        meta_features = self.metadata_mlp(metadata)   # (B, C)

        # 融合
        fused = torch.cat([cnn_features, meta_features], dim=-1)  # (B, 2C)

        # 双头输出
        policy_logits = self.policy_head(fused)       # (B, 77)
        value = self.value_head(fused)                # (B, 1)

        # 动作掩码
        if action_mask is not None:
            mask_value = torch.finfo(policy_logits.dtype).min
            policy_logits = policy_logits.masked_fill(
                action_mask == 0, mask_value)

        if single_input:
            policy_logits = policy_logits.squeeze(0)
            value = value.squeeze(0)

        return policy_logits, value

    def get_action(self, state_tensor: torch.Tensor,
                   action_mask: torch.Tensor,
                   deterministic: bool = False) -> Tuple[int, torch.Tensor]:
        """根据策略采样一个动作。

        Args:
            state_tensor: (354,) 游戏状态
            action_mask: (77,) 合法动作掩码
            deterministic: True 时选最大概率动作（推理），False 时按分布采样

        Returns:
            action_idx: 选中的动作索引 (0-76)
            log_prob: 该动作的对数概率
        """
        logits, _ = self.forward(state_tensor, action_mask)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        if deterministic:
            action_idx = probs.argmax().item()
        else:
            action_idx = torch.multinomial(probs, 1).item()

        log_prob = log_probs[action_idx]
        return action_idx, log_prob

    def evaluate_actions(self, state_tensor: torch.Tensor,
                         action_mask: torch.Tensor,
                         action_indices: torch.Tensor
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """批量评估动作：返回对数概率、价值和熵（PPO 更新用）。

        Args:
            state_tensor: (B, 354)
            action_mask: (B, 77)
            action_indices: (B,) 选中的动作索引

        Returns:
            log_probs: (B,) 所选动作的对数概率
            values: (B, 1) 状态价值
            entropy: (B,) 策略熵
        """
        logits, values = self.forward(state_tensor, action_mask)
        log_probs_all = F.log_softmax(logits, dim=-1)
        probs = F.softmax(logits, dim=-1)

        log_probs = log_probs_all.gather(1, action_indices.unsqueeze(-1)).squeeze(-1)
        entropy = -(probs * log_probs_all).sum(dim=-1)

        return log_probs, values, entropy

    def count_parameters(self) -> int:
        """返回可训练参数总数。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
# 中文注释：监督学习和强化学习共用的策略-价值网络，输出动作 logits 与局面价值。
