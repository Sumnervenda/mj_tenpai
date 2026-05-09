"""Multi-Task Heads — 从 Concept Embedding 输出各任务预测。

每个 Head 从 Concept Token 的子集中读取信息：
  - Shanten Head:  Concept[0:2]  → 7 类向听数 (CE)
  - Efficiency Head: Concept[2:4] → 3 标量牌效指标 (MSE)
  - Danger Head:     Concept[4:6] → 34 种牌危险度 (MSE)
  - Score Head:      Concept[6:7] → 预期打点 (MSE)
  - Policy Head:     Concept[7:8] → 77 维动作 logits (CE, masked)
  - Value Head:      Concept[8:10] → [-1, 1] 局面价值 (MSE)
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ShantenHead(nn.Module):
    """向听数预测头。输出 7 类（0-6 向听）。"""

    def __init__(self, d_model: int = 256, num_classes: int = 7):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, concept: torch.Tensor) -> torch.Tensor:
        """concept: (B, 2, d_model) → (B, 7)"""
        x = concept.reshape(concept.shape[0], -1)
        return self.fc(x)


class EfficiencyHead(nn.Module):
    """牌效预测头。输出 3 标量：有效进张数、好型概率、改良率。"""

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 3),
        )

    def forward(self, concept: torch.Tensor) -> torch.Tensor:
        """concept: (B, 2, d_model) → (B, 3)"""
        x = concept.reshape(concept.shape[0], -1)
        return self.fc(x)


class DangerHead(nn.Module):
    """危险度预测头。输出 34 种牌的放铳危险度。"""

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 34),
        )

    def forward(self, concept: torch.Tensor) -> torch.Tensor:
        """concept: (B, 2, d_model) → (B, 34)"""
        x = concept.reshape(concept.shape[0], -1)
        return self.fc(x)


class ScoreHead(nn.Module):
    """预期打点头。输出 1 标量。"""

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 1),
        )

    def forward(self, concept: torch.Tensor) -> torch.Tensor:
        """concept: (B, d_model) → (B, 1)"""
        return self.fc(concept.squeeze(1))


class PolicyHead(nn.Module):
    """策略头。输出 77 维动作 logits（需配合 Action Mask 屏蔽非法动作）。"""

    def __init__(self, d_model: int = 256, action_dim: int = 77):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, action_dim),
        )

    def forward(self, concept: torch.Tensor,
                action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """concept: (B, d_model) → (B, 77)

        Args:
            concept: (B, d_model) 单个 Concept Token 的 embedding
            action_mask: (B, 77) 1=合法, 0=非法。None 时不屏蔽。

        Returns:
            (B, 77) logits（非法动作已被设为 -inf 或保持原样）
        """
        logits = self.fc(concept.squeeze(1))

        if action_mask is not None:
            mask_value = torch.finfo(logits.dtype).min
            logits = logits.masked_fill(action_mask == 0, mask_value)

        return logits


class ValueHead(nn.Module):
    """价值头。Tanh 输出 [-1, 1] 的局面价值估计。"""

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, concept: torch.Tensor) -> torch.Tensor:
        """concept: (B, 2, d_model) → (B, 1)"""
        x = concept.reshape(concept.shape[0], -1)
        return self.fc(x)


class MultiTaskHeads(nn.Module):
    """所有 MTL Heads 的容器。

    Concept Token 分配：
      - Shanten:  [0, 1]  → 2 tokens
      - Efficiency: [2, 3] → 2 tokens
      - Danger:   [4, 5]  → 2 tokens
      - Score:    [6]     → 1 token
      - Policy:   [7]     → 1 token
      - Value:    [8, 9]  → 2 tokens
    """

    def __init__(self, d_model: int = 256, action_dim: int = 77):
        super().__init__()
        self.shanten = ShantenHead(d_model)
        self.efficiency = EfficiencyHead(d_model)
        self.danger = DangerHead(d_model)
        self.score = ScoreHead(d_model)
        self.policy = PolicyHead(d_model, action_dim)
        self.value = ValueHead(d_model)

    def forward(self, concept_outputs: torch.Tensor,
                action_mask: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:
        """所有 Head 的前向传播。

        Args:
            concept_outputs: (B, 10, d_model) — 10 个 Concept Token 的输出
            action_mask: (B, 77) 或 None

        Returns:
            dict with keys: policy_logits, value, shanten, efficiency, danger, score_value
        """
        return {
            'shanten': self.shanten(concept_outputs[:, 0:2, :]),
            'efficiency': self.efficiency(concept_outputs[:, 2:4, :]),
            'danger': self.danger(concept_outputs[:, 4:6, :]),
            'score_value': self.score(concept_outputs[:, 6:7, :]),
            'policy_logits': self.policy(concept_outputs[:, 7:8, :], action_mask),
            'value': self.value(concept_outputs[:, 8:10, :]),
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
# 中文注释：多任务学习预测头集合，从 Concept Token Embedding 输出向听/牌效/危险度/打点/策略/价值。
