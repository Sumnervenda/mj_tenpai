"""354 维平铺特征 → (10, 34) 空间通道 + (14,) 元数据的零开销拆分。

设计原则：
  - 直接 view reshape，无学习参数，保证推理零延迟。
  - 空间轴 = 34 种牌型，供 1D 卷积沿该轴捕捉相邻牌（搭子/顺子）关联。
  - 元数据（分数、风位等）不参与卷积，池化后与 CNN 特征 concat 融合。
"""

from typing import Tuple

import torch


class StateFeatureEncoder:
    """将 GameEngine.get_state_tensor() 输出的 354 维向量拆分为双路输入。"""

    SPATIAL_CHANNELS = 10
    TILE_TYPES = 34
    METADATA_DIM = 14

    @staticmethod
    def encode(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """拆分平铺特征。

        Args:
            tensor: (..., 354) 或 (354,) 的 float32 张量

        Returns:
            spatial: (..., 10, 34) —— 10 通道 × 34 牌型的空间特征
            metadata: (..., 14) —— 分数/风位等标量元数据
        """
        spatial = tensor[..., :340].reshape(*tensor.shape[:-1], 10, 34)
        metadata = tensor[..., 340:354]
        return spatial, metadata

    @staticmethod
    def encode_batch(states: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        """反向操作：将空间+元数据拼回 354 维平铺向量（用于数据增强等场景）。"""
        flat_spatial = states.reshape(*states.shape[:-2], 340)
        return torch.cat([flat_spatial, metadata], dim=-1)
