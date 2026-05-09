"""1D 残差网络骨干 —— 轻量级 Conv1d 残差块，沿 34 牌型轴提取面子/搭子特征。"""

import torch
import torch.nn as nn


class ResidualBlock1D(nn.Module):
    """标准 1D 残差块：Conv1d → BN → ReLU → Conv1d → BN + skip connection。

    kernel_size=3, padding=1 保持 34 牌型轴长度不变，
    多层堆叠后感受野可覆盖全部 34 牌型。
    """

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size,
                               padding=kernel_size // 2, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size,
                               padding=kernel_size // 2, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return self.relu(out)


class ResNet1DBackbone(nn.Module):
    """1D 残差骨干：输入投影 + N 个残差块 + 全局平均池化。

    Args:
        in_channels: 输入通道数（= StateFeatureEncoder.SPATIAL_CHANNELS，10）
        base_channels: 残差块通道数（默认 128）
        num_blocks: 残差块数量（默认 6）
    """

    def __init__(self, in_channels: int = 10, base_channels: int = 128,
                 num_blocks: int = 6):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
        )

        self.res_blocks = nn.Sequential(*[
            ResidualBlock1D(base_channels) for _ in range(num_blocks)
        ])

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.output_dim = base_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: (B, C_in, 34) 空间特征

        Returns:
            (B, base_channels) 池化后的特征向量
        """
        x = self.input_proj(x)
        x = self.res_blocks(x)
        x = self.global_pool(x).squeeze(-1)
        return x
# 中文注释：一维残差网络骨干，用于在 34 种牌型维度上提取局面特征。
