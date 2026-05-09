"""Transformer Encoder Backbone — Pre-LN, 6 层, 8 头, GELU。

架构：
  Token Embedding (B, S, 256) → + Positional Embedding → 6× TransformerBlock → (B, S, 256)

每层 TransformerBlock：
  Pre-LayerNorm → Multi-Head Self-Attention → Residual
  Pre-LayerNorm → FFN (GELU, 1024) → Residual

支持：
  - Key Padding Mask（变长序列）
  - Learnable Positional Embedding（max_len=256）
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    """Pre-LN Transformer Encoder Block。

    Args:
        d_model: 隐藏层维度（默认 256）
        n_heads: 注意力头数（默认 8）
        d_ff: FFN 中间层维度（默认 1024）
        dropout: Dropout 概率（默认 0.1）
    """

    def __init__(self, d_model: int = 256, n_heads: int = 8,
                 d_ff: int = 1024, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None
                ) -> torch.Tensor:
        """Pre-LN 前向传播。

        Args:
            x: (B, S, d_model)
            key_padding_mask: (B, S) bool，True 表示 padding 位置

        Returns:
            (B, S, d_model)
        """
        # Self-Attention with Pre-LN
        residual = x
        x = self.norm1(x)
        x = self.attention(x, x, x, key_padding_mask=key_padding_mask,
                          need_weights=False)[0]
        x = self.dropout1(x)
        x = residual + x

        # FFN with Pre-LN
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x


class TransformerBackbone(nn.Module):
    """Transformer Encoder Backbone。

    Args:
        layers: Encoder 层数（默认 6）
        d_model: 隐藏层维度（默认 256）
        n_heads: 注意力头数（默认 8）
        d_ff: FFN 中间层维度（默认 1024）
        dropout: Dropout 概率（默认 0.1）
        max_len: 最大序列长度（默认 256）
    """

    def __init__(self, layers: int = 6, d_model: int = 256,
                 n_heads: int = 8, d_ff: int = 1024,
                 dropout: float = 0.1, max_len: int = 256):
        super().__init__()

        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_len, d_model) * 0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.max_len = max_len

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        """前向传播。

        Args:
            x: (B, S, d_model) — Token Embedding 后的序列
            mask: (B, S) bool，True 表示 padding 位置

        Returns:
            (B, S, d_model)
        """
        B, S, D = x.shape
        assert D == self.d_model, \
            f"Expected d_model={self.d_model}, got {D}"

        # 添加位置编码
        pos = self.pos_embedding[:, :S, :]
        x = x + pos
        x = self.dropout(x)

        # 多个 Encoder Block
        for block in self.blocks:
            x = block(x, key_padding_mask=mask)

        return x

    def count_parameters(self) -> int:
        """返回可训练参数总数。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
# 中文注释：Pre-LN Transformer Encoder 骨干网络，支持变长输入和注意力掩码。
