"""TransformerPolicyValueNet — 完整 Transformer + MTL 网络。

集成：
  1. Token/Type/Behavior Embedding Tables
  2. Learnable Concept Tokens（Semantic Bottleneck）
  3. Transformer Encoder Backbone
  4. Multi-Task Prediction Heads

输入:
  - token_ids:    (B, S) — 主 Token ID
  - token_types:  (B, S) — Token 类别 ID
  - behavior_ids: (B, S) — 行为属性 ID
  - attention_mask: (B, S) — bool, True=padding
  - action_mask:  (B, 77) — 合法动作掩码

输出（Student 模式）:
  - policy_logits: (B, 77)
  - value: (B, 1)
  - shanten: (B, 7)
  - efficiency: (B, 3)
  - danger: (B, 34)
  - score_value: (B, 1)

输出（Teacher 模式，额外）:
  - oracle_value: (B, 1) — 上帝视角价值
  - teacher_policy_logits: (B, 77) — 教师策略（用于蒸馏）
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenizer import TokenVocab, TokenType
from .transformer_backbone import TransformerBackbone
from .multi_task_heads import MultiTaskHeads


class TransformerPolicyValueNet(nn.Module):
    """Transformer 策略-价值网络（替代 MahjongPolicyValueNet 的 ResNet 版本）。

    Args:
        vocab_size: Token 词表大小（默认 128）
        num_token_types: Token 类型数量（默认 6）
        num_behavior_types: 行为类型数量（默认 64）
        d_model: 隐藏层维度（默认 256）
        n_concept: Concept Token 数量（默认 10）
        n_layers: Transformer 层数（默认 6）
        n_heads: 注意力头数（默认 8）
        d_ff: FFN 中间层维度（默认 1024）
        dropout: Dropout 概率（默认 0.1）
        max_len: 最大序列长度（默认 256）
        action_dim: 动作空间维度（默认 77）
    """

    def __init__(self,
                 vocab_size: int = TokenVocab.VOCAB_SIZE,
                 num_token_types: int = TokenType.NUM_TYPES,
                 num_behavior_types: int = TokenVocab.MAX_BEHAVIOR_ID,
                 d_model: int = 256,
                 n_concept: int = 10,
                 n_private_concept: int = 4,
                 n_layers: int = 6,
                 n_heads: int = 8,
                 d_ff: int = 1024,
                 dropout: float = 0.1,
                 max_len: int = 256,
                 action_dim: int = 77):
        super().__init__()

        if n_concept < 10:
            raise ValueError(
                f"n_concept must be >= 10 for MTL heads, got {n_concept}")

        # Embedding Tables
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.type_embedding = nn.Embedding(num_token_types, d_model)
        self.behavior_embedding = nn.Embedding(num_behavior_types, d_model)

        # LayerNorm for embedding outputs
        self.embed_norm = nn.LayerNorm(d_model)
        self.embed_dropout = nn.Dropout(dropout)

        # Concept Tokens (Semantic Bottleneck) — public + private
        self.concept_tokens = nn.Parameter(
            torch.randn(n_concept, d_model) * 0.02)
        self.private_concept_tokens = nn.Parameter(
            torch.randn(n_private_concept, d_model) * 0.02)

        # Transformer Backbone
        self.backbone = TransformerBackbone(
            layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            max_len=max_len,
        )

        # Multi-Task Heads
        self.heads = MultiTaskHeads(d_model, action_dim)

        # Save config
        self.d_model = d_model
        self.n_concept = n_concept
        self.n_private_concept = n_private_concept
        self.action_dim = action_dim

    def forward(self,
                token_ids: torch.Tensor,
                token_types: torch.Tensor,
                behavior_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                action_mask: Optional[torch.Tensor] = None,
                private_token_ids: Optional[torch.Tensor] = None,
                private_token_types: Optional[torch.Tensor] = None,
                private_behavior_ids: Optional[torch.Tensor] = None,
                private_attention_mask: Optional[torch.Tensor] = None,
                mode: str = "student",
                ) -> Dict[str, torch.Tensor]:
        """前向传播。

        Args:
            token_ids: (B, S) public Token ID
            token_types: (B, S) public Token Type ID
            behavior_ids: (B, S) public Behavior ID, 可选
            attention_mask: (B, S) bool, True=padding
            action_mask: (B, 77) 合法动作掩码
            private_token_ids: (B, S_priv) private Token ID (teacher mode)
            private_token_types: (B, S_priv) private Token Type ID
            private_behavior_ids: (B, S_priv) private Behavior ID
            private_attention_mask: (B, S_priv) bool
            mode: "student" | "teacher"

        Returns:
            dict: policy_logits, value, shanten, efficiency, danger, score_value,
                  and in teacher mode: oracle_value
        """
        B, S = token_ids.shape

        # ── Public Token Embedding ─────────────────────────────────────
        x = self.token_embedding(token_ids)
        x = x + self.type_embedding(token_types)

        if behavior_ids is not None:
            x = x + self.behavior_embedding(behavior_ids)

        x = self.embed_norm(x)
        x = self.embed_dropout(x)

        # ── Prepend Public Concept Tokens ──────────────────────────────
        concepts = self.concept_tokens.unsqueeze(0).expand(B, -1, -1)
        full_seq = torch.cat([x, concepts], dim=1)

        # ── Teacher: embed private tokens + private concept ────────────
        private_concept_out = None
        has_private = (mode == "teacher" and private_token_ids is not None
                       and private_token_ids.shape[1] > 0)
        if has_private:
            x_priv = self.token_embedding(private_token_ids)
            x_priv = x_priv + self.type_embedding(private_token_types)
            if private_behavior_ids is not None:
                x_priv = x_priv + self.behavior_embedding(private_behavior_ids)
            x_priv = self.embed_norm(x_priv)
            x_priv = self.embed_dropout(x_priv)

            priv_concepts = self.private_concept_tokens.unsqueeze(0).expand(
                B, -1, -1)
            full_seq = torch.cat([x, x_priv, priv_concepts, concepts], dim=1)

        # ── Build combined padding mask ────────────────────────────────
        full_mask = None
        if attention_mask is not None:
            concept_mask = torch.zeros(
                B, self.n_concept, device=attention_mask.device,
                dtype=attention_mask.dtype)
            if has_private:
                priv_concept_mask = torch.zeros(
                    B, self.n_private_concept, device=attention_mask.device,
                    dtype=attention_mask.dtype)
                priv_mask = (private_attention_mask
                             if private_attention_mask is not None
                             else torch.zeros(
                                 B, private_token_ids.shape[1],
                                 device=attention_mask.device,
                                 dtype=attention_mask.dtype))
                full_mask = torch.cat(
                    [attention_mask, priv_mask, priv_concept_mask, concept_mask],
                    dim=1)
            else:
                full_mask = torch.cat([attention_mask, concept_mask], dim=1)

        # ── Transformer Forward ────────────────────────────────────────
        total_len = full_seq.size(1)
        if total_len > self.backbone.max_len:
            raise RuntimeError(
                f"Sequence length {total_len} exceeds backbone max_len "
                f"{self.backbone.max_len}. In teacher mode the total length is "
                f"public({token_ids.size(1)}) + private({private_token_ids.size(1) if has_private else 0})"
                f" + public_concepts({self.n_concept})"
                f" + private_concepts({self.n_private_concept if has_private else 0})."
                f" Use --max_len {total_len} or higher.")
        hidden = self.backbone(full_seq, mask=full_mask)

        # ── Extract Concept Outputs ────────────────────────────────────
        concept_outputs = hidden[:, -self.n_concept:, :]

        if has_private:
            # Private concepts are between public concepts and the end
            priv_start = -(self.n_concept + self.n_private_concept)
            priv_end = -self.n_concept
            private_concept_out = hidden[:, priv_start:priv_end, :]

        # ── Multi-Task Heads ───────────────────────────────────────────
        return self.heads(concept_outputs, action_mask, private_concept_out)

    def get_action(self,
                   token_ids: torch.Tensor,
                   token_types: torch.Tensor,
                   behavior_ids: Optional[torch.Tensor] = None,
                   attention_mask: Optional[torch.Tensor] = None,
                   action_mask: Optional[torch.Tensor] = None,
                   deterministic: bool = False,
                   ) -> Tuple[int, torch.Tensor]:
        """根据策略采样一个动作。

        Args:
            same as forward, plus:
            deterministic: True=argmax, False=multinomial sample

        Returns:
            action_idx: 选中的动作索引 (0-76)
            log_prob: 该动作的对数概率
        """
        # single sample → add batch dim
        single = token_ids.dim() == 1
        if single:
            token_ids = token_ids.unsqueeze(0)
            token_types = token_types.unsqueeze(0)
            if behavior_ids is not None:
                behavior_ids = behavior_ids.unsqueeze(0)
            if attention_mask is not None:
                attention_mask = attention_mask.unsqueeze(0)
            if action_mask is not None:
                action_mask = action_mask.unsqueeze(0)

        outputs = self.forward(
            token_ids, token_types, behavior_ids,
            attention_mask, action_mask)
        logits = outputs['policy_logits']  # (1, 77)
        probs = F.softmax(logits, dim=-1)
        log_probs_all = F.log_softmax(logits, dim=-1)

        if deterministic:
            action_idx = probs.argmax(dim=-1, keepdim=True)
        else:
            action_idx = torch.multinomial(probs, 1)

        log_prob = log_probs_all.gather(1, action_idx)

        if single:
            action_idx = action_idx.squeeze(0)
            log_prob = log_prob.squeeze(0)

        return action_idx.item(), log_prob

    def evaluate_actions(self,
                         token_ids: torch.Tensor,
                         token_types: torch.Tensor,
                         behavior_ids: torch.Tensor,
                         attention_mask: torch.Tensor,
                         action_mask: torch.Tensor,
                         action_indices: torch.Tensor,
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """批量评估动作（PPO 更新用）。

        Returns:
            log_probs: (B,) 所选动作的对数概率
            values: (B, 1) 状态价值
            entropy: (B,) 策略熵
        """
        outputs = self.forward(
            token_ids, token_types, behavior_ids,
            attention_mask, action_mask)
        logits = outputs['policy_logits']
        values = outputs['value']

        log_probs_all = F.log_softmax(logits, dim=-1)
        probs = F.softmax(logits, dim=-1)

        log_probs = log_probs_all.gather(
            1, action_indices.unsqueeze(-1)).squeeze(-1)
        entropy = -(probs * log_probs_all).sum(dim=-1)

        return log_probs, values, entropy

    def compute_diversity_loss(self) -> torch.Tensor:
        """计算 Concept Token 的多样性损失（防 Collapse）。

        鼓励 public + private concept tokens 各自内部保持区分度。
        """
        def _off_diag_loss(tokens):
            n = tokens.shape[0]
            if n < 2:
                return torch.tensor(0.0, device=tokens.device)
            normalized = F.normalize(tokens, dim=-1)
            sim = normalized @ normalized.T
            mask = ~torch.eye(n, dtype=torch.bool, device=sim.device)
            return sim[mask].pow(2).mean()

        loss = _off_diag_loss(self.concept_tokens)
        loss = loss + _off_diag_loss(self.private_concept_tokens)
        return loss

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
# 中文注释：完整的 Transformer + MTL 网络，组合 Token 嵌入、Concept Token、Backbone 和预测头。
