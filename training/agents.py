"""统一 Agent 接口与具体实现。

设计目标：让 SelfPlayEnv / SelfPlayRecorder 只依赖 Agent 协议，
不直接耦合模型结构，从而同时支持 ResNet、Transformer 和 Oracle Teacher。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from engine.game import GameEngine
from engine.actions import LegalActions


class Agent(ABC):
    """自对弈/录制 agent 的抽象接口。"""

    @abstractmethod
    def select_action(
        self,
        engine: GameEngine,
        player_idx: int,
        deterministic: bool = False,
    ) -> Tuple[int, float, float]:
        """为 player_idx 在当前引擎状态下选择动作。

        Returns:
            (action_idx, log_prob, value)
            - action_idx: 0-76 动作索引（与 LegalActions.mask 对齐）
            - log_prob: 所选动作的对数概率
            - value: 状态价值估计
        """
        ...

    def on_game_start(self, engine: GameEngine) -> None:
        """每局开始时回调，子类可重写以初始化局级状态。"""

    def on_game_end(self, engine: GameEngine) -> None:
        """每局结束时回调，子类可重写以清理局级状态。"""


# ── ResNet Agent ─────────────────────────────────────────────────────────────


class ResNetAgent(Agent):
    """MahjongPolicyValueNet (354 维 ResNet) 的 Agent 包装。"""

    def __init__(self, model: 'MahjongPolicyValueNet', device: str = 'cpu'):
        self.model = model
        self.device = device

    @torch.no_grad()
    def select_action(
        self,
        engine: GameEngine,
        player_idx: int,
        deterministic: bool = False,
    ) -> Tuple[int, float, float]:
        state_np = engine.get_state_tensor(player_idx)
        legal = engine.get_legal_actions(player_idx)

        state_t = torch.from_numpy(state_np).float().to(self.device)
        mask_t = torch.tensor(legal.mask, dtype=torch.float32).to(self.device)

        self.model.eval()
        action_idx, log_prob = self.model.get_action(
            state_t, mask_t, deterministic=deterministic)
        _, value_t = self.model.forward(state_t, mask_t)

        return int(action_idx), log_prob.item(), value_t.item()


# ── Transformer Agent ────────────────────────────────────────────────────────


class TransformerAgent(Agent):
    """TransformerPolicyValueNet (tokenized) 的 Agent 包装。"""

    def __init__(self,
                 model: 'TransformerPolicyValueNet',
                 tokenizer: Optional['MahjongTokenizer'] = None,
                 device: str = 'cpu',
                 max_len: int = 256):
        self.model = model
        self.device = device
        self.max_len = max_len

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            from models.tokenizer import MahjongTokenizer
            self.tokenizer = MahjongTokenizer(max_sequence_length=max_len)

    @torch.no_grad()
    def select_action(
        self,
        engine: GameEngine,
        player_idx: int,
        deterministic: bool = False,
    ) -> Tuple[int, float, float]:
        seq = self.tokenizer.tokenize_engine_state(engine, player_idx)
        legal = engine.get_legal_actions(player_idx)
        batch = _build_student_batch(seq, legal, self.device)
        self.model.eval()
        outputs = self.model(**batch, mode='student')
        return _sample_action(outputs, deterministic)


# ── Shared helper functions ──────────────────────────────────────────────────


def _build_student_batch(seq, legal, device):
    """Build student-mode batch dict from tokenized sequence + legal actions."""
    token_ids = torch.tensor([seq.token_ids], dtype=torch.long, device=device)
    token_types = torch.tensor([seq.token_types], dtype=torch.long, device=device)
    behavior_ids = torch.tensor([seq.behavior_ids], dtype=torch.long, device=device)
    attention_mask = torch.zeros(1, len(seq.token_ids), dtype=torch.bool,
                                 device=device)
    action_mask = torch.tensor([legal.mask], dtype=torch.float32, device=device)
    return dict(token_ids=token_ids, token_types=token_types,
                behavior_ids=behavior_ids, attention_mask=attention_mask,
                action_mask=action_mask)


def _sample_action(outputs, deterministic):
    """Sample action from model outputs, return (action_idx, log_prob, value)."""
    logits = outputs['policy_logits'][0]
    value = outputs['value'][0].item()
    probs = torch.softmax(logits, dim=-1)
    if deterministic:
        action_idx = int(torch.argmax(probs).item())
    else:
        action_idx = int(torch.multinomial(probs, 1).item())
    log_prob = torch.log(probs[action_idx] + 1e-8).item()
    return action_idx, log_prob, value


# ── Oracle Teacher Agent ─────────────────────────────────────────────────────


class OracleTeacherAgent(Agent):
    """上帝视角 Teacher Agent：使用 public + private 信息选择动作。

    用于生成带 privileged 信息的自博弈轨迹数据（Stage 2）。

    Args:
        model: Teacher TransformerPolicyValueNet（可训练或冻结）
        tokenizer: MahjongTokenizer 实例
        device: 推理设备
        private_visibility: private tokens 可见比例 (0.0 ~ 1.0)
            1.0 = 完整上帝视角；0.0 = 等价 public-only
        max_len: public 序列最大长度
        max_private_len: private 序列最大长度
    """

    def __init__(self,
                 model: 'TransformerPolicyValueNet',
                 tokenizer: Optional['MahjongTokenizer'] = None,
                 device: str = 'cpu',
                 private_visibility: float = 1.0,
                 max_len: int = 256,
                 max_private_len: int = 256):
        self.model = model
        self.device = device
        self.private_visibility = private_visibility
        self.max_len = max_len
        self.max_private_len = max_private_len

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            from models.tokenizer import MahjongTokenizer
            self.tokenizer = MahjongTokenizer(max_sequence_length=max_len)

    @torch.no_grad()
    def select_action(
        self,
        engine: GameEngine,
        player_idx: int,
        deterministic: bool = False,
    ) -> Tuple[int, float, float]:
        # visibility=0.0 → public-only student forward
        if self.private_visibility <= 0.0:
            return self._select_public_only(engine, player_idx, deterministic)

        pub_seq, priv_seq = self.tokenizer \
            .tokenize_public_private_engine_state(engine, player_idx)
        legal = engine.get_legal_actions(player_idx)
        batch = self._to_batch(pub_seq, priv_seq, legal)
        self.model.eval()
        outputs = self.model(**batch, mode='teacher')

        logits = outputs['policy_logits'][0]
        value = outputs['value'][0].item()
        oracle_v = outputs.get('oracle_value')
        if oracle_v is not None:
            value = oracle_v[0].item()

        probs = torch.softmax(logits, dim=-1)
        if deterministic:
            action_idx = int(torch.argmax(probs).item())
        else:
            action_idx = int(torch.multinomial(probs, 1).item())
        log_prob = torch.log(probs[action_idx] + 1e-8).item()
        return action_idx, log_prob, value

    def _select_public_only(self, engine, player_idx, deterministic):
        """Public-only student forward（与 TransformerAgent 逻辑相同）。"""
        seq = self.tokenizer.tokenize_engine_state(engine, player_idx)
        legal = engine.get_legal_actions(player_idx)
        batch = _build_student_batch(seq, legal, self.device)
        self.model.eval()
        outputs = self.model(**batch, mode='student')
        return _sample_action(outputs, deterministic)

    def _to_batch(self, pub_seq, priv_seq, legal):
        pub_ids = torch.tensor([pub_seq.token_ids], dtype=torch.long,
                               device=self.device)
        pub_types = torch.tensor([pub_seq.token_types], dtype=torch.long,
                                 device=self.device)
        pub_bids = torch.tensor([pub_seq.behavior_ids], dtype=torch.long,
                                device=self.device)
        pub_attn = torch.zeros(1, len(pub_seq.token_ids), dtype=torch.bool,
                               device=self.device)
        action_mask = torch.tensor([legal.mask], dtype=torch.float32,
                                   device=self.device)

        priv_ids = torch.tensor([priv_seq.token_ids], dtype=torch.long,
                                device=self.device)
        priv_types = torch.tensor([priv_seq.token_types], dtype=torch.long,
                                  device=self.device)
        priv_bids = torch.tensor([priv_seq.behavior_ids], dtype=torch.long,
                                 device=self.device)
        priv_attn = torch.zeros(1, len(priv_seq.token_ids), dtype=torch.bool,
                                device=self.device)

        # Apply visibility mask: Bernoulli dropout on private tokens
        if self.private_visibility < 1.0 and priv_ids.size(1) > 0:
            keep_prob = self.private_visibility
            keep_mask = torch.rand(priv_ids.size(1), device=self.device) < keep_prob
            # Always keep at least 1 token if visibility > 0
            if keep_prob > 0 and not keep_mask.any():
                keep_mask[0] = True
            priv_ids = priv_ids[:, keep_mask]
            priv_types = priv_types[:, keep_mask]
            priv_bids = priv_bids[:, keep_mask]
            priv_attn = priv_attn[:, keep_mask]

        return dict(
            token_ids=pub_ids, token_types=pub_types,
            behavior_ids=pub_bids, attention_mask=pub_attn,
            action_mask=action_mask,
            private_token_ids=priv_ids, private_token_types=priv_types,
            private_behavior_ids=priv_bids, private_attention_mask=priv_attn,
        )


def build_agent(
    model_type: str,
    model: Any,
    device: str = 'cpu',
    tokenizer: Optional[Any] = None,
    private_visibility: float = 1.0,
    max_len: int = 256,
    checkpoint_metadata: Optional[Dict] = None,
) -> Agent:
    """工厂函数：根据模型类型构造对应 Agent。

    Args:
        model_type: 'resnet' 或 'transformer' 或 'oracle_teacher'
        model: 模型实例
        device: 推理设备
        tokenizer: MahjongTokenizer 实例（Transformer 类 agent 需要）
        private_visibility: private token 可见比例（仅 oracle_teacher）
        max_len: 序列最大长度
        checkpoint_metadata: checkpoint 元数据，用于推断 max_len 等
    """
    if checkpoint_metadata is not None:
        max_len = checkpoint_metadata.get('max_len', max_len)

    if model_type == 'resnet':
        return ResNetAgent(model, device=device)
    elif model_type == 'transformer':
        return TransformerAgent(model, tokenizer=tokenizer, device=device,
                                max_len=max_len)
    elif model_type == 'oracle_teacher':
        return OracleTeacherAgent(model, tokenizer=tokenizer, device=device,
                                  private_visibility=private_visibility,
                                  max_len=max_len)
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")
