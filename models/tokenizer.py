"""麻将局面 Tokenizer — 将 MJSON 牌谱/引擎状态转换为 Token 序列。

每个 Token 包含三个字段：
  - token_id: 主内容 ID（用于词嵌入查找）
  - token_type: 类别（HAND/DORA/DISCARD/MELD/RIICHI/GLOBAL）
  - behavior_id: 行为属性（弃牌摸切/手切标记、副露类型等）

序列长度 ~120 tokens，随巡目增长。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from data.mjson_parser import MJSONGameTracker, mjson_str_to_type
from engine.tile import NUM_TYPES, abs_to_type, is_aka


# ── 词表常量 ─────────────────────────────────────────────────────────────────

class TokenVocab:
    """Token ID 词汇表。"""
    # 牌型 ID（0-33，直接使用 engine.tile 的定义）
    TILE_MIN = 0
    TILE_MAX = 33

    # 赤宝牌标记（34-36）
    RED_5M = 34
    RED_5P = 35
    RED_5S = 36

    # 玩家标记（37-40）
    P0 = 37
    P1 = 38
    P2 = 39
    P3 = 40

    # 动作类型（41-48）
    ACTION_TSUMOGIRI = 41      # 摸切
    ACTION_TEDASHI = 42        # 手切
    ACTION_RIICHI_DISCARD = 43 # 立直宣言切牌
    ACTION_CHI = 44            # 吃
    ACTION_PON = 45            # 碰
    ACTION_KAN = 46            # 明槓
    ACTION_RIICHI = 47         # 立直事件
    ACTION_KAKAN = 48          # 加槓

    # 副露类型（49-53）
    MELD_CHI = 49
    MELD_PON = 50
    MELD_KAN = 51
    MELD_KAKAN = 52
    MELD_ANKAN = 53

    # 全局标记（54+）
    ROUND_EAST = 54
    ROUND_SOUTH = 55
    ROUND_WEST = 56
    ROUND_NORTH = 57
    SELF_EAST = 58
    SELF_SOUTH = 59
    SELF_WEST = 60
    SELF_NORTH = 61
    REMAINING_BASE = 62   # 剩余牌数标记从 62 开始（62 + remaining）
    HONBA = 92            # 本场数
    RIICHI_STICK = 93     # 立直棒标记
    DIFF_BASE = 94        # 分差标记从 94 开始（94 + 分差归一化区间）

    VOCAB_SIZE = 128      # 总词表大小（预留扩展空间）


class TokenType:
    """Token 类型（用于 Type Embedding）。"""
    HAND = 0
    DORA = 1
    DISCARD = 2
    MELD = 3
    RIICHI = 4
    GLOBAL = 5
    NUM_TYPES = 6


# ── 数据结构 ─────────────────────────────────────────────────────────────────

@dataclass
class Token:
    """单个 Token。

    Attributes:
        token_id: 主内容 ID（用于 torch.nn.Embedding 查找）
        token_type: 类别 ID（用于 Type Embedding）
        behavior_id: 行为属性 ID（弃牌摸切/手切、副露类型等），0=无
    """
    token_id: int
    token_type: int
    behavior_id: int = 0


@dataclass
class TokenSequence:
    """Token 序列 + 辅助字段。"""
    tokens: List[Token] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.tokens)

    def add(self, token_id: int, token_type: int, behavior_id: int = 0) -> None:
        self.tokens.append(Token(token_id, token_type, behavior_id))

    @property
    def token_ids(self) -> List[int]:
        return [t.token_id for t in self.tokens]

    @property
    def token_types(self) -> List[int]:
        return [t.token_type for t in self.tokens]

    @property
    def behavior_ids(self) -> List[int]:
        return [t.behavior_id for t in self.tokens]


# ── Tokenizer ────────────────────────────────────────────────────────────────

class MahjongTokenizer:
    """将麻将局面转换为 Token 序列。

    用法:
        tokenizer = MahjongTokenizer()
        seq = tokenizer.tokenize_game_snapshot(tracker, player_idx=0)
    """

    def __init__(self, max_sequence_length: int = 192):
        self.max_len = max_sequence_length

    # ── 公共接口 ────────────────────────────────────────────────────────

    def tokenize_game_snapshot(self, tracker: MJSONGameTracker,
                               player_idx: int) -> TokenSequence:
        """从 MJSONGameTracker 生成 Token 序列（监督学习/标签生成时使用）。"""
        seq = TokenSequence()

        # 1. 手牌 Token（14 张：手牌 13 张 + 最新摸牌）
        self._add_hand_tokens(seq, tracker, player_idx)

        # 2. 宝牌指示牌 Token（已翻开的）
        self._add_dora_tokens(seq, tracker)

        # 3. 四家舍牌序列（按顺序排列，包含摸切/手切标记）
        self._add_discard_sequence(seq, tracker, player_idx)

        # 4. 四家副露记录
        self._add_meld_sequence(seq, tracker, player_idx)

        # 5. 立直事件
        self._add_riichi_events(seq, tracker)

        # 6. 全局状态 Token
        self._add_global_tokens(seq, tracker, player_idx)

        # 7. 截断到最大长度
        if len(seq) > self.max_len:
            seq.tokens = seq.tokens[:self.max_len]

        return seq

    def tokenize_engine_state(self, engine, player_idx: int) -> TokenSequence:
        """从引擎状态生成 Token 序列（推理时使用）。

        需要 engine 暴露 tracker 类似的状态接口。
        """
        # TODO: 在 RL 推理时从 GameEngine 构建 token 序列
        # 当前留空，后续实现
        raise NotImplementedError(
            "Engine state tokenization will be implemented in Phase B")

    # ── 内部方法 ────────────────────────────────────────────────────────

    def _add_hand_tokens(self, seq: TokenSequence,
                         tracker: MJSONGameTracker, player_idx: int) -> None:
        """添加手牌 Token。

        手牌 13 张 + 最新摸牌（通过 tracker 的 remaining_tiles 变化推断）。
        由于 tracker 没有明确的"最新摸牌"标记，我们简化为：
        - 对于 14 张手牌的情况，全部添加
        - 对于 13 张手牌（摸牌前），只添加 13 张
        """
        hand = tracker.hands[player_idx]
        total_tiles = sum(hand)

        # 展开 int[34] 为 tile 列表
        tiles = []
        for t in range(NUM_TYPES):
            count = hand[t]
            if count > 0:
                # 判断是否为赤宝牌
                is_red = False
                if t in (4, 13, 22):
                    is_red = True
                for _ in range(count):
                    tiles.append((t, is_red))

        # 摸牌前 = 13 张，摸牌后 = 14 张
        # 如果 total_tiles == 14，最后一张是刚摸的牌
        for i, (tile_type, red) in enumerate(tiles):
            # 赤宝牌使用赤牌 ID，普通牌直接用 tile type
            tile_id = tile_type
            if red:
                if tile_type == 4:
                    tile_id = TokenVocab.RED_5M
                elif tile_type == 13:
                    tile_id = TokenVocab.RED_5P
                elif tile_type == 22:
                    tile_id = TokenVocab.RED_5S
            seq.add(tile_id, TokenType.HAND)

    def _add_dora_tokens(self, seq: TokenSequence,
                         tracker: MJSONGameTracker) -> None:
        """添加宝牌指示牌 Token。"""
        for dora_type in tracker.dora_indicators:
            if 0 <= dora_type < NUM_TYPES:
                seq.add(dora_type, TokenType.DORA)

    def _add_discard_sequence(self, seq: TokenSequence,
                              tracker: MJSONGameTracker,
                              player_idx: int) -> None:
        """按顺序添加四家舍牌序列。

        每张舍牌包含：
        - token_id: 牌型
        - behavior_id: 摸切/手切/立直切标记
        """
        # 收集所有玩家的舍牌
        all_discards: List[Tuple[int, int, int]] = []  # (player, tile_type, is_tsumogiri)

        for p in range(4):
            for tile_type in tracker.discards[p]:
                if 0 <= tile_type < NUM_TYPES:
                    all_discards.append((p, tile_type, 0))

        # 简化：按玩家分组添加
        # 先加自己，再加其他三家
        opponents = [i for i in range(4) if i != player_idx]
        ordered_players = [player_idx] + opponents

        for p in ordered_players:
            for tile_type in tracker.discards[p]:
                if 0 <= tile_type < NUM_TYPES:
                    # behavior_id = player << 8 | action_type
                    # player = 0-3, action_type = 0 (默认为手切)
                    behavior_id = (p << 8) | TokenVocab.ACTION_TEDASHI
                    seq.add(tile_type, TokenType.DISCARD, behavior_id)

    def _add_meld_sequence(self, seq: TokenSequence,
                           tracker: MJSONGameTracker,
                           player_idx: int) -> None:
        """添加四家副露记录。"""
        meld_type_map = {
            'chi': TokenVocab.MELD_CHI,
            'pon': TokenVocab.MELD_PON,
            'daiminkan': TokenVocab.MELD_KAN,
            'ankan': TokenVocab.MELD_ANKAN,
            'kakan': TokenVocab.MELD_KAKAN,
        }

        opponents = [i for i in range(4) if i != player_idx]
        ordered_players = [player_idx] + opponents

        for p in ordered_players:
            for meld_type_str, tiles, called_from in tracker.melds[p]:
                mt = meld_type_map.get(meld_type_str, TokenVocab.MELD_CHI)
                first_tile = tiles[0] if tiles else 0

                # token_id = 副露的第一张牌型
                # behavior_id = player << 8 | meld_type
                behavior_id = (p << 8) | mt
                seq.add(first_tile if 0 <= first_tile < NUM_TYPES else 0,
                       TokenType.MELD, behavior_id)

                # 添加副露中剩余牌
                for t in tiles[1:]:
                    if 0 <= t < NUM_TYPES:
                        behavior_id = (p << 8) | mt
                        seq.add(t, TokenType.MELD, behavior_id)

    def _add_riichi_events(self, seq: TokenSequence,
                           tracker: MJSONGameTracker) -> None:
        """添加立直事件 Token。"""
        for p in range(4):
            if tracker.is_riichi[p]:
                seq.add(TokenVocab.ACTION_RIICHI, TokenType.RIICHI, p)

    def _add_global_tokens(self, seq: TokenSequence,
                           tracker: MJSONGameTracker,
                           player_idx: int) -> None:
        """添加全局状态 Token。

        包括：场风、自风、巡目、剩余牌数、本场、立直棒、分差。
        """
        # 场风
        if tracker.bakaze == 27:
            seq.add(TokenVocab.ROUND_EAST, TokenType.GLOBAL)
        elif tracker.bakaze == 28:
            seq.add(TokenVocab.ROUND_SOUTH, TokenType.GLOBAL)
        elif tracker.bakaze == 29:
            seq.add(TokenVocab.ROUND_WEST, TokenType.GLOBAL)
        elif tracker.bakaze == 30:
            seq.add(TokenVocab.ROUND_NORTH, TokenType.GLOBAL)

        # 自风
        seat_wind = self._seat_wind(tracker, player_idx)
        if seat_wind == 27:
            seq.add(TokenVocab.SELF_EAST, TokenType.GLOBAL)
        elif seat_wind == 28:
            seq.add(TokenVocab.SELF_SOUTH, TokenType.GLOBAL)
        elif seat_wind == 29:
            seq.add(TokenVocab.SELF_WEST, TokenType.GLOBAL)
        elif seat_wind == 30:
            seq.add(TokenVocab.SELF_NORTH, TokenType.GLOBAL)

        # 剩余牌数
        remaining = tracker.remaining_tiles
        remaining_id = TokenVocab.REMAINING_BASE + min(remaining, 70)
        seq.add(remaining_id, TokenType.GLOBAL)

        # 本场数
        seq.add(TokenVocab.HONBA, TokenType.GLOBAL, tracker.honba)

        # 立直棒数量
        seq.add(TokenVocab.RIICHI_STICK, TokenType.GLOBAL, tracker.kyotaku)

        # 分差（对其他三家）
        opponents = [i for i in range(4) if i != player_idx]
        for opp in opponents:
            diff = tracker.scores[opp] - tracker.scores[player_idx]
            # 分差归一化到 -30~30（+/-30000点），映射到 DIFF_BASE+0~DIFF_BASE+60
            diff_clamped = max(-30, min(30, diff // 1000))
            diff_id = TokenVocab.DIFF_BASE + (diff_clamped + 30)
            seq.add(diff_id, TokenType.GLOBAL)

    def _seat_wind(self, tracker: MJSONGameTracker, player_idx: int) -> int:
        """计算玩家自风。"""
        winds = [27, 28, 29, 30]
        offset = (player_idx - tracker.oya) % 4
        return winds[offset]
# 中文注释：将麻将局面从 MJSON 解析器状态转换为 Transformer 可消费的 Token 序列。
