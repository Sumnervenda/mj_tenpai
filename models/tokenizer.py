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
    """Token ID 词汇表。

    ID 0 保留为 PAD（padding），真实牌型从 1 开始。
    这解决了 padding_idx=0 与 tile type 0 (1m) 的冲突。
    """
    PAD = 0

    # 牌型 ID（1-34，对应 engine.tile 的 0-33 偏移 +1）
    TILE_MIN = 1
    TILE_MAX = 34

    # 赤宝牌标记（35-37）
    RED_5M = 35
    RED_5P = 36
    RED_5S = 37

    # 玩家标记（38-41）
    P0 = 38
    P1 = 39
    P2 = 40
    P3 = 41

    # 动作类型（42-49）
    ACTION_TSUMOGIRI = 42      # 摸切
    ACTION_TEDASHI = 43        # 手切
    ACTION_RIICHI_DISCARD = 44 # 立直宣言切牌
    ACTION_CHI = 45            # 吃
    ACTION_PON = 46            # 碰
    ACTION_KAN = 47            # 明槓
    ACTION_RIICHI = 48         # 立直事件
    ACTION_KAKAN = 49          # 加槓

    # 副露类型（50-54）
    MELD_CHI = 50
    MELD_PON = 51
    MELD_KAN = 52
    MELD_KAKAN = 53
    MELD_ANKAN = 54

    # 全局标记（55+）
    ROUND_EAST = 55
    ROUND_SOUTH = 56
    ROUND_WEST = 57
    ROUND_NORTH = 58
    SELF_EAST = 59
    SELF_SOUTH = 60
    SELF_WEST = 61
    SELF_NORTH = 62
    REMAINING_BASE = 63   # 剩余牌数标记从 63 开始（63 + remaining）
    HONBA = 93            # 本场数
    RIICHI_STICK = 94     # 立直棒标记
    DIFF_BASE = 95        # 分差标记从 95 开始（95 + 分差归一化区间）

    VOCAB_SIZE = 192      # 总词表大小（覆盖 DIFF_BASE+60=155，留余量）


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
        """从 GameEngine 状态生成 Token 序列（RL/self-play 推理时使用）。

        与 tokenize_game_snapshot() 输出相同格式的 TokenSequence，
        但数据来源是 GameEngine 的实时状态而非 MJSONGameTracker。
        """
        from engine.hand import MeldType

        state = engine.get_game_state()
        seq = TokenSequence()

        # 1. 手牌 Token（int[34] 直方图，与 tracker 相同）
        hand = state.hands_concealed[player_idx]
        aka_set = set(engine.players[player_idx].hand.aka_tiles)
        # 收集该玩家手牌中赤牌的 type_id
        aka_types_in_hand = set()
        for abs_id in aka_set:
            aka_types_in_hand.add(abs_to_type(abs_id))

        tiles = []
        for t in range(NUM_TYPES):
            count = hand[t]
            if count > 0:
                is_red = t in aka_types_in_hand
                for _ in range(count):
                    tiles.append((t, is_red))

        for _tile_type, red in tiles:
            if red:
                if _tile_type == 4:
                    tile_id = TokenVocab.RED_5M
                elif _tile_type == 13:
                    tile_id = TokenVocab.RED_5P
                elif _tile_type == 22:
                    tile_id = TokenVocab.RED_5S
                else:
                    tile_id = _tile_type + 1
            else:
                tile_id = _tile_type + 1
            seq.add(tile_id, TokenType.HAND)

        # 2. 宝牌指示牌 Token（绝对 ID → type ID）
        for abs_id in state.dora_indicators:
            dora_type = abs_to_type(abs_id)
            if 0 <= dora_type < NUM_TYPES:
                seq.add(dora_type + 1, TokenType.DORA)

        # 3. 四家舍牌序列（绝对 ID → type ID）
        opponents = [i for i in range(4) if i != player_idx]
        ordered_players = [player_idx] + opponents
        for p in ordered_players:
            for abs_id in state.discards[p]:
                tile_type = abs_to_type(abs_id)
                if 0 <= tile_type < NUM_TYPES:
                    behavior_id = p * 4
                    seq.add(tile_type + 1, TokenType.DISCARD, behavior_id)

        # 4. 四家副露记录（MeldType 枚举 → 字符串）
        meld_type_to_str = {
            MeldType.CHI: 'chi',
            MeldType.PON: 'pon',
            MeldType.KAN_CLOSED: 'ankan',
            MeldType.KAN_OPEN: 'kakan',
            MeldType.KAN_DAIMIN: 'daiminkan',
        }
        meld_type_map = {
            'chi': TokenVocab.MELD_CHI,
            'pon': TokenVocab.MELD_PON,
            'daiminkan': TokenVocab.MELD_KAN,
            'ankan': TokenVocab.MELD_ANKAN,
            'kakan': TokenVocab.MELD_KAKAN,
        }
        for p in ordered_players:
            for meld in state.open_melds[p]:
                mts = meld_type_to_str.get(meld.meld_type, 'chi')
                mt = meld_type_map.get(mts, TokenVocab.MELD_CHI)
                first_tile = meld.tiles[0] if meld.tiles else 0
                behavior_id = p * 8 + (mt - TokenVocab.MELD_CHI)
                seq.add(first_tile + 1 if 0 <= first_tile < NUM_TYPES else 0,
                       TokenType.MELD, behavior_id)
                for t in meld.tiles[1:]:
                    if 0 <= t < NUM_TYPES:
                        seq.add(t + 1, TokenType.MELD, behavior_id)

        # 5. 立直事件
        for p in range(4):
            if state.is_riichi[p]:
                seq.add(TokenVocab.ACTION_RIICHI, TokenType.RIICHI, p)

        # 6. 全局状态 Token
        # 场风
        if state.round_wind == 27:
            seq.add(TokenVocab.ROUND_EAST, TokenType.GLOBAL)
        elif state.round_wind == 28:
            seq.add(TokenVocab.ROUND_SOUTH, TokenType.GLOBAL)

        # 自风
        winds = [27, 28, 29, 30]
        offset = (player_idx - engine.dealer_idx) % 4
        seat_wind = winds[offset]
        if seat_wind == 27:
            seq.add(TokenVocab.SELF_EAST, TokenType.GLOBAL)
        elif seat_wind == 28:
            seq.add(TokenVocab.SELF_SOUTH, TokenType.GLOBAL)
        elif seat_wind == 29:
            seq.add(TokenVocab.SELF_WEST, TokenType.GLOBAL)
        elif seat_wind == 30:
            seq.add(TokenVocab.SELF_NORTH, TokenType.GLOBAL)

        # 剩余牌数
        remaining = state.remaining_tiles
        remaining_id = TokenVocab.REMAINING_BASE + min(remaining, 70)
        seq.add(remaining_id, TokenType.GLOBAL)

        # 本场数
        seq.add(TokenVocab.HONBA, TokenType.GLOBAL, state.honba)

        # 立直棒数量
        seq.add(TokenVocab.RIICHI_STICK, TokenType.GLOBAL, state.riichi_sticks)

        # 分差（对其他三家）
        for opp in opponents:
            diff = state.scores[opp] - state.scores[player_idx]
            diff_clamped = max(-30, min(30, diff // 1000))
            diff_id = TokenVocab.DIFF_BASE + (diff_clamped + 30)
            seq.add(diff_id, TokenType.GLOBAL)

        # 7. 截断到最大长度
        if len(seq) > self.max_len:
            seq.tokens = seq.tokens[:self.max_len]

        return seq

    def tokenize_public_private_snapshot(self, tracker: MJSONGameTracker,
                                         player_idx: int
                                         ) -> Tuple[TokenSequence, TokenSequence]:
        """返回 (public_seq, private_seq) 两个 TokenSequence。

        Public: 自己手牌、四家牌河、四家副露、宝牌指示牌、场风自风、分数、
                分差、巡目、立直 — 与 tokenize_game_snapshot() 相同。
        Private: 对手暗手、ura 宝牌 — God's-eye 教师可见的信息。
        """
        public_seq = self.tokenize_game_snapshot(tracker, player_idx)

        private_seq = TokenSequence()
        opponents = [i for i in range(4) if i != player_idx]

        # 1. 对手暗手 Token
        for opp in opponents:
            hand = tracker.hands[opp]
            aka = tracker.aka_counts[opp]
            for t in range(NUM_TYPES):
                count = hand[t]
                if count > 0:
                    aka_count = aka.get(t, 0)
                    normal_count = count - aka_count
                    for _ in range(aka_count):
                        if t == 4:
                            tid = TokenVocab.RED_5M
                        elif t == 13:
                            tid = TokenVocab.RED_5P
                        elif t == 22:
                            tid = TokenVocab.RED_5S
                        else:
                            tid = t + 1
                        private_seq.add(tid, TokenType.HAND, opp)
                    for _ in range(normal_count):
                        private_seq.add(t + 1, TokenType.HAND, opp)

        # 2. Ura 宝牌指示牌（已在 _on_hora 中记录）
        for ura_type in tracker.ura_indicators:
            if 0 <= ura_type < NUM_TYPES:
                private_seq.add(ura_type + 1, TokenType.DORA)

        # 3. 截断
        if len(private_seq) > self.max_len:
            private_seq.tokens = private_seq.tokens[:self.max_len]

        return public_seq, private_seq

    def tokenize_public_private_engine_state(self, engine, player_idx: int
                                             ) -> Tuple[TokenSequence,
                                                        TokenSequence]:
        """从 GameEngine 状态生成 (public, private) Token 序列。

        Private 包含：对手暗手、完整牌山剩余牌型、ura 宝牌。
        """
        public_seq = self.tokenize_engine_state(engine, player_idx)

        state = engine.get_game_state()
        private_seq = TokenSequence()
        opponents = [i for i in range(4) if i != player_idx]

        # 1. 对手暗手
        for opp in opponents:
            hand = state.hands_concealed[opp]
            aka_set = set(engine.players[opp].hand.aka_tiles)
            aka_types = set()
            for abs_id in aka_set:
                aka_types.add(abs_to_type(abs_id))
            for t in range(NUM_TYPES):
                count = hand[t]
                if count > 0:
                    is_red = t in aka_types
                    aka_c = 1 if is_red else 0
                    normal_c = count - aka_c
                    for _ in range(aka_c):
                        if t == 4:
                            tid = TokenVocab.RED_5M
                        elif t == 13:
                            tid = TokenVocab.RED_5P
                        elif t == 22:
                            tid = TokenVocab.RED_5S
                        else:
                            tid = t + 1
                        private_seq.add(tid, TokenType.HAND, opp)
                    for _ in range(normal_c):
                        private_seq.add(t + 1, TokenType.HAND, opp)

        # 2. 牌山剩余牌（保留摸牌顺序）
        wall = engine.wall
        for i in range(wall._live_ptr, wall._dead_wall_start):
            t = abs_to_type(wall.tiles[i])
            if 0 <= t < NUM_TYPES:
                private_seq.add(t + 1, TokenType.GLOBAL)

        # 3. Ura 宝牌指示牌
        for abs_id in state.ura_dora_indicators:
            ura_type = abs_to_type(abs_id)
            if 0 <= ura_type < NUM_TYPES:
                private_seq.add(ura_type + 1, TokenType.DORA)

        if len(private_seq) > self.max_len:
            private_seq.tokens = private_seq.tokens[:self.max_len]

        return public_seq, private_seq

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
        aka = tracker.aka_counts[player_idx]

        tiles = []
        for t in range(NUM_TYPES):
            count = hand[t]
            if count > 0:
                aka_count = aka.get(t, 0)
                normal_count = count - aka_count
                for _ in range(aka_count):
                    tiles.append((t, True))
                for _ in range(normal_count):
                    tiles.append((t, False))

        for _tile_type, red in tiles:
            # 赤宝牌使用赤牌 ID，普通牌偏移 +1（PAD=0 保留）
            if red:
                if _tile_type == 4:
                    tile_id = TokenVocab.RED_5M
                elif _tile_type == 13:
                    tile_id = TokenVocab.RED_5P
                elif _tile_type == 22:
                    tile_id = TokenVocab.RED_5S
                else:
                    tile_id = _tile_type + 1
            else:
                tile_id = _tile_type + 1
            seq.add(tile_id, TokenType.HAND)

    def _add_dora_tokens(self, seq: TokenSequence,
                         tracker: MJSONGameTracker) -> None:
        """添加宝牌指示牌 Token。"""
        for dora_type in tracker.dora_indicators:
            if 0 <= dora_type < NUM_TYPES:
                seq.add(dora_type + 1, TokenType.DORA)

    def _add_discard_sequence(self, seq: TokenSequence,
                              tracker: MJSONGameTracker,
                              player_idx: int) -> None:
        """按顺序添加四家舍牌序列。

        每张舍牌包含：
        - token_id: 牌型（偏移 +1）
        - behavior_id: 连续编码 player*4 + action_offset
          offset: TEDASHI=0, TSUMOGIRI=1, RIICHI_DISCARD=2
        """
        opponents = [i for i in range(4) if i != player_idx]
        ordered_players = [player_idx] + opponents

        for p in ordered_players:
            for tile_type in tracker.discards[p]:
                if 0 <= tile_type < NUM_TYPES:
                    behavior_id = p * 4  # TEDASHI offset = 0
                    seq.add(tile_type + 1, TokenType.DISCARD, behavior_id)

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

                # behavior_id = player * 8 + meld_type_offset
                # meld_type 50-54 → offset 0-4
                behavior_id = p * 8 + (mt - TokenVocab.MELD_CHI)
                seq.add(first_tile + 1 if 0 <= first_tile < NUM_TYPES else 0,
                       TokenType.MELD, behavior_id)

                # 添加副露中剩余牌
                for t in tiles[1:]:
                    if 0 <= t < NUM_TYPES:
                        seq.add(t + 1, TokenType.MELD, behavior_id)

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
