"""MJSON 人类牌谱解析器 —— 将天凤/雀魂格式的对局事件流转换为训练样本。

格式：gzip 压缩的 JSONL，每行一个事件对象。
解析方式：逐事件回放 → 跟踪 4 家手牌/副露/舍牌 → 在每个决策点重建 354 维状态张量
和当前合法动作掩码 → 记录人类实际选择的动作为标签。
"""

import gzip
import json
import glob
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from engine.tile import NUM_TYPES
from engine.agari import get_legal_discards_for_riichi
from .record_parser import TrainingSample

# ── MJSON tile code → engine type_id (0-33) 映射 ──────────────────────────

# 数牌: "1m".."9m" → 0-8, "1p".."9p" → 9-17, "1s".."9s" → 18-26
# 赤宝牌: "5mr"→4, "5pr"→13, "5sr"→22
# 风牌: E→27(東), S→28(南), W→29(西), N→30(北)
# 三元牌: P→31(白), F→32(発), C→33(中)

_MJSON_TO_TYPE: Dict[str, int] = {}
_TYPE_TO_MJSON: Dict[int, str] = {}

def _build_tile_map():
    if _MJSON_TO_TYPE:
        return
    suits = {'m': 0, 'p': 9, 's': 18}
    for code, base in suits.items():
        for n in range(1, 10):
            _MJSON_TO_TYPE[f"{n}{code}"] = base + n - 1
    # Red fives
    _MJSON_TO_TYPE["5mr"] = 4
    _MJSON_TO_TYPE["5pr"] = 13
    _MJSON_TO_TYPE["5sr"] = 22
    # Winds
    _MJSON_TO_TYPE["E"] = 27
    _MJSON_TO_TYPE["S"] = 28
    _MJSON_TO_TYPE["W"] = 29
    _MJSON_TO_TYPE["N"] = 30
    # Dragons
    _MJSON_TO_TYPE["P"] = 31
    _MJSON_TO_TYPE["F"] = 32
    _MJSON_TO_TYPE["C"] = 33
    for k, v in _MJSON_TO_TYPE.items():
        _TYPE_TO_MJSON[v] = k


def mjson_str_to_type(tile_str: str) -> int:
    """将 MJSON 牌码字符串转换为引擎 type_id (0-33)。"""
    _build_tile_map()
    return _MJSON_TO_TYPE.get(tile_str, -1)


def type_to_mjson_str(type_id: int) -> str:
    """将引擎 type_id 转换为 MJSON 牌码字符串。"""
    _build_tile_map()
    return _TYPE_TO_MJSON.get(type_id, "?")


# ── 宝牌计算 ──────────────────────────────────────────────────────────────

def _next_dora_type(indicator_type: int) -> int:
    """根据宝牌指示牌类型，计算实际的宝牌类型。"""
    if indicator_type < 27:
        suit_base = (indicator_type // 9) * 9
        return suit_base + (indicator_type + 1) % 9
    elif indicator_type < 31:
        return 27 + (indicator_type - 27 + 1) % 4
    else:
        return 31 + (indicator_type - 31 + 1) % 3


# ── 对局状态追踪器 ─────────────────────────────────────────────────────────

@dataclass
class MJSONGameTracker:
    """回放 MJSON 事件流，实时追踪 4 家完整对局状态。"""

    hands: List[List[int]] = field(default_factory=lambda: [[0]*34 for _ in range(4)])
    melds: List[List[Tuple[str, List[int], int]]] = field(default_factory=lambda: [[] for _ in range(4)])
    discards: List[List[int]] = field(default_factory=lambda: [[] for _ in range(4)])
    dora_indicators: List[int] = field(default_factory=list)
    ura_indicators: List[int] = field(default_factory=list)
    scores: List[int] = field(default_factory=lambda: [25000]*4)
    is_riichi: List[bool] = field(default_factory=lambda: [False]*4)
    is_ippatsu: List[bool] = field(default_factory=lambda: [False]*4)
    is_double_riichi: List[bool] = field(default_factory=lambda: [False]*4)
    bakaze: int = 27
    kyoku: int = 1
    honba: int = 0
    kyotaku: int = 0
    oya: int = 0
    pending_reach: Optional[int] = None
    last_discard_type: int = -1
    last_discard_by: int = -1
    remaining_tiles: int = 70
    aka_flag: bool = True
    tenpai_at_ryuukyoku: List[bool] = field(default_factory=lambda: [False]*4)

    def reset_round(self):
        self.hands = [[0]*34 for _ in range(4)]
        self.melds = [[] for _ in range(4)]
        self.discards = [[] for _ in range(4)]
        self.dora_indicators = []
        self.ura_indicators = []
        self.is_riichi = [False]*4
        self.is_ippatsu = [False]*4
        self.is_double_riichi = [False]*4
        self.pending_reach = None
        self.last_discard_type = -1
        self.last_discard_by = -1
        self.remaining_tiles = 70
        self.tenpai_at_ryuukyoku = [False]*4

    # ── 事件应用 ────────────────────────────────────────────────────────

    def apply_event(self, event: dict) -> None:
        etype = event.get('type', '')
        handler = getattr(self, f'_on_{etype}', None)
        if handler:
            handler(event)

    def _on_start_game(self, e: dict):
        self.aka_flag = e.get('aka_flag', True)

    def _on_start_kyoku(self, e: dict):
        self.reset_round()
        self.bakaze = _MJSON_TO_TYPE.get(e['bakaze'], 27)
        self.kyoku = e.get('kyoku', 1)
        self.honba = e.get('honba', 0)
        self.kyotaku = e.get('kyotaku', 0)
        self.oya = e.get('oya', 0)
        self.scores = list(e.get('scores', [25000]*4))
        dora_str = e.get('dora_marker', '')
        self.dora_indicators = [mjson_str_to_type(dora_str)] if dora_str else []
        for p_idx, hand_list in enumerate(e.get('tehais', [])):
            for tile_str in hand_list:
                t = mjson_str_to_type(tile_str)
                if t >= 0:
                    self.hands[p_idx][t] += 1

    def _on_tsumo(self, e: dict):
        actor = e['actor']
        t = mjson_str_to_type(e['pai'])
        if t >= 0:
            self.hands[actor][t] += 1
        self.remaining_tiles -= 1

    def _on_dahai(self, e: dict):
        actor = e['actor']
        t = mjson_str_to_type(e['pai'])
        if t >= 0:
            self.hands[actor][t] -= 1
            self.discards[actor].append(t)
        self.last_discard_type = t
        self.last_discard_by = actor
        if self.pending_reach == actor:
            self.pending_reach = None

    def _on_reach(self, e: dict):
        self.pending_reach = e['actor']

    def _on_reach_accepted(self, e: dict):
        actor = e['actor']
        self.is_riichi[actor] = True
        self.is_ippatsu[actor] = True
        if len(self.discards[actor]) == 0:
            self.is_double_riichi[actor] = True
        self.scores[actor] -= 1000
        self.kyotaku += 1

    def _on_chi(self, e: dict):
        actor = e['actor']
        called_tile = mjson_str_to_type(e['pai'])
        consumed = [mjson_str_to_type(s) for s in e.get('consumed', [])]
        for t in consumed:
            if t >= 0:
                self.hands[actor][t] -= 1
        all_tiles = consumed + [called_tile]
        all_tiles.sort()
        self.melds[actor].append(('chi', all_tiles, e.get('target', -1)))
        self.last_discard_type = -1

    def _on_pon(self, e: dict):
        actor = e['actor']
        called_tile = mjson_str_to_type(e['pai'])
        consumed = [mjson_str_to_type(s) for s in e.get('consumed', [])]
        for t in consumed:
            if t >= 0:
                self.hands[actor][t] -= 1
        all_tiles = consumed + [called_tile]
        self.melds[actor].append(('pon', all_tiles, e.get('target', -1)))
        self.last_discard_type = -1

    def _on_daiminkan(self, e: dict):
        actor = e['actor']
        called_tile = mjson_str_to_type(e['pai'])
        consumed = [mjson_str_to_type(s) for s in e.get('consumed', [])]
        for t in consumed:
            if t >= 0:
                self.hands[actor][t] -= 1
        all_tiles = consumed + [called_tile]
        self.melds[actor].append(('daiminkan', all_tiles, e.get('target', -1)))
        self.last_discard_type = -1

    def _on_ankan(self, e: dict):
        actor = e['actor']
        consumed = [mjson_str_to_type(s) for s in e.get('consumed', [])]
        for t in consumed:
            if t >= 0:
                self.hands[actor][t] -= 1
        self.melds[actor].append(('ankan', consumed, -1))

    def _on_kakan(self, e: dict):
        actor = e['actor']
        added_tile = mjson_str_to_type(e['pai'])
        if added_tile >= 0:
            self.hands[actor][added_tile] -= 1
        consumed = [mjson_str_to_type(s) for s in e.get('consumed', [])]
        self.melds[actor].append(('kakan', consumed + [added_tile], -1))

    def _on_dora(self, e: dict):
        dora_str = e.get('dora_marker', '')
        if dora_str:
            self.dora_indicators.append(mjson_str_to_type(dora_str))

    def _on_hora(self, e: dict):
        self.scores = [self.scores[i] + e.get('deltas', [0]*4)[i] for i in range(4)]
        for ura_str in e.get('ura_markers', []):
            ura_t = mjson_str_to_type(ura_str)
            if ura_t >= 0:
                self.ura_indicators.append(ura_t)

    def _on_ryukyoku(self, e: dict):
        self.scores = [self.scores[i] + e.get('deltas', [0]*4)[i] for i in range(4)]

    def _on_end_kyoku(self, e: dict):
        pass

    def _on_end_game(self, e: dict):
        pass

    # ── 状态张量构建 ────────────────────────────────────────────────────

    def build_state_tensor(self, player_idx: int) -> np.ndarray:
        """重建 354 维状态张量，与 GameEngine.get_state_tensor() 布局一致。

        354 维布局:
          [0:34]     自手牌直方图
          [34:68]    自副露计数
          [68:102]   自舍牌计数
          [102:136]  宝牌指示牌 one-hot（含已翻开的）
          [136:170]  对家 0 舍牌
          [170:204]  对家 1 舍牌
          [204:238]  对家 2 舍牌
          [238:272]  对家 0 副露
          [272:306]  对家 1 副露
          [306:340]  对家 2 副露
          [340:347]  全局特征 (score, honba, kyotaku, remaining, riichi, bakaze, kyoku)
          [347:350]  分差 ×3
          [350:354]  自风 one-hot
        """
        state = np.zeros(354, dtype=np.float32)

        # 自手牌
        for t in range(34):
            state[t] = float(self.hands[player_idx][t])

        # 自副露
        for _, tiles, _ in self.melds[player_idx]:
            for t in tiles:
                if 0 <= t < 34:
                    state[34 + t] += 1.0

        # 自舍牌
        for t in self.discards[player_idx]:
            state[68 + t] += 1.0

        # 宝牌指示牌
        for dt in self.dora_indicators:
            if 0 <= dt < 34:
                state[102 + dt] = 1.0

        # 对手舍牌和副露（按顺序排列除自己外的 3 家）
        opponents = [i for i in range(4) if i != player_idx]
        for opp_idx, opp in enumerate(opponents):
            base_disc = 136 + opp_idx * 34
            base_meld = 238 + opp_idx * 34
            for t in self.discards[opp]:
                if 0 <= t < 34:
                    state[base_disc + t] += 1.0
            for _, tiles, _ in self.melds[opp]:
                for t in tiles:
                    if 0 <= t < 34:
                        state[base_meld + t] += 1.0

        # 全局特征
        state[340] = float(self.scores[player_idx]) / 1000.0
        state[341] = float(self.honba)
        state[342] = float(self.kyotaku)
        state[343] = float(self.remaining_tiles) / 122.0
        state[344] = 1.0 if self.is_riichi[player_idx] else 0.0
        state[345] = float(self.bakaze - 27)
        state[346] = float(self.kyoku)

        # 分差
        for opp_idx, opp in enumerate(opponents):
            state[347 + opp_idx] = (float(self.scores[opp]) -
                                     float(self.scores[player_idx])) / 1000.0

        # 自风 one-hot
        seat_wind = self._seat_wind(player_idx)
        if 27 <= seat_wind <= 30:
            state[350 + (seat_wind - 27)] = 1.0

        return state

    def _seat_wind(self, player_idx: int) -> int:
        """根据玩家与庄家的相对位置计算自风。

        庄家 = 東，下家 = 南，对家 = 西，上家 = 北。
        自风与场风无关；场风只影响 bakaze 役。
        """
        winds = [27, 28, 29, 30]  # 東南西北
        offset = (player_idx - self.oya) % 4
        return winds[offset]

    # ── 合法动作掩码 ────────────────────────────────────────────────────

    def build_draw_mask(self, player_idx: int) -> np.ndarray:
        """构建摸牌后切牌阶段的合法动作掩码。

        包括：切牌 (0-33)、自摸 (34)、立直切牌 (37-70)、暗槓 (74)、加槓 (75)。
        """
        mask = np.zeros(77, dtype=np.float32)
        hand = self.hands[player_idx]

        # 切牌：所有持有的牌
        for t in range(34):
            if hand[t] > 0:
                mask[t] = 1.0

        # 立直判定：门清、未立直、分数 >= 1000、听牌
        can_riichi = self._can_riichi(player_idx)
        if can_riichi:
            riichi_discards = get_legal_discards_for_riichi(list(hand))
            for t in riichi_discards:
                mask[37 + t] = 1.0

        # 暗槓
        if not self.is_riichi[player_idx]:
            for t in range(34):
                if hand[t] == 4:
                    mask[74] = 1.0  # ANKAN
                    break

        # 加槓
        for _, tiles, _ in self.melds[player_idx]:
            if tiles and len(tiles) == 3 and tiles[0] == tiles[1]:
                t = tiles[0]
                if hand[t] > 0:
                    mask[75] = 1.0  # KAKAN
                    break

        return mask

    def _can_riichi(self, player_idx: int) -> bool:
        """判定是否可立直：门清 + 未立直 + 分数 >= 1000 + 存在听牌切牌。"""
        if self.is_riichi[player_idx]:
            return False
        if self.scores[player_idx] < 1000:
            return False
        # 检查门清
        for _, _, called_from in self.melds[player_idx]:
            if called_from >= 0:
                return False
        # 检查是否存在至少一种切牌能听牌
        hand = list(self.hands[player_idx])
        return len(get_legal_discards_for_riichi(hand)) > 0

    def build_response_mask(self, player_idx: int) -> np.ndarray:
        """构建别家切牌后的响应动作掩码。

        包括：荣和 (35)、碰 (71)、吃 (72)、大明槓 (73)、パス (76)。
        """
        mask = np.zeros(77, dtype=np.float32)
        if self.last_discard_type < 0 or player_idx == self.last_discard_by:
            mask[76] = 1.0  # PASS only
            return mask

        hand = self.hands[player_idx]
        discard_t = self.last_discard_type

        # 碰
        if hand[discard_t] >= 2 and not self.is_riichi[player_idx]:
            mask[71] = 1.0

        # 大明槓
        if hand[discard_t] >= 3 and not self.is_riichi[player_idx]:
            mask[73] = 1.0

        # 吃（仅限数牌，且仅限上家）
        if discard_t < 27 and not self.is_riichi[player_idx]:
            is_kami = (self.last_discard_by + 1) % 4 == player_idx
            if is_kami:
                suit_base = (discard_t // 9) * 9
                suit_idx = discard_t % 9
                # 检查两侧的吃组合
                if (suit_idx >= 2 and hand[discard_t - 2] > 0
                        and hand[discard_t - 1] > 0):
                    mask[72] = 1.0
                elif (suit_idx >= 1 and suit_idx <= 7
                        and hand[discard_t - 1] > 0
                        and hand[discard_t + 1] > 0):
                    mask[72] = 1.0
                elif (suit_idx <= 6 and hand[discard_t + 1] > 0
                        and hand[discard_t + 2] > 0):
                    mask[72] = 1.0

        # パス — 始终合法
        mask[76] = 1.0

        return mask


# ── 主解析器 ──────────────────────────────────────────────────────────────

class MJSONRecordParser:
    """解析 MJSON 格式的对局记录，提取训练样本。

    用法:
        parser = MJSONRecordParser()
        samples = parser.parse_file("path/to/game.mjson")
        samples = parser.parse_directory("dataset/datasets_years/2026/", max_files=100)
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    @staticmethod
    def _is_gzipped(filepath: str) -> bool:
        with open(filepath, 'rb') as f:
            return f.read(2) == b'\x1f\x8b'

    def parse_file(self, filepath: str) -> List[TrainingSample]:
        """解析单个 .mjson 文件（自动检测 gzip / 纯文本）。"""
        events = []
        try:
            if self._is_gzipped(filepath):
                opener = gzip.open(filepath, 'rt', encoding='utf-8')
            else:
                opener = open(filepath, 'r', encoding='utf-8')
            with opener as f:
                for line in f:
                    line = line.strip()
                    if line:
                        events.append(json.loads(line))
        except (EOFError, json.JSONDecodeError) as e:
            if self.verbose:
                print(f"  Skipping {filepath}: {e}")
            return []
        return self._replay_game(events)

    def parse_directory(self, dirpath: str,
                         max_files: Optional[int] = None) -> List[TrainingSample]:
        """解析目录下所有 .mjson 文件（递归子目录）。"""
        all_samples = []
        pattern = os.path.join(dirpath, '**', '*.mjson')
        files = sorted(glob.glob(pattern, recursive=True))
        if max_files:
            files = files[:max_files]

        for i, fp in enumerate(files):
            if self.verbose and (i + 1) % 500 == 0:
                print(f"  Parsed {i+1}/{len(files)} files, "
                      f"{len(all_samples)} samples")
            all_samples.extend(self.parse_file(fp))

        if self.verbose:
            print(f"Parsed {len(files)} files, {len(all_samples)} samples total")
        return all_samples

    def _replay_game(self, events: List[dict]) -> List[TrainingSample]:
        """回放一局完整对局事件流，提取所有训练样本。"""
        tracker = MJSONGameTracker()
        samples: List[TrainingSample] = []

        for i, event in enumerate(events):
            etype = event.get('type', '')

            # ── 切牌决策点：在 dahai 之前捕获样本 ──
            if etype == 'dahai':
                actor = event['actor']
                tile_str = event.get('pai', '')
                tile_t = mjson_str_to_type(tile_str)

                state = tracker.build_state_tensor(actor)
                mask = tracker.build_draw_mask(actor)

                if tracker.pending_reach == actor:
                    # 立直宣言 + 切牌 → RIICHI_DISCARD
                    action = 37 + tile_t
                else:
                    action = tile_t  # DISCARD tile

                # 标记自摸如果合法（简化：如果事件流中下一个 actor 同玩家的
                # hora 事件 target==actor 则是自摸；这里我们检查后续事件）
                # 注意：在 dahai 时刻也应标记自摸选项
                next_events = events[i:]
                for ne in next_events:
                    nt = ne.get('type', '')
                    if nt == 'dahai' or nt == 'chi' or nt == 'pon' \
                            or nt == 'daiminkan' or nt == 'ankan' or nt == 'kakan' \
                            or nt == 'reach' or nt == 'hora' or nt == 'ryukyoku':
                        break
                if mask[tile_t] > 0 or mask[37 + tile_t] > 0:
                    samples.append(TrainingSample(
                        state_tensor=state,
                        action_mask=mask,
                        chosen_action=action,
                        player_idx=actor,
                    ))

            # 处理事件（更新状态）
            tracker.apply_event(event)

            # ── 鸣牌/和牌决策点：在动作事件后捕获 ──
            if etype in ('chi', 'pon', 'daiminkan'):
                actor = event['actor']
                # 重建动作前状态（需要回滚事件影响来重建状态）
                # 这里用 apply 后的状态简化处理
                tile_t = mjson_str_to_type(event.get('pai', ''))

                # 回滚以获取动作前的状态
                self._rollback_event(tracker, event)
                state = tracker.build_state_tensor(actor)
                mask = tracker.build_response_mask(actor)
                tracker.apply_event(event)  # 重新应用

                action_map = {'chi': 72, 'pon': 71, 'daiminkan': 73}
                action = action_map.get(etype, 76)
                mask[action] = 1.0  # 确保人类动作在掩码中

                samples.append(TrainingSample(
                    state_tensor=state,
                    action_mask=mask,
                    chosen_action=action,
                    player_idx=actor,
                ))

            elif etype == 'ankan':
                actor = event['actor']
                self._rollback_event(tracker, event)
                state = tracker.build_state_tensor(actor)
                mask = tracker.build_draw_mask(actor)
                mask[74] = 1.0
                tracker.apply_event(event)

                samples.append(TrainingSample(
                    state_tensor=state,
                    action_mask=mask,
                    chosen_action=74,
                    player_idx=actor,
                ))

            elif etype == 'kakan':
                actor = event['actor']
                self._rollback_event(tracker, event)
                state = tracker.build_state_tensor(actor)
                mask = tracker.build_draw_mask(actor)
                mask[75] = 1.0
                tracker.apply_event(event)

                samples.append(TrainingSample(
                    state_tensor=state,
                    action_mask=mask,
                    chosen_action=75,
                    player_idx=actor,
                ))

            elif etype == 'hora':
                actor = event['actor']
                target = event.get('target', actor)
                self._rollback_event(tracker, event)

                if actor == target:
                    # 自摸
                    state = tracker.build_state_tensor(actor)
                    mask = tracker.build_draw_mask(actor)
                    mask[34] = 1.0  # TSUMO
                    samples.append(TrainingSample(
                        state_tensor=state,
                        action_mask=mask,
                        chosen_action=34,
                        player_idx=actor,
                    ))
                else:
                    # 荣和
                    state = tracker.build_state_tensor(actor)
                    mask = tracker.build_response_mask(actor)
                    mask[35] = 1.0  # RON
                    samples.append(TrainingSample(
                        state_tensor=state,
                        action_mask=mask,
                        chosen_action=35,
                        player_idx=actor,
                    ))

                tracker.apply_event(event)

            # ── 其他玩家 pass 决策 ──
            elif etype == 'dahai':
                # 当 A 切牌后，为有响应能力但未鸣牌/和牌的玩家生成 pass 样本
                discarding_player = event['actor']
                responding_players: set = set()
                # 扫描后续事件，找出实际做出了响应的玩家
                for ne in events[i+1:]:
                    nt = ne.get('type', '')
                    if nt in ('chi', 'pon', 'daiminkan', 'hora', 'ankan', 'kakan'):
                        responding_players.add(ne.get('actor', -1))
                    elif nt in ('tsumo', 'dahai', 'reach', 'ryukyoku'):
                        break  # 响应窗口关闭
                for p_idx in range(4):
                    if p_idx == discarding_player:
                        continue
                    if tracker.is_riichi[p_idx]:
                        continue  # 立直者必须 pass
                    if p_idx in responding_players:
                        continue  # 已鸣牌/和牌
                    resp_mask = tracker.build_response_mask(p_idx)
                    # 仅当玩家确实有鸣牌/和牌选项（不只是纯 pass）时才记录
                    if resp_mask[35] > 0 or resp_mask[71] > 0 \
                            or resp_mask[72] > 0 or resp_mask[73] > 0:
                        state = tracker.build_state_tensor(p_idx)
                        samples.append(TrainingSample(
                            state_tensor=state,
                            action_mask=resp_mask,
                            chosen_action=76,  # PASS
                            player_idx=p_idx,
                        ))

        return samples

    def _rollback_event(self, tracker: MJSONGameTracker, event: dict):
        """回滚单个事件对 tracker 的状态修改。"""
        etype = event.get('type', '')
        # 简单的回滚：撤销分数变动和手牌变动
        if etype == 'hora':
            deltas = event.get('deltas', [0]*4)
            for i in range(4):
                tracker.scores[i] -= deltas[i]
            for ura_str in event.get('ura_markers', []):
                ura_t = mjson_str_to_type(ura_str)
                if ura_t >= 0 and ura_t in tracker.ura_indicators:
                    tracker.ura_indicators.remove(ura_t)
        elif etype == 'chi':
            actor = event['actor']
            consumed = [mjson_str_to_type(s) for s in event.get('consumed', [])]
            for t in consumed:
                if t >= 0:
                    tracker.hands[actor][t] += 1
            if tracker.melds[actor]:
                tracker.melds[actor].pop()
        elif etype == 'pon':
            actor = event['actor']
            consumed = [mjson_str_to_type(s) for s in event.get('consumed', [])]
            for t in consumed:
                if t >= 0:
                    tracker.hands[actor][t] += 1
            if tracker.melds[actor]:
                tracker.melds[actor].pop()
        elif etype == 'daiminkan':
            actor = event['actor']
            consumed = [mjson_str_to_type(s) for s in event.get('consumed', [])]
            for t in consumed:
                if t >= 0:
                    tracker.hands[actor][t] += 1
            if tracker.melds[actor]:
                tracker.melds[actor].pop()
        elif etype == 'ankan':
            actor = event['actor']
            consumed = [mjson_str_to_type(s) for s in event.get('consumed', [])]
            for t in consumed:
                if t >= 0:
                    tracker.hands[actor][t] += 1
            if tracker.melds[actor]:
                tracker.melds[actor].pop()
        elif etype == 'kakan':
            actor = event['actor']
            tile_str = event.get('pai', '')
            t = mjson_str_to_type(tile_str)
            if t >= 0:
                tracker.hands[actor][t] += 1
            if tracker.melds[actor]:
                tracker.melds[actor].pop()
