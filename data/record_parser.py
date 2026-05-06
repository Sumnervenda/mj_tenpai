"""JSONL 对局记录解析器 —— 从 engine 输出的 JSONL 记录中提取 (state, mask, action) 训练样本。"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np

from engine.tile import NUM_TYPES, abs_to_type, TILE_NAMES


@dataclass
class TrainingSample:
    """一个训练样本：某一时刻的状态 + 选择的动作。"""
    state_tensor: np.ndarray      # (354,) float32
    action_mask: np.ndarray       # (77,) int (0/1)
    chosen_action: int            # 0-76 动作索引
    player_idx: int               # 做出该动作的玩家


class JSONLRecordParser:
    """解析 engine 批量对局 JSONL 记录，重建 training samples。

    从 JSONL 中的 agari 事件 + game_summary 提取每局中每个玩家的
    状态张量和所选动作，用于监督学习预训练。
    """

    def __init__(self):
        pass

    def parse_file(self, filepath: str) -> List[TrainingSample]:
        """解析一个 JSONL 文件，返回所有训练样本。"""
        samples = []
        path = Path(filepath)
        with path.open('r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line.strip())
                extracted = self._extract_from_record(record)
                samples.extend(extracted)
        return samples

    def parse_directory(self, dirpath: str) -> List[TrainingSample]:
        """解析目录下所有 JSONL 文件。"""
        samples = []
        for p in Path(dirpath).glob('*.jsonl'):
            samples.extend(self.parse_file(str(p)))
        return samples

    def _extract_from_record(self, record: dict) -> List[TrainingSample]:
        """从一条 JSONL 记录中提取训练样本。"""
        record_type = record.get('type', '')

        if record_type == 'agari':
            return self._from_agari(record)
        elif record_type == 'game_summary':
            return self._from_game_summary(record)
        return []

    def _from_agari(self, record: dict) -> List[TrainingSample]:
        """从 agari 事件中提取和牌者最后状态的近似训练样本。

        注意：这仅重建和牌瞬间的状态，不是完整轨迹。
        """
        samples = []
        for winner in record.get('winners', []):
            player_data = winner.get('winner_hand', {})
            if not player_data:
                continue

            state = self._build_state_from_player_record(
                player_data, record, winner['winner'])
            if state is None:
                continue

            # 和牌动作索引
            if winner.get('win_type') == 'tsumo':
                action_idx = 34  # TSUMO
            else:
                action_idx = 35  # RON

            # 构建掩码（简化：仅标记合法动作，假设其他合法）
            mask = np.ones(77, dtype=np.float32)

            samples.append(TrainingSample(
                state_tensor=state,
                action_mask=mask,
                chosen_action=action_idx,
                player_idx=winner['winner'],
            ))
        return samples

    def _from_game_summary(self, record: dict) -> List[TrainingSample]:
        """从 game_summary 中获取终局状态（可用于价值头训练）。"""
        # game_summary 目前主要用于价值标注，策略训练主要靠 agari 事件
        return []

    def _build_state_from_player_record(self, player_data: dict,
                                         record: dict,
                                         player_idx: int) -> Optional[np.ndarray]:
        """从 JSONL 中的玩家数据重建 354 维状态张量。"""
        try:
            hand_counts = player_data.get('hand_counts', [0] * NUM_TYPES)
            hand = player_data.get('hand', [])
            melds = player_data.get('melds', [])
            discards_list = player_data.get('discards', [])

            state = np.zeros(354, dtype=np.float32)

            # 0-33: 手牌直方图
            if hand_counts:
                for i, c in enumerate(hand_counts[:34]):
                    state[i] = float(c)
            else:
                for tile_name in hand:
                    for i in range(NUM_TYPES):
                        if TILE_NAMES[i] == tile_name:
                            state[i] += 1.0

            # 34-67: 副露计数
            for meld in melds:
                for tile_name in meld.get('tiles', []):
                    for i in range(NUM_TYPES):
                        if TILE_NAMES[i] == tile_name:
                            state[34 + i] += 1.0

            # 68-101: 舍牌计数
            for disc in discards_list:
                for i in range(NUM_TYPES):
                    if TILE_NAMES[i] == disc:
                        state[68 + i] += 1.0

            # 102-135: 宝牌指示牌 —— 从记录中通常不可直接获取，留空
            # 136-339: 对手舍牌/副露 —— 需要完整四家信息，此处简化
            # 340-346: 元数据
            scores = record.get('scores_after_payment',
                                [25000, 25000, 25000, 25000])
            state[340] = float(scores[player_idx]) / 1000.0
            state[341] = float(record.get('honba', 0))
            state[342] = float(record.get('riichi_sticks', 0))
            state[343] = 0.5   # 剩余牌数未知，用中间值
            state[344] = float(player_data.get('riichi', False))
            state[345] = 0.0   # 场风从 round 字段推断
            state[346] = 1.0   # 局数

            # 347-349: 分差
            for opp in range(4):
                if opp != player_idx:
                    offset = 347 + (opp if opp < player_idx else opp - 1)
                    if offset < 350:
                        state[offset] = (float(scores[opp]) -
                                         float(scores[player_idx])) / 1000.0

            # 350-353: 自风 one-hot
            seat_idx = player_idx
            if seat_idx < 4:
                state[350 + seat_idx] = 1.0

            return state
        except Exception:
            return None
