"""牌山（Wall）—— 洗牌、发牌、摸牌、王牌 / 宝牌管理。

136 张牌中，最后 14 张为王牌（王牌 = 宝牌指示牌 10 张 + 嶺上牌 4 张）。

王牌布局（索引 122～135）：
  122～131：宝牌指示牌（5 组：每组含 1 张表宝牌 + 1 张里宝牌指示牌）
  132～135：嶺上牌（杠后补充牌，从后往前摸）
"""

import random
from typing import List, Optional, Tuple

from .tile import (
    NUM_ABS, COPIES_PER_TYPE, AKA_TYPES, AKA_COPY_INDEX,
    abs_to_type, is_aka, all_abs_ids,
)

# ── 常量 ─────────────────────────────────────────────────────────────────────

TOTAL_TILES = NUM_ABS                     # 136
DEAD_WALL_SIZE = 14                       # 王牌数量
DEAD_WALL_START = TOTAL_TILES - DEAD_WALL_SIZE  # 122（王牌起始位置）
RINSHAN_SIZE = 4                          # 嶺上牌数量
DORA_PAIR_COUNT = 5                       # 5 组表/里宝牌指示牌
MAX_DORA_INDICATORS = DORA_PAIR_COUNT     # 最多翻 5 次宝牌

# 王牌内部索引
DORA_INDICATOR_BASE = DEAD_WALL_START                     # 122
RINSHAN_BASE = DEAD_WALL_START + 2 * DORA_PAIR_COUNT      # 132


class Wall:
    """牌山，管理摸牌和宝牌。

    用法示例：
        wall = Wall()
        wall.shuffle()               # 洗牌
        hands = wall.deal()          # 发牌 → 4 手 (14, 13, 13, 13)
        tile = wall.draw()           # 从牌山摸牌
        rinshan = wall.draw_rinshan()# 从嶺上牌摸牌（杠后）
        dora = wall.flip_dora()      # 翻下一张宝牌指示牌
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)                 # 可复现的随机数
        self.tiles: List[int] = []                     # 牌数组（绝对 ID）
        self._live_ptr: int = 0                        # 牌山摸牌指针
        self._rinshan_ptr: int = 3                     # 嶺上牌指针（从后往前：3,2,1,0）
        self._dora_count: int = 0                      # 已翻宝牌指示牌数量
        self.dora_indicators: List[int] = []           # 已翻开的宝牌指示牌（绝对 ID）

        self.reset()

    def reset(self) -> None:
        """重置牌山，按绝对 ID 顺序排列。"""
        self.tiles = all_abs_ids()
        # 赤宝牌自动位于 AKA_COPY_INDEX 位置（copy_index=3）
        # 即 type_id*4+3 的牌为赤牌，通过 is_aka() 识别
        self._live_ptr = 0
        self._rinshan_ptr = 3
        self._dora_count = 0
        self.dora_indicators = []

    def shuffle(self) -> None:
        """洗牌：随机打乱牌山顺序。"""
        self.rng.shuffle(self.tiles)

    # ── 发牌 ──────────────────────────────────────────────────────────────

    def deal(self) -> Tuple[List[int], List[int], List[int], List[int]]:
        """发牌给 4 家。返回 (東家, 南家, 西家, 北家)。

        发牌流程：3 轮每轮每家 4 张 → 每家 13 张 → 东家再摸 1 张（共 14 张）。
        """
        hands: List[List[int]] = [[] for _ in range(4)]

        # 3 轮：每轮每家 4 张
        for _ in range(3):
            for p in range(4):
                for _ in range(4):
                    hands[p].append(self._take_live())

        # 每家再发 1 张
        for p in range(4):
            hands[p].append(self._take_live())

        # 东家（庄家）额外 1 张
        hands[0].append(self._take_live())

        return hands[0], hands[1], hands[2], hands[3]

    # ── 摸牌 ──────────────────────────────────────────────────────────────

    def draw(self) -> int:
        """从牌山摸一张牌。摸尽时抛出 IndexError。"""
        if self._live_ptr >= DEAD_WALL_START:
            raise IndexError("牌山已摸尽")
        return self._take_live()

    def draw_rinshan(self) -> int:
        """从嶺上牌摸一张（杠后补充牌）。"""
        if self._rinshan_ptr < 0:
            raise IndexError("嶺上牌已摸尽")
        idx = RINSHAN_BASE + self._rinshan_ptr
        self._rinshan_ptr -= 1
        return self.tiles[idx]

    def _take_live(self) -> int:
        """内部：从牌山取下一张。"""
        tile = self.tiles[self._live_ptr]
        self._live_ptr += 1
        return tile

    # ── 宝牌管理 ──────────────────────────────────────────────────────────

    def flip_dora(self) -> Optional[int]:
        """翻开下一张宝牌指示牌。返回其绝对 ID，已翻满时返回 None。

        第 1 次调用：开局翻初始表宝牌。
        之后每次杠后：翻新的宝牌指示牌 → 再摸嶺上牌。
        """
        if self._dora_count >= MAX_DORA_INDICATORS:
            return None
        indicator_idx = DORA_INDICATOR_BASE + 2 * self._dora_count
        self._dora_count += 1
        indicator = self.tiles[indicator_idx]
        self.dora_indicators.append(indicator)
        return indicator

    def flip_ura_dora(self, count: int) -> List[int]:
        """立直和牌后翻开里宝牌指示牌。返回里宝牌指示牌绝对 ID 列表。"""
        result = []
        for i in range(min(count, self._dora_count)):
            ura_idx = DORA_INDICATOR_BASE + 2 * i + 1
            result.append(self.tiles[ura_idx])
        return result

    def get_dora_types(self, with_ura: bool = False,
                       ura_dora_indicators: Optional[List[int]] = None) -> List[int]:
        """返回当前所有宝牌的类型 ID 列表。

        宝牌 = 指示牌的"下一张"：
          - 数牌：同花色内数字 +1（9 的下一张是 1）
          - 风牌：東→南→西→北→東
          - 三元牌：白→発→中→白

        with_ura=True 时同时返回里宝牌。
        """
        indicators = list(self.dora_indicators)
        if with_ura and ura_dora_indicators:
            indicators.extend(ura_dora_indicators)
        return [_dora_type(indicator) for indicator in indicators]


def _dora_type(indicator_abs: int) -> int:
    """给定宝牌指示牌的绝对 ID，计算宝牌的类型 ID。

    宝牌是"下一张"牌型：
      - 数牌 0～8, 9～17, 18～26：indicator + 1 同花色内循环
      - 风牌 27～30：東→南→西→北→東
      - 三元牌 31～33：白→発→中→白
    """
    t = abs_to_type(indicator_abs)
    if t <= 7:        return t + 1     # 1～8万/筒/条
    elif t == 8:      return 0         # 9万 → 1万
    elif t <= 16:     return t + 1     # 1～8筒
    elif t == 17:     return 9         # 9筒 → 1筒
    elif t <= 25:     return t + 1     # 1～8条
    elif t == 26:     return 18        # 9条 → 1条
    elif t <= 29:     return t + 1     # 東→南→西→北
    elif t == 30:     return 27        # 北→東
    elif t == 31:     return 32        # 白→発
    elif t == 32:     return 33        # 発→中
    else:             return 31        # 中→白


def count_dora_in_hand(hand_types: List[int],
                       dora_types: List[int],
                       aka_types: Optional[List[int]] = None) -> int:
    """计算手牌中的宝牌数量。

    每张匹配宝牌类型的牌 +1 宝，赤宝牌额外 +1（不依赖指示牌）。
    """
    count = 0
    for t in hand_types:
        if t in dora_types:
            count += 1
    if aka_types:
        count += sum(1 for t in aka_types if t in AKA_TYPES)
    return count


def lives_remaining(wall: Wall) -> int:
    """牌山中剩余的可摸牌数量（不含王牌）。"""
    return DEAD_WALL_START - wall._live_ptr


def is_last_tile(wall: Wall) -> bool:
    """判断下一张是否为海底牌（牌山最后一张）。"""
    return wall._live_ptr == DEAD_WALL_START - 1
