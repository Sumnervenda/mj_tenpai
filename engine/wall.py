"""牌山（Wall）—— 预构建牌山、发牌、摸牌、王牌 / 宝牌管理。

牌山结构（136 张牌，游戏开始前一次性预构建，顺序固定）：

  索引 0～51 （52 张）：4 家初始手牌（各 13 张）
  索引 52～121（70 张）：可摸牌墙 —— 玩家摸牌区
  索引 122～135（14 张）：王牌（玩家不可摸）

王牌布局（122～135）：
  122～131：宝牌指示牌（5 组：每组 1 张表宝牌 + 1 张里宝牌指示牌）
  132～135：嶺上牌（杠后补充摸牌）

关键设计：
  - 摸牌顺序完全由预构建决定，玩家操作只影响"谁"摸下一张，不影响"什么牌"
  - 摸岭上牌后，可摸牌墙末尾一张牌自动补入王牌，维持王牌始终为 14 张
  - 因此可摸牌数随岭上摸牌递减（初始 70 → 每杠 -1）
"""

import random
from typing import List, Optional, Tuple

from .tile import (
    NUM_ABS, COPIES_PER_TYPE, AKA_TYPES, AKA_COPY_INDEX,
    abs_to_type, is_aka, all_abs_ids,
)

# ── 常量 ─────────────────────────────────────────────────────────────────────

TOTAL_TILES = NUM_ABS                     # 136
TILES_PER_HAND = 13                       # 每人手牌数
DEAL_SIZE = TILES_PER_HAND * 4            # 52（发牌总数）
DEAD_WALL_SIZE = 14                       # 王牌数量
DRAWABLE_SIZE = TOTAL_TILES - DEAL_SIZE - DEAD_WALL_SIZE  # 70

DRAWABLE_START = DEAL_SIZE                # 52（可摸牌墙起始）
DEAD_WALL_START = DEAL_SIZE + DRAWABLE_SIZE  # 122（王牌起始位置，初始值）
RINSHAN_SIZE = 4                          # 嶺上牌数量
DORA_PAIR_COUNT = 5                       # 5 组表/里宝牌指示牌
MAX_DORA_INDICATORS = DORA_PAIR_COUNT     # 最多翻 5 次宝牌

# 王牌内部固定索引
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
        self._live_ptr: int = 0                        # 摸牌指针（可摸牌墙位置）
        self._rinshan_ptr: int = 3                     # 嶺上牌指针（从后往前：3,2,1,0）
        self._dora_count: int = 0                      # 已翻宝牌指示牌数量
        self.dora_indicators: List[int] = []           # 已翻开的宝牌指示牌（绝对 ID）
        self._dead_wall_start: int = DEAD_WALL_START   # 王牌起始位置（可变，岭上牌后递减）

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
        self._dead_wall_start = DEAD_WALL_START

    def shuffle(self) -> None:
        """洗牌：随机打乱牌山顺序。"""
        self.rng.shuffle(self.tiles)

    # ── 发牌 ──────────────────────────────────────────────────────────────

    def deal(self) -> Tuple[List[int], List[int], List[int], List[int]]:
        """发牌给 4 家。返回 (東家, 南家, 西家, 北家)，各 13 张。

        发牌流程：3 轮每轮每家 4 张 → 每家 13 张（无庄家特权）。
        庄家的第 14 张将在 _start_round 中从可摸牌墙摸取。
        """
        hands: List[List[int]] = [[] for _ in range(4)]

        # 3 轮：每轮每家 4 张
        for _ in range(3):
            for p in range(4):
                for _ in range(4):
                    hands[p].append(self._take_live())

        # 每家再发 1 张（共 13 张 / 家）
        for p in range(4):
            hands[p].append(self._take_live())

        return tuple(hands)

    # ── 摸牌 ──────────────────────────────────────────────────────────────

    def draw(self) -> int:
        """从可摸牌墙摸一张牌。摸尽（触及王牌区域）时抛出 IndexError。"""
        if self._live_ptr >= self._dead_wall_start:
            raise IndexError("牌山已摸尽")
        return self._take_live()

    def draw_rinshan(self) -> int:
        """从嶺上牌摸一张（杠后补充牌）。

        岭上牌从王牌末尾向前摸（135→134→133→132）。
        摸完后将可摸牌墙末尾一张牌补入王牌，维持王牌始终为 14 张。
        """
        if self._rinshan_ptr < 0:
            raise IndexError("嶺上牌已摸尽")
        idx = RINSHAN_BASE + self._rinshan_ptr
        self._rinshan_ptr -= 1
        # 维持王牌数量：将可摸牌墙末尾一张牌并入王牌
        self._dead_wall_start -= 1
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
                       dora_types: List[int]) -> int:
    """计算手牌中的宝牌数量。

    每张匹配宝牌类型的牌 +1 宝。
    赤宝牌由调用方通过 len(player.hand.aka_tiles) 单独计算。
    """
    count = 0
    for t in hand_types:
        if t in dora_types:
            count += 1
    return count


def lives_remaining(wall: Wall) -> int:
    """牌山中剩余的可摸牌数量（不含王牌）。"""
    return wall._dead_wall_start - wall._live_ptr


def is_last_tile(wall: Wall) -> bool:
    """判断下一张是否为海底牌（牌山最后一张）。"""
    return wall._live_ptr == wall._dead_wall_start - 1
# 中文注释：实现牌山、王牌、宝牌指示牌、发牌和摸牌等牌堆行为。
