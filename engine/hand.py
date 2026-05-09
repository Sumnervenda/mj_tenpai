"""手牌表示 —— int[34] 直方图 + 副露管理。

核心数据结构：
  - tiles[34]: 每种牌型在手牌中的数量（最大 4），只含门内（未副露）的牌
  - melds: 副露列表（吃/碰/明槓/暗槓）
  - aka_tiles: 手牌中赤宝牌的绝对 ID 列表（用于宝牌计数）

暗槓：从手牌中移除 4 张牌，加入 melds（标记为暗槓）
明槓/碰/吃：从手牌中移除相应数量，加入 melds
"""

from copy import deepcopy
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, Set, Tuple

from .tile import (
    NUM_TYPES, MANZU, PINZU, SOUZU, JIHAI,
    abs_to_type, is_aka, is_jihai, YAOCHUHAI_TYPES,
)


class MeldType(IntEnum):
    """副露类型。"""
    CHI = 0          # 吃（顺子，只能明）
    PON = 1          # 碰（刻子，明）
    KAN_CLOSED = 2   # 暗槓（门内杠）
    KAN_OPEN = 3     # 明槓（加槓，由碰升杠）
    KAN_DAIMIN = 4   # 大明槓（从别家切牌直接杠）


@dataclass
class Meld:
    """一个副露面子（吃/碰/杠）。

    Attributes:
        meld_type: 副露类型
        tiles: 面子中的牌类型列表（3 或 4 张）
        called_from: 从哪家收的牌（-1 表示门内）
        source_tile: 从别家收的那张牌的类型 ID
    """
    meld_type: MeldType
    tiles: List[int]        # 3 或 4 张牌的类型 ID 列表
    called_from: int = -1   # 被收牌玩家的索引（-1 表示门内暗槓）
    source_tile: int = -1   # 从别家收的那张牌的类型

    @property
    def is_open(self) -> bool:
        """是否为明面（对手可见）的副露。暗槓不是明面。"""
        return self.meld_type != MeldType.KAN_CLOSED

    @property
    def is_kan(self) -> bool:
        """是否为杠子。"""
        return self.meld_type in (MeldType.KAN_CLOSED, MeldType.KAN_OPEN, MeldType.KAN_DAIMIN)

    @property
    def tile_type(self) -> int:
        """面子代表牌型。"""
        return self.tiles[0] if self.meld_type == MeldType.CHI else self.tiles[0]

    @property
    def size(self) -> int:
        """面子牌数（3 或 4）。"""
        return len(self.tiles)


@dataclass
class Hand:
    """玩家手牌：门内牌直方图 + 副露列表。

    Usage:
        h = Hand()
        h.add(tile_type, abs_id)    # 摸入一张牌
        h.remove(tile_type)         # 切出一张牌
        h.add_meld(meld)            # 添加副露
        h.is_menzen                 # 门前清判定
    """

    tiles: List[int] = field(default_factory=lambda: [0] * NUM_TYPES)  # 门内牌 int[34]
    melds: List[Meld] = field(default_factory=list)                    # 副露列表
    aka_tiles: List[int] = field(default_factory=list)                 # 赤宝牌绝对 ID

    def __init__(self):
        self.tiles = [0] * NUM_TYPES
        self.melds = []
        self.aka_tiles = []

    # ── 访问 ──────────────────────────────────────────────────────────────

    def count(self, tile_type: int) -> int:
        """门内某牌型的持有张数。"""
        return self.tiles[tile_type]

    def total_concealed(self) -> int:
        """门内牌总数。"""
        return sum(self.tiles)

    def total_in_melds(self) -> int:
        """副露中的牌总数。"""
        return sum(m.size for m in self.melds)

    def total_tiles(self) -> int:
        """总牌数（门内 + 副露）。"""
        return self.total_concealed() + self.total_in_melds()

    @property
    def is_menzen(self) -> bool:
        """门前清判定：无明面副露即为门清（暗槓不破门清）。"""
        return all(not m.is_open for m in self.melds)

    @property
    def open_meld_count(self) -> int:
        """明面副露数量。"""
        return sum(1 for m in self.melds if m.is_open)

    # ── 修改操作 ──────────────────────────────────────────────────────────

    def add(self, tile_type: int, abs_id: int = -1) -> None:
        """摸入一张牌，加入门内。自动追踪赤宝牌。"""
        if self.tiles[tile_type] >= 4:
            raise ValueError(f"牌型 {tile_type} 已有 4 张，无法再摸入")
        self.tiles[tile_type] += 1
        if is_aka(abs_id):
            self.aka_tiles.append(abs_id)

    def remove(self, tile_type: int, abs_id: int = -1) -> None:
        """从门内移除一张牌（切牌或副露消耗）。"""
        if self.tiles[tile_type] <= 0:
            raise ValueError(f"牌型 {tile_type} 手牌中无此牌，无法移除")
        self.tiles[tile_type] -= 1
        if is_aka(abs_id) and abs_id in self.aka_tiles:
            self.aka_tiles.remove(abs_id)

    def add_meld(self, meld: Meld) -> None:
        """添加副露面子，并从门内移除相应牌。

        不同类型消耗门内牌数：
          - 暗槓：消耗 4 张
          - 加槓：消耗 1 张（原有碰的 3 张已在副露中）
          - 碰：消耗 2 张（第 3 张从别家收）
          - 大明槓：消耗 3 张（第 4 张从别家收）
          - 吃：消耗 2 张（第 3 张从别家收）
        """
        if meld.meld_type == MeldType.KAN_CLOSED:
            t = meld.tile_type
            if self.tiles[t] < 4:
                raise ValueError(f"暗槓需要 4 张 {t}，但手牌只有 {self.tiles[t]} 张")
            self.tiles[t] -= 4
        elif meld.meld_type == MeldType.KAN_OPEN:
            t = meld.tile_type
            if self.tiles[t] < 1:
                raise ValueError(f"加槓需要 1 张 {t}")
            self.tiles[t] -= 1
        elif meld.meld_type in (MeldType.PON, MeldType.KAN_DAIMIN):
            t = meld.tile_type
            needed = 2 if meld.meld_type == MeldType.PON else 3
            if self.tiles[t] < needed:
                raise ValueError(f"碰/明槓需要 {needed} 张 {t}，但手牌只有 {self.tiles[t]} 张")
            self.tiles[t] -= needed
        elif meld.meld_type == MeldType.CHI:
            called = meld.source_tile
            for t in meld.tiles:
                if t != called:
                    if self.tiles[t] < 1:
                        raise ValueError(f"吃需要牌型 {t}")
                    self.tiles[t] -= 1
        self.melds.append(meld)

    # ── 花色视图 ──────────────────────────────────────────────────────────

    def suit_counts(self, suit_range: range) -> List[int]:
        """获取特定花色的手牌计数列表。"""
        return [self.tiles[i] for i in suit_range]

    @property
    def manzu(self) -> List[int]:  return self.suit_counts(MANZU)
    @property
    def pinzu(self) -> List[int]:  return self.suit_counts(PINZU)
    @property
    def souzu(self) -> List[int]:  return self.suit_counts(SOUZU)
    @property
    def jihai_counts(self) -> List[int]:  return self.suit_counts(JIHAI)

    def non_zero_types(self) -> List[int]:
        """返回门内所有非零牌型列表。"""
        return [i for i, c in enumerate(self.tiles) if c > 0]

    # ── 转换 ──────────────────────────────────────────────────────────────

    def to_type_list(self) -> List[int]:
        """将直方图展开为平铺的类型 ID 列表（仅门内）。"""
        result = []
        for t in range(NUM_TYPES):
            result.extend([t] * self.tiles[t])
        return result

    @classmethod
    def from_type_list(cls, types: List[int]) -> "Hand":
        """从类型 ID 列表构建手牌。"""
        h = cls()
        for t in types:
            h.add(t)
        return h

    def copy(self) -> "Hand":
        """深拷贝手牌。"""
        return deepcopy(self)

    # ── 显示 ──────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        from .tile import TILE_NAMES
        parts = []
        for t in range(NUM_TYPES):
            if self.tiles[t] > 0:
                parts.append(f"{TILE_NAMES[t]}×{self.tiles[t]}")
        if self.melds:
            parts.append("|")
            for m in self.melds:
                mt = "".join(TILE_NAMES[tt] for tt in m.tiles)
                parts.append(f"[{mt}]")
        return " ".join(parts) if parts else "<空手>"


# ── 副露可行性判断 ──────────────────────────────────────────────────────────

def can_chi(hand_tiles: List[int], source_tile: int) -> List[List[int]]:
    """判断手牌能否吃上家打出的 source_tile。返回所有可行的顺子组合。

    source_tile 在顺子中可能有 3 种位置：低、中、高。
    返回值为 List[List[int]]，每个内层列表为 3 张牌的顺子组合。
    """
    if is_jihai(source_tile):
        return []

    suit_start = (source_tile // 9) * 9
    num = source_tile - suit_start  # 0～8
    results = []

    # source 为顺子低张 (n, n+1, n+2)
    if num <= 6:
        if hand_tiles[source_tile + 1] > 0 and hand_tiles[source_tile + 2] > 0:
            results.append([source_tile, source_tile + 1, source_tile + 2])
    # source 为顺子中张 (n-1, n, n+1)
    if 1 <= num <= 7:
        if hand_tiles[source_tile - 1] > 0 and hand_tiles[source_tile + 1] > 0:
            results.append([source_tile - 1, source_tile, source_tile + 1])
    # source 为顺子高张 (n-2, n-1, n)
    if num >= 2:
        if hand_tiles[source_tile - 2] > 0 and hand_tiles[source_tile - 1] > 0:
            results.append([source_tile - 2, source_tile - 1, source_tile])

    return results


def can_pon(hand_tiles: List[int], source_tile: int) -> bool:
    """判断能否碰（手牌已有 ≥2 张）。"""
    return hand_tiles[source_tile] >= 2


def can_daiminkan(hand_tiles: List[int], source_tile: int) -> bool:
    """判断能否大明槓（手牌已有 ≥3 张）。"""
    return hand_tiles[source_tile] >= 3


def can_kakan(hand_tiles: List[int], melds: List[Meld]) -> List[int]:
    """返回可以加槓（碰 → 槓）的牌型列表。

    条件：已有碰的副露，且门内有至少 1 张同种牌。
    """
    result = []
    for meld in melds:
        if meld.meld_type == MeldType.PON:
            t = meld.tile_type
            if hand_tiles[t] >= 1:
                result.append(t)
    return result


def can_ankan(hand_tiles: List[int]) -> List[int]:
    """返回可以暗槓的牌型列表（门内有 4 张同种牌）。"""
    return [t for t in range(NUM_TYPES) if hand_tiles[t] >= 4]


def count_aka_in_list(tile_list: List[int], abs_ids: List[int]) -> int:
    """计算绝对 ID 列表中的赤宝牌数量。"""
    return sum(1 for aid in abs_ids if is_aka(aid))
# 中文注释：封装玩家手牌、鸣牌、弃牌和手牌增删查等低层数据结构。
