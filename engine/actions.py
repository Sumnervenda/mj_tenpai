"""动作定义、合法动作计算与动作掩码向量。

动作是游戏引擎与 AI Agent 之间的接口（77 维离散动作空间）。

动作掩码向量布局（77 维）：
  索引 0-33:   切牌（Discard）每种牌型 0-33
  索引 34:     自摸（Tsumo）
  索引 35:     荣和（Ron）
  索引 36:     立直宣言按钮（Riichi）
  索引 37-70:  立直 + 切牌选择（Riichi + Discard）每种牌型
  索引 71:     碰（Pon）
  索引 72:     吃（Chi）
  索引 73:     大明槓（Daiminkan）
  索引 74:     暗槓（Ankan）
  索引 75:     加槓（Kakan）
  索引 76:     通过（Pass）
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, Tuple

from .tile import NUM_TYPES
from .hand import (
    Hand, Meld, MeldType,
    can_chi, can_pon, can_daiminkan, can_kakan, can_ankan,
)


# ── 动作类型枚举 ────────────────────────────────────────────────────────────

class ActionType(IntEnum):
    """日麻全部可能动作。"""
    DISCARD = 0        # 切牌（打出手中一张牌）
    TSUMO = 1          # 自摸和（自摸和了）
    RON = 2            # 栄和（荣和，别家放铳）
    RIICHI = 3         # 立直宣言（与切牌组合使用）
    PON = 4            # 碰（鸣别家弃牌成刻子）
    CHI = 5            # 吃（鸣上家弃牌成顺子）
    KAN_DAIMIN = 6     # 大明槓（鸣别家弃牌成槓子）
    KAN_ANKAN = 7      # 暗槓（门内四张成槓）
    KAN_KAKAN = 8      # 加槓（已有碰 → 加一张成槓）
    PASS = 9           # パス（不鸣牌）
    RYUUKYOKU = 10     # 流局（荒牌）—— 系统动作
    NUKI = 11          # 抜きドラ（三人麻将用，标准日麻不用）


@dataclass
class Action:
    """玩家可执行的具体动作。"""
    action_type: ActionType
    tile: int = -1          # 相关牌型（切牌/鸣牌/和了牌）
    meld_tiles: List[int] = field(default_factory=list)  # 吃时：3 张顺子组合
    actor: int = -1         # 执行玩家索引

    def __repr__(self):
        from .tile import TILE_NAMES
        tile_str = TILE_NAMES[self.tile] if self.tile >= 0 else "?"
        if self.action_type == ActionType.DISCARD:       return f"切[{tile_str}]"
        elif self.action_type == ActionType.TSUMO:       return f"ツモ[{tile_str}]"
        elif self.action_type == ActionType.RON:         return f"ロン[{tile_str}]"
        elif self.action_type == ActionType.RIICHI:      return f"立直→切[{tile_str}]"
        elif self.action_type == ActionType.PON:         return f"碰[{tile_str}]"
        elif self.action_type == ActionType.CHI:
            meld_s = "".join(TILE_NAMES[t] for t in self.meld_tiles)
            return f"吃{meld_s}"
        elif self.action_type == ActionType.KAN_DAIMIN:  return f"明槓[{tile_str}]"
        elif self.action_type == ActionType.KAN_ANKAN:   return f"暗槓[{tile_str}]"
        elif self.action_type == ActionType.KAN_KAKAN:   return f"加槓[{tile_str}]"
        elif self.action_type == ActionType.PASS:        return "パス"
        return str(self.action_type)


# ── 动作空间定义 ────────────────────────────────────────────────────────────

MAX_ACTIONS = 128
ACTION_SPACE_SIZE = 77      # 实际使用的动作掩码长度

# 动作掩码子区间
DISCARD_OFFSET = 0           # 切牌: 0-33
TSUMO_INDEX = 34             # 自摸
RON_INDEX = 35               # 荣和
RIICHI_INDEX = 36            # 立直按钮
RIICHI_DISCARD_OFFSET = 37   # 立直切牌: 37-70
PON_INDEX = 71               # 碰
CHI_INDEX = 72               # 吃
KAN_DAIMIN_INDEX = 73        # 大明槓
KAN_ANKAN_INDEX = 74         # 暗槓
KAN_KAKAN_INDEX = 75         # 加槓
PASS_INDEX = 76              # 通过


def create_action_mask() -> List[int]:
    """创建全零的动作掩码向量。"""
    return [0] * ACTION_SPACE_SIZE


def set_discard_actions(mask: List[int], hand: List[int]) -> None:
    """标记手牌中持有的牌型为合法切牌。"""
    for t in range(NUM_TYPES):
        if hand[t] > 0:
            mask[DISCARD_OFFSET + t] = 1


def set_tsumo_action(mask: List[int], can_tsumo: bool) -> None:
    """标记自摸是否合法。"""
    mask[TSUMO_INDEX] = 1 if can_tsumo else 0


def set_ron_action(mask: List[int], can_ron: bool) -> None:
    """标记荣和是否合法。"""
    mask[RON_INDEX] = 1 if can_ron else 0


def set_riichi_actions(mask: List[int], riichi_discards: List[int]) -> None:
    """标记立直切牌选择（索引 37-70，每个索引绑定具体切牌）。

    注：不开放 mask[36]（独立立直按钮），因当前动作空间用 37-70 表示完整立直+切牌。
    """
    if riichi_discards:
        for t in riichi_discards:
            mask[RIICHI_DISCARD_OFFSET + t] = 1


def set_call_actions(mask: List[int],
                     can_pon: bool = False,
                     chi_options: Optional[List[List[int]]] = None,
                     can_daiminkan: bool = False,
                     can_ron: bool = False) -> None:
    """标记鸣牌（碰/吃/槓/荣和）动作。"""
    mask[RON_INDEX] = 1 if can_ron else 0
    mask[PON_INDEX] = 1 if can_pon else 0
    mask[CHI_INDEX] = 1 if chi_options else 0
    mask[KAN_DAIMIN_INDEX] = 1 if can_daiminkan else 0
    mask[PASS_INDEX] = 1  # 始终可パス


def set_kan_actions(mask: List[int],
                    ankan_options: List[int],
                    kakan_options: List[int]) -> None:
    """标记杠动作（暗槓/加槓）。"""
    mask[KAN_ANKAN_INDEX] = 1 if ankan_options else 0
    mask[KAN_KAKAN_INDEX] = 1 if kakan_options else 0


# ── 合法动作计算 ────────────────────────────────────────────────────────────

@dataclass
class LegalActions:
    """当前玩家在当前状态的全部合法动作。"""
    actions: List[Action]              # 动作列表
    mask: List[int]                    # 77 维掩码向量（供神经网络使用）

    def __repr__(self):
        return str(self.actions)


def compute_draw_actions(
    hand: List[int],
    is_menzen: bool,
    can_tsumo: bool,
    can_riichi: bool,
    riichi_discards: List[int],
    ankan_options: List[int],
    kakan_options: List[int],
    last_drawn_tile: int = -1,
    is_drawn_tile_available: bool = True,
) -> LegalActions:
    """计算摸牌阶段（DRAW_STATE）玩家的全部合法动作。

    玩家必须选择：切牌（可附立直）、自摸、或杠。
    """
    actions = []
    mask = create_action_mask()

    # 自摸
    if can_tsumo:
        actions.append(Action(ActionType.TSUMO, tile=last_drawn_tile))
        mask[TSUMO_INDEX] = 1

    # 立直 + 切牌组合（动作索引 37-70，每个索引绑定具体切牌）
    # 注：不开放 mask[36]（独立立直按钮），因当前动作空间用 37-70 表示完整立直+切牌
    riichi_tiles = set(riichi_discards) if can_riichi else set()
    if riichi_tiles:
        for t in riichi_tiles:
            actions.append(Action(ActionType.RIICHI, tile=t))
            mask[RIICHI_DISCARD_OFFSET + t] = 1

    # 普通切牌（立直可切牌只开放 riichi discard 通道，不开放普通 discard）
    for t in range(NUM_TYPES):
        if hand[t] > 0 and t not in riichi_tiles:
            mask[DISCARD_OFFSET + t] = 1
            actions.append(Action(ActionType.DISCARD, tile=t))

    # 暗槓
    for t in ankan_options:
        actions.append(Action(ActionType.KAN_ANKAN, tile=t))
    mask[KAN_ANKAN_INDEX] = 1 if ankan_options else 0

    # 加槓
    for t in kakan_options:
        actions.append(Action(ActionType.KAN_KAKAN, tile=t))
    mask[KAN_KAKAN_INDEX] = 1 if kakan_options else 0

    return LegalActions(actions=actions, mask=mask)


def compute_response_actions(
    hand: List[int],
    source_tile: int,
    source_player: int,
    my_position: int,
    can_ron: bool = False,
    chi_options: Optional[List[List[int]]] = None,
) -> LegalActions:
    """计算别家切牌后，我方的合法响应动作。

    优先级由游戏引擎仲裁（不在此处处理）：
      1. 荣和（多家可）
      2. 槓 / 碰
      3. 吃（仅上家）
      4. パス
    """
    chi_options = chi_options or []
    can_pon_flag = can_pon(hand, source_tile)
    can_daiminkan_flag = can_daiminkan(hand, source_tile)

    # 吃仅限上家（下家）
    is_left = (source_player + 1) % 4 == my_position
    effective_chi = chi_options if is_left else []

    actions = []
    mask = create_action_mask()

    if can_ron:
        actions.append(Action(ActionType.RON, tile=source_tile))
        mask[RON_INDEX] = 1

    if can_pon_flag:
        actions.append(Action(ActionType.PON, tile=source_tile))
        mask[PON_INDEX] = 1

    if can_daiminkan_flag:
        actions.append(Action(ActionType.KAN_DAIMIN, tile=source_tile))
        mask[KAN_DAIMIN_INDEX] = 1

    if effective_chi:
        for meld in effective_chi:
            actions.append(Action(ActionType.CHI, tile=source_tile, meld_tiles=meld))
        mask[CHI_INDEX] = 1

    # 始终可パス
    actions.append(Action(ActionType.PASS))
    mask[PASS_INDEX] = 1

    return LegalActions(actions=actions, mask=mask)
# 中文注释：定义麻将动作、动作空间编码以及合法动作 mask 的构造逻辑。
