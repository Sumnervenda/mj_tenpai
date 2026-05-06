"""点数计算 —— 符（フ）计算 + 翻数 → 得点表 + 授受精算。

流程：
  1. 根据手牌分解 + 和牌方式计算符数（20～110，向上取整到 10）
  2. 查翻数 × 符数 → 基本点
  3. 荣和：放铳者全额支付；自摸：三家分摊
  4. 终局：马点（ウマ）+ オカ调整

分数变动可作为强化学习的即时奖励信号（Delta Score）。
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .tile import (
    NUM_TYPES, YAOCHUHAI_TYPES, ROTOHAI_TYPES,
    is_jihai, is_kazehai, is_sangenhai, is_yaochuhai,
)
from .hand import Meld, MeldType


# ── 符（フ）计算 ───────────────────────────────────────────────────────────

def calculate_fu(
    melds: List[Meld],
    pair: Tuple[int, int],
    winning_tile: int,
    is_menzen: bool,
    is_tsumo: bool,
    is_pinfu: bool,
    bakaze: int,
    jikaze: int,
    wait_type: str = 'ryanmen',  # 'ryanmen' | 'kanchan' | 'penchan' | 'tanki'
) -> int:
    """计算和牌的符数。

    基准 20 符。加算：
      + 门清荣和 10 符
      + 自摸 2 符（平和门清自摸除外）
      + 面子符（明刻 2/4, 暗刻 4/8, 明槓 8/16, 暗槓 16/32）
      + 雀头符（役牌对子 +2）
      + 听牌形符（嵌張/辺張/単騎 +2）
      → 向上取整到 10 的倍数（平和自摸固定 20 符）
    """
    fu = 20

    # 门清荣和 = 30 符基准
    if is_menzen and not is_tsumo:
        fu += 10

    # 自摸 +2 符（平和除外）
    if is_tsumo and not (is_menzen and is_pinfu):
        fu += 2

    # 副露面子的符
    for meld in melds:
        is_terminal = is_yaochuhai(meld.tile_type)
        if meld.meld_type == MeldType.CHI:
            pass  # 顺子 0 符
        elif meld.meld_type == MeldType.PON:
            fu += 2 if not is_terminal else 4
        elif meld.meld_type == MeldType.KAN_CLOSED:
            fu += 16 if not is_terminal else 32
        elif meld.meld_type in (MeldType.KAN_OPEN, MeldType.KAN_DAIMIN):
            fu += 8 if not is_terminal else 16

    # 雀头符
    pair_tile = pair[0]
    if _is_yakuhai_pair(pair_tile, bakaze, jikaze):
        fu += 2

    # 听牌形符
    if wait_type in ('kanchan', 'penchan', 'tanki'):
        fu += 2

    # 平和门清自摸 = 20 符固定
    if is_menzen and is_tsumo and is_pinfu:
        return 20

    # 向上取整到 10 的倍数
    return int(math.ceil(fu / 10.0)) * 10


def calculate_fu_from_decomp(
    concealed_melds: List[Tuple[str, List[int]]],  # 分解结果中的面子
    open_melds: List[Meld],                         # 副露面子的符
    pair: Tuple[int, int],
    is_menzen: bool,
    is_tsumo: bool,
    is_pinfu: bool,
    bakaze: int,
    jikaze: int,
    wait_type: str = 'ryanmen',
) -> int:
    """通过完整分解结果计算符数。

    concealed_melds: decompose_hand() 的 melds 字段输出。
    """
    fu = 20

    if is_menzen and not is_tsumo:
        fu += 10
    if is_tsumo and not (is_menzen and is_pinfu):
        fu += 2

    # 门内面子的符
    for mtype, tiles in concealed_melds:
        if mtype == 'shuntsu':
            continue  # 顺子 0 符
        t = tiles[0]
        is_term = is_yaochuhai(t)
        if mtype == 'koutsu':
            fu += 4 if not is_term else 8  # 暗刻

    # 副露面子的符
    for meld in open_melds:
        is_term = is_yaochuhai(meld.tile_type)
        if meld.meld_type == MeldType.CHI:
            pass
        elif meld.meld_type == MeldType.PON:
            fu += 2 if not is_term else 4
        elif meld.meld_type in (MeldType.KAN_OPEN, MeldType.KAN_DAIMIN):
            fu += 8 if not is_term else 16
        elif meld.meld_type == MeldType.KAN_CLOSED:
            fu += 16 if not is_term else 32

    pair_tile = pair[0]
    if _is_yakuhai_pair(pair_tile, bakaze, jikaze):
        fu += 2

    if wait_type in ('kanchan', 'penchan', 'tanki'):
        fu += 2

    if is_menzen and is_tsumo and is_pinfu:
        return 20

    return int(math.ceil(fu / 10.0)) * 10


def _is_yakuhai_pair(tile_type: int, bakaze: int, jikaze: int) -> bool:
    """判断是否为役牌对子（场风/自风/三元牌）。"""
    if tile_type == bakaze or tile_type == jikaze:
        return True
    if is_sangenhai(tile_type):
        return True
    return False


# ── 翻数 → 得点表 ──────────────────────────────────────────────────────────

# 得分表格式: scores[翻数][符数] → (荣和_子, 荣和_親, 自摸_子支払, 自摸_子_親支払, 自摸_親全員支払)
ScoreEntry = Tuple[int, int, int, int, int]

SCORE_TABLE: Dict[int, Dict[int, ScoreEntry]] = {
    1: {
        30:  (1000, 1500, 300, 500, 500),
        40:  (1300, 2000, 400, 700, 700),
        50:  (1600, 2400, 400, 800, 800),
        60:  (2000, 2900, 500, 1000, 1000),
        70:  (2300, 3400, 600, 1200, 1200),
        80:  (2600, 3900, 700, 1300, 1300),
        90:  (2900, 4400, 800, 1500, 1500),
        100: (3200, 4800, 800, 1600, 1600),
        110: (3600, 5300, 900, 1800, 1800),
    },
    2: {
        20:  (1300, 2000, 400, 700, 700),    # 平和自摸
        25:  (1600, 2400, 400, 800, 800),    # 七対子
        30:  (2000, 2900, 500, 1000, 1000),
        40:  (2600, 3900, 700, 1300, 1300),
        50:  (3200, 4800, 800, 1600, 1600),
        60:  (3900, 5800, 1000, 2000, 2000),
        70:  (4500, 6800, 1200, 2300, 2300),
        80:  (5200, 7700, 1300, 2600, 2600),
        90:  (5800, 8700, 1500, 2900, 2900),
        100: (6400, 9600, 1600, 3200, 3200),
        110: (7100, 10600, 1800, 3600, 3600),
    },
    3: {
        20:  (2600, 3900, 700, 1300, 1300),
        25:  (3200, 4800, 800, 1600, 1600),
        30:  (3900, 5800, 1000, 2000, 2000),
        40:  (5200, 7700, 1300, 2600, 2600),
        50:  (6400, 9600, 1600, 3200, 3200),
        60:  (7700, 11600, 2000, 3900, 3900),
        70:  (8000, 12000, 2000, 4000, 4000),  # 満貫
    },
    4: {
        20:  (5200, 7700, 1300, 2600, 2600),
        25:  (6400, 9600, 1600, 3200, 3200),
        30:  (7700, 11600, 2000, 3900, 3900),
        40:  (8000, 12000, 2000, 4000, 4000),  # 満貫
    },
}

# 満貫以上（翻数 ≥ 5）的固定得点
MANGAN_SCORES: Dict[int, ScoreEntry] = {
    5:  (8000, 12000, 2000, 4000, 4000),    # 満貫
    6:  (12000, 18000, 3000, 6000, 6000),    # 跳満
    7:  (12000, 18000, 3000, 6000, 6000),    # 跳満
    8:  (16000, 24000, 4000, 8000, 8000),    # 倍満
    9:  (16000, 24000, 4000, 8000, 8000),    # 倍満
    10: (16000, 24000, 4000, 8000, 8000),    # 倍満
    11: (24000, 36000, 6000, 12000, 12000),  # 三倍満
    12: (24000, 36000, 6000, 12000, 12000),  # 三倍満
}

# 役满基准得点
YAKUMAN_BASE: ScoreEntry = (32000, 48000, 8000, 16000, 16000)


def get_base_points(han: int, fu: int) -> int:
    """计算基本点（基本点 = 符 × 2^(2+翻)，上限 2000）。"""
    if han >= 13:  # 数え役満
        return 8000 * (han // 13)
    bp = fu * (2 ** (2 + min(han, 4)))
    if bp > 2000:
        if han >= 5:
            if han <= 5:      bp = 2000
            elif han <= 7:    bp = 3000
            elif han <= 10:   bp = 4000
            elif han <= 12:   bp = 6000
            else:             bp = 8000
        else:
            bp = min(bp, 2000)
    return bp


def lookup_score(han: int, fu: int, is_dealer: bool, is_tsumo: bool,
                 num_yakuman: int = 0) -> Tuple[int, int, int]:
    """查表获取支付金额。

    Returns:
        (荣和支付额, 自摸子家支付额, 自摸親家支付额)
    """
    if num_yakuman > 0:
        base = num_yakuman * 32000
        if is_dealer:
            ron_pay = base * 3 // 2
            tsumo_pay = base // 2
            return (ron_pay, tsumo_pay, tsumo_pay)
        else:
            ron_pay = base
            tsumo_ko = base // 4
            tsumo_oya = base // 2
            return (ron_pay, tsumo_ko, tsumo_oya)

    if han >= 5 and han <= 12:
        entry = MANGAN_SCORES.get(han, MANGAN_SCORES[5])
    elif han >= 13:
        entry = YAKUMAN_BASE
    else:
        if han not in SCORE_TABLE:
            return (0, 0, 0)
        fu_table = SCORE_TABLE[han]
        if fu in fu_table:
            entry = fu_table[fu]
        else:
            # 取最近的符数（查表近似）
            available_fu = sorted(fu_table.keys())
            closest = min(available_fu, key=lambda f: abs(f - fu))
            entry = fu_table[closest]

    ron_ko, ron_oya, tsumo_ko_ko, tsumo_ko_oya, tsumo_oya_all = entry
    if is_dealer:
        return (ron_oya, tsumo_oya_all, tsumo_oya_all)
    else:
        return (ron_ko, tsumo_ko_ko, tsumo_ko_oya)


# ── 支付计算 ────────────────────────────────────────────────────────────────

@dataclass
class PaymentInfo:
    """一次和牌的支付信息。"""
    winner: int                        # 和牌者索引 (0-3)
    loser: int                         # 放铳者索引（荣和时），-1 表示自摸
    han: int                           # 翻数
    fu: int                            # 符数
    yaku_names: List[str]              # 役种名称列表
    dora_count: int                    # 宝牌数量
    score_name: str                    # 得点名称（満貫/跳満/倍満/役満等）
    payments: List[int]                # [p0, p1, p2, p3] 各玩家的分数变动
    total_win: int                     # 和牌者总收入


def compute_payments(
    han: int,
    fu: int,
    winner: int,
    is_dealer: bool,
    is_tsumo: bool,
    loser: int = -1,
    num_yakuman: int = 0,
    honba: int = 0,
    riichi_sticks_on_table: int = 0,
) -> PaymentInfo:
    """计算一次和牌的各家支付额。

    Args:
        han: 总翻数
        fu: 符数（已取整）
        winner: 和牌者索引
        is_dealer: 和牌者是否为庄家
        is_tsumo: 是否自摸
        loser: 放铳者索引（荣和时）
        num_yakuman: 役满倍数（0 = 非役满）
        honba: 本场数
        riichi_sticks_on_table: 场上积存的立直棒数量
    """
    payments = [0, 0, 0, 0]
    ron_pay, tsumo_ko_pay, tsumo_oya_pay = lookup_score(
        han, fu, is_dealer, is_tsumo, num_yakuman
    )

    total_win = 0
    if is_tsumo:
        if is_dealer:
            # 庄家自摸：三家各付 tsumo_oya_pay
            for p in range(4):
                if p != winner:
                    payments[p] = -tsumo_oya_pay
                    total_win += tsumo_oya_pay
        else:
            # 闲家自摸：子家付 tsumo_ko_pay，親家付 tsumo_oya_pay
            for p in range(4):
                if p != winner:
                    if p == 0:  # 庄家始终为 player 0（此处语境下）
                        payments[p] = -tsumo_oya_pay
                        total_win += tsumo_oya_pay
                    else:
                        payments[p] = -tsumo_ko_pay
                        total_win += tsumo_ko_pay
    else:
        # 荣和：放铳者全额支付
        payments[loser] = -(ron_pay + honba * 300)
        total_win = ron_pay + honba * 300

    # 本场积点（自摸时每家 +100 点/本场）
    if is_tsumo and honba > 0:
        for p in range(4):
            if p != winner:
                payments[p] -= honba * 100
                total_win += honba * 100

    payments[winner] = total_win

    # 立直棒归和牌者
    if riichi_sticks_on_table > 0:
        riichi_value = riichi_sticks_on_table * 1000
        payments[winner] += riichi_value

    score_name = _score_name(han, fu, num_yakuman)

    return PaymentInfo(
        winner=winner, loser=loser,
        han=han, fu=fu,
        yaku_names=[], dora_count=0,
        score_name=score_name,
        payments=payments, total_win=total_win,
    )


def _score_name(han: int, fu: int, num_yakuman: int) -> str:
    """得点名称。"""
    if num_yakuman > 0:
        return f"{num_yakuman}倍役満" if num_yakuman > 1 else "役満"
    if han >= 13: return "数え役満"
    if han >= 11: return "三倍満"
    if han >= 8:  return "倍満"
    if han >= 6:  return "跳満"
    if han >= 5 or (han == 4 and fu >= 40) or (han == 3 and fu >= 70):
        return "満貫"
    return f"{han}翻{fu}符"


# ── 终局精算 ────────────────────────────────────────────────────────────────

@dataclass
class GameResult:
    """半荘终局结果。"""
    final_scores: List[int]       # 精算前分数
    adjusted_scores: List[int]    # 马点/オカ 调整后分数
    ranks: List[int]              # 排名 [0=1位, 3=4位]
    uma: Tuple[int, int, int, int]
    oka: int


def compute_final_result(
    scores: List[int],
    uma: Tuple[int, int, int, int] = (20, 10, -10, -20),
    oka: int = 0,
    target_score: int = 30000,
) -> GameResult:
    """终局精算：顺位马点 + オカ调整。

    Args:
        scores: 各玩家最终分数
        uma: 顺位马点（千点单位），如 (20, 10, -10, -20) 表示 1位+20k, 2位+10k, 3位-10k, 4位-20k
        oka: オカ（返点补差），0 表示不使用
        target_score: 返点目标
    """
    # 按分数降序排名
    sorted_indices = sorted(range(4), key=lambda i: scores[i], reverse=True)
    ranks = [0] * 4
    for rank, idx in enumerate(sorted_indices):
        ranks[idx] = rank

    adjusted = list(scores)
    for rank, idx in enumerate(sorted_indices):
        adjusted[idx] += uma[rank] * 1000

    if oka > 0:
        for i in range(4):
            adjusted[i] -= target_score
        adjusted[sorted_indices[0]] += oka * 1000

    return GameResult(
        final_scores=scores,
        adjusted_scores=adjusted,
        ranks=ranks,
        uma=uma,
        oka=oka,
    )
