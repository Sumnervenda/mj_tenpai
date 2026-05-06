"""牌编码系统 —— 双轨制：绝对 ID / 类型 ID。

绝对 ID (0-135): 每张牌的物理唯一标识，用于牌山、牌河。同种牌有 4 张不同 ID。
类型 ID (0-33):  34 种逻辑牌型，用于手牌直方图 int[34]，不区分个体。

编码公式: abs_id = type_id × 4 + copy_index (0-3)
赤宝牌: 5万/5筒/5条 的 copy_index=3 为赤牌（共 3 张）。
"""

from enum import IntEnum
from typing import List, Tuple

# ── 常量定义 ──────────────────────────────────────────────────────────────────

NUM_TYPES = 34          # 牌的种类数
NUM_ABS = 136           # 牌的绝对数量
COPIES_PER_TYPE = 4     # 每种牌 4 张

# 赤宝牌（红宝牌）设定
AKA_TYPES = {4, 13, 22}     # 5万, 5筒, 5条 的类型 ID
AKA_COPY_INDEX = 3          # 第 4 张（索引 3）为赤牌

# 花色范围
MANZU = range(0, 9)         # 万子 1～9 万
PINZU = range(9, 18)        # 筒子 1～9 筒
SOUZU = range(18, 27)       # 条子 1～9 条
JIHAI = range(27, 34)       # 字牌（风牌 4 种 + 三元牌 3 种）
KAZEHAI = range(27, 31)     # 风牌: 東 南 西 北
SANGENHAI = range(31, 34)   # 三元牌: 白 発 中


class TileType(IntEnum):
    """34 种牌的逻辑类型枚举。"""
    # 万子 (Manzu) 1～9 万
    M1, M2, M3, M4, M5, M6, M7, M8, M9 = range(0, 9)
    # 筒子 (Pinzu) 1～9 筒
    P1, P2, P3, P4, P5, P6, P7, P8, P9 = range(9, 18)
    # 条子 (Souzu) 1～9 条
    S1, S2, S3, S4, S5, S6, S7, S8, S9 = range(18, 27)
    # 字牌 (Jihai)
    TON = 27   # 東
    NAN = 28   # 南
    SHA = 29   # 西
    PEI = 30   # 北
    HAK = 31   # 白
    HAT = 32   # 発
    CHU = 33   # 中


# ── 牌名显示 ─────────────────────────────────────────────────────────────────

TILE_NAMES: List[str] = [
    # 万子
    "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
    # 筒子
    "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
    # 条子
    "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
    # 字牌
    "東", "南", "西", "北", "白", "発", "中",
]

TILE_NAMES_CN: List[str] = [
    "一万", "二万", "三万", "四万", "五万", "六万", "七万", "八万", "九万",
    "一筒", "二筒", "三筒", "四筒", "五筒", "六筒", "七筒", "八筒", "九筒",
    "一条", "二条", "三条", "四条", "五条", "六条", "七条", "八条", "九条",
    "東", "南", "西", "北", "白", "発", "中",
]

# 花色内数字（数牌 1～9，字牌统一为 0）
TILE_NUMBERS: List[int] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9,       # 万子
    1, 2, 3, 4, 5, 6, 7, 8, 9,       # 筒子
    1, 2, 3, 4, 5, 6, 7, 8, 9,       # 条子
    0, 0, 0, 0, 0, 0, 0,             # 字牌（无限定数字）
]

# ── 花色判断工具函数 ─────────────────────────────────────────────────────────

def suit_of(t: int) -> int:
    """返回花色索引：0=万, 1=筒, 2=条, 3=字。"""
    if t < 9:       return 0
    elif t < 18:    return 1
    elif t < 27:    return 2
    else:           return 3


def is_manzu(t: int) -> bool: return 0 <= t <= 8       # 是否为万子
def is_pinzu(t: int) -> bool: return 9 <= t <= 17      # 是否为筒子
def is_souzu(t: int) -> bool: return 18 <= t <= 26     # 是否为条子
def is_jihai(t: int) -> bool: return 27 <= t <= 33     # 是否为字牌
def is_kazehai(t: int) -> bool: return 27 <= t <= 30   # 是否为风牌
def is_sangenhai(t: int) -> bool: return 31 <= t <= 33 # 是否为三元牌
def is_shupai(t: int) -> bool: return 0 <= t <= 26     # 是否为数牌（万/筒/条）
def is_yaochuhai(t: int) -> bool:
    """是否为幺九牌（1/9 数牌 或 字牌）。"""
    return t in {0, 8, 9, 17, 18, 26} or is_jihai(t)
def is_tsupai(t: int) -> bool: return is_jihai(t)      # 字牌的别称


# ── 绝对 ID ↔ 类型 ID 转换 ──────────────────────────────────────────────────

def abs_to_type(abs_id: int) -> int:
    """绝对 ID (0-135) → 类型 ID (0-33)。"""
    return abs_id // COPIES_PER_TYPE


def type_to_abs(type_id: int) -> List[int]:
    """类型 ID → 对应的 4 个绝对 ID 列表。"""
    base = type_id * COPIES_PER_TYPE
    return [base, base + 1, base + 2, base + 3]


def is_aka(abs_id: int) -> bool:
    """判断某张绝对牌是否为赤宝牌（红 5）。"""
    if abs_id % COPIES_PER_TYPE != AKA_COPY_INDEX:
        return False
    return abs_to_type(abs_id) in AKA_TYPES


def is_aka_type(type_id: int) -> bool:
    """判断某牌型是否为赤宝牌类型（5万/5筒/5条）。"""
    return type_id in AKA_TYPES


def tile_name(abs_id: int) -> str:
    """获取绝对牌的显示名称，赤宝牌加"赤"前缀。"""
    t = abs_to_type(abs_id)
    name = TILE_NAMES[t]
    if is_aka(abs_id):
        return f"赤{name}"
    return name


# ── 幺九 / 数牌 / 字牌 类型集合 ──────────────────────────────────────────────

# 全部幺九牌类型：1万, 9万, 1筒, 9筒, 1条, 9条, 東, 南, 西, 北, 白, 発, 中（共 13 种）
YAOCHUHAI_TYPES = frozenset({0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33})
SHUPAI_TYPES = frozenset(range(0, 27))     # 全部数牌类型（共 27 种）
JIHAI_TYPES = frozenset(range(27, 34))     # 全部字牌类型（共 7 种）

# ── 常用牌型查询集合 ────────────────────────────────────────────────────────

# 可以形成顺子的牌型（仅限于数牌）
CAN_SEQUENCE = frozenset(range(0, 27))

# 老頭牌：只含 1 和 9 的数牌（用于混老頭 / 清老頭等役判定）
ROTOHAI_TYPES = frozenset({0, 8, 9, 17, 18, 26})

# 断幺九牌：全部数牌的 2～8（共 21 种）
TANYAO_TYPES = frozenset({1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16,
                          19, 20, 21, 22, 23, 24, 25})

# 绿一色适用的绿色牌：2s, 3s, 4s, 6s, 8s, 発（共 6 种）
GREEN_TYPES = frozenset({19, 20, 21, 23, 25, 32})


def all_tile_types() -> List[int]:
    """返回全部 34 种牌的类型 ID。"""
    return list(range(NUM_TYPES))


def all_abs_ids() -> List[int]:
    """返回全部 136 张牌的绝对 ID。"""
    return list(range(NUM_ABS))
