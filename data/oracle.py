"""Oracle 算法：为 Transformer MTL 模型生成牌理标签。

三种算法的"老师"（Oracle）：
  1. calculate_shanten  — 向听数计算（标准形 + 七对子 + 国士无双）
  2. compute_ukeire     — 有效进张计算（34 维布尔掩码）
  3. classify_wait      — 待牌类型分类 + 质量评分

所有函数以 int[34] 手牌直方图作为输入（与 engine.agari 一致）。
"""

from functools import lru_cache
from typing import List, Tuple

from engine.agari import is_tenpai, is_agari, get_waits
from engine.tile import (
    NUM_TYPES, YAOCHUHAI_TYPES, SHUPAI_TYPES, JIHAI_TYPES,
    is_jihai,
)

# ─── 花色常量 ────────────────────────────────────────────────────────────────

_MANZU = range(0, 9)
_PINZU = range(9, 18)
_SOUZU = range(18, 27)
_JIHAI = range(27, 34)
_ALL_SUITS = [
    (0, 9, True),    # 万子：9 种，允许顺子
    (9, 18, True),   # 筒子：9 种，允许顺子
    (18, 27, True),  # 条子：9 种，允许顺子
    (27, 34, False),  # 字牌：7 种，不允许顺子
]


# ==============================================================================
# 1. 向听数计算 (Shanten Calculation)
# ==============================================================================

def _suit_counts(tiles: List[int], start: int, end: int) -> Tuple[int, ...]:
    """提取花色计数元组（用于 memoization key）。"""
    return tuple(tiles[start:end])


@lru_cache(maxsize=50000)
def _max_score_suit(counts: Tuple[int, ...], allow_sequence: bool) -> Tuple[int, int, int]:
    """返回这种花色能够形成的最大 (melds, taatsu, has_pair)。

    melds: 完整面子数（刻子/顺子）
    taatsu: 搭子数（两面/坎张/边张，不含对子）
    has_pair: 是否使用了一个对子（0 或 1）

    评分标准：
      - 形成面子 → +2
      - 形成搭子 → +1
      - 形成对子 → +0（对子价值体现在 has_pair 标志位，不计入 taatsu）
      - 剩余单张 → +0

    使用带 memoization 的递归、贪心处理第一个非零位置。
    """
    n = len(counts)

    # 找到第一个非零位置
    idx = 0
    while idx < n and counts[idx] == 0:
        idx += 1

    # 基础情况：花色已为空
    if idx >= n:
        return (0, 0, 0)

    best_m, best_t, best_p = 0, 0, 0
    c = list(counts)  # mutable copy

    tile = c[idx]

    # ── A. 跳过 1 张（当剩余单张处理，贡献为 0） ──
    c[idx] -= 1
    m, t, p = _max_score_suit(tuple(c), allow_sequence)
    # 优先用 melds 大、taatsu 大的组合
    _update_best((best_m, best_t, best_p), (m, t, p))
    best_m, best_t, best_p = m, t, p
    c[idx] += 1

    # ── B. 形成对子（has_pair=1，不计入 taatsu） ──
    if tile >= 2 and best_p == 0:
        c[idx] -= 2
        m, t, p = _max_score_suit(tuple(c), allow_sequence)
        _update_best((best_m, best_t, best_p), (m, t, 1))
        best_m, best_t, best_p = m, t, 1
        c[idx] += 2

    # ── C. 形成搭子（taatsu+1） ──

    # C1. 两面/边张：连续 2 张
    if allow_sequence and idx + 1 < n and c[idx + 1] >= 1:
        c[idx] -= 1
        c[idx + 1] -= 1
        m, t, p = _max_score_suit(tuple(c), allow_sequence)
        _update_best((best_m, best_t, best_p), (m, t + 1, p))
        best_m, best_t, best_p = m, t + 1, p
        c[idx] += 1
        c[idx + 1] += 1

    # C2. 坎张：隔 1 张
    if allow_sequence and idx + 2 < n and c[idx + 2] >= 1:
        c[idx] -= 1
        c[idx + 2] -= 1
        m, t, p = _max_score_suit(tuple(c), allow_sequence)
        _update_best((best_m, best_t, best_p), (m, t + 1, p))
        best_m, best_t, best_p = m, t + 1, p
        c[idx] += 1
        c[idx + 2] += 1

    # ── D. 形成面子（meld+1, 得分+2） ──

    # D1. 刻子
    if tile >= 3:
        c[idx] -= 3
        m, t, p = _max_score_suit(tuple(c), allow_sequence)
        _update_best((best_m, best_t, best_p), (m + 1, t, p))
        best_m, best_t, best_p = m + 1, t, p
        c[idx] += 3

    # D2. 顺子
    if allow_sequence and idx + 2 < n and c[idx + 1] >= 1 and c[idx + 2] >= 1:
        c[idx] -= 1
        c[idx + 1] -= 1
        c[idx + 2] -= 1
        m, t, p = _max_score_suit(tuple(c), allow_sequence)
        _update_best((best_m, best_t, best_p), (m + 1, t, p))
        best_m, best_t, best_p = m + 1, t, p
        c[idx] += 1
        c[idx + 1] += 1
        c[idx + 2] += 1

    return (best_m, best_t, best_p)


def _update_best(current: Tuple[int, int, int],
                 candidate: Tuple[int, int, int]) -> None:
    """比较两个 (melds, taatsu, has_pair)，保留更好的。

    评分规则（用于标准形的向听计算公式）：
      score = 8 - 2*m - t - p
      值越小越好。等价于最大化 (2*m + t + p)。
    """
    c_m, c_t, c_p = current
    n_m, n_t, n_p = candidate
    c_score = 2 * c_m + c_t + c_p
    n_score = 2 * n_m + n_t + n_p
    # 平局时优先选 has_pair 高的（对 shanten 更有用）
    if n_score > c_score or (n_score == c_score and n_p > c_p):
        return True
    return False


def _suit_analysis(tiles: List[int], start: int, end: int,
                   allow_sequence: bool) -> Tuple[int, int, int]:
    """分析单一花色的 (melds, taatsu, has_pair)。"""
    counts = _suit_counts(tiles, start, end)
    return _max_score_suit(counts, allow_sequence)


def shanten_standard(tiles: List[int]) -> int:
    """计算标准形（4 面子 + 1 雀头）的向听数。

    公式：shanten = 8 - 2*M - T - P
      M = 总面子数
      T = 总搭子数
      P = 1（有雀头）或 0（无雀头）
    约束：M + T + P ≤ 5

    返回 0-8 的整数（0 = 听牌）。
    """
    total_m = 0
    total_t = 0
    total_p = 0

    # 对每个花色分析
    for start, end, allow_seq in _ALL_SUITS:
        m, t, p = _suit_analysis(tiles, start, end, allow_seq)
        total_m += m
        total_t += t
        total_p += p

    # 处理雀头：最多 1 个
    if total_p > 1:
        total_p = 1

    # 处理超额分组：M + T + P ≤ 5
    groups = total_m + total_t + total_p
    excess = max(0, groups - 5)
    if excess > 0:
        # 优先从搭子中扣除（面子更值钱）
        total_t = max(0, total_t - excess)

    # 计算向听（允许负值，由上层函数 clamp）
    return 8 - 2 * total_m - total_t - total_p


def shanten_chiitoitsu(tiles: List[int]) -> int:
    """计算七对子向听数。

    公式：shanten = 6 - unique_pairs
    其中 unique_pairs = 拥有 ≥ 2 张的牌种数。
    """
    pairs = sum(1 for c in tiles if c >= 2)
    return 6 - pairs


def shanten_kokushi(tiles: List[int]) -> int:
    """计算国士无双向听数。

    公式：shanten = 13 - unique_yaochu - (1 if any pair else 0)
    """
    unique = sum(1 for t in YAOCHUHAI_TYPES if tiles[t] >= 1)
    has_pair = any(tiles[t] >= 2 for t in YAOCHUHAI_TYPES)
    return 13 - unique - (1 if has_pair else 0)


def _calculate_shanten_raw(tiles: List[int]) -> int:
    """计算向听数（允许负值，用于内部判断进张）。"""
    s = shanten_standard(tiles)
    s = min(s, shanten_chiitoitsu(tiles))
    s = min(s, shanten_kokushi(tiles))
    return s


def calculate_shanten(tiles: List[int]) -> int:
    """计算手牌的向听数（取三种形的最小值）。

    Args:
        tiles: int[34] 手牌直方图。

    Returns:
        向听数 (0 = 听牌, 1 = 一向听, ..., 6 = 六向听及以上)。
    """
    return max(0, _calculate_shanten_raw(tiles))


# ==============================================================================
# 2. 有效进张计算 (Ukeire / Efficiency)
# ==============================================================================

def compute_ukeire(tiles: List[int],
                   remaining: int = 0) -> Tuple[int, List[bool], int]:
    """计算有效进张。

    算法：依次向手牌中加入 34 种牌中的每一种 T，
    若加入后向听数下降，则 T 为有效进张。

    Args:
        tiles: int[34] 手牌直方图。
        remaining: 场上剩余牌数（可选，用于估计有效张数存续概率）。

    Returns:
        (ukeire_count, ukeire_mask, unique_tiles):
          - ukeire_count: 有效进张的总种数 (0-34)。
          - ukeire_mask: 长度为 34 的布尔列表，True 表示该牌是有效进张。
          - unique_tiles: 有效进张中，场上还有剩余的不同牌种数。
    """
    base_shanten = _calculate_shanten_raw(tiles)

    ukeire = [False] * NUM_TYPES
    for t in range(NUM_TYPES):
        if tiles[t] >= 4:
            continue  # 这种牌已摸满，不可能再摸到
        tiles[t] += 1
        if _calculate_shanten_raw(tiles) < base_shanten:
            ukeire[t] = True
        tiles[t] -= 1

    count = sum(ukeire)
    # 有效进张中还有剩余张数的种类（用于计算纯正进张数）
    unique_available = sum(1 for t in range(NUM_TYPES)
                           if ukeire[t] and tiles[t] < 4)
    return count, ukeire, unique_available


# ==============================================================================
# 3. 待牌质量分类 (Wait Quality)
# ==============================================================================

def _find_wait_meld_type(tiles: List[int], wait_tile: int) -> Tuple[str, float]:
    """判断听牌形状的类型。

    尝试将 wait_tile 加入手牌，形成完整 14 张，
    然后分析哪组搭子接收了 wait_tile。

    Args:
        tiles: int[34] 听牌手牌（13 张或 3k+1 张）。
        wait_tile: 待牌的类型 ID (0-33)。

    Returns:
        (wait_type, factor):
          wait_type: 'ryanmen' | 'kanchan' | 'penchan' | 'tanki' | 'shanpon'
                     | 'multi_sided' | 'unknown'
          factor: 修正系数，用于质量评分：
            两面 1.0 / 多面 1.2 / 双碰 0.8 / 单骑 0.7 / 坎张 0.5 / 边张 0.4
    """
    if tiles[wait_tile] >= 4:
        return ('unknown', 0.6)

    # 加入和牌牌
    tiles[wait_tile] += 1

    # ── 检查七对子形 ──
    pairs = [t for t in range(NUM_TYPES) if tiles[t] == 2]
    if len(pairs) >= 6:
        # 七对子听牌（单骑）
        tiles[wait_tile] -= 1
        # 检查：是否是单骑（6 种对子 + 1 种单张 → 等单张成对）
        singles = [t for t in range(NUM_TYPES) if tiles[t] == 1]
        if len(singles) == 1 and singles[0] == wait_tile:
            return ('tanki', 0.7)
        return ('multi_sided', 1.2)  # 七对子多面听

    # ── 一般形分析：找到 wait_tile 所在的搭子 ──
    wait_pair = tiles[wait_tile] >= 2  # wait 牌是否是对子的一部分

    # 寻找怪形（多面听）：wait_tile 可以嵌入多个搭子
    # 拆回 13 张，看 get_waits 的数量
    tiles[wait_tile] -= 1
    waits = get_waits(tiles)

    if len(waits) >= 4:
        # 4 面以上听牌
        tiles[wait_tile] += 1
        return ('multi_sided', 1.2)

    tiles[wait_tile] += 1

    # 检查 wait_tile 是否是单骑（只有 1 张，不成对/搭子）
    if tiles[wait_tile] == 1:
        # 它可能是一个单张 → 单骑
        # 验证：移除 wait_tile 后手牌不是听牌状态
        tiles[wait_tile] -= 1
        still_tenpai = is_tenpai(tiles)
        tiles[wait_tile] += 1
        if still_tenpai:
            # 移除后还听牌 → wait_tile 不是唯一的待牌 → shanpon
            factor = 0.8
            wait_type = 'shanpon'
        else:
            factor = 0.7
            wait_type = 'tanki'

    elif wait_pair and tiles[wait_tile] == 2:
        # 对子：可能是双碰
        tiles[wait_tile] -= 2
        still_tenpai = is_tenpai(tiles)
        tiles[wait_tile] += 2
        if still_tenpai:
            wait_type = 'shanpon'
            factor = 0.8
        else:
            # 单骑（7 对子被排除）
            wait_type = 'tanki'
            factor = 0.7

    elif tiles[wait_tile] >= 1 and tiles[wait_tile] <= 2:
        # wait_tile 在顺子搭子中
        # 检查是否坎张
        is_kanchan = False
        # 寻找 wait_tile 作为"中间牌"的搭子
        w = wait_tile
        suit_start = (w // 9) * 9
        suit_end = suit_start + 9

        # 检查是否是边张搭子
        is_penchan = False
        # 12 → 听 3 (边张)
        if (w - suit_start) == 2 and tiles[w - 2] >= 1 and tiles[w - 1] >= 1:
            is_penchan = True
        # 89 → 听 7 (边张)
        elif (w - suit_start) == 6 and tiles[w + 1] >= 1 and tiles[w + 2] >= 1:
            is_penchan = True

        # 检查坎张
        # 13 → 听 2 (坎张)
        if not is_penchan and (w - suit_start) >= 1 and (w - suit_start) <= 7:
            if tiles[w - 1] >= 1 and tiles[w + 1] >= 1:
                # 三面：345 听 36，36 是两面+坎张的复合
                # 3456 → 2,5,7,8 都需要特殊检测
                # 简化处理：如果 wait_tile 同时出现在多个搭子中，判为多面
                pass  # fall through

        # 使用暴力枚举判断
        # 对 13 张手牌（移除 wait_tile 后移除一种），检查听牌
        tiles[wait_tile] -= 1
        # 找出所有使移除 wait_tile 后剩下的 13 张仍能听牌的切牌
        is_tanki_like = True
        for d in range(NUM_TYPES):
            if tiles[d] > 0:
                tiles[d] -= 1
                if is_tenpai(tiles):
                    # 发现某种切牌后仍听牌 → wait_tile 处于搭子中
                    is_tanki_like = False
                    # 判断搭子类型：如果是连张且 wait_tile 在中间 → 两面
                    # 如果 wait_tile 在两端 → 边张或两面
                    tiles[d] += 1
                    break
                tiles[d] += 1
        tiles[wait_tile] += 1

        if not is_tanki_like:
            # 判断两面 vs 坎张 vs 边张
            # 检查 wait_tile 的左右邻牌
            left_adj = (w - suit_start) >= 1 and tiles[w - 1] >= 1
            right_adj = (w - suit_start) <= 7 and tiles[w + 1] >= 1

            # 边张：12 听 3 或 89 听 7
            if (w == 2 and tiles[0] >= 1 and tiles[1] >= 1) or \
               (w == 6 and tiles[7] >= 1 and tiles[8] >= 1) or \
               (w == 11 and tiles[9] >= 1 and tiles[10] >= 1) or \
               (w == 15 and tiles[16] >= 1 and tiles[17] >= 1) or \
               (w == 20 and tiles[18] >= 1 and tiles[19] >= 1) or \
               (w == 24 and tiles[25] >= 1 and tiles[26] >= 1):
                wait_type = 'penchan'
                factor = 0.4
            # 两面
            elif left_adj and right_adj:
                wait_type = 'ryanmen'
                factor = 1.0
            else:
                # 检查是否多面听
                if len(waits) >= 3:
                    wait_type = 'multi_sided'
                    factor = 1.2
                elif left_adj or right_adj:
                    wait_type = 'ryanmen'  # 带一个搭子的两面听
                    factor = 1.0
                else:
                    wait_type = 'kanchan'
                    factor = 0.5
        else:
            # 已经是 tanki（之前已判断）
            wait_type = 'tanki'
            factor = 0.7

    else:
        wait_type = 'unknown'
        factor = 0.6

    return (wait_type, factor)


def classify_wait(tiles: List[int]) -> dict:
    """对听牌手牌进行分类和评分。

    计算：
      - 听牌类型（两面/坎张/边张/单骑/双碰/多面）
      - 质量评分（0-1 连续值）
      - 有效待牌数
      - 修正系数

    Args:
        tiles: int[34] 听牌手牌（13 张或 3k+1 张）。
              调用前需确保 is_tenpai(tiles) == True。

    Returns:
        dict 包含:
          - is_tenpai: bool
          - waits: List[int] 待牌列表
          - wait_types: List[str] 每种待牌的类型
          - main_type: str 主要待牌类型
          - quality_score: float (0-1) 质量评分
          - wait_count: int 待牌种数
          - total_available: int 剩余待牌总数
    """
    if not is_tenpai(tiles):
        return {
            'is_tenpai': False,
            'waits': [],
            'wait_types': [],
            'main_type': 'noten',
            'quality_score': 0.0,
            'wait_count': 0,
            'total_available': 0,
        }

    waits = get_waits(tiles)
    wait_types = []
    total_available = 0

    for w in waits:
        wt, factor = _find_wait_meld_type(tiles, w)
        wait_types.append(wt)

        # 计算剩余张数
        available = 4 - tiles[w]
        total_available += available

    # 主要待牌类型：取最常见的类型
    from collections import Counter
    type_counts = Counter(wait_types)
    main_type = type_counts.most_common(1)[0][0]

    # 质量评分：0.0-1.0
    factor_map = {
        'ryanmen': 1.0, 'multi_sided': 1.2, 'shanpon': 0.8,
        'tanki': 0.7, 'kanchan': 0.5, 'penchan': 0.4, 'unknown': 0.6,
    }

    # 综合评分 = 加权平均 factor × 张数归一化
    # 归一化到 0-1（满分 = 希望有 3 种两面听 × 4 张 × 1.0 = 12 分）
    weighted_score = 0.0
    for w, wt in zip(waits, wait_types):
        available = 4 - tiles[w]
        weighted_score += available * factor_map.get(wt, 0.6)
    max_possible = 12.0  # 3 ryanmen × 4 tiles = 12
    quality_score = min(1.0, weighted_score / max_possible)

    return {
        'is_tenpai': True,
        'waits': waits,
        'wait_types': wait_types,
        'main_type': main_type,
        'quality_score': round(quality_score, 4),
        'wait_count': len(waits),
        'total_available': total_available,
    }


# ==============================================================================
# 4. 组合接口（单次计算所有 Oracle 标签）
# ==============================================================================

def compute_all_oracle_labels(tiles: List[int]) -> dict:
    """一次调用计算所有 Oracle 标签。

    Args:
        tiles: int[34] 手牌直方图。

    Returns:
        dict 包含 shanten, ukeire, wait_quality 的所有标签。
    """
    shanten = calculate_shanten(tiles)
    ukeire_count, ukeire_mask, ukeire_available = compute_ukeire(tiles)
    tenpai = is_tenpai(tiles)

    result = {
        'shanten': shanten,                          # int 0-6
        'ukeire_count': ukeire_count,                # int 0-34
        'ukeire_mask': ukeire_mask,                  # List[bool] length 34
        'ukeire_available': ukeire_available,        # int 有效进张剩余牌种数
        'is_tenpai': tenpai,                         # bool
        'efficiency_score': 0.0,                     # 占位，后续计算
        'danger_map': [0.0] * NUM_TYPES,             # 占位，后续计算
        'score_estimate': 0.0,                       # 占位，后续计算
    }

    if tenpai:
        wq = classify_wait(tiles)
        result['wait_quality'] = wq
        result['quality_score'] = wq['quality_score']
        result['wait_type'] = wq['main_type']
        result['waits'] = wq['waits']
        result['wait_types'] = wq['wait_types']
        result['total_available'] = wq['total_available']
    else:
        result['wait_quality'] = None
        result['quality_score'] = 0.0
        result['wait_type'] = 'noten'
        result['waits'] = []
        result['wait_types'] = []
        result['total_available'] = 0

    return result
# 中文注释：向听数、有效进张和待牌质量三种 Oracle 算法集合，为 MTL 模型生成监督学习标签。
