"""查表法（LUT）胡牌判定 —— 引擎的算法基石。

核心思想：
  1. 将每种花色（万/筒/条 9 种，字牌 7 种）的手牌计数编码为五进制整数
  2. BFS 预计算所有合法面子/雀头组合，存入查表
  3. 完整手牌 = 4 花色 LUT 查询 × 枚举雀头归属 → O(1) 判定

状态空间：数牌 5^9 ≈ 200 万，字牌 5^7 ≈ 8 万 → 构建 ~250ms

查表内容：
  - melds[i]: 状态 i 能否完全分解为面子（无雀头）
  - with_pair[i]: 状态 i 能否分解为面子 + 1 个雀头

支持三种胡牌形：
  - 一般形：4 面子 + 1 雀头
  - 七对子：7 种不同牌各 2 张
  - 国士无双：13 种幺九牌各 1 张 + 任意幺九牌 1 张
"""

from typing import List, Optional, Tuple

from .tile import (
    NUM_TYPES, MANZU, PINZU, SOUZU, JIHAI,
    YAOCHUHAI_TYPES, is_jihai,
)

# ── 全局查表（LUT）—— 首次使用时构建 ─────────────────────────────────────────

_SUIT_LUT_MELDS = None        # List[bool]   9 位五进制 → 是否可分解为全面子
_SUIT_LUT_WITH_PAIR = None    # List[bool]   9 位五进制 → 是否可分解为面子 + 雀头
_HONOR_LUT_MELDS = None       # List[bool]   7 位五进制 → 字牌全面子
_HONOR_LUT_WITH_PAIR = None   # List[bool]   7 位五进制 → 字牌面子 + 雀头


def _encode(counts: List[int]) -> int:
    """将手牌计数列表（每位 0～4）编码为五进制整数。"""
    val = 0
    for c in reversed(counts):
        val = val * 5 + c
    return val


def _decode(state: int, n: int) -> List[int]:
    """将五进制整数解码为 n 位计数列表。"""
    counts = [0] * n
    for i in range(n - 1, -1, -1):
        counts[i] = state % 5
        state //= 5
    return counts


def _build_lut(n_positions: int, allow_sequence: bool) -> Tuple[List[bool], List[bool]]:
    """为一种花色构建全面子 / 面子+雀头 两个查表。

    BFS 算法：从空手牌（状态 0）出发，每次添加一个面子或雀头，
    只访问可到达的合法状态。比遍历全部 5^n 状态快 ~20 倍。

    Args:
        n_positions: 花色位数（数牌 9，字牌 7）
        allow_sequence: 是否允许顺子（字牌不允许）
    """
    from collections import deque

    size = 5 ** n_positions
    melds = [False] * size         # 能否分解为全面子
    with_pair = [False] * size     # 能否分解为面子 + 1 雀头
    melds[0] = True                # 空手牌 = 全面子（0 个面子）

    pow5 = [5 ** i for i in range(n_positions)]

    # 预计算顺子对应的增量（delta = 5^i + 5^(i+1) + 5^(i+2)）
    seq_deltas = []
    if allow_sequence:
        for i in range(n_positions - 2):
            seq_deltas.append((i, pow5[i] + pow5[i+1] + pow5[i+2]))

    queue: deque = deque()
    queue.append((0, False))  # (状态, 是否已有雀头)

    while queue:
        state, has_pair = queue.popleft()

        # 提取每位数字（用于判断是否有空间加牌）
        digits = [(state // pw) % 5 for pw in pow5]

        # ── 从"全面子"状态出发：可以加雀头或加面子 ──
        if not has_pair:
            # 加雀头：需要该位 ≤ 2（加 2 后 ≤ 4）
            for i in range(n_positions):
                if digits[i] <= 2:
                    ns = state + 2 * pow5[i]
                    if not with_pair[ns]:
                        with_pair[ns] = True
                        queue.append((ns, True))

        # 加刻子：需要该位 ≤ 1（加 3 后 ≤ 4）
        for i in range(n_positions):
            if digits[i] <= 1:
                ns = state + 3 * pow5[i]
                if has_pair:
                    if not with_pair[ns]:
                        with_pair[ns] = True
                        queue.append((ns, True))
                else:
                    if not melds[ns]:
                        melds[ns] = True
                        queue.append((ns, False))

        # 加顺子：需要连续 3 位各 ≤ 3（各加 1 后 ≤ 4）
        for i, delta in seq_deltas:
            if digits[i] <= 3 and digits[i+1] <= 3 and digits[i+2] <= 3:
                ns = state + delta
                if has_pair:
                    if not with_pair[ns]:
                        with_pair[ns] = True
                        queue.append((ns, True))
                else:
                    if not melds[ns]:
                        melds[ns] = True
                        queue.append((ns, False))

    return melds, with_pair


def _build_luts():
    """构建所有查表。仅首次调用时执行，全局只构建一次。"""
    global _SUIT_LUT_MELDS, _SUIT_LUT_WITH_PAIR
    global _HONOR_LUT_MELDS, _HONOR_LUT_WITH_PAIR

    if _SUIT_LUT_MELDS is None:
        _SUIT_LUT_MELDS, _SUIT_LUT_WITH_PAIR = _build_lut(9, allow_sequence=True)
    if _HONOR_LUT_MELDS is None:
        _HONOR_LUT_MELDS, _HONOR_LUT_WITH_PAIR = _build_lut(7, allow_sequence=False)


# ── 公开 API ────────────────────────────────────────────────────────────────

def is_agari(tiles: List[int]) -> bool:
    """判断 int[34] 手牌是否胡牌。

    支持：一般形（4 面子 + 1 雀头）、七对子、国士无双。
    也兼容已副露的手牌（只传入门内部分即可）。
    """
    _build_luts()
    n = sum(tiles)

    # 胡牌至少需要 3k+2 张牌
    if n % 3 != 2:
        return False

    # 特殊形：国士无双（13 幺九 + 1 对）
    if n == 14:
        if _is_kokushi(tiles):
            return True
        # 特殊形：七对子（7 对不同牌各 2 张）
        if _is_chiitoitsu(tiles):
            return True

    # 一般形：拆为 4 个花色分别查表
    man = tiles[0:9]    # 万子计数 (9)
    pin = tiles[9:18]   # 筒子计数 (9)
    sou = tiles[18:27]  # 条子计数 (9)
    ji = tiles[27:34]   # 字牌计数 (7)

    # 快速拒绝：每个花色的总牌数必须为 3k 或 3k+2
    for suit_counts in [man, pin, sou, ji]:
        s = sum(suit_counts)
        if s % 3 == 1:
            return False

    return _is_standard_agari(man, pin, sou, ji)


def _is_standard_agari(man, pin, sou, ji) -> bool:
    """一般形判定：恰好一个花色贡献雀头，其余花色全为面子。"""
    suits = [man, pin, sou, ji]
    lut_melds_list = [_SUIT_LUT_MELDS, _SUIT_LUT_MELDS, _SUIT_LUT_MELDS, _HONOR_LUT_MELDS]
    lut_pair_list = [_SUIT_LUT_WITH_PAIR, _SUIT_LUT_WITH_PAIR,
                     _SUIT_LUT_WITH_PAIR, _HONOR_LUT_WITH_PAIR]

    # 尝试让每种花色分别充当"雀头提供者"
    for pair_idx in range(4):
        ok = True
        for suit_idx in range(4):
            counts = suits[suit_idx]
            encoded = _encode(counts)
            lut = lut_pair_list[suit_idx] if suit_idx == pair_idx else lut_melds_list[suit_idx]
            if not lut[encoded]:
                ok = False
                break
        if ok:
            return True
    return False


def _is_chiitoitsu(tiles: List[int]) -> bool:
    """七对子判定：恰好 7 种牌各 2 张，无其他牌。"""
    pairs = 0
    for c in tiles:
        if c == 2:
            pairs += 1
        elif c != 0:
            return False
    return pairs == 7


def _is_kokushi(tiles: List[int]) -> bool:
    """国士无双判定：13 种幺九牌各至少 1 张，其中 1 种为对子（2 张），无其他牌。"""
    yaochu = list(YAOCHUHAI_TYPES)  # 13 种幺九牌
    has_all = True
    pair_count = 0
    for t in yaochu:
        c = tiles[t]
        if c == 0:
            has_all = False
        elif c == 2:
            pair_count += 1
        elif c > 2:
            return False
    # 确保没有幺九牌以外的牌
    for t in range(NUM_TYPES):
        if t not in YAOCHUHAI_TYPES and tiles[t] > 0:
            return False
    return has_all and pair_count == 1


# ── 探测函数（听牌 / 听牌列表）──────────────────────────────────────────────

def is_tenpai(tiles: List[int]) -> bool:
    """判断 13 张（或 3k+1 张）手牌是否听牌。

    算法：依次加入 34 种牌型，只要有一种能使 is_agari 为 True 即为听牌。
    """
    _build_luts()
    for t in range(NUM_TYPES):
        if tiles[t] >= 4:
            continue
        tiles[t] += 1
        if is_agari(tiles):
            tiles[t] -= 1
            return True
        tiles[t] -= 1
    return False


def get_waits(tiles: List[int]) -> List[int]:
    """返回手牌的听牌列表（所有能使之胡牌的牌型列表）。

    用于振听（フリテン）判定：玩家不能荣和自己打出过或漏过的听牌。
    """
    _build_luts()
    waits = []
    for t in range(NUM_TYPES):
        if tiles[t] >= 4:
            continue
        tiles[t] += 1
        if is_agari(tiles):
            waits.append(t)
        tiles[t] -= 1
    return waits


def is_agari_with_tile(tiles: List[int], new_tile: int) -> bool:
    """探测：加入 new_tile 后是否能胡牌（用于荣和/自摸判定）。"""
    if tiles[new_tile] >= 4:
        return False
    tiles[new_tile] += 1
    result = is_agari(tiles)
    tiles[new_tile] -= 1
    return result


def get_legal_discards_for_riichi(tiles: List[int]) -> List[int]:
    """返回可以立直的切牌选择列表。

    从 14 张手牌中逐一尝试切出一张，若剩余 13 张听牌则该切牌合法。
    """
    results = []
    for t in range(NUM_TYPES):
        if tiles[t] > 0:
            tiles[t] -= 1
            if is_tenpai(tiles):
                results.append(t)
            tiles[t] += 1
    return results


def can_riichi(tiles: List[int], is_menzen: bool,
               score: int, riichi_stick: int = 1000) -> bool:
    """判断是否可以宣告立直。

    条件：
      1. 门清（无明面副露）
      2. 分数 ≥ 立直棒费用（通常 1000 点）
      3. 至少有一种切牌能让手牌听牌
    """
    if not is_menzen:
        return False
    if score < riichi_stick:
        return False
    return len(get_legal_discards_for_riichi(tiles)) > 0


# ── 手牌分解工具 ────────────────────────────────────────────────────────────

def _split_hand(tiles: List[int]) -> Tuple[List[int], List[int], List[int], List[int]]:
    """将手牌直方图拆分为 4 个花色部分。"""
    return tiles[0:9], tiles[9:18], tiles[18:27], tiles[27:34]


def count_pairs(tiles: List[int]) -> int:
    """统计数据中出现了几个对子（每种牌 ≥ 2 张计为 1 对）。"""
    return sum(1 for c in tiles if c >= 2)


def count_melds_possible(tiles: List[int]) -> int:
    """估算最多能组成的面子数。"""
    total = sum(tiles)
    return total // 3
