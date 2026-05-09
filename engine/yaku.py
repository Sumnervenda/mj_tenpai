"""役种判定系统 —— 检测和牌手牌中成立的全部役。

判定流程：
  1. 构造 WinContext（手牌、场况、和牌方式）
  2. YakuChecker 将手牌分解为面子 + 雀头
  3. 依次检查 1 翻～役满共 38 种役
  4. 役满优先；非役满累加翻数 + 宝牌翻数

役种列表（雀魂规则）：

  1翻（门清限定）：立直, 一発, 门清自摸, 平和, 断幺九, 一盃口
  1翻（副露可）：役牌（场风/自风/白/発/中）
  2翻（门清）：双立直, 七対子
  2翻（副露 1 翻）：混全帯么九, 一気通貫, 三色同順
  2翻（副露可）：対々和, 三暗刻, 三槓子, 小三元, 混老頭
  3翻（副露 2 翻）：混一色, 純全帯么九
  3翻（门清）：二盃口
  6翻（副露 5 翻）：清一色
  役满：国士無双, 四暗刻, 大三元, 小四喜, 大四喜, 字一色, 緑一色, 清老頭, 九蓮宝燈, 天和, 地和
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Set, Tuple

from .tile import (
    NUM_TYPES, TANYAO_TYPES, YAOCHUHAI_TYPES, ROTOHAI_TYPES,
    GREEN_TYPES, JIHAI_TYPES,
    is_jihai, is_kazehai, is_sangenhai, is_shupai, is_yaochuhai,
    TILE_NUMBERS,
)
from .agari import _is_kokushi, _is_chiitoitsu
from .hand import Meld, MeldType


# ── 役种枚举 ─────────────────────────────────────────────────────────────────

class Yaku(IntEnum):
    """全部役种，大致按翻数排列。"""
    # 1翻（门清限定）
    RIICHI = 1              # 立直
    IPPATSU = 2             # 一発
    MENZEN_TSUMO = 3        # 門前清自摸和
    PINFU = 4               # 平和
    TANYAO = 5              # 断幺九
    IIPEIKOU = 6            # 一盃口
    # 1翻（副露可）
    YAKUHAI_BAKAZE = 7      # 役牌：場風
    YAKUHAI_JIKAZE = 8      # 役牌：自風
    YAKUHAI_HAK = 9         # 役牌：白
    YAKUHAI_HAT = 10        # 役牌：發
    YAKUHAI_CHU = 11        # 役牌：中
    # 2翻
    DOUBLE_RIICHI = 12      # 両立直
    CHIITOITSU = 13         # 七対子
    CHANTA = 14             # 混全帯么九（门清 2 翻, 副露 1 翻）
    ITTSUU = 15             # 一気通貫（门清 2 翻, 副露 1 翻）
    SANSHOKU_DOUJUN = 16    # 三色同順（门清 2 翻, 副露 1 翻）
    TOITOI = 17             # 対々和
    SANANKOU = 18           # 三暗刻
    SANKANTSU = 19          # 三槓子
    SHOUSANGEN = 20         # 小三元
    HONROUTOU = 21          # 混老頭
    # 3翻
    HONITSU = 22            # 混一色（门清 3 翻, 副露 2 翻）
    JUNCHAN = 23            # 純全帯么九（门清 3 翻, 副露 2 翻）
    RYANPEIKOU = 24         # 二盃口（门清限定）
    # 6翻
    CHINITSU = 25           # 清一色（门清 6 翻, 副露 5 翻）
    # 役满
    KOKUSHI_MUSOU = 26      # 国士無双
    SUUANKOU = 27           # 四暗刻
    DAISANGEN = 28          # 大三元
    SHOUSUUSHII = 29        # 小四喜
    DAISUUSHII = 30         # 大四喜
    TSUUIISOU = 31          # 字一色
    RYUUIISOU = 32          # 緑一色
    CHINROUTOU = 33         # 清老頭
    CHUUREN_POUTOU = 34     # 九蓮宝燈
    TENHOU = 35             # 天和
    CHIIHOU = 36            # 地和
    SUUANKOU_TANKI = 37     # 四暗刻単騎
    KOKUSHI_13_WAIT = 38    # 国士無双13面待ち


# 役种日文名
YAKU_NAMES_JP: Dict[Yaku, str] = {
    Yaku.RIICHI: "立直", Yaku.IPPATSU: "一発", Yaku.MENZEN_TSUMO: "門前清自摸和",
    Yaku.PINFU: "平和", Yaku.TANYAO: "断幺九", Yaku.IIPEIKOU: "一盃口",
    Yaku.YAKUHAI_BAKAZE: "場風牌", Yaku.YAKUHAI_JIKAZE: "自風牌",
    Yaku.YAKUHAI_HAK: "白", Yaku.YAKUHAI_HAT: "發", Yaku.YAKUHAI_CHU: "中",
    Yaku.DOUBLE_RIICHI: "両立直", Yaku.CHIITOITSU: "七対子",
    Yaku.CHANTA: "混全帯么九", Yaku.ITTSUU: "一気通貫",
    Yaku.SANSHOKU_DOUJUN: "三色同順", Yaku.TOITOI: "対々和",
    Yaku.SANANKOU: "三暗刻", Yaku.SANKANTSU: "三槓子",
    Yaku.SHOUSANGEN: "小三元", Yaku.HONROUTOU: "混老頭",
    Yaku.HONITSU: "混一色", Yaku.JUNCHAN: "純全帯么九",
    Yaku.RYANPEIKOU: "二盃口", Yaku.CHINITSU: "清一色",
    Yaku.KOKUSHI_MUSOU: "国士無双", Yaku.SUUANKOU: "四暗刻",
    Yaku.DAISANGEN: "大三元", Yaku.SHOUSUUSHII: "小四喜",
    Yaku.DAISUUSHII: "大四喜", Yaku.TSUUIISOU: "字一色",
    Yaku.RYUUIISOU: "緑一色", Yaku.CHINROUTOU: "清老頭",
    Yaku.CHUUREN_POUTOU: "九蓮宝燈", Yaku.TENHOU: "天和",
    Yaku.CHIIHOU: "地和", Yaku.SUUANKOU_TANKI: "四暗刻単騎",
    Yaku.KOKUSHI_13_WAIT: "国士無双13面",
}


# ── 和牌上下文 ───────────────────────────────────────────────────────────────

@dataclass
class WinContext:
    """役种判定所需的全部上下文信息。"""
    is_menzen: bool = True          # 是否门前清
    is_tsumo: bool = True           # 自摸(True) vs 荣和(False)
    is_riichi: bool = False         # 是否立直
    is_ippatsu: bool = False        # 是否一発（立直后一巡内和牌）
    is_double_riichi: bool = False  # 是否双立直
    is_tenhou: bool = False         # 天和（庄家配牌即和）
    is_chiihou: bool = False        # 地和（闲家第一巡自摸和）
    is_haitei: bool = False         # 海底撈月 / 河底撈魚
    is_rinshan: bool = False        # 嶺上開花
    is_chankan: bool = False        # 槍槓
    bakaze: int = 27                # 場風（27=東, 28=南）
    jikaze: int = 27                # 自風
    kuitan: bool = True             # 是否允许食断（副露断幺九）
    open_melds: List[Meld] = field(default_factory=list)
    concealed_tiles: List[int] = field(default_factory=lambda: [0] * NUM_TYPES)
    winning_tile: int = -1          # 和了牌
    dora_count: int = 0             # 宝牌数（仅显示用）


# ── 手牌分解器 ───────────────────────────────────────────────────────────────

@dataclass
class HandDecomposition:
    """手牌分解结果：面子列表 + 雀头。"""
    melds: List[Tuple[str, List[int]]]   # ('shuntsu'|'koutsu'|'kantsu', [牌型列表])
    pair: Tuple[int, int]                # (雀头牌型, 雀头牌型)
    waits: List[int]                     # 听牌列表


def decompose_hand(tiles: List[int]) -> Optional[HandDecomposition]:
    """将已胡牌的门内手牌分解为 4 面子 + 1 雀头。

    穷举所有合法分解，返回顺子数最多（刻子数最少）的分解，
    以最大化平和、一盃口等顺子系役种的检测率。
    前提：tiles 已经 is_agari 验证，不会返回 None。
    """
    counts = list(tiles)
    best_melds = None
    best_pair = None
    best_shuntsu = -1

    def _find_first(c: List[int]) -> Optional[int]:
        for i in range(NUM_TYPES):
            if c[i] > 0:
                return i
        return None

    def _solve(c: List[int], need_pair: bool,
               melds: list, pair: Optional[tuple]) -> None:
        nonlocal best_melds, best_pair, best_shuntsu
        first = _find_first(c)
        if first is None:
            if not need_pair:
                shuntsu_count = sum(1 for m in melds if m[0] == 'shuntsu')
                if shuntsu_count > best_shuntsu:
                    best_melds = list(melds)
                    best_pair = pair
                    best_shuntsu = shuntsu_count
            return

        cnt = c[first]

        # 尝试取雀头
        if need_pair and cnt >= 2:
            c[first] -= 2
            _solve(c, False, melds, (first, first))
            c[first] += 2

        # 尝试取刻子
        if cnt >= 3:
            c[first] -= 3
            melds.append(('koutsu', [first, first, first]))
            _solve(c, need_pair, melds, pair)
            melds.pop()
            c[first] += 3

        # 尝试取顺子
        if is_shupai(first) and (first % 9) <= 6:
            if c[first + 1] > 0 and c[first + 2] > 0:
                c[first] -= 1
                c[first + 1] -= 1
                c[first + 2] -= 1
                melds.append(('shuntsu', [first, first + 1, first + 2]))
                _solve(c, need_pair, melds, pair)
                melds.pop()
                c[first] += 1
                c[first + 1] += 1
                c[first + 2] += 1

    _solve(counts, True, [], None)

    if best_melds is not None:
        return HandDecomposition(
            melds=best_melds, pair=best_pair or (-1, -1), waits=[])
    return None


def _decompose_koutsu_first(tiles: List[int]) -> Optional[HandDecomposition]:
    """刻子优先的手牌分解，供対々和/三暗刻等刻子系役种独立使用。

    与旧版 decompose_hand 逻辑一致：递归回溯，先试刻子再试顺子，
    返回第一个合法分解。
    """
    melds = []
    pair = None
    counts = list(tiles)

    def _find_first(c: List[int]) -> Optional[int]:
        for i in range(NUM_TYPES):
            if c[i] > 0:
                return i
        return None

    def _solve(c: List[int], need_pair: bool) -> bool:
        nonlocal melds, pair
        first = _find_first(c)
        if first is None:
            return not need_pair

        cnt = c[first]

        if need_pair and cnt >= 2:
            c[first] -= 2
            pair = (first, first)
            if _solve(c, False):
                return True
            c[first] += 2
            pair = None

        if cnt >= 3:
            c[first] -= 3
            melds.append(('koutsu', [first, first, first]))
            if _solve(c, need_pair):
                return True
            melds.pop()
            c[first] += 3

        if is_shupai(first) and (first % 9) <= 6:
            if c[first + 1] > 0 and c[first + 2] > 0:
                c[first] -= 1
                c[first + 1] -= 1
                c[first + 2] -= 1
                melds.append(('shuntsu', [first, first + 1, first + 2]))
                if _solve(c, need_pair):
                    return True
                melds.pop()
                c[first] += 1
                c[first + 1] += 1
                c[first + 2] += 1

        return False

    if _solve(counts, True):
        return HandDecomposition(
            melds=list(melds), pair=pair or (-1, -1), waits=[])
    return None


def decompose_chiitoitsu(tiles: List[int]) -> HandDecomposition:
    """为七对子创建伪分解（7 个"对子"面子）。"""
    pairs = []
    for t in range(NUM_TYPES):
        if tiles[t] == 2:
            pairs.append((t, t))
    return HandDecomposition(
        melds=[('pair', [a, b]) for a, b in pairs],
        pair=pairs[-1],
        waits=[]
    )


# ── 役种判定结果 ─────────────────────────────────────────────────────────────

@dataclass
class YakuResult:
    """役种判定结果。"""
    yaku_list: List[Tuple[Yaku, int]]   # [(役种, 翻数), ...]
    total_han: int                       # 总翻数（不含宝牌）
    yakuman_list: List[Yaku]             # 役满役种列表
    is_yakuman: bool                     # 是否役满


class YakuChecker:
    """役种判定器：对一副已胡牌的手牌检测全部役种。

    Usage:
        ctx = WinContext(...)
        checker = YakuChecker(ctx)
        result = checker.check_all()
        print(result.total_han, result.yaku_list)
    """

    def __init__(self, ctx: WinContext):
        self.ctx = ctx
        # 合并门内牌 + 副露牌为完整手牌视图
        self.all_tiles = list(ctx.concealed_tiles)
        for meld in ctx.open_melds:
            for t in meld.tiles:
                self.all_tiles[t] += 1

        self.decomp: Optional[HandDecomposition] = None
        self._init_decomp()
        self.all_melds = self._build_all_melds()

    def _build_all_melds(self) -> List[Tuple[str, List[int]]]:
        """构建全部面子列表：副露面子（固定单位）+ 门内分解面子。"""
        all_melds: List[Tuple[str, List[int]]] = []
        for meld in self.ctx.open_melds:
            if meld.meld_type == MeldType.CHI:
                all_melds.append(('shuntsu', list(meld.tiles)))
            else:
                # PON / KAN → 全部视为刻子（杠取前 3 张表示刻子）
                all_melds.append(('koutsu', list(meld.tiles[:3])))
        if self.decomp:
            all_melds.extend(self.decomp.melds)
        return all_melds

    def _init_decomp(self):
        """初始化解：识别特殊形（国士）或进行一般形分解。

        优先尝试一般形分解（4 面子 + 1 雀头），失败时再尝试七对子。
        这是为二盃口（3 翻）优先级高于七対子（2 翻）所做的处理：
        当手牌可同时解释为二盃口和七対子时，取高翻数的二盃口。
        """
        tiles = self.ctx.concealed_tiles
        if _is_kokushi(tiles):
            self.decomp = None  # 国士无面子/雀头结构
            return
        self.decomp = decompose_hand(tiles)
        if self.decomp is None and _is_chiitoitsu(tiles):
            self.decomp = decompose_chiitoitsu(tiles)

    def check_all(self) -> YakuResult:
        """执行全部役种判定，返回结果。"""
        # 先检查役满（役满优先，不计普通役）
        yakuman_list = list(self._check_yakuman())
        if yakuman_list:
            return YakuResult(
                yaku_list=[],
                total_han=0,
                yakuman_list=yakuman_list,
                is_yakuman=True,
            )

        # 检查普通役
        yaku_list = list(self._check_regular_yaku())
        total_han = sum(han for _, han in yaku_list)
        return YakuResult(
            yaku_list=yaku_list,
            total_han=total_han,
            yakuman_list=[],
            is_yakuman=False,
        )

    def _check_regular_yaku(self) -> List[Tuple[Yaku, int]]:
        """检查全部非役满役种。

        役种互斥规则（按日麻标准）：
          1. 包含关系 — 纯全帯么九 ⊃ 混全帯么九；二盃口 ⊃ 一盃口；清一色 ⊃ 混一色
          2. 定义冲突 — 断幺九 vs 役牌/混一色等（自然互斥，无需额外处理）
          3. 结构冲突 — 七対子 vs 平和/一盃口/対々和（自然互斥）
        """
        results = []
        ctx = self.ctx
        tiles = ctx.concealed_tiles

        # —— 3 翻以上先判断，供后续互斥排除 ——
        junchan = self._check_junchan()        # 純全帯么九（⊃ 混全帯么九）
        ryanpeikou = ctx.is_menzen and self._check_ryanpeikou()  # 二盃口（⊃ 一盃口）
        chinitsu = self._check_chinitsu()      # 清一色（⊃ 混一色）
        honitsu = self._check_honitsu()        # 混一色

        # ── 1翻役 ────────────────────────────────────────────────────────

        # 立直
        if ctx.is_riichi:
            if ctx.is_double_riichi:
                results.append((Yaku.DOUBLE_RIICHI, 2))
            else:
                results.append((Yaku.RIICHI, 1))

        # 一発（可与立直重叠）
        if ctx.is_ippatsu:
            results.append((Yaku.IPPATSU, 1))

        # 門前清自摸和（七対子除外）
        is_chiitoitsu_decomp = self.decomp and self.decomp.melds and self.decomp.melds[0][0] == 'pair'
        if ctx.is_menzen and ctx.is_tsumo and not is_chiitoitsu_decomp:
            results.append((Yaku.MENZEN_TSUMO, 1))

        # 平和（门清限定，结构上与刻子类自然互斥）
        if ctx.is_menzen and self._check_pinfu():
            results.append((Yaku.PINFU, 1))

        # 断幺九（食断开关控制副露时是否可成立）
        if self._check_tanyao():
            if ctx.is_menzen or ctx.kuitan:
                results.append((Yaku.TANYAO, 1))

        # 一盃口（门清限定；二盃口成立时不重复计算）
        if ctx.is_menzen and self._check_iipeikou() and not ryanpeikou:
            results.append((Yaku.IIPEIKOU, 1))

        # 役牌（可与小三元重叠）
        for yaku in self._check_yakuhai():
            results.append((yaku, 1))

        # ── 2翻役 ────────────────────────────────────────────────────────

        if self.decomp and self.decomp.melds and self.decomp.melds[0][0] == 'pair':
            results.append((Yaku.CHIITOITSU, 2))

        # 混全帯么九（純全帯么九成立时被包含，不重复计算）
        if self._check_chanta() and not junchan:
            han = 2 if ctx.is_menzen else 1
            results.append((Yaku.CHANTA, han))

        ittsuu = self._check_ittsuu()
        if ittsuu:
            han = 2 if ctx.is_menzen else 1
            results.append((Yaku.ITTSUU, han))

        sanshoku = self._check_sanshoku_doujun()
        if sanshoku:
            han = 2 if ctx.is_menzen else 1
            results.append((Yaku.SANSHOKU_DOUJUN, han))

        if self._check_toitoi():
            results.append((Yaku.TOITOI, 2))

        if self._check_sanankou():
            results.append((Yaku.SANANKOU, 2))

        if self._check_sankantsu():
            results.append((Yaku.SANKANTSU, 2))

        # 小三元（与两个役牌可重叠）
        if self._check_shousangen():
            results.append((Yaku.SHOUSANGEN, 2))

        if self._check_honroutou():
            results.append((Yaku.HONROUTOU, 2))

        # ── 3翻以上 ──────────────────────────────────────────────────────

        # 混一色（清一色成立时不重复计算）
        if honitsu and not chinitsu:
            han = 3 if ctx.is_menzen else 2
            results.append((Yaku.HONITSU, han))

        if junchan:
            han = 3 if ctx.is_menzen else 2
            results.append((Yaku.JUNCHAN, han))

        if ryanpeikou:
            results.append((Yaku.RYANPEIKOU, 3))

        if chinitsu:
            han = 6 if ctx.is_menzen else 5
            results.append((Yaku.CHINITSU, han))

        return results

    def _check_yakuman(self) -> List[Yaku]:
        """检查全部役满。"""
        results = []
        ctx = self.ctx
        tiles = ctx.concealed_tiles

        if ctx.is_tenhou:
            results.append(Yaku.TENHOU)
        if ctx.is_chiihou:
            results.append(Yaku.CHIIHOU)

        if _is_kokushi(tiles):
            waits = _kokushi_waits(tiles)
            if len(waits) == 13:
                results.append(Yaku.KOKUSHI_13_WAIT)
            else:
                results.append(Yaku.KOKUSHI_MUSOU)

        sanko = self._check_suuankou()
        if sanko:
            results.append(sanko)

        if self._check_daisangen():
            results.append(Yaku.DAISANGEN)

        suushii = self._check_suushii()
        if suushii:
            results.append(suushii)

        if self._check_tsuuiisou():
            results.append(Yaku.TSUUIISOU)

        if self._check_ryuuiisou():
            results.append(Yaku.RYUUIISOU)

        if self._check_chinroutou():
            results.append(Yaku.CHINROUTOU)

        if self._check_chuuren_poutou():
            results.append(Yaku.CHUUREN_POUTOU)

        return results

    # ── 各役判定逻辑 ────────────────────────────────────────────────────

    def _check_pinfu(self) -> bool:
        """平和：全部顺子 + 雀头非役牌 + 两面听牌。"""
        if not self.decomp:
            return False
        melds = self.decomp.melds
        if not all(m[0] == 'shuntsu' for m in melds):
            return False
        pair_tile = self.decomp.pair[0]
        if _is_yakuhai_pair(pair_tile, self.ctx.bakaze, self.ctx.jikaze):
            return False
        wt = self.ctx.winning_tile
        if wt >= 0:
            if self._is_tanki_wait(wt) or self._is_kanchan_wait(wt) \
                    or self._is_penchan_wait(wt):
                return False
        return True

    def _check_tanyao(self) -> bool:
        """断幺九：手牌中没有任何幺九牌（1/9 和字牌）。"""
        return all(self.all_tiles[t] == 0 for t in YAOCHUHAI_TYPES)

    def _check_iipeikou(self) -> bool:
        """一盃口：同花色内有两组完全相同的顺子（门清限定）。"""
        if not self.decomp:
            return False
        shuntsu_list = []
        for mtype, tiles in self.decomp.melds:
            if mtype == 'shuntsu':
                shuntsu_list.append(tuple(tiles))
        return len(shuntsu_list) != len(set(shuntsu_list))

    def _check_yakuhai(self) -> List[Yaku]:
        """返回所有成立的役牌（场风/自风/三元牌各计 1 翻）。

        役牌成立条件：含有该风/三元牌的刻子（含暗刻、明刻、暗槓、明槓）。
        注意：雀头不计役牌（役牌雀头只加符不计翻）。
        """
        results = []
        honors_present: Set[int] = set()
        if self.decomp:
            # 全部面子（含副露刻子）中的字牌刻子
            for mtype, tiles in self.all_melds:
                if mtype == 'koutsu' and is_jihai(tiles[0]):
                    honors_present.add(tiles[0])

        ctx = self.ctx
        for t in honors_present:
            if t == ctx.bakaze:
                results.append(Yaku.YAKUHAI_BAKAZE)
            if t == ctx.jikaze:
                results.append(Yaku.YAKUHAI_JIKAZE)
            if t == 31:  results.append(Yaku.YAKUHAI_HAK)
            if t == 32:  results.append(Yaku.YAKUHAI_HAT)
            if t == 33:  results.append(Yaku.YAKUHAI_CHU)
        return results

    def _check_chanta(self) -> bool:
        """混全帯么九：每个面子 + 雀头均含至少 1 张幺九牌（含副露）。"""
        if not self.decomp:
            return False
        for mtype, tiles in self.all_melds:
            if not any(is_yaochuhai(t) for t in tiles):
                return False
        if not is_yaochuhai(self.decomp.pair[0]):
            return False
        return True

    def _check_ittsuu(self) -> bool:
        """一気通貫：同花色内 123 + 456 + 789 三组顺子（含副露）。"""
        if not self.decomp:
            return False
        for suit_start in [0, 9, 18]:
            has_low = has_mid = has_high = False
            for mtype, tiles in self.all_melds:
                if mtype == 'shuntsu':
                    if tiles == [suit_start, suit_start + 1, suit_start + 2]:
                        has_low = True
                    if tiles == [suit_start + 3, suit_start + 4, suit_start + 5]:
                        has_mid = True
                    if tiles == [suit_start + 6, suit_start + 7, suit_start + 8]:
                        has_high = True
            if has_low and has_mid and has_high:
                return True
        return False

    def _check_sanshoku_doujun(self) -> bool:
        """三色同順：三种花色有同一数字序列的顺子（含副露）。"""
        if not self.decomp:
            return False
        seq_set = set()
        for mtype, tiles in self.all_melds:
            if mtype == 'shuntsu':
                seq_set.add(tuple(TILE_NUMBERS[t] for t in tiles))
        for seq in seq_set:
            found_suits = set()
            for mtype, tiles in self.all_melds:
                if mtype == 'shuntsu' and tuple(TILE_NUMBERS[t] for t in tiles) == seq:
                    found_suits.add(tiles[0] // 9)
            if len(found_suits) >= 3:
                return True
        return False

    def _check_toitoi(self) -> bool:
        """対々和：全部 4 组面子为刻子 + 1 对雀头（含副露）。

        使用独立的刻子优先分解，不依赖主（顺子优先）分解。
        """
        # 副露中不可有吃
        for meld in self.ctx.open_melds:
            if meld.meld_type == MeldType.CHI:
                return False
        # 门内牌可用刻子优先分解
        return _decompose_koutsu_first(list(self.ctx.concealed_tiles)) is not None

    def _check_sanankou(self) -> bool:
        """三暗刻：3 组门内刻子（含暗槓）。

        使用独立的刻子优先分解以最大化暗刻计数。
        """
        closed_koutsu = 0
        koutsu_decomp = _decompose_koutsu_first(list(self.ctx.concealed_tiles))
        if koutsu_decomp:
            for mtype, tiles in koutsu_decomp.melds:
                if mtype == 'koutsu':
                    if not self.ctx.is_tsumo and self.ctx.winning_tile in tiles:
                        continue  # 荣和完成的刻子不算暗刻
                    closed_koutsu += 1
        for meld in self.ctx.open_melds:
            if meld.meld_type == MeldType.KAN_CLOSED:
                closed_koutsu += 1
        return closed_koutsu >= 3

    def _check_sankantsu(self) -> bool:
        """三槓子：3 组杠子（含明杠）。"""
        kan_count = sum(1 for m in self.ctx.open_melds if m.is_kan)
        return kan_count >= 3

    def _check_shousangen(self) -> bool:
        """小三元：2 组三元刻子 + 三元雀头（含副露）。"""
        if not self.decomp:
            return False
        dragon_koutsu = 0
        for mtype, tiles in self.all_melds:
            if mtype == 'koutsu' and is_sangenhai(tiles[0]):
                dragon_koutsu += 1
        dragon_pair = is_sangenhai(self.decomp.pair[0])
        return dragon_koutsu >= 2 and dragon_pair

    def _check_honroutou(self) -> bool:
        """混老頭：全部牌均为幺九牌（含至少一种数牌）。"""
        return all(
            is_yaochuhai(t) for t in range(NUM_TYPES) if self.all_tiles[t] > 0
        ) and any(is_shupai(t) for t in range(NUM_TYPES) if self.all_tiles[t] > 0)

    def _check_honitsu(self) -> bool:
        """混一色：仅含一种数牌花色 + 字牌。"""
        suits_present = set()
        for t in range(NUM_TYPES):
            if self.all_tiles[t] > 0:
                if is_shupai(t):
                    suits_present.add(t // 9)
        return len(suits_present) == 1 and any(
            is_jihai(t) for t in range(NUM_TYPES) if self.all_tiles[t] > 0
        )

    def _check_junchan(self) -> bool:
        """純全帯么九：每个面子 + 雀头均含至少 1 张老頭牌（1 或 9），不含字牌（含副露）。"""
        if not self.decomp:
            return False
        for mtype, tiles in self.all_melds:
            if not any(t in ROTOHAI_TYPES for t in tiles):
                return False
        if self.decomp.pair[0] not in ROTOHAI_TYPES:
            return False
        if any(self.all_tiles[t] > 0 for t in JIHAI_TYPES):
            return False
        if all(t in ROTOHAI_TYPES for t in range(NUM_TYPES) if self.all_tiles[t] > 0):
            return False  # 排除清老頭
        return True

    def _check_ryanpeikou(self) -> bool:
        """二盃口：两对不同的一盃口（4 组顺子中两两相同）。"""
        if not self.decomp or not self.ctx.is_menzen:
            return False
        shuntsu_list = [tuple(tiles) for mtype, tiles in self.decomp.melds
                        if mtype == 'shuntsu']
        if len(shuntsu_list) != 4:
            return False
        from collections import Counter
        counts = Counter(shuntsu_list)
        pairs = [seq for seq, cnt in counts.items() if cnt == 2]
        return len(pairs) == 2

    def _check_chinitsu(self) -> bool:
        """清一色：全部牌来自同一种数牌花色，无字牌。"""
        suits_present = set()
        for t in range(NUM_TYPES):
            if self.all_tiles[t] > 0:
                if is_shupai(t):
                    suits_present.add(t // 9)
                elif is_jihai(t):
                    return False
        return len(suits_present) == 1

    # ── 役满判定 ────────────────────────────────────────────────────────

    def _check_suuankou(self) -> Yaku:
        """四暗刻：4 组门内刻子 + 1 对雀头（含暗槓）。单骑待ち为四暗刻単騎。"""
        if not self.decomp:
            return None
        # 四暗刻必须全部为门内刻子，有明面副露则不可能
        for meld in self.ctx.open_melds:
            if meld.is_open:
                return None
        closed_koutsu = 0
        for mtype, tiles in self.decomp.melds:
            if mtype == 'koutsu':
                if not self.ctx.is_tsumo and self.ctx.winning_tile in tiles:
                    return None  # 荣和完成的刻子不算暗刻
                closed_koutsu += 1
        for meld in self.ctx.open_melds:
            if meld.meld_type == MeldType.KAN_CLOSED:
                closed_koutsu += 1
        if closed_koutsu == 4:
            pair_tile = self.decomp.pair[0]
            if self.ctx.winning_tile == pair_tile:
                return Yaku.SUUANKOU_TANKI
            return Yaku.SUUANKOU
        return None

    def _check_daisangen(self) -> bool:
        """大三元：白・発・中三组刻子（含副露）。"""
        if not self.decomp:
            return False
        dragon_koutsu = 0
        for mtype, tiles in self.all_melds:
            if mtype == 'koutsu' and is_sangenhai(tiles[0]):
                dragon_koutsu += 1
        return dragon_koutsu >= 3

    def _check_suushii(self) -> Yaku:
        """小四喜（3 风刻 + 风雀头）/ 大四喜（4 风刻，含副露）。"""
        if not self.decomp:
            return None
        wind_koutsu = 0
        for mtype, tiles in self.all_melds:
            if mtype == 'koutsu' and is_kazehai(tiles[0]):
                wind_koutsu += 1
        wind_pair = is_kazehai(self.decomp.pair[0])
        if wind_koutsu == 4:
            return Yaku.DAISUUSHII
        elif wind_koutsu == 3 and wind_pair:
            return Yaku.SHOUSUUSHII
        return None

    def _check_tsuuiisou(self) -> bool:
        """字一色：全部牌均为字牌。"""
        return all(
            self.all_tiles[t] == 0 for t in range(NUM_TYPES) if not is_jihai(t)
        )

    def _check_ryuuiisou(self) -> bool:
        """緑一色：仅含绿牌（2s, 3s, 4s, 6s, 8s, 発）。"""
        return all(
            self.all_tiles[t] == 0 for t in range(NUM_TYPES) if t not in GREEN_TYPES
        )

    def _check_chinroutou(self) -> bool:
        """清老頭：全部牌为老頭牌（1 和 9 的数牌），无字牌。"""
        return all(
            self.all_tiles[t] == 0 for t in range(NUM_TYPES) if t not in ROTOHAI_TYPES
        )

    def _check_chuuren_poutou(self) -> bool:
        """九蓮宝燈：同花色 1112345678999 + 任意一张（门前清限定）。"""
        if not self.ctx.is_menzen:
            return False
        for suit_start in [0, 9, 18]:
            suit_end = suit_start + 9
            suit_tiles = self.all_tiles[suit_start:suit_end]
            if suit_tiles[0] >= 3 and suit_tiles[8] >= 3 and \
               all(suit_tiles[i] >= 1 for i in range(1, 8)):
                other = sum(self.all_tiles[:suit_start]) + \
                        sum(self.all_tiles[suit_end:27]) + \
                        sum(self.all_tiles[27:])
                if other == 0:
                    return True
        return False

    # ── 听牌形判定 ──────────────────────────────────────────────────────

    def _is_tanki_wait(self, winning_tile: int) -> bool:
        """単騎待ち：雀头待ち。"""
        if not self.decomp:
            return False
        return winning_tile == self.decomp.pair[0]

    def _is_kanchan_wait(self, winning_tile: int) -> bool:
        """嵌張待ち：听顺子中间张（如 46 等 5）。"""
        if not self.decomp:
            return False
        for mtype, tiles in self.decomp.melds:
            if mtype == 'shuntsu' and winning_tile == tiles[1]:
                return True
        return False

    def _is_penchan_wait(self, winning_tile: int) -> bool:
        """辺張待ち：听 12 的 3 或 89 的 7。"""
        if not self.decomp:
            return False
        for mtype, tiles in self.decomp.melds:
            if mtype == 'shuntsu':
                seq = tiles
                if seq[0] % 9 == 0 and winning_tile == seq[2]:    # 12 + 3
                    return True
                if seq[2] % 9 == 8 and winning_tile == seq[0]:    # 7 + 89
                    return True
        return False


# ── 独立辅助函数 ─────────────────────────────────────────────────────────────

def _is_yakuhai_pair(tile_type: int, bakaze: int, jikaze: int) -> bool:
    """判断某牌型是否为役牌对子。"""
    if tile_type == bakaze or tile_type == jikaze:
        return True
    if is_sangenhai(tile_type):
        return True
    return False


def _kokushi_waits(tiles: List[int]) -> List[int]:
    """返回国士手牌的听牌列表（用于判断 13 面听）。"""
    yaochu = list(YAOCHUHAI_TYPES)
    waits = []
    for t in yaochu:
        if tiles[t] == 1:
            waits.append(t)
    return waits
# 中文注释：检测立直麻将役种与部分符计算辅助逻辑，为 scoring 模块提供役种结果。
