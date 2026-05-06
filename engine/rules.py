"""规则配置 —— 可开关的日麻规则参数。

默认值采用雀魂（Mahjong Soul）标准竞技规则。
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class GameConfig:
    """可配置的日麻游戏规则。

    Attributes:
        start_score: 初始持点（每人均分）
        target_score: 返点目标（用于オカ计算和庄家返点判定）
        riichi_stick_cost: 立直棒费用（宣告立直时扣除）
        honba_bonus: 每本场的额外支付额
        kuitan: 食い断（副露断幺九是否成立）
        aka_dora: 是否使用赤宝牌（3 张赤 5）
        ryanhan_shibari: 二翻縛り（5 本场以上需要 2 翻起和）
        open_riichi: 是否允许开立直
        uma: 顺位马点（1位, 2位, 3位, 4位），千点单位
        oka: オカ（额外返点差）
        east_only: 東風戦（True）还是半荘戦（False）
        agari_yame: 和了り止め（最终局庄家和牌后可选择终局）
        tenpai_renchan: 听牌连荘（荒牌流局时庄家听牌则连荘）
        tobi: 飛び（分数为负即淘汰）
        multiple_ron: 允许多家荣和（ダブロン/トリプルロン）
        use_red_dora: 牌山中是否包含赤宝牌
    """
    # ── 基础设定 ──
    start_score: int = 25000          # 初始持点
    target_score: int = 30000         # 返点目标
    riichi_stick_cost: int = 1000     # 立直棒费用
    honba_bonus: int = 300            # 本场积点

    # ── 役种开关 ──
    kuitan: bool = True               # 食い断（副露断幺允许）
    aka_dora: bool = True             # 赤ドラ（赤宝牌）
    ryanhan_shibari: bool = False     # 二翻縛（5 本场以上强制 2 翻）
    kuikae: bool = False              # 食い替え禁止
    atozuke: bool = True              # 後付け（无役也能立直）
    open_riichi: bool = False         # 开立直

    # ── 终局设定 ──
    uma: Tuple[int, int, int, int] = (20, 10, -10, -20)  # 顺位马点（千点）
    oka: int = 0                      # オカ
    yakuman_multiple: bool = True     # 複合役満（役满叠加）

    # ── 对局长度 ──
    rounds: int = 1                   # 半荘数（通常为 1）
    east_only: bool = False           # 東風戦（True = 4 局）/ 半荘（False = 8 局）
    agari_yame: bool = True           # 和了り止め
    tenpai_renchan: bool = True       # 听牌连荘

    # ── 特殊规则 ──
    tobi: bool = False                # 飛び（负分即终局）
    wareme: bool = False              # 割れ目（非标准）
    multiple_ron: bool = True         # 多家荣和

    # ── 特殊牌 ──
    use_red_dora: bool = True         # 牌山中包含 3 张赤宝牌

    @property
    def total_start_points(self) -> int:
        """全场初始总点数。"""
        return self.start_score * 4

    @property
    def rounds_to_play(self) -> int:
        """总局数（東風 = 4, 半荘 = 8）。"""
        if self.east_only:
            return 4   # 東1～東4
        return 8        # 東1～東4 + 南1～南4
