"""游戏引擎 —— 事件驱动的日麻状态机。

协调完整的游戏流程：
  DEAL → DRAW_STATE ↔ DISCARD_STATE → AGARI / RYUUKYOKU → 下一局 → END

支持 AI 训练接口：
  - step(action) → (next_state_info, reward, done)    类 Gym 接口
  - get_legal_actions() → 动作掩码向量
  - clone() → 深拷贝，供 MCTS 模拟使用
  - get_state_tensor(player_idx) → 神经网络输入特征张量
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .tile import (
    NUM_TYPES, NUM_ABS,
    MANZU, PINZU, SOUZU, JIHAI,
    abs_to_type, type_to_abs, is_aka, is_aka_type,
    is_jihai, is_shupai, is_yaochuhai,
    TILE_NAMES, TILE_NUMBERS,
    YAOCHUHAI_TYPES,
)
from .wall import Wall, count_dora_in_hand
from .hand import (
    Hand, Meld, MeldType,
    can_chi, can_pon, can_daiminkan, can_kakan, can_ankan,
)
from .agari import (
    is_agari, is_tenpai, get_waits, is_agari_with_tile,
    get_legal_discards_for_riichi, can_riichi,
)
from .yaku import (
    YakuChecker, WinContext, YakuResult,
    _is_kokushi, _is_chiitoitsu, Yaku, YAKU_NAMES_JP,
)
from .scoring import (
    calculate_fu_from_decomp, compute_payments, compute_final_result,
    PaymentInfo, GameResult,
)
from .actions import (
    Action, ActionType, LegalActions,
    compute_draw_actions, compute_response_actions,
    MAX_ACTIONS,
)
from .rules import GameConfig


# ── 游戏阶段枚举 ──────────────────────────────────────────────────────────────

class GamePhase(IntEnum):
    """游戏阶段枚举。"""
    DEAL = 0           # 配牌阶段（发牌）
    DRAW = 1           # 摸牌阶段（当前玩家摸牌后需决策）
    DISCARD = 2        # 舍牌阶段（等待其他玩家响应）
    RESPONSE = 3       # 响应收集阶段（内部子状态）
    AGARI = 4          # 和了（有人胡牌）
    RYUUKYOKU = 5      # 流局（荒牌流局）
    ROUND_END = 6      # 局间过渡
    GAME_END = 7       # 游戏结束


# ── 玩家状态 ──────────────────────────────────────────────────────────────────

@dataclass
class PlayerState:
    """一局游戏中单个玩家的可变状态。

    Attributes:
        hand: 手牌对象（直方图 + 副露面子的列表）
        score: 当前分数（持点）
        seat_wind: 自风（27=東, 28=南, 29=西, 30=北）
        is_riichi: 是否已立直
        is_double_riichi: 是否为ダブル立直（第一巡立直）
        is_ippatsu: 是否在一発有效期内
        has_won: 本局是否已和了
        discards: 舍牌列表（绝对 ID）
        discard_types: 舍牌的牌型集合（用于振听判定）
        furiten_types: 振听牌型集合（永久振听）
        temp_furiten: 临时振听标记（摸牌后清除）
        is_riichi_furiten: 立直后永久振听（错过和牌机会后永久生效）
        is_tenpai_at_ryuukyoku: 流局时是否听牌
    """
    hand: Hand = field(default_factory=Hand)
    score: int = 25000
    seat_wind: int = 27          # 東=27, 南=28, 西=29, 北=30
    is_riichi: bool = False
    is_double_riichi: bool = False
    is_ippatsu: bool = False     # 立直后一巡内有效
    has_won: bool = False
    discards: List[int] = field(default_factory=list)          # 舍牌的绝对 ID 列表
    discard_types: Set[int] = field(default_factory=set)       # 舍牌的牌型集合（振听判定用）
    furiten_types: Set[int] = field(default_factory=set)       # 振听牌型集合
    temp_furiten: bool = False   # 临时振听（摸牌后自动清除）
    is_riichi_furiten: bool = False  # 立直后永久振听（错过和牌机会后永久生效）
    is_tenpai_at_ryuukyoku: bool = False  # 流局听牌标记（用于听牌连荘）

    @property
    def is_menzen(self) -> bool:
        """是否是门清状态（无明面副露）。"""
        return self.hand.is_menzen

    def add_discard(self, abs_id: int) -> None:
        """记录一张舍牌，同时更新 discard_types（振听判定用）。"""
        self.discards.append(abs_id)
        t = abs_to_type(abs_id)
        self.discard_types.add(t)

    def clear_round_state(self) -> None:
        """重置每局状态，保留分数。"""
        self.hand = Hand()
        self.is_riichi = False
        self.is_double_riichi = False
        self.is_ippatsu = False
        self.has_won = False
        self.discards = []
        self.discard_types = set()
        self.furiten_types = set()
        self.temp_furiten = False
        self.is_riichi_furiten = False
        self.is_tenpai_at_ryuukyoku = False


# ── 游戏主引擎 ────────────────────────────────────────────────────────────────

@dataclass
class GameState:
    """游戏状态快照，供 AI 消费。

    包含完整信息用于神经网络输入和训练数据记录。
    """
    phase: GamePhase                     # 当前游戏阶段
    current_player: int                  # 当前决策玩家索引 (0-3)
    round_wind: int                      # 场风（27=東, 28=南）
    round_number: int                    # 局数（1-4 東, 1-4 南）
    honba: int                           # 本场数
    riichi_sticks: int                   # 场上立直棒数量
    scores: List[int]                    # 各玩家分数 [p0, p1, p2, p3]
    hands_concealed: List[List[int]]     # 各玩家门内手牌（int[34] 直方图）
    open_melds: List[List[Meld]]        # 各玩家副露面子的列表
    discards: List[List[int]]            # 各玩家舍牌的绝对 ID 列表
    dora_indicators: List[int]           # 已翻开的宝牌指示牌的绝对 ID
    ura_dora_indicators: List[int]       # 已翻开的里宝牌指示牌的绝对 ID
    is_riichi: List[bool]                # 各玩家立直状态
    last_discard: int                    # 最近一次舍牌的绝对 ID，-1 表示无
    last_discard_by: int                 # 最近舍牌玩家的索引
    remaining_tiles: int                 # 牌山中剩余牌数
    legal_actions: Optional[LegalActions] = None  # 当前合法动作（可选）
    rewards: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])  # 分数变动
    done: bool = False                   # 游戏是否结束


class GameEngine:
    """完整的竞技麻将游戏引擎。

    用法:
        engine = GameEngine()
        engine.init_game()

        while not engine.is_game_over():
            state = engine.get_game_state()
            actions = engine.get_legal_actions()
            action = choose_action(actions)  # AI 或人类选择动作
            engine.step(action)

        result = engine.get_result()
    """

    def __init__(self, config: Optional[GameConfig] = None, seed: Optional[int] = None):
        self.config = config or GameConfig()
        self.wall = Wall(seed=seed)
        self.players: List[PlayerState] = []
        self.phase = GamePhase.DEAL
        self.round_wind = 27     # 场风初始为東
        self.round_number = 1
        self.honba = 0           # 本场数
        self.riichi_sticks = 0   # 场上积存的立直棒数量
        self.dealer_idx = 0      # 当前庄家索引 (0-3)
        self.current_player = 0  # 当前摸牌/切牌玩家
        self.last_discard = -1   # 最近舍牌的绝对 ID
        self.last_discard_by = -1  # 最近舍牌的玩家
        self.ura_dora_indicators: List[int] = []  # 里宝牌指示牌
        self._round_history: List[Dict] = []       # 局历史记录
        self._game_finished = False                # 游戏终结标记

        self._last_drawn_tile: int = -1  # 最近一次摸到的牌型
        self._last_winner: int = -1      # 最近一次和了的玩家
        self._last_agari_payments: List[PaymentInfo] = []  # 最近一次和牌事件的支付详情
        self._last_yaku_result: Optional[YakuResult] = None # 最近一次和牌役种结果
        self._last_win_is_tsumo: bool = False              # 最近一次和牌是否自摸
        self._last_win_loser: int = -1                     # 最近一次荣和放铳者
        self._last_win_tile: int = -1                      # 最近一次和牌牌型

        self.init_game()

    # ── 初始化 ────────────────────────────────────────────────────────────

    def init_game(self) -> None:
        """初始化游戏：创建 4 位玩家，开始第一局。"""
        self.players = [
            PlayerState(score=self.config.start_score, seat_wind=27),  # 東
            PlayerState(score=self.config.start_score, seat_wind=28),  # 南
            PlayerState(score=self.config.start_score, seat_wind=29),  # 西
            PlayerState(score=self.config.start_score, seat_wind=30),  # 北
        ]
        self.phase = GamePhase.ROUND_END
        self._start_round()

    def _start_round(self) -> None:
        """开始新的一局：洗牌、配牌、翻开宝牌指示牌。"""
        # 清除每局状态
        for p_idx, p in enumerate(self.players):
            p.clear_round_state()
            p.seat_wind = self._seat_wind_for(p_idx)

        self.wall.reset()
        self.wall.shuffle()

        # 配牌：庄家 14 张，闲家各 13 张
        hands = self.wall.deal()  # 返回 (14, 13, 13, 13) 绝对 ID 四元组
        for p_idx, tiles in enumerate(hands):
            for abs_id in tiles:
                t = abs_to_type(abs_id)
                self.players[p_idx].hand.add(t, abs_id)

        # 翻开第一张宝牌指示牌（东家第一巡前翻开）
        self.wall.flip_dora()
        self.ura_dora_indicators = []

        self.current_player = self.dealer_idx   # 庄家先手
        self.last_discard = -1
        self.last_discard_by = -1
        # 本场数由 _step_agari_settlement / _step_ryuukyoku_settlement 管理，
        # _start_round 不重置，以保留连荘积点。
        self.phase = GamePhase.DRAW

    def _seat_wind_for(self, player_idx: int) -> int:
        """根据玩家位置和场风计算自风。

        规则：庄家 = 场风，其余顺时针排列（東→南→西→北）。
        """
        winds = [27, 28, 29, 30]  # 東南西北
        offset = (player_idx - self.dealer_idx) % 4
        base_wind_idx = winds.index(self.round_wind)
        return winds[(base_wind_idx + offset) % 4]

    # ── 主步进函数 ────────────────────────────────────────────────────────

    def step(self, action: Action) -> GameState:
        """执行一个动作，推进游戏状态。

        Args:
            action: 当前玩家的合法动作

        Returns:
            GameState: 包含 rewards（分数变动，可作为强化学习即时奖励）。
        """
        prev_scores = [p.score for p in self.players]

        if self.phase == GamePhase.DRAW:
            self._step_draw(action)
        elif self.phase == GamePhase.DISCARD:
            self._step_response(action)
        elif self.phase == GamePhase.AGARI:
            self._step_agari_settlement()
        elif self.phase == GamePhase.RYUUKYOKU:
            self._step_ryuukyoku_settlement()

        # 计算分数变动（即时奖励信号）
        rewards = [p.score - prev_scores[i] for i, p in enumerate(self.players)]

        return self.get_game_state(rewards=rewards)

    def _step_draw(self, action: Action) -> None:
        """处理摸牌阶段的动作。

        可选动作：自摸、立直切牌、普通切牌、暗槓、加槓。
        """
        player = self.players[self.current_player]
        atype = action.action_type

        if atype == ActionType.TSUMO:
            self._handle_tsumo(self.current_player)
        elif atype == ActionType.RIICHI:
            self._handle_riichi_discard(action.tile)
        elif atype == ActionType.DISCARD:
            self._handle_discard(action.tile)
        elif atype == ActionType.KAN_ANKAN:
            self._handle_ankan(action.tile)
        elif atype == ActionType.KAN_KAKAN:
            self._handle_kakan(action.tile)

    def _step_response(self, action: Action) -> None:
        """处理响应阶段的动作（别家切牌后）。

        引擎仲裁优先级：
          1. 荣和（允许多家）
          2. 大明槓 / 碰
          3. 吃（仅限上家）
          4. パス → 下家摸牌
        """
        atype = action.action_type

        if atype == ActionType.RON:
            self._handle_ron(action.actor, action.tile)
        elif atype == ActionType.KAN_DAIMIN:
            self._handle_daiminkan(action.actor, action.tile)
        elif atype == ActionType.PON:
            self._handle_pon(action.actor, action.tile)
        elif atype == ActionType.CHI:
            self._handle_chi(action.actor, action.tile, action.meld_tiles)
        elif atype == ActionType.PASS:
            self._handle_pass()

    def _step_agari_settlement(self) -> None:
        """和了后精算：处理支付，判断庄家连荘或轮庄。"""
        winner = self._last_winner
        if winner == self.dealer_idx:
            # 庄家和了：本场 +1，不轮庄
            self.honba += 1
        else:
            # 闲家和了：轮庄，本场清零
            self._rotate_winds()
            self.honba = 0

        # 检查游戏是否应该结束（在轮庄后检查）
        if self._check_game_end():
            self.phase = GamePhase.GAME_END
            self._game_finished = True
        else:
            self.phase = GamePhase.ROUND_END
            self._start_round()

    def _step_ryuukyoku_settlement(self) -> None:
        """荒牌流局精算。

        听牌玩家从不听牌玩家处获得点数（合计 3000 点均分）。
        规则：
          - 听牌者 1-3 人时，不听牌者各付 3000/N 点，听牌者各得 3000/M 点
          - 全员听牌或全员不听牌：无点棒移动
          - 庄家听牌则连荘（由 tenpai_renchan 规则决定）
        """
        # 检查听牌状态
        for p_idx, player in enumerate(self.players):
            if not player.has_won:
                concealed = player.hand.tiles
                player.is_tenpai_at_ryuukyoku = is_tenpai(concealed)

        tenpai_players = [i for i, p in enumerate(self.players)
                          if p.is_tenpai_at_ryuukyoku and not p.has_won]
        noten_players = [i for i, p in enumerate(self.players)
                         if not p.is_tenpai_at_ryuukyoku and not p.has_won]

        # 听牌者从不听牌者处获得点数
        if 1 <= len(tenpai_players) <= 3:
            total_penalty = 3000
            per_noten = total_penalty // len(noten_players)
            per_tenpai = total_penalty // len(tenpai_players)
            for n_idx in noten_players:
                self.players[n_idx].score -= per_noten
            for t_idx in tenpai_players:
                self.players[t_idx].score += per_tenpai

        # 场上立直棒保留至下一局

        # 听牌连荘规则：庄家听牌则连荘
        if self.players[self.dealer_idx].is_tenpai_at_ryuukyoku and self.config.tenpai_renchan:
            self.honba += 1
        else:
            self._rotate_winds()
            self.honba = 0

        if self._check_game_end():
            self.phase = GamePhase.GAME_END
            self._game_finished = True
        else:
            self.phase = GamePhase.ROUND_END
            self._start_round()

    # ── 动作处理器 ─────────────────────────────────────────────────────────

    def _has_rinshan_tile(self) -> bool:
        """是否还有可摸的岭上牌。四次杠之后必须禁止继续杠。"""
        return self.wall._rinshan_ptr >= 0

    def _handle_tsumo(self, winner: int) -> None:
        """处理自摸和了。

        手牌 + 刚摸的牌 → 确认和牌。检查役种并计算得点。
        """
        player = self.players[winner]
        player.has_won = True
        self._last_winner = winner
        self._last_agari_payments = []
        self._last_win_is_tsumo = True
        self._last_win_loser = -1
        self._last_win_tile = self._last_drawn_tile

        # 役种检查
        ctx = self._build_win_context(winner, is_tsumo=True, winning_tile=self._last_drawn_tile)
        checker = YakuChecker(ctx)
        result = checker.check_all()
        self._last_yaku_result = result

        # 计算并执行支付
        self._settle_payments(winner, result, is_tsumo=True, loser=-1)
        self.phase = GamePhase.AGARI

    def _handle_ron(self, winner: int, winning_tile_type: int) -> None:
        """处理荣和（放铳）。

        获胜者手牌 + 放铳牌 → 确认和牌。放铳者全额支付。
        """
        player = self.players[winner]
        player.has_won = True
        self._last_winner = winner
        if self.phase != GamePhase.AGARI:
            self._last_agari_payments = []
        self._last_win_is_tsumo = False
        self._last_win_loser = self.last_discard_by
        self._last_win_tile = winning_tile_type

        # 将放铳牌临时加入手牌进行役种检查
        player.hand.add(winning_tile_type)

        ctx = self._build_win_context(winner, is_tsumo=False, winning_tile=winning_tile_type)
        checker = YakuChecker(ctx)
        result = checker.check_all()
        self._last_yaku_result = result

        # 放铳牌保留在手中（用于手牌分解和符计算）

        self._settle_payments(winner, result, is_tsumo=False, loser=self.last_discard_by)
        self.phase = GamePhase.AGARI

    def _handle_riichi_discard(self, tile_type: int) -> None:
        """处理立直宣告 + 切牌。

        扣减立直棒费用（1000 点），标记立直/一発状态，切出指定牌。
        """
        player = self.players[self.current_player]

        # 支付立直棒
        player.score -= self.config.riichi_stick_cost
        self.riichi_sticks += 1

        player.is_riichi = True
        player.is_ippatsu = True  # 一発在下一巡摸牌前有效

        # 双立直检查（第一巡无人鸣牌即立直）
        if len(player.discards) == 0:
            player.is_double_riichi = True

        # 切牌
        abs_id = self._find_abs_to_discard(player, tile_type)
        player.hand.remove(tile_type, abs_id)
        player.add_discard(abs_id)

        # 手牌听牌，记录听牌牌型用于振听判定
        player.furiten_types = set()
        player.temp_furiten = False

        self.last_discard = abs_id
        self.last_discard_by = self.current_player
        self.phase = GamePhase.DISCARD

    def _handle_discard(self, tile_type: int) -> None:
        """处理普通切牌。

        更新振听状态（检查自己的舍牌是否包含自己的听牌）。
        """
        player = self.players[self.current_player]

        # 更新永久振听：检查舍牌中是否包含听牌
        waits = get_waits(player.hand.tiles)
        for dt in player.discard_types:
            if dt in waits:
                player.furiten_types.add(dt)

        abs_id = self._find_abs_to_discard(player, tile_type)
        player.hand.remove(tile_type, abs_id)
        player.add_discard(abs_id)

        # 清除临时振听
        player.temp_furiten = False

        # 立直后切牌：再次检查振听（切出的牌如果在听牌列表中）
        if player.is_riichi:
            waits = get_waits(player.hand.tiles)
            for dt in player.discard_types:
                if dt in waits:
                    player.furiten_types.add(dt)

        self.last_discard = abs_id
        self.last_discard_by = self.current_player
        self.phase = GamePhase.DISCARD

    def _handle_ankan(self, tile_type: int) -> None:
        """处理暗槓（门内四张相同牌成槓子）。

        流程：移除 4 张手牌 → 添加暗槓副露 → 翻开新宝牌 → 摸岭上牌。
        """
        if not self._has_rinshan_tile():
            self._handle_ryuukyoku()
            return

        player = self.players[self.current_player]

        meld = Meld(
            meld_type=MeldType.KAN_CLOSED,
            tiles=[tile_type, tile_type, tile_type, tile_type],
        )
        player.hand.add_meld(meld)

        # 槓 → 翻开新宝牌指示牌
        self.wall.flip_dora()

        # 摸岭上牌
        rinshan = self.wall.draw_rinshan()
        t = abs_to_type(rinshan)
        player.hand.add(t, rinshan)
        self._last_drawn_tile = t

        # 保持 DRAW 阶段（玩家需切牌或自摸）
        self.phase = GamePhase.DRAW

    def _handle_kakan(self, tile_type: int) -> None:
        """处理加槓（已有碰的面子加一张成槓）。

        将碰的面子升级为明槓，翻开新宝牌，摸岭上牌。
        """
        if not self._has_rinshan_tile():
            self._handle_ryuukyoku()
            return

        player = self.players[self.current_player]

        # 找到对应的碰面子和升级为明槓
        for meld in player.hand.melds:
            if meld.meld_type == MeldType.PON and meld.tile_type == tile_type:
                meld.meld_type = MeldType.KAN_OPEN
                meld.tiles.append(tile_type)
                player.hand.remove(tile_type)  # 从手牌中移除第四张
                break

        # 翻开新宝牌指示牌
        self.wall.flip_dora()

        # 摸岭上牌
        rinshan = self.wall.draw_rinshan()
        t = abs_to_type(rinshan)
        player.hand.add(t, rinshan)
        self._last_drawn_tile = t

        self.phase = GamePhase.DRAW

    def _handle_daiminkan(self, caller: int, tile_type: int) -> None:
        """处理大明槓（鸣别人的舍牌成槓）。

        移除手中 3 张 + 别家舍牌 1 张 → 槓子。翻开新宝牌，摸岭上牌。
        """
        if not self._has_rinshan_tile():
            self._handle_ryuukyoku()
            return

        player = self.players[caller]

        meld = Meld(
            meld_type=MeldType.KAN_DAIMIN,
            tiles=[tile_type, tile_type, tile_type, tile_type],
            called_from=self.last_discard_by,
            source_tile=tile_type,
        )
        player.hand.add_meld(meld)

        # 翻开新宝牌指示牌
        self.wall.flip_dora()

        # 摸岭上牌
        rinshan = self.wall.draw_rinshan()
        t = abs_to_type(rinshan)
        player.hand.add(t, rinshan)
        self._last_drawn_tile = t

        # 鸣牌者成为当前玩家，跳过摸牌直接切牌
        self.current_player = caller
        self.last_discard = -1
        self.phase = GamePhase.DRAW

    def _handle_pon(self, caller: int, tile_type: int) -> None:
        """处理碰（鸣别家舍牌成刻子）。

        移除手中 2 张 + 别家舍牌 1 张 → 刻子副露。鸣牌者跳上摸牌直接进入切牌阶段。
        """
        player = self.players[caller]

        meld = Meld(
            meld_type=MeldType.PON,
            tiles=[tile_type, tile_type, tile_type],
            called_from=self.last_discard_by,
            source_tile=tile_type,
        )
        player.hand.add_meld(meld)

        self.last_discard = -1

        # 鸣牌者跳上摸牌直接切牌
        self.current_player = caller
        self.phase = GamePhase.DRAW

    def _handle_chi(self, caller: int, source_tile: int, meld_tiles: List[int]) -> None:
        """处理吃（鸣上家舍牌成顺子）。

        仅可吃上家（左侧相邻玩家）的舍牌。
        """
        player = self.players[caller]

        meld = Meld(
            meld_type=MeldType.CHI,
            tiles=list(meld_tiles),
            called_from=self.last_discard_by,
            source_tile=source_tile,
        )
        player.hand.add_meld(meld)

        self.last_discard = -1
        self.current_player = caller
        self.phase = GamePhase.DRAW  # 鸣牌者跳过摸牌直接切牌

    def _handle_pass(self) -> None:
        """处理パス（不鸣牌）—— 推进到下一位玩家摸牌。

        流程：
          1. 清除 last_discard
          2. 下一位玩家摸牌
          3. 如果牌山耗尽 → 流局
          4. 检查并清除一発标志（如果是立直玩家摸牌）
        """
        self.last_discard = -1
        # 按下家顺序推进
        self.current_player = (self.last_discard_by + 1) % 4

        # 检查牌山是否耗尽（进入死牌区域之前）
        if self.wall._live_ptr >= 122:  # 死牌区域起点
            self._handle_ryuukyoku()
            return

        # 摸牌
        drawn = self.wall.draw()
        t = abs_to_type(drawn)
        self.players[self.current_player].hand.add(t, drawn)
        self._last_drawn_tile = t

        # 一発在立直玩家再次摸牌时失效
        player = self.players[self.current_player]
        if player.is_riichi:
            player.is_ippatsu = False

        self.phase = GamePhase.DRAW

    def _handle_ryuukyoku(self) -> None:
        """处理荒牌流局（牌山耗尽无人和牌）。"""
        self.phase = GamePhase.RYUUKYOKU

    # ── 支付精算 ──────────────────────────────────────────────────────────

    def _settle_payments(self, winner: int, yaku_result: YakuResult,
                         is_tsumo: bool, loser: int) -> None:
        """计算和牌支付并更新各玩家分数。

        流程：
          1. 计算符数（如果有役满则跳过）
          2. 计算宝牌数
          3. 检查二翻缚规则
          4. 查表计算得点
          5. 执行各玩家分数变动
          6. 立直棒归和牌者
        """
        winner_player = self.players[winner]
        is_dealer = winner == self.dealer_idx

        # 役满处理：不使用符/翻表，直接查役满得点
        han = yaku_result.total_han
        if yaku_result.is_yakuman:
            num_yakuman = len(yaku_result.yakuman_list)
            han = 0
            fu = 0
        else:
            num_yakuman = 0
            # 通过手牌分解计算符数
            from .yaku import decompose_hand
            decomp = decompose_hand(winner_player.hand.tiles)
            if decomp:
                fu = calculate_fu_from_decomp(
                    concealed_melds=decomp.melds,
                    open_melds=winner_player.hand.melds,
                    pair=decomp.pair,
                    is_menzen=winner_player.hand.is_menzen,
                    is_tsumo=True,
                    is_pinfu=any(y == Yaku.PINFU for y, _ in yaku_result.yaku_list),
                    bakaze=self.round_wind,
                    jikaze=winner_player.seat_wind,
                )
            else:
                fu = 30  # 七对子/国士无双 默认 30 符

        # 宝牌计算（表宝牌 + 里宝牌 + 赤宝牌）
        dora_types = self.wall.get_dora_types(with_ura=winner_player.is_riichi,
                                               ura_dora_indicators=self.ura_dora_indicators)
        if winner_player.is_riichi and not self.ura_dora_indicators:
            # 立直和牌时翻开里宝牌
            self.ura_dora_indicators = self.wall.flip_ura_dora(self.wall._dora_count)
            dora_types = self.wall.get_dora_types(
                with_ura=True, ura_dora_indicators=self.ura_dora_indicators)

        all_tiles = winner_player.hand.to_type_list()
        dora_count = count_dora_in_hand(all_tiles, dora_types)
        if self.config.aka_dora:
            dora_count += len(winner_player.hand.aka_tiles)

        total_han = han + dora_count

        if yaku_result.is_yakuman:
            total_han = 0  # 役满使用独立得点体系

        # 二翻缚规则：5 本场以上需要 2 翻起和
        if self.config.ryanhan_shibari and self.honba >= 5 and total_han < 2:
            return  # 不足 2 翻 → 不能和牌

        payment = compute_payments(
            han=total_han,
            fu=fu,
            winner=winner,
            is_dealer=is_dealer,
            is_tsumo=is_tsumo,
            loser=loser,
            num_yakuman=num_yakuman,
            honba=self.honba,
            riichi_sticks_on_table=self.riichi_sticks,
        )
        payment.yaku_names = [
            YAKU_NAMES_JP.get(yaku, str(yaku))
            for yaku, _ in yaku_result.yaku_list
        ] or [
            YAKU_NAMES_JP.get(yaku, str(yaku))
            for yaku in yaku_result.yakuman_list
        ]
        payment.dora_count = dora_count
        self._last_agari_payments.append(payment)

        # 执行支付
        for i, delta in enumerate(payment.payments):
            self.players[i].score += delta

        # 立直棒归和牌者，清空场上立直棒
        self.riichi_sticks = 0

    # ── 振听检查 ──────────────────────────────────────────────────────────

    def _check_furiten(self, player_idx: int, winning_tile_type: int) -> bool:
        """检查玩家是否处于振听状态（不能荣和）。

        三种振听（按日麻标准）：
          1. 捨牌振听：自己舍牌中包含任意一张听牌 → 持续，改变听牌后可解除
          2. 同巡振听：当前巡错过和牌机会 → 1巡后自动解除
          3. 立直振听：立直后错过和牌机会 → 本局永久

        振听是手牌整体状态，不是针对某一张牌。
        即使荣和牌不是舍牌振听的那张，只要手牌整体振听就不许荣和。

        Returns:
            True 表示处于振听状态，禁止荣和。
        """
        player = self.players[player_idx]
        waits = get_waits(player.hand.tiles)

        # 1. 捨牌振听：自己打出过的牌中，有任意一张是当前的听牌
        for dt in player.discard_types:
            if dt in waits:
                return True

        # 2. 立直振听：立直后错过和牌机会 → 永久禁止荣和
        if player.is_riichi_furiten:
            return True

        # 3. 同巡振听：当前巡错过和牌机会 → 禁止荣和直到下一次切牌
        if player.temp_furiten:
            return True

        return False

    # ── 和牌上下文构建 ────────────────────────────────────────────────────

    def _build_win_context(self, winner: int, is_tsumo: bool,
                           winning_tile: int) -> WinContext:
        """构建 WinContext 供役种检查器使用。

        汇总所有影响役种判定的参数：
          - 门清状态、自摸/荣和
          - 立直/一発/双立直
          - 天和/地和判定
          - 场风/自风
          - 副露信息
        """
        player = self.players[winner]
        return WinContext(
            is_menzen=player.hand.is_menzen,
            is_tsumo=is_tsumo,
            is_riichi=player.is_riichi,
            is_ippatsu=player.is_ippatsu and player.is_riichi,
            is_double_riichi=player.is_double_riichi,
            # 天和：庄家配牌 14 张即和牌（非摸牌后），故 winning_tile=-1
            is_tenhou=(winner == self.dealer_idx and len(player.discards) == 0 and is_tsumo
                       and self._last_drawn_tile == -1),
            # 地和：闲家第一巡摸牌即和（此前无人鸣牌/舍牌）
            is_chiihou=(winner != self.dealer_idx and len(player.discards) == 0 and is_tsumo),
            bakaze=self.round_wind,
            jikaze=player.seat_wind,
            kuitan=self.config.kuitan,
            open_melds=list(player.hand.melds),
            concealed_tiles=list(player.hand.tiles),
            winning_tile=winning_tile,
        )

    # ── 场风轮转 ──────────────────────────────────────────────────────────

    def _rotate_winds(self) -> None:
        """局间风位轮转。

        规则：
          - 庄家按顺时针轮转（P0→P1→P2→P3→P0）
          - 東 4 局后（東風战直接结束），场风变为南，再 4 局后游戏结束
        """
        self.round_number += 1

        if self.round_number > 4:
            if not self.config.east_only and self.round_wind == 27:
                # 東風战结束 → 进入南風战
                self.round_wind = 28
                self.round_number = 1
            elif not self.config.east_only and self.round_wind == 28:
                # 南風战结束 → 游戏终结
                self._game_finished = True
            else:
                # 東風战 4 局结束 → 游戏终结
                self._game_finished = True

        # 庄家顺时针轮转
        self.dealer_idx = (self.dealer_idx + 1) % 4

    def _check_game_end(self) -> bool:
        """检查游戏是否应该结束。

        条件：
          - 已到达最终局（半荘战 8 局或東風战 4 局）
          - 飛び规则：任意玩家分数为负
        """
        if self._game_finished:
            return True
        # 飛び检查（负分淘汰）
        if self.config.tobi:
            for p in self.players:
                if p.score < 0:
                    return True
        return False

    def is_game_over(self) -> bool:
        """游戏是否已结束。"""
        return self._game_finished or self.phase == GamePhase.GAME_END

    # ── 批量响应接口（供 AI 自对弈使用）──────────────────────────────────

    def get_response_options(self) -> Dict[int, LegalActions]:
        """舍牌后，获取所有 3 位非舍牌玩家的合法响应动作。

        Returns:
            Dict[玩家索引, LegalActions] —— AI 应为每位玩家各选一个动作，
            然后调用 resolve_responses() 来按优先级执行。
        """
        if self.phase != GamePhase.DISCARD:
            return {}
        options = {}
        for p_idx in range(4):
            if p_idx != self.last_discard_by:
                options[p_idx] = self._get_response_actions(p_idx)
        return options

    def resolve_responses(self, responses: Dict[int, Action],
                          force_draw: bool = False) -> None:
        """按优先级规则处理舍牌后所有玩家的响应动作。

        优先级：
          1. 荣和（允许多家 → 放铳者全额支付给各家和牌）
          2. 大明槓 / 碰（第一家声明者获得优先）
          3. 吃（仅限上家）
          4. 无响应 → 下一家摸牌

        Args:
            responses: 玩家索引 → 动作的映射
            force_draw: 如果为 True，跳过荣和强制执行摸牌流程
                        （用于立直玩家必须パス的情况）
        """
        # ── 振听触发：检查错过和牌机会的玩家 ──
        discard_type = abs_to_type(self.last_discard)
        for p_idx, action in responses.items():
            if p_idx == self.last_discard_by:
                continue
            if action.action_type == ActionType.RON:
                continue  # 选择了荣和，不触发振听
            player = self.players[p_idx]
            if is_agari_with_tile(player.hand.tiles, discard_type):
                # 该牌能完成手牌形 → 不论有无役，错过即触发振听
                if not self._check_furiten(p_idx, discard_type):
                    if player.is_riichi:
                        player.is_riichi_furiten = True  # 立直振听：永久
                    else:
                        player.temp_furiten = True       # 同巡振听：临时

        ron_players = []
        kan_players = []
        pon_players = []
        chi_players = []

        for p_idx, action in responses.items():
            if force_draw or action.action_type == ActionType.PASS:
                continue
            elif action.action_type == ActionType.RON:
                ron_players.append(p_idx)
            elif action.action_type == ActionType.KAN_DAIMIN:
                kan_players.append(p_idx)
            elif action.action_type == ActionType.PON:
                pon_players.append(p_idx)
            elif action.action_type == ActionType.CHI:
                chi_players.append((p_idx, action))

        # 优先级 1: 荣和
        if ron_players:
            if self.config.multiple_ron:
                # 多家荣和：各家均和牌
                for winner in ron_players:
                    self._handle_ron(winner, abs_to_type(self.last_discard))
            else:
                # 头跳：只离放铳者最近的玩家和牌
                closest = min(ron_players,
                              key=lambda p: (p - self.last_discard_by) % 4)
                self._handle_ron(closest, abs_to_type(self.last_discard))
            return

        # 优先级 2: 大明槓 / 碰（头跳原则）
        if kan_players:
            caller = kan_players[0]
            self._handle_daiminkan(caller, abs_to_type(self.last_discard))
            return
        if pon_players:
            caller = pon_players[0]
            self._handle_pon(caller, abs_to_type(self.last_discard))
            return

        # 优先级 3: 吃
        if chi_players:
            caller, action = chi_players[0]
            self._handle_chi(caller, action.tile, action.meld_tiles)
            return

        # 优先级 4: 全員パス → 下家摸牌
        self._handle_pass()

    # ── 合法动作计算 ──────────────────────────────────────────────────────

    def get_legal_actions(self, player_idx: Optional[int] = None) -> LegalActions:
        """获取当前决策玩家的合法动作列表。

        摸牌阶段 (DRAW): 切牌、自摸、立直、槓。
        响应阶段 (DISCARD): 荣和、碰、吃、大明槓、パス。

        Returns:
            LegalActions 包含动作列表和 77 维掩码向量。
        """
        if player_idx is None:
            player_idx = self.current_player

        if self.phase == GamePhase.DRAW:
            return self._get_draw_actions(player_idx)
        elif self.phase == GamePhase.DISCARD:
            return self._get_response_actions(player_idx)
        else:
            return LegalActions(actions=[], mask=[0] * 77)

    def _get_draw_actions(self, player_idx: int) -> LegalActions:
        """计算摸牌阶段的合法动作。

        包括：
          - 自摸（如果手牌和了）
          - 立直 + 切牌（如果满足立直条件）
          - 普通切牌（每种手牌中持有 ≥1 张的牌型）
          - 暗槓 / 加槓（有条件的）
        """
        if player_idx != self.current_player:
            return LegalActions(actions=[], mask=[0] * 77)

        player = self.players[player_idx]
        hand = player.hand.tiles
        is_menzen_flag = player.hand.is_menzen

        # 自摸判定
        can_tsumo = is_agari(hand) and self._can_win(player_idx)

        # 立直判定（门清 + 分数足够 + 存在听牌切牌选择）
        riichi_ok = can_riichi(hand, is_menzen_flag, player.score,
                               self.config.riichi_stick_cost)
        riichi_discards = get_legal_discards_for_riichi(hand) if riichi_ok else []

        # 槓选项
        ankan_opts = can_ankan(hand)
        # 立直后禁止暗槓（除非不改变听牌牌型）
        if player.is_riichi:
            ankan_opts = []
        # 牌山耗尽时禁止槓（无法摸岭上牌）
        if self.wall._live_ptr >= 122 or not self._has_rinshan_tile():
            ankan_opts = []

        kakan_opts = can_kakan(hand, player.hand.melds)
        if player.is_riichi:
            kakan_opts = []  # 立直后禁止加槓
        if not self._has_rinshan_tile():
            kakan_opts = []

        return compute_draw_actions(
            hand=hand,
            is_menzen=is_menzen_flag,
            can_tsumo=can_tsumo,
            can_riichi=riichi_ok,
            riichi_discards=riichi_discards,
            ankan_options=ankan_opts,
            kakan_options=kakan_opts,
            last_drawn_tile=getattr(self, '_last_drawn_tile', -1),
        )

    def _get_response_actions(self, player_idx: int) -> LegalActions:
        """计算别家舍牌后指定玩家的合法响应动作。

        注意：吃仅限上家（左侧相邻玩家），振听玩家不能荣和。
        """
        if player_idx == self.last_discard_by:
            return LegalActions(actions=[Action(ActionType.PASS)], mask=[0] * 77)

        player = self.players[player_idx]
        hand = player.hand.tiles
        discard_type = abs_to_type(self.last_discard)

        # 荣和判定（需排除振听 + 满足至少 1 役）
        can_ron = False
        if is_agari_with_tile(hand, discard_type):
            if not self._check_furiten(player_idx, discard_type):
                if self._can_win(player_idx, winning_tile=discard_type, is_tsumo=False):
                    can_ron = True

        # 吃选项（立直后不可吃）
        chi_opts = can_chi(hand, discard_type) if not player.is_riichi else []

        legal = compute_response_actions(
            hand=hand,
            source_tile=discard_type,
            source_player=self.last_discard_by,
            my_position=player_idx,
            can_ron=can_ron,
            chi_options=chi_opts,
        )
        if not self._has_rinshan_tile():
            legal.actions = [
                action for action in legal.actions
                if action.action_type != ActionType.KAN_DAIMIN
            ]
            legal.mask[73] = 0
        return legal

    def _can_win(self, player_idx: int, winning_tile: Optional[int] = None,
                 is_tsumo: bool = True) -> bool:
        """检查玩家是否能和牌（至少需要 1 役）。

        宝牌、赤宝牌、里宝牌不计为役——必须是至少一种役种（含役满）成立方可和牌。
        无役和牌是日麻最常见的犯规，本检查在摸牌/听牌阶段即拦截。
        """
        player = self.players[player_idx]

        # 构建门内牌（荣和时需临时加入和了牌）
        tiles = list(player.hand.tiles)
        if winning_tile is not None:
            tiles[winning_tile] += 1

        ctx = WinContext(
            is_menzen=player.hand.is_menzen,
            is_tsumo=is_tsumo,
            is_riichi=player.is_riichi,
            is_ippatsu=player.is_ippatsu and player.is_riichi,
            is_double_riichi=player.is_double_riichi,
            is_tenhou=(player_idx == self.dealer_idx and len(player.discards) == 0 and is_tsumo
                       and self._last_drawn_tile == -1),
            is_chiihou=(player_idx != self.dealer_idx and len(player.discards) == 0 and is_tsumo),
            bakaze=self.round_wind,
            jikaze=player.seat_wind,
            kuitan=self.config.kuitan,
            open_melds=list(player.hand.melds),
            concealed_tiles=tiles,
            winning_tile=winning_tile if winning_tile is not None
                        else getattr(self, '_last_drawn_tile', -1),
        )

        checker = YakuChecker(ctx)
        result = checker.check_all()

        return bool(result.yakuman_list or result.yaku_list)

    def _find_abs_to_discard(self, player: PlayerState, tile_type: int) -> int:
        """为玩家找到要切出的牌的绝对 ID。

        优先切非赤宝牌（赤宝牌价值更高）。
        """
        candidates = type_to_abs(tile_type)
        # 优先非赤牌
        for aid in candidates:
            if not is_aka(aid) and aid not in player.hand.aka_tiles:
                return aid
        # 回退到任意牌
        return candidates[0]

    # ── 游戏状态快照 ──────────────────────────────────────────────────────

    def get_game_state(self, rewards: Optional[List[float]] = None) -> GameState:
        """返回完整游戏状态快照。

        Args:
            rewards: 分数变动列表（强化学习即时奖励）
        """
        return GameState(
            phase=self.phase,
            current_player=self.current_player,
            round_wind=self.round_wind,
            round_number=self.round_number,
            honba=self.honba,
            riichi_sticks=self.riichi_sticks,
            scores=[p.score for p in self.players],
            hands_concealed=[list(p.hand.tiles) for p in self.players],
            open_melds=[list(p.hand.melds) for p in self.players],
            discards=[list(p.discards) for p in self.players],
            dora_indicators=list(self.wall.dora_indicators),
            ura_dora_indicators=list(self.ura_dora_indicators),
            is_riichi=[p.is_riichi for p in self.players],
            last_discard=self.last_discard,
            last_discard_by=self.last_discard_by,
            remaining_tiles=122 - self.wall._live_ptr,
            rewards=rewards or [0.0, 0.0, 0.0, 0.0],
            done=self.is_game_over(),
        )

    def get_state_tensor(self, player_idx: int) -> np.ndarray:
        """构建神经网络输入特征张量（从指定玩家视角）。

        特征通道（354 维向量）：
          - 自己手牌：34 维
          - 自己副露：34 维
          - 自己舍牌：34 维
          - 宝牌指示牌：34 维（one-hot）
          - 对手舍牌 ×3：102 维
          - 对手副露 ×3：102 维
          - 全局特征：7 维（分数、本场、立直棒、剩余牌数、立直状态、场风、局数）
          - 分数差：3 维
          - 自风：4 维（one-hot）
        """
        state = self.get_game_state()

        features = []

        # 自己手牌 (34 维)
        features.extend(state.hands_concealed[player_idx])

        # 自己副露指示器 (34 维)
        own_melds_array = [0] * NUM_TYPES
        for meld in state.open_melds[player_idx]:
            for t in meld.tiles:
                own_melds_array[t] += 1
        features.extend(own_melds_array)

        # 自己舍牌 (34 维)
        own_disc = [0] * NUM_TYPES
        for aid in state.discards[player_idx]:
            own_disc[abs_to_type(aid)] += 1
        features.extend(own_disc)

        # 可见宝牌指示牌 (34 维 one-hot)
        dora_types = self.wall.get_dora_types()
        dora_feat = [0] * NUM_TYPES
        for dt in dora_types:
            dora_feat[dt] = 1
        features.extend(dora_feat)

        # 对手舍牌 (3 × 34 维)
        for opp in range(4):
            if opp == player_idx:
                continue
            opp_disc = [0] * NUM_TYPES
            for aid in state.discards[opp]:
                opp_disc[abs_to_type(aid)] += 1
            features.extend(opp_disc)

        # 对手副露 (3 × 34 维)
        for opp in range(4):
            if opp == player_idx:
                continue
            opp_meld = [0] * NUM_TYPES
            for meld in state.open_melds[opp]:
                for t in meld.tiles:
                    opp_meld[t] += 1
            features.extend(opp_meld)

        # 全局特征 (7 维)
        features.extend([
            state.scores[player_idx] / 1000.0,             # 自己分数（千点单位）
            state.honba,                                    # 本场数
            state.riichi_sticks,                            # 场上立直棒数
            state.remaining_tiles / 122.0,                  # 剩余牌数（归一化）
            1.0 if state.is_riichi[player_idx] else 0.0,   # 自己是否立直
            float(state.round_wind - 27),                   # 场风（0=東, 1=南）
            float(state.round_number),                      # 局数
        ])

        # 对手分数差 (3 维)
        for opp in range(4):
            if opp != player_idx:
                features.append((state.scores[opp] - state.scores[player_idx]) / 1000.0)

        # 自风 one-hot (4 维)
        for w in range(4):
            features.append(1.0 if self.players[player_idx].seat_wind == 27 + w else 0.0)

        return np.array(features, dtype=np.float32)

    def get_state_tensor_dim(self) -> int:
        """返回状态张量的维度（354）。

        计算公式：
          34 (手牌) + 34 (自副露) + 34 (自舍牌) + 34 (宝牌) +
          3×34 (对手舍牌) + 3×34 (对手副露) +
          7 (全局) + 3 (分差) + 4 (风位 one-hot) = 354
        """
        return 34 * 4 + 34 * 6 + 7 + 3 + 4

    # ── 克隆支持（供 MCTS 使用）───────────────────────────────────────────

    def clone(self) -> "GameEngine":
        """深拷贝整个引擎状态，保证 MCTS 模拟不会影响原始游戏。"""
        return deepcopy(self)

    # ── 结果 ──────────────────────────────────────────────────────────────

    def get_result(self) -> GameResult:
        """获取最终游戏结果（含顺位马点和オカ调整）。"""
        scores = [p.score for p in self.players]
        return compute_final_result(
            scores=scores,
            uma=self.config.uma,
            oka=self.config.oka,
            target_score=self.config.target_score,
        )

    def get_winner(self) -> int:
        """返回获胜玩家索引（按调整后分数最高者）。"""
        result = self.get_result()
        scores = result.adjusted_scores
        return max(range(4), key=lambda i: scores[i])

    # ── 显示 ──────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        parts = []
        parts.append(f"Round: {TILE_NAMES[self.round_wind]}{self.round_number}局 "
                     f"{self.honba}本場")
        parts.append(f"Phase: {self.phase.name}  Player: {self.current_player}")
        parts.append(f"Dora: {[TILE_NAMES[abs_to_type(d)] for d in self.wall.dora_indicators]}")
        for i, p in enumerate(self.players):
            riichi = " [立直]" if p.is_riichi else ""
            parts.append(f"  P{i}({TILE_NAMES[p.seat_wind]}): {p.score}pts{riichi} {p.hand}")
        return "\n".join(parts)
