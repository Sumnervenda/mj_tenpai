"""启发式 Agent —— 基于牌效和安全牌的规则 AI，为 SL 阶段生成"优于随机"的训练数据。

优先级链（按日麻常识）：
  1. 自摸（有役）
  2. 荣和（有役）
  3. 立直 + 安全切牌
  4. 切孤张（优先客风→三元牌→断幺牌→宝牌周边）
"""

import random
from typing import List, Optional

from engine import (
    GameEngine, GameConfig, GamePhase,
    Action, ActionType, LegalActions,
    NUM_TYPES, TILE_NAMES,
    is_jihai, is_yaochuhai, is_kazehai, is_sangenhai,
)


class HeuristicAgent:
    """简单规则 Agent：牌效优先 + 基本安全策略。

    用于对局数据生成，行为介于"纯随机"和"人类雀士"之间。
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def select_action(self, engine: GameEngine, player_idx: int) -> Action:
        """根据当前游戏状态选择一个合法动作。"""
        legal = engine.get_legal_actions(player_idx)
        if not legal.actions:
            return Action(ActionType.PASS)

        # 1. 自摸优先
        tsumo_acts = [a for a in legal.actions if a.action_type == ActionType.TSUMO]
        if tsumo_acts:
            return tsumo_acts[0]

        # 2. 荣和优先
        ron_acts = [a for a in legal.actions if a.action_type == ActionType.RON]
        if ron_acts:
            return ron_acts[0]

        # 3. 立直 + 切牌（选择最安全的切牌）
        riichi_acts = [a for a in legal.actions if a.action_type == ActionType.RIICHI]
        if riichi_acts:
            return self._best_riichi_discard(riichi_acts, engine, player_idx)

        # 4. 槓（有则用之）
        kan_acts = [a for a in legal.actions
                    if a.action_type in (ActionType.KAN_ANKAN, ActionType.KAN_KAKAN)]
        if kan_acts:
            return self.rng.choice(kan_acts)

        # 5. 切牌（切孤张策略）
        discard_acts = [a for a in legal.actions if a.action_type == ActionType.DISCARD]
        if discard_acts:
            return self._best_discard(discard_acts, engine, player_idx)

        # 回退
        return legal.actions[0]

    def select_response(self, engine: GameEngine, player_idx: int) -> Action:
        """在 DISCARD 阶段选择响应动作。"""
        legal = engine.get_legal_actions(player_idx)
        if not legal.actions:
            return Action(ActionType.PASS)

        # 荣和优先
        ron_acts = [a for a in legal.actions if a.action_type == ActionType.RON]
        if ron_acts:
            return ron_acts[0]

        # 碰/大明槓
        pon_acts = [a for a in legal.actions if a.action_type == ActionType.PON]
        kan_acts = [a for a in legal.actions if a.action_type == ActionType.KAN_DAIMIN]
        if pon_acts:
            return pon_acts[0]
        if kan_acts:
            return kan_acts[0]

        # 吃（仅在推进向听数时使用）
        chi_acts = [a for a in legal.actions if a.action_type == ActionType.CHI]
        if chi_acts:
            return self.rng.choice(chi_acts)

        return Action(ActionType.PASS)

    def _best_riichi_discard(self, riichi_acts: List[Action],
                              engine: GameEngine, player_idx: int) -> Action:
        """从多个立直切牌中选择最安全的一张。"""
        hand = engine.players[player_idx].hand
        # 优先切非宝牌、非役牌的单张
        scored = []
        for act in riichi_acts:
            t = act.tile
            score = 0
            if hand.tiles[t] == 1:
                score += 10  # 切孤张不破坏搭子
            if is_jihai(t) and not is_sangenhai(t):
                score += 5   # 客风优先切
            scored.append((score, act))
        scored.sort(key=lambda x: -x[0])
        return scored[0][1]

    def _best_discard(self, discard_acts: List[Action],
                       engine: GameEngine, player_idx: int) -> Action:
        """选择最优切牌：孤张字牌 > 孤张幺九 > 孤张中张 > 拆搭子（最后手段）。"""
        hand = engine.players[player_idx].hand

        scored = []
        for act in discard_acts:
            t = act.tile
            count = hand.tiles[t]
            score = 0

            # 孤张优先
            if count == 1:
                score += 20
            elif count == 2:
                score -= 5   # 可能在对子/搭子中

            # 客风优先切
            if is_kazehai(t) and t != engine.players[player_idx].seat_wind \
               and t != engine.round_wind:
                score += 30
            # 三元牌次之
            elif is_sangenhai(t):
                score += 15
            # 幺九次之
            elif is_yaochuhai(t):
                score += 10

            scored.append((score, act))

        scored.sort(key=lambda x: -x[0])
        return scored[0][1]
