"""奖励塑形器 —— 通过修改 Reward 信号塑造 AI 打牌风格。

基类提供分数变动归一化；个性子类实现 TRAIN_README.md 中三种人格。
"""

from typing import Optional

from engine.game import GameEngine
from engine.actions import Action, ActionType


class RewardShaper:
    """奖励塑形基类：分数变动归一化。"""

    def __call__(self, engine: GameEngine, player_idx: int,
                 action: Optional[Action] = None) -> float:
        """计算即时奖励。

        Args:
            engine: 游戏引擎（已执行 action 后）
            player_idx: 玩家索引
            action: 刚执行的动作（可选，用于动作相关奖励）

        Returns:
            归一化奖励值
        """
        state = engine.get_game_state()
        return state.rewards[player_idx] / 10000.0


class TurtleShaper(RewardShaper):
    """千年老龟：强手进攻、弱手死守型。

    手牌 4 番以上时鼓励和牌（高打点奖励），手牌差时惩罚放铳和第四名，
    迫使 AI 学会弃和和安全牌策略。

    中间奖励（增加信号密度，避免稀疏奖励导致策略退化）：
      - 立直 +riichi_bonus/10000（乌龟也能立直，尤其在好牌时）
    """

    def __init__(self,
                 fourth_place_penalty: float = -200.0,
                 dealing_in_penalty: float = -50.0,
                 high_value_bonus: float = 200.0,
                 high_value_threshold: int = 4,
                 riichi_bonus: float = 100.0):
        self.fourth_place_penalty = fourth_place_penalty
        self.dealing_in_penalty = dealing_in_penalty
        self.high_value_bonus = high_value_bonus
        self.high_value_threshold = high_value_threshold
        self.riichi_bonus = riichi_bonus

    def __call__(self, engine: GameEngine, player_idx: int,
                 action: Optional[Action] = None) -> float:
        reward = super().__call__(engine, player_idx, action)

        # 放铳即时惩罚
        payments = getattr(engine, '_last_agari_payments', [])
        for p in payments:
            if p.loser == player_idx:
                reward += self.dealing_in_penalty / 10000.0

        # 和牌奖励：4 番以上额外加分
        if action is not None and action.action_type in (ActionType.TSUMO,
                                                          ActionType.RON):
            for p in payments:
                if p.winner == player_idx and p.han >= self.high_value_threshold:
                    reward += self.high_value_bonus / 10000.0
                    break

        # 立直中间奖励
        if action is not None and action.action_type == ActionType.RIICHI:
            reward += self.riichi_bonus / 10000.0

        # 第四名惩罚
        if engine.is_game_over():
            result = engine.get_result()
            if result.ranks[player_idx] == 3:
                reward += self.fourth_place_penalty / 10000.0

        return reward


class MadDogShaper(RewardShaper):
    """疯狗狂战：极限争一型。

    巨大的一位奖励，赢者通吃。早巡胡牌额外加分（鼓励速攻）。
    """

    def __init__(self,
                 first_place_bonus: float = 1000.0,
                 early_win_bonus: float = 50.0,
                 early_win_turn_threshold: int = 6):
        self.first_place_bonus = first_place_bonus
        self.early_win_bonus = early_win_bonus
        self.early_win_turn_threshold = early_win_turn_threshold

    def __call__(self, engine: GameEngine, player_idx: int,
                 action: Optional[Action] = None) -> float:
        reward = super().__call__(engine, player_idx, action)

        # 一位大奖励
        if engine.is_game_over():
            result = engine.get_result()
            if result.ranks[player_idx] == 0:  # 0-indexed，0 = 第1位
                reward += self.first_place_bonus / 10000.0

        # 早巡胡牌奖励（通过检查 action 判断）
        if action is not None and action.action_type in (ActionType.TSUMO,
                                                          ActionType.RON):
            player = engine.players[player_idx]
            if len(player.discards) <= self.early_win_turn_threshold:
                reward += self.early_win_bonus / 10000.0

        return reward


class RiichiFundamentalistShaper(RewardShaper):
    """立直原教旨：浪漫门清型。

    惩罚副露，奖励立直，鼓励门前清打法。
    """

    def __init__(self,
                 chi_pon_penalty: float = -50.0,
                 riichi_bonus: float = 300.0):
        self.chi_pon_penalty = chi_pon_penalty
        self.riichi_bonus = riichi_bonus

    def __call__(self, engine: GameEngine, player_idx: int,
                 action: Optional[Action] = None) -> float:
        reward = super().__call__(engine, player_idx, action)

        if action is not None:
            # 副露惩罚
            if action.action_type in (ActionType.CHI, ActionType.PON):
                reward += self.chi_pon_penalty / 1000.0

            # 立直奖励
            if action.action_type == ActionType.RIICHI:
                reward += self.riichi_bonus / 1000.0

        return reward


def load_shaper_from_config(config_path: str) -> RewardShaper:
    """从 YAML 配置文件加载对应奖励塑形器。"""
    import yaml
    from pathlib import Path

    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    personality = cfg.get('personality', 'base')
    reward_cfg = cfg.get('reward', {})

    if personality == 'turtle':
        return TurtleShaper(
            fourth_place_penalty=reward_cfg.get('fourth_place_penalty', -200.0),
            dealing_in_penalty=reward_cfg.get('dealing_in_penalty', -50.0),
            high_value_bonus=reward_cfg.get('high_value_bonus', 200.0),
            high_value_threshold=reward_cfg.get('high_value_threshold', 4),
            riichi_bonus=reward_cfg.get('riichi_bonus', 100.0),
        )
    elif personality == 'mad_dog':
        return MadDogShaper(
            first_place_bonus=reward_cfg.get('first_place_bonus', 1000.0),
            early_win_bonus=reward_cfg.get('early_win_bonus', 50.0),
            early_win_turn_threshold=reward_cfg.get('early_win_turn_threshold', 6),
        )
    elif personality == 'riichi_fundamentalist':
        return RiichiFundamentalistShaper(
            chi_pon_penalty=reward_cfg.get('chi_pon_penalty', -50.0),
            riichi_bonus=reward_cfg.get('riichi_bonus', 300.0),
        )
    else:
        return RewardShaper()
# 中文注释：奖励塑形模块，把牌局事件转换为更密集的强化学习训练信号。
