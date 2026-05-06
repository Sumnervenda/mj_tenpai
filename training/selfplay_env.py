"""自对弈环境 —— 4 人共用单一模型进行对局，收集训练轨迹。

对局流程:
  1. 摸牌阶段 (DRAW): 当前玩家获取 state → model 推理 → sample action → engine.step()
  2. 舍牌阶段 (DISCARD): 所有非舍牌玩家获取各自 state → model 推理 → 收集 responses
     → engine.resolve_responses() → 引擎按优先级执行
  3. 和了/流局/局末/终局: engine.step(PASS) 推进状态机
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from engine.game import GameEngine, GameConfig, GamePhase, GameState
from engine.actions import Action, ActionType, LegalActions
from models import MahjongPolicyValueNet


@dataclass
class RolloutStep:
    """自对弈中单步轨迹数据。"""
    state: np.ndarray              # (354,) 状态特征
    mask: np.ndarray               # (77,) 动作掩码
    action: int                    # 选择的动作索引
    log_prob: float                # 所选动作的对数概率
    value: float                   # 状态价值预测
    reward: float                  # 即时奖励
    player_idx: int                # 所属玩家索引 (0-3)
    done: bool = False             # 该步后游戏是否结束


@dataclass
class GameTrajectory:
    """一局完整自对弈轨迹。"""
    steps: List[RolloutStep] = field(default_factory=list)
    final_scores: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    winner: int = -1
    total_steps: int = 0


class SelfPlayEnv:
    """4 人自对弈环境包装器。

    同一模型服务所有 4 位玩家。每步收集 (state, mask, action, log_prob, reward, value)。

    Args:
        model: 策略-价值双头网络
        device: 推理设备
        deterministic: True 时选最大概率动作（评估），False 时按分布采样（训练）
    """

    def __init__(self,
                 model: MahjongPolicyValueNet,
                 device: str = 'cpu',
                 deterministic: bool = False):
        self.model = model
        self.device = device
        self.deterministic = deterministic

    @torch.no_grad()
    def _select_action(self, player_idx: int,
                        engine: GameEngine) -> Tuple[int, float, float]:
        """为指定玩家选择动作。

        Returns:
            action_idx: 选中的动作索引 (0-76)
            log_prob: 该动作的对数概率
            value: 状态价值
        """
        state_np = engine.get_state_tensor(player_idx)
        legal = engine.get_legal_actions(player_idx)

        state_t = torch.from_numpy(state_np).float().to(self.device)
        mask_t = torch.tensor(legal.mask, dtype=torch.float32).to(self.device)

        action_idx, log_prob = self.model.get_action(
            state_t, mask_t, deterministic=self.deterministic)

        # 获取价值
        _, value_t = self.model.forward(state_t, mask_t)
        value = value_t.item()

        return action_idx, log_prob.item(), value

    def _action_from_index(self, action_idx: int, actor: int,
                            legal: LegalActions) -> Action:
        """将动作索引转换回 Action 对象。

        优先从 legal.actions 中反查匹配的具体 Action（保留 tile/meld_tiles）。
        兜底：按动作索引直接构造（兼容未在 actions 列表中的动作类型）。
        """
        # ── 从 legal.actions 反查，保留完整动作信息 ──
        for action in legal.actions:
            if self._action_to_index(action) == action_idx:
                action.actor = actor
                return action

        # ── 兜底：按动作索引直接构造 ──
        if 0 <= action_idx <= 33:
            return Action(ActionType.DISCARD, tile=action_idx, actor=actor)
        elif action_idx == 34:
            return Action(ActionType.TSUMO, actor=actor)
        elif action_idx == 35:
            return Action(ActionType.RON, actor=actor)
        elif 37 <= action_idx <= 70:
            return Action(ActionType.RIICHI,
                          tile=action_idx - 37, actor=actor)
        elif action_idx == 71:
            return Action(ActionType.PON, actor=actor)
        elif action_idx == 72:
            return Action(ActionType.CHI, actor=actor)
        elif action_idx == 73:
            return Action(ActionType.KAN_DAIMIN, actor=actor)
        elif action_idx == 74:
            return Action(ActionType.KAN_ANKAN, actor=actor)
        elif action_idx == 75:
            return Action(ActionType.KAN_KAKAN, actor=actor)
        elif action_idx == 76:
            return Action(ActionType.PASS, actor=actor)
        else:
            return Action(ActionType.PASS, actor=actor)

    @staticmethod
    def _action_to_index(action: Action) -> int:
        """将 Action 对象映射到 77 维动作空间中的索引。"""
        at = action.action_type
        if at == ActionType.DISCARD:
            return action.tile  # 0-33
        elif at == ActionType.TSUMO:
            return 34
        elif at == ActionType.RON:
            return 35
        elif at == ActionType.RIICHI:
            if 0 <= action.tile <= 33:
                return 37 + action.tile
            return 36
        elif at == ActionType.PON:
            return 71
        elif at == ActionType.CHI:
            return 72
        elif at == ActionType.KAN_DAIMIN:
            return 73
        elif at == ActionType.KAN_ANKAN:
            return 74
        elif at == ActionType.KAN_KAKAN:
            return 75
        elif at == ActionType.PASS:
            return 76
        return 76

    def run_game(self, seed: Optional[int] = None) -> GameTrajectory:
        """运行一局完整自对弈，收集全轨迹。

        Args:
            seed: 对局随机种子

        Returns:
            GameTrajectory 包含所有 RolloutStep
        """
        config = GameConfig()
        engine = GameEngine(config=config, seed=seed)
        trajectory = GameTrajectory()
        game_step = 0
        max_steps = 2000

        while not engine.is_game_over() and game_step < max_steps:
            game_step += 1

            if engine.phase == GamePhase.DRAW:
                p = engine.current_player

                # 收集摸牌前状态
                state_np = engine.get_state_tensor(p)
                legal = engine.get_legal_actions(p)

                if not legal.actions:
                    engine.step(Action(ActionType.PASS))
                    continue

                action_idx, log_prob, value = self._select_action(p, engine)
                action = self._action_from_index(action_idx, p, legal)

                # 确保 action 在合法列表中（模型采样有可能因 mask 失效出界）
                prev_scores = [pl.score for pl in engine.players]
                engine.step(action)
                reward = (engine.players[p].score - prev_scores[p]) / 1000.0

                trajectory.steps.append(RolloutStep(
                    state=state_np,
                    mask=np.array(legal.mask, dtype=np.float32),
                    action=action_idx,
                    log_prob=log_prob,
                    value=value,
                    reward=reward,
                    player_idx=p,
                ))

            elif engine.phase == GamePhase.DISCARD:
                # 获取所有非舍牌玩家的响应选项
                options = engine.get_response_options()

                if not options:
                    engine.step(Action(ActionType.PASS))
                    continue

                # 收集各方响应
                prev_scores = [pl.score for pl in engine.players]
                responses: Dict[int, Action] = {}
                for p_idx, legal in options.items():
                    if not legal.actions:
                        continue

                    action_idx, log_prob, value = self._select_action(
                        p_idx, engine)
                    action = self._action_from_index(action_idx, p_idx, legal)
                    responses[p_idx] = action

                    trajectory.steps.append(RolloutStep(
                        state=engine.get_state_tensor(p_idx),
                        mask=np.array(legal.mask, dtype=np.float32),
                        action=action_idx,
                        log_prob=log_prob,
                        value=value,
                        reward=0.0,  # 响应时 reward 暂记为 0，结算后由引擎统一计算
                        player_idx=p_idx,
                    ))

                engine.resolve_responses(responses)

                # 回溯更新 reward（结算后各玩家分数变动）
                for p_idx in range(4):
                    delta = (engine.players[p_idx].score -
                              prev_scores[p_idx]) / 1000.0
                    if delta != 0.0:
                        for step in reversed(trajectory.steps):
                            if step.player_idx == p_idx and step.reward == 0.0:
                                step.reward = delta
                                break

            elif engine.phase in (GamePhase.AGARI, GamePhase.RYUUKYOKU,
                                   GamePhase.ROUND_END, GamePhase.GAME_END):
                engine.step(Action(ActionType.PASS))

            else:
                engine.step(Action(ActionType.PASS))

        # 终局结果
        result = engine.get_result()
        trajectory.final_scores = list(result.adjusted_scores)
        trajectory.winner = engine.get_winner()
        trajectory.total_steps = game_step

        # 标记最后一步为 done
        if trajectory.steps:
            trajectory.steps[-1].done = True

        return trajectory


def run_games(model: MahjongPolicyValueNet,
              num_games: int,
              device: str = 'cpu',
              base_seed: int = 0,
              deterministic: bool = False) -> List[GameTrajectory]:
    """批量运行自对弈，返回轨迹列表。"""
    env = SelfPlayEnv(model, device=device, deterministic=deterministic)
    trajectories = []
    for i in range(num_games):
        seed = base_seed + i if base_seed > 0 else None
        traj = env.run_game(seed=seed)
        trajectories.append(traj)
    return trajectories
