"""自对弈环境 —— 4 人共用单一模型进行对局，收集训练轨迹。

对局流程:
  1. 摸牌阶段 (DRAW): 当前玩家获取 state → model 推理 → sample action → engine.step()
  2. 舍牌阶段 (DISCARD): 所有非舍牌玩家获取各自 state → model 推理 → 收集 responses
     → engine.resolve_responses() → 引擎按优先级执行
  3. 和了/流局/局末/终局: engine.step(PASS) 推进状态机
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from engine.game import GameEngine, GameConfig, GamePhase, GameState
from engine.actions import Action, ActionType, LegalActions

# Agent 抽象接口（优先使用），同时保持对裸 MahjongPolicyValueNet 的向后兼容
from .agents import Agent, ResNetAgent
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
    game_id: int = 0               # 所属对局 ID（用于 GAE 分组）
    done: bool = False             # 该步后游戏是否结束
    sl_log_prob: float = 0.0       # SL 冻结策略的对数概率（KL 正则化用）


@dataclass
class GameTrajectory:
    """一局完整自对弈轨迹。"""
    steps: List[RolloutStep] = field(default_factory=list)
    final_scores: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    winner: int = -1
    total_steps: int = 0


class SelfPlayEnv:
    """4 人自对弈环境包装器，支持单玩家训练 + 冻结对手。

    当 baseline_model 为 None 时，同一模型控制四家（旧行为）。
    当 baseline_model 传入时：trainee_idx 玩家使用 self.model + reward_shaper，
    其余三家使用 frozen baseline_model，仅收集 trainee 的轨迹数据。

    支持两种初始化方式（向后兼容）：
      - SelfPlayEnv(model=MahjongPolicyValueNet, ...)  # 原有接口，自动包装为 ResNetAgent
      - SelfPlayEnv(agent=Agent, ...)                   # 新接口，直接传入 Agent 实例

    Args:
        model: 策略-价值双头网络（trainee / 待训练模型），与 agent 二选一
        agent: Agent 实例（优先），与 model 二选一
        device: 推理设备
        deterministic: True 时选最大概率动作（评估），False 时按分布采样（训练）
        reward_shaper: 奖励塑形器，仅对 trainee 生效
        trainee_idx: 被训练玩家的座位索引 (0-3)
        baseline_model: 对手使用的冻结模型（自动包装为 ResNetAgent）
        baseline_agent: 对手使用的 Agent 实例（优先于 baseline_model）
        kl_coef: KL 正则系数
    """

    def __init__(self,
                 model: Optional[MahjongPolicyValueNet] = None,
                 device: str = 'cpu',
                 deterministic: bool = False,
                 reward_shaper: object = None,
                 trainee_idx: int = 0,
                 baseline_model: Optional[MahjongPolicyValueNet] = None,
                 baseline_agent: Optional[Agent] = None,
                 kl_coef: float = 0.01,
                 agent: Optional[Agent] = None):
        self.device = device
        self.deterministic = deterministic
        self.reward_shaper = reward_shaper
        self.trainee_idx = trainee_idx
        self.kl_coef = kl_coef
        self._game_id_counter = 0

        # 解析 trainee agent：优先 agent 参数，否则从 model 包装
        if agent is not None:
            self.agent = agent
            self.model = None  # 无裸模型引用（不用于反向兼容 run_eval 等）
        elif model is not None:
            self.agent = ResNetAgent(model, device=device)
            self.model = model  # 保留裸模型引用，用于反向兼容
        else:
            raise ValueError("Must provide either 'agent' or 'model'")

        # 解析 baseline agent
        if baseline_agent is not None:
            self.baseline_agent: Optional[Agent] = baseline_agent
        elif baseline_model is not None:
            self.baseline_agent = ResNetAgent(baseline_model, device=device)
        else:
            self.baseline_agent = None
        # 保留裸模型引用，用于 run_eval 等直接访问的场景
        self.baseline_model = baseline_model

    def _agent_for(self, player_idx: int) -> Agent:
        """返回指定玩家的 Agent：trainee 用训练 agent，对手用 baseline。"""
        if player_idx == self.trainee_idx:
            return self.agent
        if self.baseline_agent is not None:
            return self.baseline_agent
        return self.agent

    @torch.no_grad()
    def _select_action(self, player_idx: int,
                        engine: GameEngine) -> Tuple[int, float, float, float]:
        """为指定玩家选择动作。

        Returns:
            action_idx: 选中的动作索引 (0-76)
            log_prob: 该动作的对数概率
            value: 状态价值
            sl_log_prob: SL 冻结策略的 log_prob（KL 正则化用，baseline 不可用时为 0）
        """
        ag = self._agent_for(player_idx)
        action_idx, log_prob, value = ag.select_action(
            engine, player_idx, deterministic=self.deterministic)

        # KL 正则化：baseline agent 的 deterministic log_prob
        sl_log_prob = 0.0
        if player_idx == self.trainee_idx and self.baseline_agent is not None:
            try:
                _, sl_lp, _ = self.baseline_agent.select_action(
                    engine, player_idx, deterministic=True)
                sl_log_prob = sl_lp
            except Exception:
                pass  # agent 可能不支持 baseline KL（如 OracleTeacher）

        return action_idx, log_prob, value, sl_log_prob

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
        game_id = seed if seed is not None else self._game_id_counter
        self._game_id_counter += 1
        game_step = 0
        max_steps = 2000

        # 推理时使用 eval 模式：BatchNorm 用 running stats，单样本推理稳定
        model_was_training = self.model.training if self.model else False
        bl_was_training = self.baseline_model.training if self.baseline_model else False
        if self.model:
            self.model.eval()
        if self.baseline_model:
            self.baseline_model.eval()

        while not engine.is_game_over() and game_step < max_steps:
            game_step += 1

            if engine.phase == GamePhase.DRAW:
                p = engine.current_player
                is_trainee = (p == self.trainee_idx)

                state_np = engine.get_state_tensor(p)
                legal = engine.get_legal_actions(p)

                if not legal.actions:
                    engine.step(Action(ActionType.PASS))
                    continue

                action_idx, log_prob, value, sl_log_prob = self._select_action(p, engine)
                action = self._action_from_index(action_idx, p, legal)

                prev_scores = [pl.score for pl in engine.players]
                engine.step(action)

                # 仅 trainee 使用 reward_shaper，对手用基础分数归一化
                if is_trainee and self.reward_shaper is not None:
                    reward = self.reward_shaper(engine, p, action)
                else:
                    reward = (engine.players[p].score - prev_scores[p]) / 10000.0

                # 仅收集 trainee 的轨迹数据
                if is_trainee:
                    trajectory.steps.append(RolloutStep(
                        state=state_np,
                        mask=np.array(legal.mask, dtype=np.float32),
                        action=action_idx,
                        log_prob=log_prob,
                        value=value,
                        reward=reward,
                        player_idx=p,
                        game_id=game_id,
                        sl_log_prob=sl_log_prob,
                    ))

            elif engine.phase == GamePhase.DISCARD:
                options = engine.get_response_options()

                if not options:
                    engine.step(Action(ActionType.PASS))
                    continue

                prev_scores = [pl.score for pl in engine.players]
                responses: Dict[int, Action] = {}
                for p_idx, legal in options.items():
                    if not legal.actions:
                        continue

                    is_trainee = (p_idx == self.trainee_idx)
                    action_idx, log_prob, value, sl_log_prob = self._select_action(
                        p_idx, engine)
                    action = self._action_from_index(action_idx, p_idx, legal)
                    responses[p_idx] = action

                    # 仅收集 trainee 的响应步骤
                    if is_trainee:
                        trajectory.steps.append(RolloutStep(
                            state=engine.get_state_tensor(p_idx),
                            mask=np.array(legal.mask, dtype=np.float32),
                            action=action_idx,
                            log_prob=log_prob,
                            value=value,
                            reward=0.0,  # 结算后回溯写入
                            player_idx=p_idx,
                            game_id=game_id,
                            sl_log_prob=sl_log_prob,
                        ))

                engine.resolve_responses(responses)

                # 回溯写入 reward（仅 trainee 使用 reward_shaper）
                for p_idx, action in responses.items():
                    is_trainee = (p_idx == self.trainee_idx)
                    if is_trainee and self.reward_shaper is not None:
                        shaped = self.reward_shaper(engine, p_idx, action)
                    else:
                        delta = (engine.players[p_idx].score -
                                 prev_scores[p_idx]) / 10000.0
                        shaped = delta
                    if is_trainee and shaped != 0.0:
                        for step in reversed(trajectory.steps):
                            if step.player_idx == p_idx and step.reward == 0.0:
                                step.reward = shaped
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

        # 终端惩罚仅来自 reward_shaper（四位惩罚、放铳等），
        # 不再额外施加排名奖励以保持 reward 信号密度均匀。

        # 恢复模型训练模式
        if model_was_training and self.model:
            self.model.train()
        if bl_was_training and self.baseline_model:
            self.baseline_model.train()

        # 标记最后一步为 done
        if trajectory.steps:
            trajectory.steps[-1].done = True

        return trajectory


def run_games(model: MahjongPolicyValueNet,
              num_games: int,
              device: str = 'cpu',
              base_seed: int = 0,
              deterministic: bool = False,
              agent: Optional[Agent] = None) -> List[GameTrajectory]:
    """批量运行自对弈，返回轨迹列表。"""
    env = SelfPlayEnv(model=model, agent=agent, device=device,
                      deterministic=deterministic)
    trajectories = []
    for i in range(num_games):
        seed = base_seed + i if base_seed > 0 else None
        traj = env.run_game(seed=seed)
        trajectories.append(traj)
    return trajectories
# 中文注释：把麻将引擎包装成自博弈环境，向 PPO 提供状态、动作 mask 和奖励。
