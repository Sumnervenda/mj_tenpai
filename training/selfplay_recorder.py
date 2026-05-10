"""ModelSelfPlayRecorder —— 基座模型驱动的全牌山自博弈轨迹录制器。

用法:
    python -m training.selfplay_recorder \\
        --base_model checkpoints/sl_best.pt \\
        --num_games 10000 --output data/oracle_trajectories.jsonl

输出格式 (JSONL):
    每行一个 OracleTrajectoryStep，包含：
      public/private token IDs、token_types、behavior_ids
      action_mask、chosen_action、scores、reward
      player_idx、game_seed、round 信息
    终局额外输出 GameSummary 行（包含 final_scores、wall、dead_wall 等）。
"""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from engine.game import GameEngine, GameConfig, GamePhase
from engine.actions import Action, ActionType
from engine.tile import abs_to_type
from models import MahjongPolicyValueNet, TransformerPolicyValueNet
from models.model_io import load_checkpoint, load_checkpoint_metadata


# ── 数据结构 ──────────────────────────────────────────────────────────────────

@dataclass
class OracleStep:
    """单步 Oracle 训练样本。"""
    public_token_ids: List[int]
    public_token_types: List[int]
    public_behavior_ids: List[int]
    private_token_ids: List[int]
    private_token_types: List[int]
    private_behavior_ids: List[int]
    action_mask: List[float]
    chosen_action: int
    scores_before: List[int]
    player_idx: int
    game_seed: int
    step: int
    round_wind: int
    round_number: int
    honba: int
    reward: float = 0.0


@dataclass
class OracleGameSummary:
    """终局摘要，包含完整牌山信息。

    字段语义说明：
    - initial_wall_order: 开局时完整牌山（含已发的 53 张 + 未摸的活牌墙），
      按 136 绝对 ID 对应的 tile type 编码，长度 = _dead_wall_start。
    - live_ptr_at_start: 配牌后 wall._live_ptr 值（通常=53），
      initial_wall_order[:live_ptr_at_start] 是已发给四家的牌。
    - dead_wall: 14 张王牌（岭上牌 + 宝牌指示牌）。
    - final_hand_counts_by_player: 终局时四家手牌的 34 维计数数组。
    """
    type: str = 'game_summary'
    game_seed: int = 0
    total_steps: int = 0
    final_scores: List[int] = field(default_factory=lambda: [0] * 4)
    final_ranks: List[int] = field(default_factory=lambda: [0] * 4)
    winner: int = -1
    initial_wall_order: List[int] = field(default_factory=list)
    live_ptr_at_start: int = 53
    dead_wall: List[int] = field(default_factory=list)
    final_hand_counts_by_player: List[List[int]] = field(default_factory=lambda: [[0]*34 for _ in range(4)])
    dora_indicators: List[int] = field(default_factory=list)
    ura_dora_indicators: List[int] = field(default_factory=list)


# ── ModelSelfPlayRecorder ─────────────────────────────────────────────────────

class ModelSelfPlayRecorder:
    """使用模型 checkpoint 驱动 GameEngine 自博弈，逐决策录制 Oracle 轨迹。

    Args:
        agent: 用于选择动作的 Agent（ResNetAgent / TransformerAgent / ...）
        tokenizer: MahjongTokenizer 实例（None 则只录原始状态）
        max_steps: 单局最大步数
        reward_mode: reward 计算方式
            'rank'   — 终局排名奖励 (1st=+1, 2nd=0, 3rd=-0.5, 4th=-1)
            'score'  — 终局归一化得分 (adjusted_score / 100000)
    """

    def __init__(self,
                 agent: Optional[Any] = None,
                 tokenizer: Optional[Any] = None,
                 max_steps: int = 2000,
                 reward_mode: str = 'rank'):
        self.agent = agent
        self.tokenizer = tokenizer
        self.max_steps = max_steps
        self.reward_mode = reward_mode

    def _compute_reward(self, rank: int, score: float) -> float:
        if self.reward_mode == 'rank':
            return {0: 1.0, 1: 0.0, 2: -0.5, 3: -1.0}.get(rank, 0.0)
        return score / 100000.0

    def _get_tokens(self, engine, player_idx):
        """返回 (pub_ids, pub_types, pub_bids, priv_ids, priv_types, priv_bids)"""
        if self.tokenizer is None:
            return ([], [], [], [], [], [])
        pub, priv = self.tokenizer \
            .tokenize_public_private_engine_state(engine, player_idx)
        return (pub.token_ids, pub.token_types, pub.behavior_ids,
                priv.token_ids, priv.token_types, priv.behavior_ids)

    def record_game(self, engine: GameEngine, game_seed: int
                    ) -> Tuple[List[OracleStep], OracleGameSummary]:
        """录制一局完整对局。

        Returns:
            (steps, summary) — 训练步列表 + 终局摘要
        """
        from .agents import Agent, ResNetAgent

        steps: List[OracleStep] = []
        step_count = 0
        start_scores = [p.score for p in engine.players]

        # 在开局时保存完整牌山（配牌后 live_ptr=53，含已发的53张）
        # init_wall 包含所有非 dead wall 的 136 张牌（含已发+未摸），
        # live_ptr_at_start 标记未摸活牌墙的起始位置
        init_wall = [int(abs_to_type(t))
                     for t in engine.wall.tiles[:engine.wall._dead_wall_start]]
        live_ptr_at_start = engine.wall._live_ptr
        init_dead_wall = [int(abs_to_type(t))
                          for t in engine.wall.tiles[
                              engine.wall._dead_wall_start:
                              engine.wall._dead_wall_start + 14]]

        while not engine.is_game_over() and step_count < self.max_steps:
            step_count += 1

            if engine.phase == GamePhase.DRAW:
                p = engine.current_player
                legal = engine.get_legal_actions(p)
                if not legal.actions:
                    engine.step(Action(ActionType.PASS))
                    continue

                # 选择动作（agent 或启发式 fallback）
                if self.agent is not None:
                    action_idx, _, _ = self.agent.select_action(
                        engine, p, deterministic=True)
                else:
                    action_idx = self._heuristic_fallback(legal)

                action = self._index_to_action(action_idx, p, legal)

                # 录制 Oracle 样本
                scores_before = [pl.score for pl in engine.players]
                (pub_ids, pub_types, pub_bids,
                 priv_ids, priv_types, priv_bids) = self._get_tokens(engine, p)

                steps.append(OracleStep(
                    public_token_ids=pub_ids,
                    public_token_types=pub_types,
                    public_behavior_ids=pub_bids,
                    private_token_ids=priv_ids,
                    private_token_types=priv_types,
                    private_behavior_ids=priv_bids,
                    action_mask=list(legal.mask),
                    chosen_action=action_idx,
                    scores_before=scores_before,
                    player_idx=p,
                    game_seed=game_seed,
                    step=step_count,
                    round_wind=engine.round_wind,
                    round_number=engine.round_number,
                    honba=engine.honba,
                ))

                engine.step(action)

            elif engine.phase == GamePhase.DISCARD:
                options = engine.get_response_options()
                if not options:
                    engine.step(Action(ActionType.PASS))
                    continue

                scores_before = [pl.score for pl in engine.players]
                responses: Dict[int, Action] = {}

                for p_idx, legal in options.items():
                    if not legal.actions:
                        continue

                    if self.agent is not None:
                        action_idx, _, _ = self.agent.select_action(
                            engine, p_idx, deterministic=True)
                    else:
                        action_idx = self._heuristic_fallback(legal)
                    responses[p_idx] = self._index_to_action(
                        action_idx, p_idx, legal)

                    (pub_ids, pub_types, pub_bids,
                     priv_ids, priv_types, priv_bids) = \
                        self._get_tokens(engine, p_idx)

                    steps.append(OracleStep(
                        public_token_ids=pub_ids,
                        public_token_types=pub_types,
                        public_behavior_ids=pub_bids,
                        private_token_ids=priv_ids,
                        private_token_types=priv_types,
                        private_behavior_ids=priv_bids,
                        action_mask=list(legal.mask),
                        chosen_action=action_idx,
                        scores_before=scores_before,
                        player_idx=p_idx,
                        game_seed=game_seed,
                        step=step_count,
                        round_wind=engine.round_wind,
                        round_number=engine.round_number,
                        honba=engine.honba,
                    ))

                engine.resolve_responses(responses)

            elif engine.phase in (GamePhase.AGARI, GamePhase.RYUUKYOKU,
                                  GamePhase.ROUND_END, GamePhase.GAME_END):
                engine.step(Action(ActionType.PASS))
            else:
                engine.step(Action(ActionType.PASS))

        # 终局
        result = engine.get_result()
        final_scores = list(result.adjusted_scores)
        ranks = [0] * 4
        sorted_pairs = sorted(enumerate(final_scores),
                              key=lambda x: -x[1])
        for rank_order, (p_idx, _) in enumerate(sorted_pairs):
            ranks[p_idx] = rank_order
        winner = engine.get_winner()

        # 回溯写入 reward
        for s in steps:
            rank = ranks[s.player_idx]
            s.reward = self._compute_reward(rank, final_scores[s.player_idx])

        # 终局时四家手牌计数（hand.tiles 已是 34 维计数数组）
        final_hand_counts = [list(engine.players[p].hand.tiles) for p in range(4)]

        summary = OracleGameSummary(
            game_seed=game_seed,
            total_steps=step_count,
            final_scores=final_scores,
            final_ranks=ranks,
            winner=winner,
            initial_wall_order=init_wall,
            live_ptr_at_start=live_ptr_at_start,
            dead_wall=init_dead_wall,
            final_hand_counts_by_player=final_hand_counts,
            dora_indicators=[int(abs_to_type(t))
                             for t in engine.get_game_state().dora_indicators],
            ura_dora_indicators=[int(abs_to_type(t))
                                 for t in engine.get_game_state().ura_dora_indicators],
        )

        return steps, summary

    @staticmethod
    def _index_to_action(action_idx: int, actor: int, legal) -> Action:
        for action in legal.actions:
            if _action_to_index(action) == action_idx:
                action.actor = actor
                return action
        # Fallback
        if 0 <= action_idx <= 33:
            return Action(ActionType.DISCARD, tile=action_idx, actor=actor)
        if action_idx == 34:
            return Action(ActionType.TSUMO, actor=actor)
        if action_idx == 35:
            return Action(ActionType.RON, actor=actor)
        if 37 <= action_idx <= 70:
            return Action(ActionType.RIICHI,
                          tile=action_idx - 37, actor=actor)
        if action_idx == 71:
            return Action(ActionType.PON, actor=actor)
        if action_idx == 72:
            return Action(ActionType.CHI, actor=actor)
        if action_idx == 73:
            return Action(ActionType.KAN_DAIMIN, actor=actor)
        if action_idx == 74:
            return Action(ActionType.KAN_ANKAN, actor=actor)
        if action_idx == 75:
            return Action(ActionType.KAN_KAKAN, actor=actor)
        return Action(ActionType.PASS, actor=actor)

    @staticmethod
    def _heuristic_fallback(legal) -> int:
        """简单启发式 fallback：优先舍牌，其次 PASS。"""
        mask = legal.mask
        for i in range(34):
            if mask[i] > 0:
                return i
        if mask[76] > 0:
            return 76
        for i in range(77):
            if mask[i] > 0:
                return i
        return 76


def _action_to_index(action: Action) -> int:
    at = action.action_type
    if at == ActionType.DISCARD:
        return action.tile
    if at == ActionType.TSUMO:
        return 34
    if at == ActionType.RON:
        return 35
    if at == ActionType.RIICHI:
        return 37 + action.tile if 0 <= action.tile <= 33 else 36
    if at == ActionType.PON:
        return 71
    if at == ActionType.CHI:
        return 72
    if at == ActionType.KAN_DAIMIN:
        return 73
    if at == ActionType.KAN_ANKAN:
        return 74
    if at == ActionType.KAN_KAKAN:
        return 75
    if at == ActionType.PASS:
        return 76
    return 76


# ── 批量录制工具 ──────────────────────────────────────────────────────────────

def record_games(output_path: str,
                 num_games: int,
                 agent: Optional[Any] = None,
                 tokenizer: Optional[Any] = None,
                 base_seed: int = 0,
                 reward_mode: str = 'rank',
                 progress_every: int = 100) -> Dict[str, Any]:
    """批量录制 Oracle 轨迹到 JSONL。

    Args:
        output_path: JSONL 输出路径
        num_games: 录制局数
        agent: 动作选择 Agent（None 则用启发式 fallback）
        tokenizer: MahjongTokenizer（None 则 token 字段为空）
        base_seed: 基础随机种子
        reward_mode: reward 计算模式
        progress_every: 进度打印间隔

    Returns:
        录制统计 dict
    """
    # 始终创建 tokenizer 以录制 public/private tokens
    if tokenizer is None:
        from models.tokenizer import MahjongTokenizer
        tokenizer = MahjongTokenizer(max_sequence_length=512)
    recorder = ModelSelfPlayRecorder(
        agent=agent, tokenizer=tokenizer, reward_mode=reward_mode)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    total_steps = 0
    total_games_ok = 0

    with open(output_path, 'w', encoding='utf-8') as f:
        for game_idx in range(num_games):
            game_seed = base_seed + game_idx
            engine = GameEngine(config=GameConfig(), seed=game_seed)
            try:
                steps, summary = recorder.record_game(engine, game_seed)
                for s in steps:
                    f.write(json.dumps(asdict(s), ensure_ascii=False) + '\n')
                f.write(json.dumps(asdict(summary), ensure_ascii=False) + '\n')
                total_steps += len(steps)
                total_games_ok += 1
            except Exception as e:
                print(f"  Game {game_idx} (seed={game_seed}) error: {e}")

            if progress_every > 0 and (game_idx + 1) % progress_every == 0:
                elapsed = time.time() - start
                speed = (game_idx + 1) / elapsed if elapsed > 0 else 0
                print(f"  Recorded {game_idx + 1}/{num_games} games, "
                      f"steps={total_steps}, {speed:.1f} games/sec")

    elapsed = time.time() - start
    stats = {
        'output_path': output_path,
        'num_games': total_games_ok,
        'total_steps': total_steps,
        'elapsed_sec': elapsed,
        'games_per_sec': total_games_ok / elapsed if elapsed > 0 else 0,
    }
    print(f"Done: {total_games_ok}/{num_games} games, {total_steps} steps, "
          f"{elapsed:.1f}s ({stats['games_per_sec']:.1f} games/sec)")
    return stats


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Record oracle trajectories from model self-play')
    p.add_argument('--base_model', type=str, default=None,
                   help='Model checkpoint path (None = heuristic only)')
    p.add_argument('--model_arch', type=str, default='resnet',
                   choices=['resnet', 'transformer', 'heuristic'],
                   help='Model architecture (heuristic = no model)')
    p.add_argument('--num_games', type=int, default=1000)
    p.add_argument('--output', type=str,
                   default='data/oracle_trajectories.jsonl')
    p.add_argument('--base_seed', type=int, default=0)
    p.add_argument('--reward_mode', type=str, default='rank',
                   choices=['rank', 'score'])
    p.add_argument('--device', type=str, default='cpu')
    p.add_argument('--progress_every', type=int, default=100)
    return p.parse_args()


def main():
    args = parse_args()
    agent = None
    tokenizer = None

    if args.base_model and args.model_arch != 'heuristic':
        meta = load_checkpoint_metadata(args.base_model)
        ckpt_arch = meta.get('model_arch', args.model_arch)
        if ckpt_arch != args.model_arch:
            print(f"Auto-detected model_arch={ckpt_arch}")
            args.model_arch = ckpt_arch

        if args.model_arch == 'transformer':
            from models.tokenizer import MahjongTokenizer
            model = TransformerPolicyValueNet(
                d_model=meta.get('d_model', 256),
                n_layers=meta.get('n_layers', 6),
                n_heads=meta.get('n_heads', 8),
                n_concept=meta.get('n_concept', 10),
                max_len=meta.get('max_len', 256),
            )
            load_checkpoint(model, args.base_model, device=args.device)
            model = model.to(args.device)
            model.eval()
            tokenizer = MahjongTokenizer(
                max_sequence_length=meta.get('max_len', 256))
            from .agents import TransformerAgent
            agent = TransformerAgent(model, tokenizer=tokenizer,
                                     device=args.device)
            print(f"Loaded Transformer agent from {args.base_model}")
        else:
            model = MahjongPolicyValueNet()
            load_checkpoint(model, args.base_model, device=args.device)
            model = model.to(args.device)
            model.eval()
            from .agents import ResNetAgent
            agent = ResNetAgent(model, device=args.device)
            print(f"Loaded ResNet agent from {args.base_model}")
        # ResNet/heuristic 也需要 tokenizer 来录制 public/private tokens
        if tokenizer is None:
            from models.tokenizer import MahjongTokenizer
            tokenizer = MahjongTokenizer(max_sequence_length=512)
    else:
        print("Using heuristic fallback (no model)")
        # heuristic 也需要 tokenizer 来录制 public/private tokens
        from models.tokenizer import MahjongTokenizer
        tokenizer = MahjongTokenizer(max_sequence_length=512)

    stats = record_games(
        output_path=args.output,
        num_games=args.num_games,
        agent=agent,
        tokenizer=tokenizer,
        base_seed=args.base_seed,
        reward_mode=args.reward_mode,
        progress_every=args.progress_every,
    )
    print(f"Output: {stats['output_path']}")


if __name__ == '__main__':
    main()
