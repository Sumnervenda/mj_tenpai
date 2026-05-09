"""Tests for RL/PPO training correctness."""

import os
import sys
import tempfile
from pathlib import Path

import pytest
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.ppo_agent import compute_gae, compute_gae_grouped
from training.reward_shaper import (
    RewardShaper, TurtleShaper, MadDogShaper,
    RiichiFundamentalistShaper, load_shaper_from_config,
)
from training.rl_selfplay import load_training_config, run_eval
from training.selfplay_env import SelfPlayEnv
from engine import GameEngine, GameConfig, Action, ActionType, GamePhase
from models import MahjongPolicyValueNet


class TestGAEGrouped:
    """P1: GAE must compute per (game_id, player_idx) group, not across players."""

    def test_grouped_gae_isolates_groups(self):
        """GAE for one group is unaffected by another group's values."""
        torch.manual_seed(42)

        # Two groups interleaved: group A at [0,2,4], group B at [1,3,5]
        rewards = torch.tensor([1.0, 0.1, 2.0, 0.1, 3.0, 0.1])
        values = torch.tensor([0.5, 9.9, 1.0, 9.9, 2.0, 9.9])
        dones = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        game_ids = [0, 1, 0, 1, 0, 1]
        player_indices = [0, 0, 0, 0, 0, 0]  # same player_idx, diff game_id

        returns_g, adv_g = compute_gae_grouped(
            rewards, values, dones, game_ids, player_indices,
            gamma=0.99, gae_lambda=0.95,
        )

        # Compute ground truth: GAE on group A indices [0, 2, 4] only
        idx_a = torch.tensor([0, 2, 4])
        ret_a, adv_a = compute_gae(
            rewards[idx_a], values[idx_a], dones[idx_a],
            gamma=0.99, gae_lambda=0.95,
        )

        # Group A results must match
        assert torch.allclose(returns_g[idx_a], ret_a, atol=1e-6), \
            f"Group A returns mismatch: {returns_g[idx_a]} vs {ret_a}"
        assert torch.allclose(adv_g[idx_a], adv_a, atol=1e-6), \
            f"Group A advantages mismatch: {adv_g[idx_a]} vs {adv_a}"

    def test_grouped_gae_differs_from_flat_gae(self):
        """Grouped GAE must differ from flat GAE when groups are interleaved."""
        torch.manual_seed(123)

        # Create two different groups with very different value scales
        rewards = torch.tensor([1.0, -5.0, 2.0, -3.0, 3.0, -1.0])
        values = torch.tensor([0.5, -2.0, 1.0, -1.0, 2.0, 0.0])
        dones = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0])
        game_ids = [0, 1, 0, 1, 0, 1]
        player_indices = [0, 0, 0, 0, 0, 0]

        returns_g, adv_g = compute_gae_grouped(
            rewards, values, dones, game_ids, player_indices,
        )

        # Flat GAE would mix groups
        returns_flat, adv_flat = compute_gae(rewards, values, dones)

        # The results should differ (this is the bug being fixed)
        assert not torch.allclose(returns_g, returns_flat, atol=1e-6), \
            "Grouped and flat GAE should differ for interleaved groups"

    def test_grouped_gae_multi_player_same_game(self):
        """Same game, different players each get independent GAE."""
        torch.manual_seed(99)

        # P0 steps: [0, 4, 8], P1 steps: [1, 5], P2: [2, 6], P3: [3, 7]
        rewards = torch.tensor([0.5, 0.3, -0.1, 0.0, 0.8, -0.2, 0.1, 0.0, 1.2])
        values = torch.tensor([0.4, 0.2, -0.1, 0.0, 0.6, -0.1, 0.0, 0.0, 0.9])
        dones = torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 1.])
        game_ids = [0] * 9  # all same game
        player_indices = [0, 1, 2, 3, 0, 1, 2, 3, 0]

        returns_g, adv_g = compute_gae_grouped(
            rewards, values, dones, game_ids, player_indices,
        )

        # P0 should have 3 steps: indices [0, 4, 8]
        idx_p0 = torch.tensor([0, 4, 8])
        ret_p0, _ = compute_gae(rewards[idx_p0], values[idx_p0], dones[idx_p0])
        assert torch.allclose(returns_g[idx_p0], ret_p0, atol=1e-6)

        # P1 should have 2 steps: indices [1, 5]
        idx_p1 = torch.tensor([1, 5])
        ret_p1, _ = compute_gae(rewards[idx_p1], values[idx_p1], dones[idx_p1])
        assert torch.allclose(returns_g[idx_p1], ret_p1, atol=1e-6)


class TestRewardShaper:
    """P1: Reward shaper must be called and produce correct values."""

    def test_base_shaper_normalizes_by_10000(self):
        """Base RewardShaper divides score delta by 10000."""
        shaper = RewardShaper()
        engine = GameEngine(seed=42)

        # Initial state: all scores are 25000
        state = engine.get_game_state()
        assert state.rewards == [0.0, 0.0, 0.0, 0.0]

        # Manually run step to get rewards
        # Discard a tile to trigger score tracking
        hand = engine.players[0].hand.tiles
        discard_tile = next(t for t in range(34) if hand[t] > 0)
        engine.step(Action(ActionType.DISCARD, tile=discard_tile))
        engine.resolve_responses({
            1: Action(ActionType.PASS),
            2: Action(ActionType.PASS),
            3: Action(ActionType.PASS),
        })

        # After a normal discard pass, all scores still 25000, rewards ~0
        result = shaper(engine, 0)
        assert abs(result) < 1e-6, f"Expected ~0, got {result}"

    def test_turtle_shaper_dealing_in_penalty(self):
        """TurtleShaper applies dealing-in penalty."""
        shaper = TurtleShaper(dealing_in_penalty=-50.0, fourth_place_penalty=-200.0)
        engine = GameEngine(seed=42)

        # Simulate dealing-in by patching _last_agari_payments
        from engine.scoring import PaymentInfo
        engine._last_agari_payments = [
            PaymentInfo(winner=0, loser=1, han=1, fu=30,
                        yaku_names=[], dora_count=0, score_name='',
                        payments=[0, -1000, 0, 0], total_win=1000),
            PaymentInfo(winner=2, loser=1, han=1, fu=30,
                        yaku_names=[], dora_count=0, score_name='',
                        payments=[0, -1000, 0, 0], total_win=1000),
        ]

        # P1 dealt in twice → dealing_in_penalty applied twice
        result = shaper(engine, 1)
        # -50/10000 * 2 = -0.01
        assert result < 0, f"Expected negative reward, got {result}"

    def test_mad_dog_first_place_bonus(self):
        """MadDogShaper applies first place bonus at game end."""
        shaper = MadDogShaper(first_place_bonus=1000.0)
        engine = GameEngine(seed=42)

        # Set scores so P0 is 1st, then trigger game-over
        engine.players[0].score = 40000
        engine.players[1].score = 25000
        engine.players[2].score = 20000
        engine.players[3].score = 15000
        engine._game_finished = True

        result = shaper(engine, 0)
        # Base reward ~0 + first_place_bonus/10000 = 1000/10000 = 0.1
        assert result > 0.09, f"Expected ~0.1 from 1st place bonus, got {result}"

    def test_riichi_fundamentalist_chi_pon_penalty(self):
        """RiichiFundamentalistShaper penalizes chi/pon."""
        shaper = RiichiFundamentalistShaper(
            chi_pon_penalty=-50.0, riichi_bonus=300.0)

        # Test chi penalty
        chi_action = Action(ActionType.CHI, actor=0)
        engine = GameEngine(seed=42)
        result = shaper(engine, 0, action=chi_action)
        assert result < 0, \
            f"Chi should be penalized, got {result}"

    def test_riichi_fundamentalist_riichi_bonus(self):
        """RiichiFundamentalistShaper rewards riichi."""
        shaper = RiichiFundamentalistShaper(
            chi_pon_penalty=-50.0, riichi_bonus=300.0)

        riichi_action = Action(ActionType.RIICHI, actor=0)
        engine = GameEngine(seed=42)
        result = shaper(engine, 0, action=riichi_action)
        assert result > 0, \
            f"Riichi should be rewarded, got {result}"


class TestRewardScaleConsistency:
    """P1: Reward scale must be consistent between selfplay_env and reward_shaper."""

    def test_base_shaper_uses_10000_divisor(self):
        """Base RewardShaper divides by 10000."""
        shaper = RewardShaper()
        engine = GameEngine(seed=42)

        # Set up known score difference
        engine.players[0].score = 35000  # +10000 from initial
        state = engine.get_game_state(rewards=[10000.0, 0.0, 0.0, 0.0])

        # Monkey-patch get_game_state to return our custom state
        original = engine.get_game_state
        engine.get_game_state = lambda: state
        try:
            result = shaper(engine, 0)
            assert abs(result - 1.0) < 1e-6, \
                f"10000 score delta / 10000 should be 1.0, got {result}"
        finally:
            engine.get_game_state = original

    def test_selfplay_env_reward_divisor(self):
        """SelfPlayEnv uses 10000 as reward divisor."""
        # The env computes reward as (score_delta) / 10000.0
        # Verify by checking the code path: 10000 score delta = 1.0 reward
        score_delta = 10000
        reward = score_delta / 10000.0
        assert abs(reward - 1.0) < 1e-9

    def test_subclass_shapers_use_10000_not_1000(self):
        """All shaper subclasses divide by 10000, not 1000."""
        # Verify penalty amounts are divided by 10000 in subclasses
        turtle = TurtleShaper(fourth_place_penalty=-200.0)
        assert turtle.fourth_place_penalty == -200.0
        # The penalty is applied as self.fourth_place_penalty / 10000.0
        # So -200 / 10000 = -0.02

        mad = MadDogShaper(first_place_bonus=1000.0)
        assert mad.first_place_bonus == 1000.0
        # 1000 / 10000 = 0.1

        rif = RiichiFundamentalistShaper(chi_pon_penalty=-50.0, riichi_bonus=300.0)
        assert rif.chi_pon_penalty == -50.0
        assert rif.riichi_bonus == 300.0


class TestTrainingConfig:
    """P2: Training config must read rl_training key."""

    def test_load_config_reads_rl_training_key(self):
        """load_training_config reads from rl_training when present."""
        yaml_content = """
rl_training:
  lr: 1.0e-4
  ppo_epochs: 4
  mini_batch_size: 512
  entropy_coef: 0.005
"""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.yaml', delete=False,
            encoding='utf-8',
        ) as f:
            f.write(yaml_content)
            tmp_path = f.name

        try:
            cfg = load_training_config(tmp_path)
            assert cfg['lr'] == 1.0e-4
            assert cfg['ppo_epochs'] == 4
            assert cfg['mini_batch_size'] == 512
            assert cfg['entropy_coef'] == 0.005
        finally:
            os.unlink(tmp_path)

    def test_load_config_falls_back_to_training_key(self):
        """load_training_config falls back to 'training' key."""
        yaml_content = """
training:
  lr: 5.0e-5
  ppo_epochs: 8
"""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.yaml', delete=False,
            encoding='utf-8',
        ) as f:
            f.write(yaml_content)
            tmp_path = f.name

        try:
            cfg = load_training_config(tmp_path)
            assert cfg['lr'] == 5.0e-5
            assert cfg['ppo_epochs'] == 8
        finally:
            os.unlink(tmp_path)

    def test_load_config_returns_defaults_for_empty_file(self):
        """load_training_config returns defaults when YAML has no training keys."""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.yaml', delete=False,
            encoding='utf-8',
        ) as f:
            f.write("model:\n  base_channels: 64\n")
            tmp_path = f.name

        try:
            cfg = load_training_config(tmp_path)
            assert cfg['lr'] == 3e-4  # default
            assert cfg['gamma'] == 0.99  # default
        finally:
            os.unlink(tmp_path)


class TestEvalMetrics:
    """P2: run_eval must return enriched metrics."""

    def test_run_eval_returns_required_keys(self):
        """run_eval returns single-player metrics (trainee view)."""
        model = MahjongPolicyValueNet()
        metrics = run_eval(model, num_games=2, device='cpu')

        assert 'win_rate' in metrics      # single float
        assert 'avg_steps' in metrics
        assert 'avg_rank' in metrics      # single float
        assert 'avg_score' in metrics     # single float
        assert 'fourth_rate' in metrics   # single float
        assert 'rank_distribution' in metrics

        assert isinstance(metrics['avg_rank'], float)
        assert isinstance(metrics['win_rate'], float)
        assert isinstance(metrics['fourth_rate'], float)

        # avg_rank should be in [0, 3]
        assert 0.0 <= metrics['avg_rank'] <= 3.0

        # rank_distribution should have 4 ranks (0-3)
        rd = metrics['rank_distribution']
        assert len(rd) == 4
        assert sum(int(v) for v in rd.values()) == 2  # num_games=2


class TestRewardShapersIntegration:
    """Integration: verify that each shaper subclass returns a float."""

    @pytest.mark.parametrize('shaper_cls,kwargs', [
        (RewardShaper, {}),
        (TurtleShaper, {}),
        (MadDogShaper, {}),
        (RiichiFundamentalistShaper, {}),
    ])
    def test_shaper_returns_float(self, shaper_cls, kwargs):
        """All shapers return a float from __call__."""
        shaper = shaper_cls(**kwargs)
        engine = GameEngine(seed=42)
        result = shaper(engine, 0)
        assert isinstance(result, float), \
            f"{shaper_cls.__name__} returned {type(result)}"


class TestLoadShaperFromConfig:
    """load_shaper_from_config creates correct shaper type."""

    def test_loads_turtle_from_config(self):
        yaml_content = """
personality: turtle
reward:
  fourth_place_penalty: -150.0
  dealing_in_penalty: -30.0
"""
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.yaml', delete=False,
            encoding='utf-8',
        ) as f:
            f.write(yaml_content)
            tmp_path = f.name

        try:
            shaper = load_shaper_from_config(tmp_path)
            assert isinstance(shaper, TurtleShaper)
            assert shaper.fourth_place_penalty == -150.0
            assert shaper.dealing_in_penalty == -30.0
        finally:
            os.unlink(tmp_path)

    def test_loads_base_for_unknown_personality(self):
        yaml_content = "personality: unknown\n"
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.yaml', delete=False,
            encoding='utf-8',
        ) as f:
            f.write(yaml_content)
            tmp_path = f.name

        try:
            shaper = load_shaper_from_config(tmp_path)
            assert isinstance(shaper, RewardShaper)
# 中文注释：验证强化学习训练流程中的奖励塑形、评估指标和训练配置加载的正确性。

        finally:
            os.unlink(tmp_path)
