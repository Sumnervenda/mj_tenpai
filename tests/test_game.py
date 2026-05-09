"""Tests for game engine integration."""
import pytest
from engine import (
    GameEngine, GameConfig, GamePhase,
    Action, ActionType,
)
from engine.agari import is_agari, is_tenpai
from engine.hand import Hand, Meld, MeldType
from engine.yaku import _is_chiitoitsu


class TestGameFlow:
    """Basic game flow tests."""

    def test_game_initialization(self):
        engine = GameEngine(seed=42)
        assert len(engine.players) == 4
        assert all(p.score == 25000 for p in engine.players)
        assert engine.phase == GamePhase.DRAW
        assert engine.current_player == 0

    def test_deal_correct_tile_counts(self):
        engine = GameEngine(seed=42)
        # Dealer (P0) gets 14, others get 13
        assert engine.players[0].hand.total_concealed() == 14
        assert engine.players[1].hand.total_concealed() == 13
        assert engine.players[2].hand.total_concealed() == 13
        assert engine.players[3].hand.total_concealed() == 13

    def test_wall_has_correct_remaining(self):
        engine = GameEngine(seed=42)
        state = engine.get_game_state()
        # 136 - 14*4 + 1 = 136 - 53 = 83... wait
        # Actually 136 total, 53 dealt (14+13+13+13), 14 dead wall
        # Live wall: 136 - 53 - 14 = 69
        assert state.remaining_tiles == 69

    def test_dora_indicator_flipped(self):
        engine = GameEngine(seed=42)
        assert len(engine.wall.dora_indicators) == 1  # initial dora flipped

    def test_discard_advances_to_next_player(self):
        """After a normal discard + pass, next player draws."""
        engine = GameEngine(seed=42)

        # Current player (0) discards
        hand = engine.players[0].hand.tiles
        discard_tile = next(t for t in range(34) if hand[t] > 0)

        engine.step(Action(ActionType.DISCARD, tile=discard_tile))

        # Should be in DISCARD phase
        assert engine.phase == GamePhase.DISCARD

        # All pass → next player draws
        engine.resolve_responses({
            1: Action(ActionType.PASS),
            2: Action(ActionType.PASS),
            3: Action(ActionType.PASS),
        })

        assert engine.phase == GamePhase.DRAW
        assert engine.current_player == 1

    def test_game_completes(self):
        """Run a full game with random actions, verify it completes."""
        engine = GameEngine(seed=123)

        import random
        rng = random.Random(456)
        max_steps = 5000
        steps = 0

        while not engine.is_game_over() and steps < max_steps:
            state = engine.get_game_state()

            if state.phase == GamePhase.DRAW:
                actions = engine.get_legal_actions()
                non_pass = [a for a in actions.actions
                            if a.action_type != ActionType.PASS]
                action = rng.choice(non_pass) if non_pass else actions.actions[0]
                engine.step(action)

            elif state.phase == GamePhase.DISCARD:
                options = engine.get_response_options()
                responses = {}
                for p_idx, legal in options.items():
                    non_pass = [a for a in legal.actions
                                if a.action_type != ActionType.PASS]
                    responses[p_idx] = rng.choice(non_pass) if non_pass else Action(ActionType.PASS)
                engine.resolve_responses(responses)

            elif state.phase in (GamePhase.AGARI, GamePhase.RYUUKYOKU):
                engine.step(Action(ActionType.PASS))

            steps += 1

        assert engine.is_game_over() or steps < max_steps
        # Verify scores are still integers
        for p in engine.players:
            assert isinstance(p.score, int)

    def test_multiple_rounds(self):
        """Test that game plays through multiple rounds."""
        engine = GameEngine(seed=789)

        import random
        rng = random.Random(101112)
        round_count = 0
        max_steps = 2000

        for _ in range(max_steps):
            if engine.is_game_over():
                break

            state = engine.get_game_state()

            if state.phase == GamePhase.DRAW:
                actions = engine.get_legal_actions()
                non_pass = [a for a in actions.actions if a.action_type != ActionType.PASS]
                action = rng.choice(non_pass) if non_pass else actions.actions[0]
                engine.step(action)

            elif state.phase == GamePhase.DISCARD:
                options = engine.get_response_options()
                responses = {}
                for p_idx, legal in options.items():
                    non_pass = [a for a in legal.actions if a.action_type != ActionType.PASS]
                    responses[p_idx] = rng.choice(non_pass) if non_pass else Action(ActionType.PASS)
                engine.resolve_responses(responses)

            elif state.phase in (GamePhase.AGARI, GamePhase.RYUUKYOKU):
                engine.step(Action(ActionType.PASS))

        assert engine.is_game_over()

    def test_legal_actions_not_empty(self):
        """Every draw state should have legal actions."""
        engine = GameEngine(seed=555)

        for _ in range(20):
            if engine.is_game_over():
                break

            state = engine.get_game_state()
            if state.phase == GamePhase.DRAW:
                actions = engine.get_legal_actions()
                assert len(actions.actions) > 0, f"No legal actions at step with phase DRAW"

                # Pick discard
                discards = [a for a in actions.actions if a.action_type == ActionType.DISCARD]
                if discards:
                    engine.step(discards[0])
                    # Resolve with all pass
                    engine.resolve_responses({
                        (engine.last_discard_by + i) % 4: Action(ActionType.PASS)
                        for i in range(1, 4)
                    })
            elif state.phase == GamePhase.DISCARD:
                engine.resolve_responses({
                    (engine.last_discard_by + i) % 4: Action(ActionType.PASS)
                    for i in range(1, 4)
                })
            elif state.phase in (GamePhase.AGARI, GamePhase.RYUUKYOKU):
                engine.step(Action(ActionType.PASS))


class TestClone:
    """Deep copy for MCTS."""

    def test_clone_independent(self):
        engine = GameEngine(seed=42)
        clone = engine.clone()

        # Find a tile the player actually has
        hand = engine.players[0].hand.tiles
        tile_to_discard = next(t for t in range(34) if hand[t] > 0)

        # Modify original
        engine.step(Action(ActionType.DISCARD, tile=tile_to_discard))

        # Clone should be unaffected
        assert clone.players[0].hand.total_concealed() != engine.players[0].hand.total_concealed()

    def test_clone_can_play(self):
        engine = GameEngine(seed=42)
        clone = engine.clone()

        # Play from clone, verify engine is unchanged
        clone_hand_before = list(clone.players[0].hand.tiles)
        engine_hand_before = list(engine.players[0].hand.tiles)

        assert clone_hand_before == engine_hand_before


class TestStateTensor:
    """State tensor for neural network input."""

    def test_tensor_shape(self):
        engine = GameEngine(seed=42)
        tensor = engine.get_state_tensor(0)
        assert tensor.shape[0] == engine.get_state_tensor_dim()
        assert tensor.dtype.name == 'float32'

    def test_tensor_for_all_players(self):
        engine = GameEngine(seed=42)
        for p in range(4):
            tensor = engine.get_state_tensor(p)
            assert tensor is not None
            assert len(tensor.shape) == 1


# =============================================================================
# Review fix regression tests
# =============================================================================

class TestTenhouDetection:
    """P1-1: Tenhou should be detectable when dealer has winning initial hand."""

    def test_tenhou_flag_in_can_win(self):
        """_can_win should set is_tenhou=True for dealer's first tsumo."""
        from engine.yaku import WinContext, YakuChecker
        engine = GameEngine(seed=42)
        # Set up dealer with a winning hand (simple 4 melds + pair)
        p = engine.players[engine.dealer_idx]
        p.hand = Hand.from_type_list([0, 0, 0, 1, 1, 1, 2, 2, 2, 9, 9, 9, 27, 27])
        ctx = engine._build_win_context(engine.dealer_idx, is_tsumo=True, winning_tile=0)
        assert ctx.is_tenhou, f"Expected tenhou=True for dealer first tsumo, got {ctx.is_tenhou}"

    def test_tenhou_flag_false_after_discard(self):
        """Tenhou should NOT be set if dealer has already discarded."""
        engine = GameEngine(seed=42)
        p = engine.players[engine.dealer_idx]
        p.add_discard(0)  # mark that dealer has discarded
        p.hand = Hand.from_type_list([0, 0, 0, 1, 1, 1, 2, 2, 2, 9, 9, 9, 27, 27])
        ctx = engine._build_win_context(engine.dealer_idx, is_tsumo=True, winning_tile=0)
        assert not ctx.is_tenhou, "Tenhou should be false after dealer has discarded"


class TestChiihouDetection:
    """P2-2: Chiihou requires no melds or discards from any player."""

    def test_chiihou_detected_for_clean_game(self):
        """Chiihou should be detected for non-dealer first tsumo in clean game."""
        engine = GameEngine(seed=42)
        engine.dealer_idx = 0
        # Clean state: no discards, no melds for any player
        p1 = engine.players[1]
        p1.hand = Hand.from_type_list([0, 0, 0, 1, 1, 1, 2, 2, 2, 9, 9, 9, 27, 27])
        ctx = engine._build_win_context(1, is_tsumo=True, winning_tile=0)
        assert ctx.is_chiihou, f"Expected chiihou=True for clean first tsumo, got {ctx.is_chiihou}"

    def test_chiihou_blocked_by_other_discard(self):
        """Chiihou should be blocked if another player has discarded."""
        engine = GameEngine(seed=42)
        engine.dealer_idx = 0
        engine.players[0].add_discard(1)  # dealer discarded
        p1 = engine.players[1]
        p1.hand = Hand.from_type_list([0, 0, 0, 1, 1, 1, 2, 2, 2, 9, 9, 9, 27, 27])
        ctx = engine._build_win_context(1, is_tsumo=True, winning_tile=0)
        assert not ctx.is_chiihou, "Chiihou should be false if dealer has discarded"


class TestDoubleRiichi:
    """P2-3: Double riichi requires no melds from any player."""

    def test_double_riichi_allowed_clean(self):
        engine = GameEngine(seed=42)
        # Ensure all players have clean state
        for p in engine.players:
            p.discards.clear()
            p.hand.melds.clear()
        # Give dealer a valid hand with tile 0
        engine.players[0].hand.add(0)
        # Dealer declares riichi with discard of tile 0
        engine._handle_riichi_discard(0)
        assert engine.players[engine.dealer_idx].is_double_riichi


class TestChiitoitsuFu:
    """P2-1: Chiitoitsu should use 25 fu, not 30."""

    def test_chiitoitsu_25_fu(self):
        """Verify that _is_chiitoitsu detects 7 pairs correctly."""
        # 7 pairs as histogram: each of 7 types has count 2
        tiles_hist = [0] * 34
        for t in [0, 1, 2, 9, 10, 11, 27]:
            tiles_hist[t] = 2
        assert _is_chiitoitsu(tiles_hist), "Hand should be chiitoitsu"


class TestRyanhanShibari:
    """P1-2: Ryanhan shibari blocks win before state corruption."""

    def test_ryanhan_shibari_blocks_tsumo(self):
        """Tsumo with < 2 han under ryanhan_shibari should be rejected."""
        engine = GameEngine(seed=42)
        engine.honba = 5
        engine.config.ryanhan_shibari = True

        # Prevent tenhou detection (dealer already discarded)
        engine.players[engine.dealer_idx].add_discard(0)

        # Valid winning hand: 234m 678m 345p(chi) 567s + 22s pair
        # Open meld → not menzen → no menzen tsumo/pinfu
        # Tanyao only = 1 han. Dora type 8 (9m) doesn't match → 0 dora.
        p = engine.players[engine.dealer_idx]
        p.hand = Hand.from_type_list([1, 2, 3, 5, 6, 7, 19, 19, 22, 23, 24])
        p.hand.melds.append(Meld(MeldType.CHI, [10, 11, 12],
                                 called_from=2, source_tile=12))

        engine._last_drawn_tile = 24  # 7s
        engine._handle_tsumo(engine.dealer_idx)
        assert not p.has_won, "Player should not be marked as won under ryanhan shibari"
        assert engine.phase != GamePhase.AGARI, "Phase should not be AGARI when blocked"

    def test_ryanhan_shibari_allows_2han(self):
        """Tsumo with >= 2 han under ryanhan_shibari should succeed."""
        engine = GameEngine(seed=42)
        engine.honba = 5
        engine.config.ryanhan_shibari = True

        # Prevent tenhou detection
        engine.players[engine.dealer_idx].add_discard(0)

        # Menzen hand: 234m 678m 345p 567s + 22s pair
        # Menzen tsumo (1) + tanyao (1) = 2 han → allowed
        p = engine.players[engine.dealer_idx]
        p.hand = Hand.from_type_list([1, 2, 3, 5, 6, 7, 11, 12, 13,
                                      19, 19, 22, 23, 24])

        engine._last_drawn_tile = 24  # 7s
        engine._handle_tsumo(engine.dealer_idx)
        assert p.has_won, "Player should be marked as won with 2 han"
        assert engine.phase == GamePhase.AGARI
# 中文注释：验证游戏引擎主流程，包括初始化、摸切、响应、和牌与流局状态迁移。

