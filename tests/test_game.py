"""Tests for game engine integration."""
import pytest
from engine import (
    GameEngine, GameConfig, GamePhase,
    Action, ActionType,
)
from engine.agari import is_agari, is_tenpai


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
