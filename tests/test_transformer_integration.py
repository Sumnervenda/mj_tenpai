"""Integration tests: real Tokenizer output → TransformerPolicyValueNet forward.

These tests close the gap between unit tests (which use random low-range IDs)
and real training/inference, where Tokenizer produces IDs that must fit within
the embedding tables.
"""

import pytest
import torch

from data.mjson_parser import MJSONGameTracker
from engine import GameEngine
from models.tokenizer import MahjongTokenizer, TokenVocab, TokenType
from models.transformer_policy_value import TransformerPolicyValueNet


def _make_extreme_tracker() -> MJSONGameTracker:
    """Build a tracker with values that stress-test the full vocab range.

    remaining_tiles=70 → token_id up to REMAINING_BASE+70=133
    wide score diffs → DIFF_BASE up to DIFF_BASE+60=155
    multiple meld types, riichi, discards from all 4 players.
    """
    tracker = MJSONGameTracker()
    tracker.oya = 0
    tracker.bakaze = 27   # east
    tracker.kyoku = 4
    tracker.honba = 3
    tracker.kyotaku = 2
    tracker.scores = [45000, 5000, 30000, 20000]  # +/-40k range
    tracker.remaining_tiles = 70  # max possible for REMAINING_BASE

    # Rich hand: mixed tiles including 1m, 5m, 9m, honors
    hand_types = [0, 0, 1, 2, 3, 4, 8, 16, 17, 25, 26, 27, 31, 33]
    for t in hand_types:
        tracker.hands[0][t] += 1

    # Opponent hands
    tracker.hands[1][4] = 3
    tracker.hands[1][13] = 1
    tracker.hands[2][22] = 3
    tracker.hands[3][30] = 2

    # Dora indicators
    tracker.dora_indicators = [8, 17, 26]  # 9m, 9p, 9s

    # Discards from all 4 players
    tracker.discards[0] = [3, 7, 15, 24, 30]
    tracker.discards[1] = [5, 10, 20, 28]
    tracker.discards[2] = [1, 11, 21]
    tracker.discards[3] = [2, 12, 19, 29]

    # Melds: chi, pon, ankan, kakan
    tracker.melds[0].append(('chi', [9, 10, 11], 1))
    tracker.melds[1].append(('pon', [27, 27, 27], 2))
    tracker.melds[2].append(('ankan', [18, 18, 18, 18], -1))
    tracker.melds[3].append(('kakan', [0, 0, 0, 0], -1))

    # Riichi: P2 in riichi, P0 in double riichi
    tracker.is_riichi[2] = True
    tracker.is_double_riichi[0] = True

    return tracker


class TestRealTokenizerToTransformer:
    """End-to-end: MJSONGameTracker → Tokenizer → Transformer forward."""

    def test_max_token_id_within_vocab(self):
        """Real tokenizer output must have every token_id < VOCAB_SIZE."""
        tracker = _make_extreme_tracker()
        tokenizer = MahjongTokenizer()
        seq = tokenizer.tokenize_game_snapshot(tracker, player_idx=0)

        max_id = max(t.token_id for t in seq.tokens)
        assert max_id < TokenVocab.VOCAB_SIZE, \
            f"max_token_id={max_id} >= VOCAB_SIZE={TokenVocab.VOCAB_SIZE}"

        # Also check known high-range values
        remaining_token = next(
            t for t in seq.tokens
            if t.token_id == TokenVocab.REMAINING
        )
        assert remaining_token.behavior_id == 70, \
            f"Expected remaining behavior_id=70, got {remaining_token.behavior_id}"

    def test_max_behavior_id_within_embedding(self):
        """All behavior_ids must fit within num_behavior_types."""
        tracker = _make_extreme_tracker()
        tokenizer = MahjongTokenizer()
        seq = tokenizer.tokenize_game_snapshot(tracker, player_idx=0)

        max_bid = max(
            (t.behavior_id for t in seq.tokens if t.behavior_id > 0),
            default=0,
        )
        num_behavior_types = TokenVocab.MAX_BEHAVIOR_ID
        assert max_bid < num_behavior_types, \
            f"max_behavior_id={max_bid} >= num_behavior_types={num_behavior_types}"

    def test_pad_token_not_in_real_sequence(self):
        """PAD=0 should not appear in non-padding token positions."""
        tracker = _make_extreme_tracker()
        tokenizer = MahjongTokenizer()
        seq = tokenizer.tokenize_game_snapshot(tracker, player_idx=0)

        for t in seq.tokens:
            assert t.token_id != TokenVocab.PAD, \
                f"PAD token (id=0) found in real token sequence: type={t.token_type}"

    def test_transformer_forward_does_not_crash(self):
        """Full pipeline: tokenize → tensor → Transformer forward → valid outputs."""
        tracker = _make_extreme_tracker()
        tokenizer = MahjongTokenizer()
        seq = tokenizer.tokenize_game_snapshot(tracker, player_idx=0)

        # Convert to batched tensors
        S = len(seq)
        token_ids = torch.tensor([seq.token_ids], dtype=torch.long)
        token_types = torch.tensor([seq.token_types], dtype=torch.long)
        behavior_ids = torch.tensor([seq.behavior_ids], dtype=torch.long)
        attention_mask = torch.zeros(1, S, dtype=torch.bool)  # no padding
        action_mask = torch.ones(1, 77, dtype=torch.float32)

        model = TransformerPolicyValueNet()
        outputs = model(token_ids, token_types, behavior_ids,
                       attention_mask, action_mask)

        B = 1
        assert outputs['policy_logits'].shape == (B, 77)
        assert outputs['value'].shape == (B, 1)
        assert outputs['shanten'].shape == (B, 7)
        assert outputs['efficiency'].shape == (B, 3)
        assert outputs['danger'].shape == (B, 34)
        assert outputs['score_value'].shape == (B, 1)

    def test_transformer_forward_multiple_players(self):
        """All 4 player perspectives tokenize and forward without errors."""
        tracker = _make_extreme_tracker()
        tokenizer = MahjongTokenizer()
        model = TransformerPolicyValueNet()

        for p in range(4):
            seq = tokenizer.tokenize_game_snapshot(tracker, player_idx=p)
            S = len(seq)

            token_ids = torch.tensor([seq.token_ids], dtype=torch.long)
            token_types = torch.tensor([seq.token_types], dtype=torch.long)
            behavior_ids = torch.tensor([seq.behavior_ids], dtype=torch.long)
            attention_mask = torch.zeros(1, S, dtype=torch.bool)
            action_mask = torch.ones(1, 77, dtype=torch.float32)

            outputs = model(token_ids, token_types, behavior_ids,
                          attention_mask, action_mask)
            assert outputs['policy_logits'].shape == (1, 77), \
                f"Player {p} forward failed"

    def test_transformer_with_padded_batch(self):
        """Batched sequences with different lengths and padding."""
        tracker = _make_extreme_tracker()
        tokenizer = MahjongTokenizer()

        # Create two sequences of different lengths
        seq0 = tokenizer.tokenize_game_snapshot(tracker, player_idx=0)
        seq1 = tokenizer.tokenize_game_snapshot(tracker, player_idx=1)

        # Pad to max length
        max_len = max(len(seq0), len(seq1))
        B = 2

        token_ids = torch.zeros(B, max_len, dtype=torch.long)  # PAD=0
        token_types = torch.zeros(B, max_len, dtype=torch.long)
        behavior_ids = torch.zeros(B, max_len, dtype=torch.long)
        attention_mask = torch.ones(B, max_len, dtype=torch.bool)  # all padding initially

        for i, seq in enumerate([seq0, seq1]):
            L = len(seq)
            token_ids[i, :L] = torch.tensor(seq.token_ids, dtype=torch.long)
            token_types[i, :L] = torch.tensor(seq.token_types, dtype=torch.long)
            behavior_ids[i, :L] = torch.tensor(seq.behavior_ids, dtype=torch.long)
            attention_mask[i, :L] = False  # real tokens, not padding

        action_mask = torch.ones(B, 77, dtype=torch.float32)

        model = TransformerPolicyValueNet()
        outputs = model(token_ids, token_types, behavior_ids,
                       attention_mask, action_mask)

        assert outputs['policy_logits'].shape == (B, 77)
        assert outputs['value'].shape == (B, 1)

    def test_token_id_zero_only_for_padding(self):
        """After padding a batch, id=0 appears only in padded positions."""
        tracker = _make_extreme_tracker()
        tokenizer = MahjongTokenizer()
        seq = tokenizer.tokenize_game_snapshot(tracker, player_idx=0)

        # Verify original sequence has no PAD
        assert 0 not in seq.token_ids

        # Build padded batch
        max_len = len(seq) + 5  # force padding
        token_ids = torch.zeros(1, max_len, dtype=torch.long)
        for i, tid in enumerate(seq.token_ids):
            token_ids[0, i] = tid

        # First S positions should be non-zero; last 5 should be 0 (PAD)
        S = len(seq)
        assert (token_ids[0, :S] != 0).all(), "Real tokens contain PAD"
        assert (token_ids[0, S:] == 0).all(), "Padding positions should be PAD=0"


class TestTokenizeEngineState:
    """Engine state → TokenSequence → Transformer forward."""

    def test_engine_state_produces_valid_tokens(self):
        """tokenize_engine_state() should produce a non-empty TokenSequence."""
        engine = GameEngine(seed=42)
        tokenizer = MahjongTokenizer()
        seq = tokenizer.tokenize_engine_state(engine, player_idx=0)

        assert len(seq) > 0, "Engine state should produce tokens"
        assert len(seq.token_ids) == len(seq.tokens)
        assert len(seq.token_types) == len(seq.tokens)

    def test_engine_state_token_ids_in_vocab_range(self):
        """All token_ids from engine state must be < VOCAB_SIZE."""
        engine = GameEngine(seed=42)
        tokenizer = MahjongTokenizer()
        seq = tokenizer.tokenize_engine_state(engine, player_idx=0)

        max_id = max(t.token_id for t in seq.tokens)
        assert max_id < TokenVocab.VOCAB_SIZE, \
            f"Engine max_token_id={max_id} >= VOCAB_SIZE={TokenVocab.VOCAB_SIZE}"

    def test_engine_state_no_pad_in_real_tokens(self):
        """PAD=0 should not appear in engine state tokens."""
        engine = GameEngine(seed=42)
        tokenizer = MahjongTokenizer()
        seq = tokenizer.tokenize_engine_state(engine, player_idx=0)

        for t in seq.tokens:
            assert t.token_id != TokenVocab.PAD, \
                f"PAD token found in engine state: type={t.token_type}"

    def test_engine_state_to_transformer_forward(self):
        """Engine state → tokenize → Transformer forward should succeed."""
        engine = GameEngine(seed=42)
        tokenizer = MahjongTokenizer()
        seq = tokenizer.tokenize_engine_state(engine, player_idx=0)

        S = len(seq)
        token_ids = torch.tensor([seq.token_ids], dtype=torch.long)
        token_types = torch.tensor([seq.token_types], dtype=torch.long)
        behavior_ids = torch.tensor([seq.behavior_ids], dtype=torch.long)
        attention_mask = torch.zeros(1, S, dtype=torch.bool)
        action_mask = torch.ones(1, 77, dtype=torch.float32)

        model = TransformerPolicyValueNet()
        outputs = model(token_ids, token_types, behavior_ids,
                       attention_mask, action_mask)

        assert outputs['policy_logits'].shape == (1, 77)
        assert outputs['value'].shape == (1, 1)

    def test_engine_state_all_four_players(self):
        """All 4 player perspectives from engine should tokenize correctly."""
        engine = GameEngine(seed=42)
        tokenizer = MahjongTokenizer()

        for p in range(4):
            seq = tokenizer.tokenize_engine_state(engine, player_idx=p)
            max_id = max(t.token_id for t in seq.tokens)
            assert max_id < TokenVocab.VOCAB_SIZE, \
                f"Player {p}: max_token_id={max_id} out of range"
