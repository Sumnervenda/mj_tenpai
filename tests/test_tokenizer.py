"""Tests for MahjongTokenizer."""
import pytest
from data.mjson_parser import MJSONGameTracker
from models.tokenizer import (
    MahjongTokenizer, TokenVocab, TokenType, TokenSequence,
)


def make_tracker() -> MJSONGameTracker:
    """Create a minimal MJSONGameTracker with known state for testing."""
    tracker = MJSONGameTracker()
    tracker.oya = 0
    tracker.bakaze = 27  # east
    tracker.kyoku = 1
    tracker.honba = 0
    tracker.kyotaku = 1
    tracker.scores = [25000, 24000, 26000, 23000]
    tracker.remaining_tiles = 42

    # Set initial hand: 123m 789p 123s EE (13 tiles)
    hand_types = [0, 1, 2, 15, 16, 17, 18, 19, 20, 27, 27, 31, 32]
    for t in hand_types:
        tracker.hands[0][t] += 1

    # Opponent hands (dummy)
    tracker.hands[1][4] = 4  # 4 copies of 5m
    tracker.hands[2][9] = 4  # 4 copies of 1p
    tracker.hands[3][27] = 4  # 4 copies of east

    # Set dora indicators
    tracker.dora_indicators = [0]  # 1m as dora indicator

    # Set discards for each player
    tracker.discards[0] = [3, 7, 15]  # P0 discards: 4m, 8m, 6p
    tracker.discards[1] = [5, 10, 28]  # P1 discards: 6m, 2p, south

    # Set melds
    # P0 has a chi
    tracker.melds[0].append(('chi', [9, 10, 11], 1))  # chi 123p from P1
    # P1 has a pon
    tracker.melds[1].append(('pon', [27, 27, 27], 2))  # pon east from P2

    # Set riichi
    tracker.is_riichi[2] = True

    return tracker


class TestTokenizerConstruction:
    """Test basic tokenizer construction and vocabulary."""

    def test_vocab_constants(self):
        assert TokenVocab.VOCAB_SIZE == 128
        assert TokenVocab.TILE_MIN == 0
        assert TokenVocab.TILE_MAX == 33
        assert TokenVocab.RED_5M == 34
        assert TokenVocab.ACTION_TSUMOGIRI == 41
        assert TokenVocab.ROUND_EAST == 54

    def test_token_type_constants(self):
        assert TokenType.HAND == 0
        assert TokenType.DISCARD == 2
        assert TokenType.RIICHI == 4
        assert TokenType.GLOBAL == 5
        assert TokenType.NUM_TYPES == 6

    def test_token_sequence_basics(self):
        seq = TokenSequence()
        assert len(seq) == 0

        seq.add(1, TokenType.HAND)
        seq.add(5, TokenType.DORA, behavior_id=2)
        assert len(seq) == 2
        assert seq.token_ids == [1, 5]
        assert seq.token_types == [TokenType.HAND, TokenType.DORA]
        assert seq.behavior_ids == [0, 2]


class TestTokenizerHand:
    """Test hand token generation."""

    def test_hand_token_count(self):
        tracker = make_tracker()
        tokenizer = MahjongTokenizer()
        seq = tokenizer.tokenize_game_snapshot(tracker, player_idx=0)

        # Count hand tokens
        hand_tokens = [t for t in seq.tokens if t.token_type == TokenType.HAND]
        # 13 tiles in hand (tracker hands[0] has 13 tiles)
        assert len(hand_tokens) == 13

    def test_hand_token_ids(self):
        tracker = make_tracker()
        tokenizer = MahjongTokenizer()
        seq = tokenizer.tokenize_game_snapshot(tracker, player_idx=0)

        hand_ids = [t.token_id for t in seq.tokens
                    if t.token_type == TokenType.HAND]
        # Expected: 0,1,2 (123m), 15,16,17 (789p), 18,19,20 (123s), 27,27 (EE), 31 (P), 32 (F)
        expected = [0, 1, 2, 15, 16, 17, 18, 19, 20, 27, 27, 31, 32]
        assert sorted(hand_ids) == sorted(expected), f"{sorted(hand_ids)} != {sorted(expected)}"


class TestTokenizerDora:
    """Test dora indicator tokens."""

    def test_dora_tokens(self):
        tracker = make_tracker()
        tokenizer = MahjongTokenizer()
        seq = tokenizer.tokenize_game_snapshot(tracker, player_idx=0)

        dora_tokens = [t for t in seq.tokens if t.token_type == TokenType.DORA]
        assert len(dora_tokens) == 1
        assert dora_tokens[0].token_id == 0  # 1m dora indicator


class TestTokenizerDiscard:
    """Test discard sequence tokens."""

    def test_discard_tokens_exist(self):
        tracker = make_tracker()
        tokenizer = MahjongTokenizer()
        seq = tokenizer.tokenize_game_snapshot(tracker, player_idx=0)

        discard_tokens = [t for t in seq.tokens
                          if t.token_type == TokenType.DISCARD]
        # P0: 3 discards, P1: 3 discards, P2: 0, P3: 0 = 6 total
        assert len(discard_tokens) == 6

    def test_discard_behavior_ids(self):
        tracker = make_tracker()
        tokenizer = MahjongTokenizer()
        seq = tokenizer.tokenize_game_snapshot(tracker, player_idx=0)

        discard_tokens = [t for t in seq.tokens
                          if t.token_type == TokenType.DISCARD]
        for t in discard_tokens:
            assert t.behavior_id > 0, "Discard tokens should have behavior_id"
            player = (t.behavior_id >> 8) & 0xFF
            assert player in (0, 1)


class TestTokenizerMeld:
    """Test meld tokens."""

    def test_meld_tokens(self):
        tracker = make_tracker()
        tokenizer = MahjongTokenizer()
        seq = tokenizer.tokenize_game_snapshot(tracker, player_idx=0)

        meld_tokens = [t for t in seq.tokens
                       if t.token_type == TokenType.MELD]
        # P0 has 1 chi (3 tiles), P1 has 1 pon (3 tiles) = 6 tokens
        assert len(meld_tokens) == 6


class TestTokenizerRiichi:
    """Test riichi event tokens."""

    def test_riichi_tokens(self):
        tracker = make_tracker()
        tokenizer = MahjongTokenizer()
        seq = tokenizer.tokenize_game_snapshot(tracker, player_idx=0)

        riichi_tokens = [t for t in seq.tokens
                         if t.token_type == TokenType.RIICHI]
        assert len(riichi_tokens) == 1
        assert riichi_tokens[0].token_id == TokenVocab.ACTION_RIICHI


class TestTokenizerGlobal:
    """Test global state tokens."""

    def test_global_tokens(self):
        tracker = make_tracker()
        tokenizer = MahjongTokenizer()
        seq = tokenizer.tokenize_game_snapshot(tracker, player_idx=0)

        global_tokens = [t for t in seq.tokens
                         if t.token_type == TokenType.GLOBAL]
        # Round wind + seat wind + remaining + honba + riichi stick + 3 diffs = 8
        assert len(global_tokens) == 8


class TestTokenizerSequence:
    """Test overall token sequence."""

    def test_total_length(self):
        tracker = make_tracker()
        tokenizer = MahjongTokenizer()
        seq = tokenizer.tokenize_game_snapshot(tracker, player_idx=0)

        # hand(13) + dora(1) + discard(6) + meld(6) + riichi(1) + global(8) = 35
        assert len(seq) == 35

    def test_max_sequence_length(self):
        """Test that sequence is capped at max_len."""
        tracker = make_tracker()
        tokenizer = MahjongTokenizer(max_sequence_length=10)
        seq = tokenizer.tokenize_game_snapshot(tracker, player_idx=0)
        assert len(seq) == 10

    def test_token_ids_length_match_types(self):
        tracker = make_tracker()
        tokenizer = MahjongTokenizer()
        seq = tokenizer.tokenize_game_snapshot(tracker, player_idx=0)

        assert len(seq.token_ids) == len(seq.tokens)
        assert len(seq.token_types) == len(seq.tokens)
        assert len(seq.behavior_ids) == len(seq.tokens)

    def test_all_token_ids_in_vocab_range(self):
        tracker = make_tracker()
        tokenizer = MahjongTokenizer()
        seq = tokenizer.tokenize_game_snapshot(tracker, player_idx=0)

        for tid in seq.token_ids:
            assert 0 <= tid < TokenVocab.VOCAB_SIZE, \
                f"token_id {tid} out of range"

    def test_all_token_types_valid(self):
        tracker = make_tracker()
        tokenizer = MahjongTokenizer()
        seq = tokenizer.tokenize_game_snapshot(tracker, player_idx=0)

        for tt in seq.token_types:
            assert 0 <= tt < TokenType.NUM_TYPES

    def test_different_player_perspective(self):
        """Test tokenization from different player's perspective."""
        tracker = make_tracker()
        tokenizer = MahjongTokenizer()
        seq_p0 = tokenizer.tokenize_game_snapshot(tracker, player_idx=0)
        seq_p1 = tokenizer.tokenize_game_snapshot(tracker, player_idx=1)

        # Different player perspectives should produce different sequences
        assert seq_p0.token_ids != seq_p1.token_ids
# 中文注释：验证 MahjongTokenizer 词表常量、手牌/宝牌/舍牌/副露/立直/全局等各种 Token 的生成正确性。

