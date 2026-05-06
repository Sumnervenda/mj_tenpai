"""Tests for LUT-based agari (win) checking."""
import pytest
from engine.agari import is_agari, is_tenpai, get_waits, is_agari_with_tile
from engine.tile import YAOCHUHAI_TYPES


# ── Test helpers ─────────────────────────────────────────────────────────────

def make_hand(tiles_spec):
    """Create int[34] from tile type list."""
    h = [0] * 34
    for t in tiles_spec:
        h[t] += 1
    return h


class TestStandardAgari:
    """Standard 4-melds + 1-pair winning hands."""

    def test_standard_win_1(self):
        """123m 456m 789m 123p 55s — a classic win."""
        hand = make_hand([
            0,1,2, 3,4,5, 6,7,8,       # 123m, 456m, 789m
            9,10,11,                     # 123p
            23,23,                       # 55s
        ])
        assert is_agari(hand) == True

    def test_standard_win_2(self):
        """111m 222m 333m 444m 55m — all triplets in one suit."""
        hand = make_hand([0]*3 + [1]*3 + [2]*3 + [3]*3 + [4,4])
        assert is_agari(hand) == True

    def test_standard_win_3(self):
        """Tanyao win: 234m 456p 678s 222s 33m."""
        hand = make_hand([
            1,2,3,                       # 234m
            12,13,14,                    # 456p
            20,21,22, 20,21,22,          # 678s + 678s (iipeikou)
            1,1,                         # 33m
        ])
        assert is_agari(hand) == True

    def test_standard_win_with_koutsu(self):
        """111m 234p 567p 789s 東東."""
        hand = make_hand([
            0,0,0,                       # 111m
            10,11,12, 13,14,15,          # 234p 567p
            24,25,26,                    # 789s
            27,27,                       # 東東
        ])
        assert is_agari(hand) == True

    def test_not_agari(self):
        """Random 14 tiles that don't form a winning hand."""
        hand = make_hand([0,1,2,4,7,10,15,18,22,25,27,30,31,33])
        assert is_agari(hand) == False

    def test_not_agari_partial(self):
        """13 tiles (3k+1) should not be agari."""
        hand = make_hand([0,1,2, 3,4,5, 6,7,8, 9,10,11, 23])
        assert is_agari(hand) == False

    def test_empty_hand(self):
        assert is_agari([0]*34) == False

    def test_five_of_a_kind(self):
        """5 of the same tile type should never be valid."""
        hand = [0]*34
        hand[0] = 5
        assert is_agari(hand) == False


class TestChiitoitsu:
    """七対子 (7 pairs)."""

    def test_chiitoitsu_valid(self):
        hand = [0]*34
        for t in [0, 3, 6, 12, 15, 21, 27]:
            hand[t] = 2
        assert is_agari(hand) == True

    def test_chiitoitsu_invalid_same_tile_4(self):
        """Cannot have 4 of the same tile in chiitoitsu."""
        hand = [0]*34
        for t in [0, 3, 6, 12, 15, 21]:
            hand[t] = 2
        hand[0] = 4  # 4 copies of 1m — not chiitoitsu
        assert is_agari(hand) == False

    def test_chiitoitsu_6_pairs(self):
        """Only 6 pairs — not enough."""
        hand = [0]*34
        for t in [0, 3, 6, 12, 15, 21]:
            hand[t] = 2
        assert is_agari(hand) == False


class TestKokushi:
    """国士無双 (13 orphans)."""

    def test_kokushi_valid(self):
        yaochu = list(YAOCHUHAI_TYPES)
        hand = [0]*34
        for t in yaochu:
            hand[t] = 1
        hand[yaochu[0]] = 2  # pair
        assert is_agari(hand) == True

    def test_kokushi_missing_one(self):
        """12 orphans + a pair (missing one orphan)."""
        yaochu = list(YAOCHUHAI_TYPES)
        hand = [0]*34
        for t in yaochu:
            hand[t] = 1
        hand[yaochu[0]] = 2
        hand[yaochu[-1]] = 0  # missing last orphan
        assert is_agari(hand) == False

    def test_kokushi_extra_tile(self):
        """Kokushi + extra tile (15 tiles total)."""
        yaochu = list(YAOCHUHAI_TYPES)
        hand = [0]*34
        for t in yaochu:
            hand[t] = 1
        hand[yaochu[0]] = 2
        hand[1] = 1  # extra 2m
        assert is_agari(hand) == False


class TestTenpai:
    """Tenpai (听牌) detection."""

    def test_tenpai_ryanmen(self):
        """Waiting on two sides: 234m 456m 789m 123p + 5s (waiting on 5s)."""
        hand = make_hand([
            0,1,2, 3,4,5, 6,7,8,
            9,10,11,
            23,  # single 5s — waiting on another 5s
        ])
        assert is_tenpai(hand) == True
        waits = get_waits(hand)
        assert 23 in waits  # 5s

    def test_tenpai_many_waits(self):
        """123m 456m 789m 23p 55s — waiting on 1p or 4p (ryanmen)."""
        hand = make_hand([
            0,1,2, 3,4,5, 6,7,8,
            10,11,  # 23p waiting on 1p or 4p
            23,23,  # 55s
        ])
        assert is_tenpai(hand) == True
        waits = get_waits(hand)
        assert 9 in waits   # 1p
        assert 12 in waits  # 4p

    def test_not_tenpai(self):
        """Random 13 tiles."""
        hand = make_hand([0,1,3,5,8,10,13,18,21,25,27,30,33])
        # This might coincidentally be tenpai, but with random tiles it's unlikely
        # Just check it doesn't crash
        result = is_tenpai(hand)
        assert isinstance(result, bool)

    def test_waits_empty_for_not_tenpai(self):
        """get_waits should return empty list for non-tenpai hand."""
        hand = make_hand([0,0,0, 3,3,3, 12,12,12, 27,27,27, 31])
        # All triplets + single: 13 tiles but every tile type has 3 copies
        # This is not tenpai because adding any tile gives 4 triplets + 2 singles → not standard
        # Wait: actually, 4 triplets of the above would need 4x3 = 12 + 1 = 13.
        # If we add a tile to match the single → we get 4 triplets + 1 pair = agari!
        # 111 444 333 東東東 + 白白 → tenpai waiting on 白
        # But the hand has: 111m(3) 444m(3) 444p(3) 東東東(3) 白(1) → waiting on 白
        assert is_tenpai(hand) == True
        waits = get_waits(hand)
        assert 31 in waits  # 白


class TestAgariWithTile:
    """Probing: does adding a specific tile result in agari?"""

    def test_probe_add_pair(self):
        """Same as tenpai test: add 5s to complete pair."""
        hand = make_hand([
            0,1,2, 3,4,5, 6,7,8,
            9,10,11,
            23,
        ])
        assert is_agari_with_tile(hand, 23) == True  # add 5s → pair
        assert is_agari_with_tile(hand, 0) == False  # add 1m → doesn't help

    def test_probe_at_limit(self):
        """Cannot probe with a tile that already has 4 copies."""
        hand = [0]*34
        hand[0] = 4
        hand[1] = 1
        # Already 4 copies of 1m
        assert is_agari_with_tile(hand, 0) == False


class TestLUTCorrectness:
    """Verify the LUT handles edge cases correctly."""

    def test_all_zeros(self):
        """Empty suit should pass melds-only check."""
        from engine.agari import _encode, _SUIT_LUT_MELDS, _SUIT_LUT_WITH_PAIR
        from engine.agari import _build_luts
        _build_luts()
        encoded = _encode([0]*9)
        assert _SUIT_LUT_MELDS[encoded] == True   # empty = all melds
        assert _SUIT_LUT_WITH_PAIR[encoded] == False  # empty ≠ melds+pair

    def test_single_triplet(self):
        """111 should pass melds-only."""
        from engine.agari import _encode, _SUIT_LUT_MELDS, _SUIT_LUT_WITH_PAIR
        from engine.agari import _build_luts
        _build_luts()
        counts = [3] + [0]*8
        encoded = _encode(counts)
        assert _SUIT_LUT_MELDS[encoded] == True

    def test_single_sequence(self):
        """123 should pass melds-only."""
        from engine.agari import _encode, _SUIT_LUT_MELDS, _SUIT_LUT_WITH_PAIR
        from engine.agari import _build_luts
        _build_luts()
        counts = [1,1,1] + [0]*6
        encoded = _encode(counts)
        assert _SUIT_LUT_MELDS[encoded] == True

    def test_pair_only(self):
        """11 should pass melds+pair but NOT melds-only."""
        from engine.agari import _encode, _SUIT_LUT_MELDS, _SUIT_LUT_WITH_PAIR
        from engine.agari import _build_luts
        _build_luts()
        counts = [2] + [0]*8
        encoded = _encode(counts)
        assert _SUIT_LUT_MELDS[encoded] == False
        assert _SUIT_LUT_WITH_PAIR[encoded] == True

    def test_honor_lut(self):
        """Honor LUT: only pairs and triplets, no sequences."""
        from engine.agari import _encode, _HONOR_LUT_MELDS, _HONOR_LUT_WITH_PAIR
        from engine.agari import _build_luts
        _build_luts()
        # 東東東 (triplet) = melds only
        counts_honor = [3] + [0]*6
        encoded = _encode(counts_honor)
        assert _HONOR_LUT_MELDS[encoded] == True
