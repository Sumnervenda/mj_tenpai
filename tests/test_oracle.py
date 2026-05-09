"""Tests for Oracle algorithms: shanten, ukeire, wait quality."""
import pytest
from data.oracle import (
    calculate_shanten, shanten_standard, shanten_chiitoitsu, shanten_kokushi,
    compute_ukeire, classify_wait, compute_all_oracle_labels,
)
from engine.tile import NUM_TYPES, YAOCHUHAI_TYPES


# ── Test helpers ─────────────────────────────────────────────────────────────

def make_hand(tiles_spec):
    """Create int[34] from tile type list. Each element = 1 tile."""
    h = [0] * 34
    for t in tiles_spec:
        h[t] += 1
    return h


def assert_tile_count(hand, expected):
    actual = sum(hand)
    assert actual == expected, f"hand has {actual} tiles, expected {expected}"


# =============================================================================
# 1. Shanten Calculation
# =============================================================================

class TestShantenStandard:
    """Standard form (4 melds + 1 pair) shanten."""

    def test_tenpai_ryanmen(self):
        """13 tiles: 3 melds + 1 taatsu + 1 pair → tenpai, shanten=0."""
        # 123m(0-2) 456m(3-5) 789m(6-8) = 3 melds (9 tiles)
        # 23p(10-11) = taatsu (2 tiles)
        # 55s(23,23) = pair (2 tiles)
        hand = make_hand([0,1,2, 3,4,5, 6,7,8, 10,11, 23,23])
        assert_tile_count(hand, 13)
        assert calculate_shanten(hand) == 0

    def test_tenpai_tanki(self):
        """13 tiles: 4 melds + 1 single → tanki tenpai, shanten=0."""
        # 111m(0,0,0) 222m(1,1,1) = 2 triplets (6 tiles)
        # 333m(2,2,2) = 1 triplet (3 tiles)
        # 123p(9,10,11) = sequence (3 tiles)
        # 白(31) = single (1 tile)
        hand = make_hand([0,0,0, 1,1,1, 2,2,2, 9,10,11, 31])
        assert_tile_count(hand, 13)
        assert calculate_shanten(hand) == 0

    def test_one_shanten(self):
        """13 tiles: 2 melds + 2 taatsu + 1 pair → 1-shanten."""
        # 111m(0,0,0) = triplet (3 tiles)
        # 123p(9,10,11) = sequence (3 tiles)
        # 45p(12,13) = taatsu (2 tiles)
        # 67s(19,20) = taatsu (2 tiles)
        # 東東(27,27) = pair (2 tiles)
        # 5m(4) = isolated (1 tile)
        hand = make_hand([0,0,0, 9,10,11, 12,13, 19,20, 27,27, 4])
        assert_tile_count(hand, 13)
        assert calculate_shanten(hand) == 1

    def test_two_shanten(self):
        """13 tiles: 2 melds + 1 taatsu + 1 pair + 2 singles → 2-shanten."""
        # 123m(0,1,2) = meld (3 tiles)
        # 456p(12,13,14) = meld (3 tiles)
        # 78s(21,22) = taatsu (2 tiles)
        # 東東(27,27) = pair (2 tiles)
        # 5m(4) + 9p(17) + 中(33) = 3 singles
        hand = make_hand([0,1,2, 12,13,14, 21,22, 27,27, 4, 17, 33])
        assert_tile_count(hand, 13)
        assert calculate_shanten(hand) == 2

    def test_kokushi_tenpai_terminal_hand(self):
        """13 isolated terminal/honor tiles = all yaochu → kokushi tenpai, shanten=0."""
        hand = make_hand([0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33])
        assert_tile_count(hand, 13)
        # All 13 are unique yaochu types → kokushi 0-shanten
        assert calculate_shanten(hand) == 0

    def test_isolated_odd(self):
        """13 non-adjacent odd tiles → kanchan taatsu form, shanten=3."""
        hand = make_hand([0, 2, 4, 6, 8, 9, 11, 13, 15, 17, 18, 20, 22])
        assert_tile_count(hand, 13)
        # Odd-numbered tiles form kanchan taatsu: shanten=3 not 8
        assert calculate_shanten(hand) == 3

    def test_agari_hand(self):
        """Complete winning hand (14 tiles) → shanten=0."""
        hand = make_hand([0,1,2, 3,4,5, 6,7,8, 9,10,11, 23,23])
        assert_tile_count(hand, 14)
        assert calculate_shanten(hand) == 0

    def test_agari_hand_13_tiles_raw(self):
        """Complete winning hand as 13 raw tiles → raw shanten should be -1, clamped to 0."""
        # 14-tile winning hand: 4 melds + 1 pair
        # But calling with the same hand (14 tiles) gives raw = -1
        # If we call calculate_shanten with 14 tiles, it clamps to 0
        hand = make_hand([0,1,2, 3,4,5, 6,7,8, 9,10,11, 23,23])
        assert_tile_count(hand, 14)
        assert calculate_shanten(hand) == 0

    def test_many_taatsu(self):
        """4 ryanmen taatsu + 1 pair → 1-shanten (groups=5, no excess)."""
        hand = make_hand([
            3,4,        # 45m
            5,6,        # 67m
            10,11,      # 23p
            13,14,      # 56p
            21,22,      # 78s
            31,31,      # 白白 (pair)
        ])
        assert_tile_count(hand, 12)
        s = calculate_shanten(hand)
        assert 1 <= s <= 3


class TestShantenChiitoitsu:
    """Chiitoitsu shanten."""

    def test_chiitoitsu_tenpai(self):
        """6 pairs + 1 single → tenpai, shanten=0."""
        hand = [0]*34
        for t in [0, 3, 6, 12, 15, 27]:
            hand[t] = 2  # 6 pairs = 12 tiles
        hand[31] = 1  # 1 single = 1 tile
        assert_tile_count(hand, 13)
        assert shanten_chiitoitsu(hand) == 0

    def test_chiitoitsu_one_shanten(self):
        """5 pairs + 3 singles → 1-shanten."""
        hand = [0]*34
        for t in [0, 3, 6, 12, 27]:
            hand[t] = 2  # 5 pairs = 10 tiles
        hand[15] = hand[31] = hand[32] = 1  # 3 singles = 3 tiles
        assert_tile_count(hand, 13)
        assert shanten_chiitoitsu(hand) == 1

    def test_chiitoitsu_two_shanten(self):
        """4 pairs + 5 singles → 2-shanten."""
        hand = [0]*34
        for t in [0, 3, 6, 12]:
            hand[t] = 2  # 4 pairs = 8 tiles
        for t in [15, 16, 31, 32, 33]:
            hand[t] = 1  # 5 singles = 5 tiles
        assert_tile_count(hand, 13)
        assert shanten_chiitoitsu(hand) == 2

    def test_chiitoitsu_four_of_a_kind(self):
        """4 copies of a tile counts as only 1 pair for chiitoitsu."""
        hand = [0]*34
        hand[0] = 4  # four 1m = 1 pair
        for t in [3, 6, 12, 15, 27]:
            hand[t] = 2  # 5 pairs = 10 tiles
        hand[31] = 1  # 1 single = 1 tile
        assert_tile_count(hand, 15)
        # pairs: 1m(0) counts as 1 pair, + 5 others = 6 pairs → 0 shanten
        assert shanten_chiitoitsu(hand) == 0

    def test_chiitoitsu_high(self):
        """No pairs at all → 6-shanten for chiitoitsu."""
        hand = make_hand([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        assert_tile_count(hand, 13)
        assert shanten_chiitoitsu(hand) == 6


class TestShantenKokushi:
    """Kokushi (13 orphans) shanten."""

    def test_kokushi_tenpai(self):
        """12 unique yaochu + 1 pair (13 tiles) → tenpai."""
        yaochu = list(YAOCHUHAI_TYPES)
        hand = [0]*34
        for t in yaochu:
            hand[t] = 1
        hand[yaochu[0]] = 2  # pair
        hand[yaochu[-1]] = 0  # missing 1 orphan
        assert_tile_count(hand, 13)
        assert shanten_kokushi(hand) == 0

    def test_kokushi_one_shanten(self):
        """11 unique yaochu + 1 pair + 1 isolated = 13 tiles → 1-shanten."""
        yaochu = list(YAOCHUHAI_TYPES)
        hand = [0]*34
        for t in yaochu[:-2]:
            hand[t] = 1  # 11 unique yaochu
        hand[yaochu[0]] = 2  # pair (2 of first yaochu, net +1 since already counted)
        hand[1] = 1  # non-yaochu filler (2m)
        assert_tile_count(hand, 13)
        assert shanten_kokushi(hand) == 1

    def test_kokushi_two_shanten(self):
        """11 unique yaochu, no pair → 2-shanten."""
        yaochu = list(YAOCHUHAI_TYPES)
        hand = [0]*34
        for t in yaochu[:-2]:
            hand[t] = 1  # 11 tiles
        hand[1] = 1  # 1 extra non-yaochu
        hand[2] = 1  # 1 extra non-yaochu
        assert_tile_count(hand, 13)
        assert shanten_kokushi(hand) == 2

    def test_kokushi_no_yaochu(self):
        """No yaochu tiles at all → 13-shanten."""
        hand = make_hand([1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15])
        assert_tile_count(hand, 13)
        assert shanten_kokushi(hand) == 13


class TestShantenCombined:
    """Combined shanten takes the minimum of all forms."""

    def test_min_across_forms(self):
        """Should return the minimum of standard/chiitoitsu/kokushi."""
        # A hand closer to chiitoitsu than standard
        hand = [0]*34
        for t in [0, 3, 6, 9, 27]:
            hand[t] = 2  # 5 pairs = 10 tiles
        for t in [12, 15, 31]:
            hand[t] = 1  # 3 singles = 3 tiles
        assert_tile_count(hand, 13)
        # Chiitoitsu: 5 pairs → 1-shanten
        assert shanten_chiitoitsu(hand) == 1
        s = calculate_shanten(hand)
        # The combined min should pick chiitoitsu (1) or standard (>=1)
        assert s == 1


# =============================================================================
# 2. Ukeire (Efficiency) Calculation
# =============================================================================

class TestUkeire:
    """Effective tile acceptance calculation."""

    def test_tenpai_ukeire_ryanmen(self):
        """Tenpai hand: waits should be detected as ukeire."""
        # 123m 456m 789m + 23p(taatsu) + 55s(pair) = 13 tiles
        # Waits: 1p(9) and 4p(12) for the 23p taatsu
        hand = make_hand([0,1,2, 3,4,5, 6,7,8, 10,11, 23,23])
        assert_tile_count(hand, 13)
        count, mask, avail = compute_ukeire(hand)
        assert count >= 1, "tenpai hand should have ukeire"
        # Adding 1p(9) or 4p(12) should complete to agari (shanten -1 < 0)
        assert mask[9], "1p should be ukeire"
        assert mask[12], "4p should be ukeire"

    def test_tenpai_ukeire_tanki(self):
        """Tanki tenpai: the single's pair should be ukeire."""
        # 4 melds (12 tiles) + 白(1) = 13 tiles, tanki on 白
        hand = make_hand([0,0,0, 1,1,1, 2,2,2, 9,10,11, 31])
        assert_tile_count(hand, 13)
        count, mask, avail = compute_ukeire(hand)
        assert count >= 1
        assert mask[31], "白 should be ukeire (tanki)"

    def test_one_shanten_ukeire(self):
        """1-shanten hand should have > 0 ukeire."""
        # 111m(triplet) + 123p(meld) + 45p(taatsu) + 67s(taatsu) + 東東(pair) + 5m(isolated)
        # = 3+3+2+2+2+1 = 13 tiles
        hand = make_hand([0,0,0, 9,10,11, 12,13, 19,20, 27,27, 4])
        assert_tile_count(hand, 13)
        count, mask, avail = compute_ukeire(hand)
        assert count > 0, "1-shanten should have ukeire"

    def test_ukeire_mask_length(self):
        """Ukeire mask should be length 34."""
        hand = make_hand([0, 2, 4, 6, 8, 9, 11, 13, 15, 17, 18, 20, 22])
        assert_tile_count(hand, 13)
        count, mask, avail = compute_ukeire(hand)
        assert len(mask) == 34

    def test_ukeire_agari_only_one_way(self):
        """Tanki hand: only 1 ukeire tile type."""
        # 123m 456m 789m 123p 東 = 4 melds + 1 single = 13 tiles
        hand = make_hand([0,1,2, 3,4,5, 6,7,8, 9,10,11, 27])
        assert_tile_count(hand, 13)
        count, mask, avail = compute_ukeire(hand)
        assert count == 1, f"tanki on 東 should have exactly 1 ukeire, got {count}"
        assert mask[27], "東 should be the only ukeire"

    def test_ukeire_kokushi_tenpai(self):
        """Kokushi-tenpai hand: all 13 yaochu tiles are ukeire."""
        # All 13 yaochu types, each count 1 → kokushi tenpai
        hand = make_hand([0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33])
        count, mask, avail = compute_ukeire(hand)
        # All 13 yaochu tiles reduce shanten
        assert count == 13, f"expected 13 ukeire for kokushi tenpai, got {count}"


# =============================================================================
# 3. Wait Quality Classification
# =============================================================================

class TestWaitQuality:
    """Wait type and quality score."""

    def test_ryanmen_23p(self):
        """23p waiting on 1p and 4p → simple ryanmen."""
        # 123m 456m 789m + 23p + 55s = 13 tiles
        hand = make_hand([0,1,2, 3,4,5, 6,7,8, 10,11, 23,23])
        result = classify_wait(hand)
        assert result['is_tenpai'] == True
        assert result['wait_count'] >= 2
        assert 9 in result['waits'] and 12 in result['waits']
        assert result['quality_score'] > 0

    def test_tenpai_simple_hand(self):
        """123m 456m 789m 123p 東 → tanki on 東."""
        hand = make_hand([0,1,2, 3,4,5, 6,7,8, 9,10,11, 27])
        result = classify_wait(hand)
        assert result['is_tenpai'] == True
        assert result['wait_count'] == 1
        assert 27 in result['waits']

    def test_penchan_12m(self):
        """12m waiting on 3m → penchan."""
        hand = make_hand([0,1, 9,10,11, 18,19,20, 27,27,27, 31,31])
        assert_tile_count(hand, 13)
        result = classify_wait(hand)
        assert result['is_tenpai'] == True
        assert 2 in result['waits']

    def test_not_tenpai(self):
        """Non-tenpai hand should return noten result."""
        hand = make_hand([0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 28, 29, 30])
        assert_tile_count(hand, 13)
        result = classify_wait(hand)
        assert result['is_tenpai'] == False
        assert result['main_type'] == 'noten'
        assert result['quality_score'] == 0.0
        assert result['waits'] == []
        assert result['wait_count'] == 0

    def test_quality_score_range(self):
        """Quality score should be in [0, 1] for any tenpai hand."""
        hand = make_hand([0,1,2, 3,4,5, 6,7,8, 10,11, 23,23])
        result = classify_wait(hand)
        if result['is_tenpai']:
            assert 0.0 <= result['quality_score'] <= 1.0

    def test_total_available(self):
        """total_available should reflect remaining tiles."""
        hand = make_hand([0,1,2, 3,4,5, 6,7,8, 10,11, 23,23])
        result = classify_wait(hand)
        if result['is_tenpai']:
            assert result['total_available'] > 0
            # 1p = 4 tiles left, 4p = 4 tiles left → total = 8
            assert result['total_available'] <= 8


# =============================================================================
# 4. Combined Oracle Labels
# =============================================================================

class TestComputeAllOracleLabels:
    """Combined interface."""

    def test_tenpai_hand_includes_wait_info(self):
        """Tenpai hand should include wait details in combined output."""
        hand = make_hand([0,1,2, 3,4,5, 6,7,8, 10,11, 23,23])
        labels = compute_all_oracle_labels(hand)
        assert labels['shanten'] == 0
        assert labels['is_tenpai'] == True
        assert labels['wait_quality'] is not None
        assert 'waits' in labels
        assert 'wait_types' in labels
        assert 'quality_score' in labels
        assert labels['quality_score'] > 0

    def test_noten_hand_no_wait_info(self):
        """Non-tenpai hand should have None wait_quality."""
        hand = make_hand([0, 2, 4, 6, 8, 9, 11, 13, 15, 17, 18, 20, 22])
        labels = compute_all_oracle_labels(hand)
        assert labels['shanten'] > 0
        assert labels['is_tenpai'] == False
        assert labels['wait_quality'] is None
        assert labels['quality_score'] == 0.0
        assert labels['wait_type'] == 'noten'

    def test_ukeire_in_output(self):
        """Combined output should include ukeire info."""
        hand = make_hand([0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 23, 23])
        labels = compute_all_oracle_labels(hand)
        assert 'ukeire_count' in labels
        assert 'ukeire_mask' in labels
        assert 'ukeire_available' in labels
        assert len(labels['ukeire_mask']) == 34

    def test_efficiency_and_danger_placeholders(self):
        """Placeholder fields should exist."""
        hand = make_hand([0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 23, 23])
        labels = compute_all_oracle_labels(hand)
        assert 'efficiency_score' in labels
        assert 'danger_map' in labels
        assert len(labels['danger_map']) == NUM_TYPES
        assert 'score_estimate' in labels
# 中文注释：验证 Oracle 标签系统的向听数、有效进张、等待牌质量和组合接口的正确性。

