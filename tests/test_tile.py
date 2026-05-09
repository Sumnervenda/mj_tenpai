"""Tests for tile encoding system."""
import pytest
from engine.tile import (
    NUM_TYPES, NUM_ABS, COPIES_PER_TYPE,
    TileType, AKA_TYPES,
    abs_to_type, type_to_abs, is_aka, is_aka_type,
    is_manzu, is_pinzu, is_souzu, is_jihai,
    is_kazehai, is_sangenhai, is_shupai, is_yaochuhai,
    TILE_NAMES, TILE_NAMES_CN, TILE_NUMBERS,
    TANYAO_TYPES, YAOCHUHAI_TYPES, GREEN_TYPES,
    suit_of, tile_name,
)


class TestTileEncoding:
    """Absolute ID ↔ Type ID conversion."""

    def test_abs_to_type_range(self):
        assert abs_to_type(0) == 0
        assert abs_to_type(3) == 0
        assert abs_to_type(4) == 1
        assert abs_to_type(135) == 33

    def test_type_to_abs(self):
        abs_ids = type_to_abs(0)
        assert abs_ids == [0, 1, 2, 3]
        assert type_to_abs(33) == [132, 133, 134, 135]

    def test_roundtrip(self):
        for abs_id in range(NUM_ABS):
            t = abs_to_type(abs_id)
            assert abs_id in type_to_abs(t)

    def test_aka_detection(self):
        # Red 5m: type 4, abs_ids [16, 17, 18, 19], copy 3 (19) is red
        assert is_aka(19) == True
        assert is_aka(16) == False
        assert is_aka(17) == False
        assert is_aka(18) == False

        # Red 5p: type 13, abs_ids [52, 53, 54, 55], copy 3 (55) is red
        assert is_aka(55) == True

        # Red 5s: type 22, abs_ids [88, 89, 90, 91], copy 3 (91) is red
        assert is_aka(91) == True

        # Non-aka tile type should never be aka
        assert is_aka(0) == False  # 1m


class TestTileTypes:
    """Tile type classification."""

    def test_suit_ranges(self):
        assert all(is_manzu(i) for i in range(0, 9))
        assert all(is_pinzu(i) for i in range(9, 18))
        assert all(is_souzu(i) for i in range(18, 27))
        assert all(is_jihai(i) for i in range(27, 34))

    def test_shupai_jihai(self):
        assert all(is_shupai(i) for i in range(0, 27))
        assert all(not is_shupai(i) for i in range(27, 34))
        assert all(not is_jihai(i) for i in range(0, 27))
        assert all(is_jihai(i) for i in range(27, 34))

    def test_yaochuhai(self):
        # Terminals
        assert is_yaochuhai(0)   # 1m
        assert is_yaochuhai(8)   # 9m
        assert is_yaochuhai(9)   # 1p
        assert is_yaochuhai(17)  # 9p
        assert is_yaochuhai(18)  # 1s
        assert is_yaochuhai(26)  # 9s
        # Honors
        assert all(is_yaochuhai(i) for i in range(27, 34))
        # Not yaochu
        assert not is_yaochuhai(1)  # 2m
        assert not is_yaochuhai(10) # 2p
        assert not is_yaochuhai(19) # 2s

    def test_tanyao_types(self):
        # 2-8 in all suits should be tanyao
        assert 1 in TANYAO_TYPES   # 2m
        assert 7 in TANYAO_TYPES   # 8m
        assert 0 not in TANYAO_TYPES  # 1m
        assert 8 not in TANYAO_TYPES  # 9m
        assert 27 not in TANYAO_TYPES  # 東

    def test_yaochuhai_count(self):
        # 13 yaochuhai types: 1,9 of each suit = 6, + 7 honors = 13
        assert len(YAOCHUHAI_TYPES) == 13

    def test_green_types(self):
        assert 19 in GREEN_TYPES  # 2s
        assert 20 in GREEN_TYPES  # 3s
        assert 21 in GREEN_TYPES  # 4s
        assert 23 in GREEN_TYPES  # 6s
        assert 25 in GREEN_TYPES  # 8s
        assert 32 in GREEN_TYPES  # 發
        assert 22 not in GREEN_TYPES  # 5s (red)

    def test_suit_of(self):
        assert suit_of(0) == 0   # 万
        assert suit_of(8) == 0
        assert suit_of(9) == 1   # 筒
        assert suit_of(17) == 1
        assert suit_of(18) == 2  # 条
        assert suit_of(26) == 2
        assert suit_of(27) == 3  # 字
        assert suit_of(33) == 3


class TestTileNames:
    """Display names."""
    def test_all_types_have_names(self):
        assert len(TILE_NAMES) == NUM_TYPES
        assert len(TILE_NAMES_CN) == NUM_TYPES

    def test_tile_name_marks_aka(self):
        assert tile_name(19) == "赤5m"
        assert tile_name(55) == "赤5p"
        assert tile_name(91) == "赤5s"
        # Normal copy of aka type
        assert tile_name(16) == "5m"
# 中文注释：验证牌编码、红五、显示名称和 ID 映射等基础工具函数。
