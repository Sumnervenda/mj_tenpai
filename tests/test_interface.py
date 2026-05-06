"""统一接口契约层测试。"""

from engine import (
    ActionType,
    GameConfig,
    GameEngine,
    create_engine_snapshot,
    deserialize_action,
    serialize_config,
    serialize_game_state,
    serialize_legal_actions,
    serialize_tile,
)


def test_serialize_tile_maps_code_and_red_five():
    """绝对 ID 和图片 code 的映射必须稳定，供前端 PNG 渲染使用。"""
    assert serialize_tile(type_id=0)["code"] == "1m"
    assert serialize_tile(type_id=27)["code"] == "E"
    red_five_man = serialize_tile(abs_id=19)
    assert red_five_man["code"] == "0m"
    assert red_five_man["red"] is True


def test_serialize_config_uses_python_field_names():
    config = GameConfig(start_score=30000, east_only=True)
    dto = serialize_config(config)
    assert dto["start_score"] == 30000
    assert dto["east_only"] is True
    assert "riichi_stick_cost" in dto
    assert "multiple_ron" in dto


def test_serialize_game_state_contains_ui_and_engine_fields():
    engine = GameEngine(seed=42)
    dto = serialize_game_state(engine)
    assert dto["phase"] == "DRAW"
    assert dto["current_player"] == 0
    assert dto["current_seat"] == "east"
    assert len(dto["players"]) == 4
    assert len(dto["hands_concealed"]) == 4
    assert dto["remaining_tiles"] == len(dto["wall"])
    assert len(dto["dead_wall"]) == 14
    assert len(dto["legal_actions"]["mask"]) == 77


def test_serialize_legal_actions_mask_length():
    engine = GameEngine(seed=7)
    legal = engine.get_legal_actions()
    dto = serialize_legal_actions(legal)
    assert len(dto["mask"]) == 77
    assert dto["actions"]


def test_deserialize_action_accepts_frontend_command():
    command = {
        "type": "discard",
        "tile": {"type_id": 4, "code": "5m"},
        "actor": 0,
    }
    action = deserialize_action(command)
    assert action.action_type == ActionType.DISCARD
    assert action.tile == 4
    assert action.actor == 0


def test_create_engine_snapshot_shape():
    engine = GameEngine(seed=11)
    snapshot = create_engine_snapshot(engine, logs=[{"level": "info", "message": "ok"}])
    assert set(snapshot) == {"state", "rules", "logs"}
    assert snapshot["logs"][0]["message"] == "ok"
