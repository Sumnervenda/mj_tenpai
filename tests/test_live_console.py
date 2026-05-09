"""Live Console SSE 辅助逻辑测试。"""

import json

from engine import GameEngine
from main import (
    LiveSnapshotHub,
    advance_random_game,
    format_sse_event,
    make_live_log,
    make_live_snapshot,
    random,
)


def test_advance_random_game_progresses_frame():
    """随机单步推进应至少产生一条可读消息。"""
    engine = GameEngine(seed=42)
    rng = random.Random(42)
    message, done = advance_random_game(engine, rng, verbose=False)
    assert isinstance(message, str)
    assert message
    assert done is False


def test_live_snapshot_json_shape():
    """Live 快照应保持 EngineSnapshotDTO 结构。"""
    engine = GameEngine(seed=42)
    logs = [make_live_log(1, "info", "ok")]
    snapshot = make_live_snapshot(engine, logs, frame=1, last_action="ok", paused=True)
    encoded = json.dumps(snapshot, ensure_ascii=False)
    decoded = json.loads(encoded)
    assert set(decoded) == {"state", "rules", "logs"}
    assert decoded["state"]["debug"]["connection"] == "sse"
    assert decoded["state"]["debug"]["replay_frame"] == 1
    assert decoded["state"]["debug"]["paused"] is True
    assert len(decoded["state"]["legal_actions"]["mask"]) == 77


def test_sse_event_format():
    payload = '{"ok": true}'
    event = format_sse_event(payload, version=3).decode("utf-8")
    assert "id: 3\n" in event
    assert "event: snapshot\n" in event
    assert "data: {\"ok\": true}\n\n" in event


def test_snapshot_hub_publish_latest():
    hub = LiveSnapshotHub()
    hub.publish({"state": {"done": True}, "rules": {}, "logs": []})
    version, snapshot_json = hub.latest()
    assert version == 1
    assert hub.done is True
    assert json.loads(snapshot_json)["state"]["done"] is True


def test_snapshot_hub_pause_state():
    hub = LiveSnapshotHub()
    assert hub.is_paused() is False
    changed = hub.set_paused(True)
    assert changed is True
    assert hub.is_paused() is True
    status = hub.status()
    assert status["paused"] is True
    assert status["ok"] is True
# 中文注释：验证本地控制台/实时接口相关 DTO，确保调试页面能消费引擎状态。
