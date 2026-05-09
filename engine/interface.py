"""前后端统一契约层。

本模块把 Python 引擎内部对象转换为 JSON-friendly 的字典结构，作为
React 控制台、未来 WebSocket/HTTP 服务和文档之间的共同接口。

设计原则：
  - 引擎核心仍以 dataclass / IntEnum / int[34] 为准，不为了前端改算法对象。
  - 契约字段尽量贴近 Python 引擎命名，便于排查状态机问题。
  - UI 需要的中文标签、图片 code 在这里一并给出，避免多处维护牌映射。
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from .actions import Action, ActionType, LegalActions
from .agari import get_waits, is_tenpai
from .game import GameEngine, GameState
from .hand import Meld, MeldType
from .rules import GameConfig
from .tile import TILE_NAMES, TILE_NAMES_CN, abs_to_type, is_aka
from .wall import DEAD_WALL_SIZE


SEAT_NAMES = ["east", "south", "west", "north"]
SEAT_LABELS = ["东家", "南家", "西家", "北家"]

MELD_TYPE_NAMES = {
    MeldType.CHI: "chi",
    MeldType.PON: "pon",
    MeldType.KAN_CLOSED: "ankan",
    MeldType.KAN_OPEN: "kakan",
    MeldType.KAN_DAIMIN: "kan",
}

ACTION_TYPE_NAMES = {
    ActionType.DISCARD: "discard",
    ActionType.TSUMO: "tsumo",
    ActionType.RON: "ron",
    ActionType.RIICHI: "riichi",
    ActionType.PON: "pon",
    ActionType.CHI: "chi",
    ActionType.KAN_DAIMIN: "kan",
    ActionType.KAN_ANKAN: "ankan",
    ActionType.KAN_KAKAN: "kakan",
    ActionType.PASS: "pass",
    ActionType.RYUUKYOKU: "ryuukyoku",
    ActionType.NUKI: "nuki",
}

ACTION_NAME_TO_TYPE = {
    "discard": ActionType.DISCARD,
    "tsumo": ActionType.TSUMO,
    "win": ActionType.TSUMO,
    "ron": ActionType.RON,
    "riichi": ActionType.RIICHI,
    "pon": ActionType.PON,
    "chi": ActionType.CHI,
    "kan": ActionType.KAN_DAIMIN,
    "daiminkan": ActionType.KAN_DAIMIN,
    "ankan": ActionType.KAN_ANKAN,
    "kakan": ActionType.KAN_KAKAN,
    "pass": ActionType.PASS,
    "draw": ActionType.PASS,  # draw 是控制台/Mock 的系统动作，真实引擎中用 step 后的状态推进表达。
    "ryuukyoku": ActionType.RYUUKYOKU,
    "nuki": ActionType.NUKI,
}


def tile_code(type_id: int, abs_id: int = -1) -> str:
    """返回前端牌图 code，例如 1m、0p、E、zhong。"""
    if is_aka(abs_id):
        if type_id == 4:
            return "0m"
        if type_id == 13:
            return "0p"
        if type_id == 22:
            return "0s"

    if 0 <= type_id <= 8:
        return f"{type_id + 1}m"
    if 9 <= type_id <= 17:
        return f"{type_id - 8}p"
    if 18 <= type_id <= 26:
        return f"{type_id - 17}s"

    honor_codes = {
        27: "E",
        28: "S",
        29: "W",
        30: "N",
        31: "white",
        32: "haku",
        33: "zhong",
    }
    return honor_codes.get(type_id, "unknown")


def serialize_tile(abs_id: int = -1, type_id: Optional[int] = None) -> Dict[str, Any]:
    """把绝对 ID / 类型 ID 转成前端可渲染的牌对象。"""
    resolved_type = abs_to_type(abs_id) if type_id is None and abs_id >= 0 else int(type_id or 0)
    red = is_aka(abs_id) if abs_id >= 0 else False
    label = TILE_NAMES_CN[resolved_type] if 0 <= resolved_type < len(TILE_NAMES_CN) else "未知"
    return {
        "type_id": resolved_type,
        "abs_id": abs_id if abs_id >= 0 else None,
        "code": tile_code(resolved_type, abs_id),
        "label": f"赤{label}" if red else label,
        "red": red,
    }


def expand_counts(counts: Iterable[int]) -> List[Dict[str, Any]]:
    """把 int[34] 直方图展开为 TileDTO 列表，主要给控制台 God View 使用。"""
    tiles: List[Dict[str, Any]] = []
    for type_id, count in enumerate(counts):
        for _ in range(count):
            tiles.append(serialize_tile(type_id=type_id))
    return tiles


def serialize_meld(meld: Meld) -> Dict[str, Any]:
    """序列化副露。tiles 保留类型 ID，同时提供 tile 对象便于 UI 渲染。"""
    return {
        "type": MELD_TYPE_NAMES.get(meld.meld_type, "unknown"),
        "tiles": [serialize_tile(type_id=t) for t in meld.tiles],
        "tile_types": list(meld.tiles),
        "called_from": meld.called_from,
        "source_tile": meld.source_tile,
    }


def serialize_action(action: Action) -> Dict[str, Any]:
    """序列化一个合法动作。"""
    return {
        "type": ACTION_TYPE_NAMES.get(action.action_type, str(int(action.action_type))),
        "action_type": int(action.action_type),
        "tile": serialize_tile(type_id=action.tile) if action.tile >= 0 else None,
        "tile_type": action.tile,
        "meld_tiles": [serialize_tile(type_id=t) for t in action.meld_tiles],
        "meld_tile_types": list(action.meld_tiles),
        "actor": action.actor,
        "label": repr(action),
    }


def serialize_legal_actions(legal: Optional[LegalActions]) -> Dict[str, Any]:
    """序列化合法动作列表和 77 维动作掩码。"""
    if legal is None:
        return {"actions": [], "mask": [0] * 77}
    return {
        "actions": [serialize_action(action) for action in legal.actions],
        "mask": list(legal.mask),
    }


def deserialize_action(command: Dict[str, Any]) -> Action:
    """把前端 ActionCommandDTO 转回 Python Action。

    支持两种牌字段：
      - tile_type: 直接传类型 ID
      - tile: 传 TileDTO，其中包含 type_id
    """
    raw_type = str(command.get("type") or command.get("action_type") or "pass").lower()
    action_type = ACTION_NAME_TO_TYPE.get(raw_type, ActionType.PASS)

    tile_value = command.get("tile_type", -1)
    if isinstance(command.get("tile"), dict):
        tile_value = command["tile"].get("type_id", tile_value)

    meld_values = command.get("meld_tile_types", [])
    if not meld_values and isinstance(command.get("meld_tiles"), list):
        meld_values = [
            tile.get("type_id", -1)
            for tile in command["meld_tiles"]
            if isinstance(tile, dict)
        ]

    return Action(
        action_type=action_type,
        tile=int(tile_value) if tile_value is not None else -1,
        meld_tiles=[int(t) for t in meld_values if int(t) >= 0],
        actor=int(command.get("actor", command.get("player", -1))),
    )


def serialize_config(config: GameConfig) -> Dict[str, Any]:
    """序列化规则配置，字段名与 Python GameConfig 保持一致。"""
    return {
        "start_score": config.start_score,
        "target_score": config.target_score,
        "riichi_stick_cost": config.riichi_stick_cost,
        "honba_bonus": config.honba_bonus,
        "kuitan": config.kuitan,
        "aka_dora": config.aka_dora,
        "akadora": 3 if config.use_red_dora else 0,
        "ryanhan_shibari": config.ryanhan_shibari,
        "kuikae": config.kuikae,
        "atozuke": config.atozuke,
        "open_riichi": config.open_riichi,
        "uma": list(config.uma),
        "oka": config.oka,
        "yakuman_multiple": config.yakuman_multiple,
        "rounds": config.rounds,
        "east_only": config.east_only,
        "agari_yame": config.agari_yame,
        "tenpai_renchan": config.tenpai_renchan,
        "tobi": config.tobi,
        "wareme": config.wareme,
        "multiple_ron": config.multiple_ron,
        "use_red_dora": config.use_red_dora,
        # 这两个字段当前 Python 引擎尚未完整实现，但前端规则面板需要稳定契约。
        "nagashi_mangan": True,
        "responsibility": True,
    }


def _serialize_players(engine: GameEngine, state: GameState) -> List[Dict[str, Any]]:
    players = []
    for index, player in enumerate(engine.players):
        waits: List[Dict[str, Any]] = []
        try:
            if is_tenpai(player.hand.tiles):
                waits = [serialize_tile(type_id=t) for t in get_waits(player.hand.tiles)]
        except Exception:
            # 听牌探测不能影响状态快照输出；调试信息仍可从 Raw JSON 看见。
            waits = []

        players.append({
            "index": index,
            "seat": SEAT_NAMES[index],
            "seat_label": SEAT_LABELS[index],
            "seat_wind": player.seat_wind,
            "score": player.score,
            "hand_counts": list(player.hand.tiles),
            "hand": expand_counts(player.hand.tiles),
            "melds": [serialize_meld(meld) for meld in player.hand.melds],
            "discards": [serialize_tile(abs_id=abs_id) for abs_id in player.discards],
            "is_riichi": player.is_riichi,
            "is_double_riichi": player.is_double_riichi,
            "is_ippatsu": player.is_ippatsu,
            "has_won": player.has_won,
            "tenpai": bool(waits),
            "waits": waits,
            "furiten": bool(player.temp_furiten or player.is_riichi_furiten
                           or any(dt in waits for dt in player.discard_types)),
            "is_tenpai_at_ryuukyoku": player.is_tenpai_at_ryuukyoku,
            "legal_actions": serialize_legal_actions(
                state.legal_actions if index == state.current_player else None
            ),
        })
    return players


def serialize_game_state(engine: GameEngine) -> Dict[str, Any]:
    """序列化 GameEngine 当前完整状态。"""
    state = engine.get_game_state()
    live_tiles = engine.wall.tiles[engine.wall._live_ptr:engine.wall._dead_wall_start]
    dead_wall = engine.wall.tiles[engine.wall._dead_wall_start:engine.wall._dead_wall_start + DEAD_WALL_SIZE]

    return {
        "phase": engine.phase.name,
        "phase_id": int(engine.phase),
        "current_player": state.current_player,
        "current_seat": SEAT_NAMES[state.current_player],
        "round_wind": state.round_wind,
        "round_wind_label": TILE_NAMES[state.round_wind] if state.round_wind >= 0 else "",
        "round_number": state.round_number,
        "round": f"{TILE_NAMES_CN[state.round_wind]}{state.round_number}局",
        "honba": state.honba,
        "riichi_sticks": state.riichi_sticks,
        "scores": list(state.scores),
        "hands_concealed": [list(hand) for hand in state.hands_concealed],
        "players": _serialize_players(engine, state),
        "open_melds": [[serialize_meld(meld) for meld in melds] for melds in state.open_melds],
        "discards": [[serialize_tile(abs_id=abs_id) for abs_id in discards] for discards in state.discards],
        "wall": [serialize_tile(abs_id=abs_id) for abs_id in live_tiles],
        "dead_wall": [serialize_tile(abs_id=abs_id) for abs_id in dead_wall],
        "dora_indicators": [serialize_tile(abs_id=abs_id) for abs_id in state.dora_indicators],
        "ura_dora_indicators": [serialize_tile(abs_id=abs_id) for abs_id in state.ura_dora_indicators],
        "is_riichi": list(state.is_riichi),
        "last_discard": serialize_tile(abs_id=state.last_discard) if state.last_discard >= 0 else None,
        "last_discard_by": state.last_discard_by,
        "remaining_tiles": state.remaining_tiles,
        "legal_actions": serialize_legal_actions(engine.get_legal_actions()),
        "rewards": list(state.rewards),
        "done": state.done,
    }


def create_engine_snapshot(
    engine: GameEngine,
    logs: Optional[List[Dict[str, Any]]] = None,
    debug: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """创建统一 EngineSnapshotDTO。

    Args:
        engine: 当前 Python 游戏引擎。
        logs: 前端可展示的日志列表。
        debug: 控制台调试元信息，例如连接来源、帧号、最近动作。
    """
    state = serialize_game_state(engine)
    if debug:
        state["debug"] = debug
        if "last_action" in debug:
            state["last_action"] = debug["last_action"]
    return {
        "state": state,
        "rules": serialize_config(engine.config),
        "logs": logs or [],
    }
# 中文注释：把引擎内部对象序列化为前端/API 可消费的数据契约，并反序列化用户动作。
