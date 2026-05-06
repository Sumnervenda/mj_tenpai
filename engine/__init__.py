"""Mahjong DL-Engine —— 面向 AI 训练的竞技麻将游戏引擎。

用法:
    from engine import GameEngine, GameConfig

    config = GameConfig()              # 雀魂标准规则
    engine = GameEngine(config)

    while not engine.is_game_over():
        # 摸牌阶段：当前玩家决策
        if engine.phase == GamePhase.DRAW:
            actions = engine.get_legal_actions()
            action = choose_action(actions)   # AI Agent 选择动作
            engine.step(action)

        # 舍牌后响应阶段：收集所有响应，按优先级执行
        elif engine.phase == GamePhase.DISCARD:
            options = engine.get_response_options()
            responses = {p: pick_action(opts) for p, opts in options.items()}
            engine.resolve_responses(responses)

    result = engine.get_result()        # 获取终局精算结果
"""

# ── 牌型系统 ──
from .tile import (
    NUM_TYPES, NUM_ABS,
    TileType,
    MANZU, PINZU, SOUZU, JIHAI,
    abs_to_type, type_to_abs, is_aka, is_aka_type,
    is_manzu, is_pinzu, is_souzu, is_jihai,
    is_kazehai, is_sangenhai, is_shupai, is_yaochuhai,
    TILE_NAMES, TILE_NAMES_CN, TILE_NUMBERS,
    TANYAO_TYPES, YAOCHUHAI_TYPES, JIHAI_TYPES,
    suit_of,
)

# ── 牌山 ──
from .wall import Wall

# ── 手牌 ──
from .hand import Hand, Meld, MeldType

# ── 胡牌判定 ──
from .agari import is_agari, is_tenpai, get_waits, is_agari_with_tile

# ── 役种 ──
from .yaku import Yaku, YakuChecker, WinContext, YakuResult, YAKU_NAMES_JP

# ── 得点计算 ──
from .scoring import (
    calculate_fu, calculate_fu_from_decomp, compute_payments,
    compute_final_result, PaymentInfo, GameResult,
)

# ── 动作系统 ──
from .actions import (
    Action, ActionType, LegalActions,
    compute_draw_actions, compute_response_actions,
    create_action_mask, ACTION_SPACE_SIZE,
)

# ── 规则配置 ──
from .rules import GameConfig

# ── 游戏引擎 ──
from .game import GameEngine, GamePhase, GameState, PlayerState

# ── 前后端统一契约层 ──
from .interface import (
    create_engine_snapshot, deserialize_action, serialize_config,
    serialize_game_state, serialize_legal_actions, serialize_tile,
)
