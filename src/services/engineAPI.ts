import {
  ActionCommand,
  ActionType,
  EngineSnapshot,
  MeldType,
  SEAT_META,
  SEATS,
  ScenarioKey,
  Seat,
  Tile,
  formatTile
} from "../types/game";
import {
  ActionCommandDTO,
  EngineSnapshotDTO,
  GameStateDTO,
  LogEntryDTO,
  RuleSettingsDTO,
  SeatDTO,
  TileDTO
} from "../types/contract";
import {
  countsFromTiles,
  createTileDTO,
  expandCounts,
  rulesUIPatchToDTO,
  snapshotDTOToUI,
  uiActionToDTO,
  uiTileToDTO
} from "./adapters";

const ACTION_INDEX: Record<ActionType, number> = {
  discard: 0,
  tsumo: 34,
  ron: 35,
  riichi: 36,
  pon: 71,
  chi: 72,
  kan: 73,
  draw: 74,
  ankan: 74,
  kakan: 75,
  win: 35,
  pass: 76,
  ryuukyoku: 76,
  nuki: 76
};

const DEFAULT_RULES: RuleSettingsDTO = {
  start_score: 25000,
  target_score: 30000,
  riichi_stick_cost: 1000,
  honba_bonus: 300,
  kuitan: true,
  aka_dora: true,
  akadora: 3,
  ryanhan_shibari: false,
  kuikae: false,
  atozuke: true,
  open_riichi: false,
  uma: [20, 10, -10, -20],
  oka: 0,
  yakuman_multiple: true,
  rounds: 1,
  east_only: false,
  agari_yame: true,
  tenpai_renchan: true,
  tobi: false,
  wareme: false,
  multiple_ron: true,
  use_red_dora: true,
  nagashi_mangan: true,
  responsibility: true
};

let logSerial = 0;
let absSerial = 0;
let currentSnapshot = createInitialSnapshotDTO();

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

function now(): string {
  return new Date().toLocaleTimeString("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false
  });
}

function makeLog(level: LogEntryDTO["level"], message: string, payload?: unknown): LogEntryDTO {
  logSerial += 1;
  return {
    id: `log-${logSerial}`,
    time: now(),
    level,
    message,
    payload
  };
}

function pushLog(level: LogEntryDTO["level"], message: string, payload?: unknown): void {
  currentSnapshot.logs = [makeLog(level, message, payload), ...currentSnapshot.logs].slice(0, 80);
}

function tile(typeId: number, red = false): TileDTO {
  absSerial += 1;
  return createTileDTO(typeId, { absId: absSerial, red });
}

function makeWall(): TileDTO[] {
  const tiles: TileDTO[] = [];
  for (let typeId = 0; typeId < 34; typeId += 1) {
    for (let copy = 0; copy < 4; copy += 1) {
      const red = copy === 3 && [4, 13, 22].includes(typeId);
      tiles.push(createTileDTO(typeId, { absId: typeId * 4 + copy, red }));
    }
  }

  return tiles
    .map((entry, index) => ({ entry, sort: (index * 37 + 17) % 136 }))
    .sort((a, b) => a.sort - b.sort)
    .map(({ entry }) => entry);
}

function makePlayer(seat: SeatDTO, typeIds: number[], score = 25000) {
  const hand = typeIds.map((typeId) => tile(typeId, [4, 13, 22].includes(typeId)));
  const index = SEAT_META[seat].index;
  return {
    index,
    seat,
    seat_label: SEAT_META[seat].label,
    seat_wind: 27 + index,
    score,
    hand_counts: countsFromTiles(hand),
    hand,
    melds: [],
    discards: [],
    is_riichi: false,
    is_double_riichi: false,
    is_ippatsu: false,
    has_won: false,
    tenpai: false,
    shanten: 2,
    waits: [tile((index * 7 + 3) % 34)],
    furiten: false,
    ai_enabled: seat !== "east",
    is_tenpai_at_ryuukyoku: false,
    legal_actions: emptyLegalActions()
  };
}

function emptyLegalActions() {
  return { actions: [], mask: Array.from({ length: 77 }, () => 0) };
}

function createActionMask(enabled: number[] = []): number[] {
  const mask = Array.from({ length: 77 }, () => 0);
  for (const index of enabled) {
    if (index >= 0 && index < mask.length) mask[index] = 1;
  }
  return mask;
}

function createLegalActions(state: GameStateDTO) {
  const enabled = state.phase === "DISCARD" || state.phase === "RESPONSE"
    ? [0, 36, 71, 72, 73, 76]
    : [0, 1, 2, 3, 4, 34, 36, 74];

  return {
    actions: enabled.map((index) => ({
      type: Object.entries(ACTION_INDEX).find(([, value]) => value === index)?.[0] as ActionType,
      actor: state.current_player,
      seat: state.current_seat,
      label: `mask[${index}]`
    })),
    mask: createActionMask(enabled)
  };
}

function createInitialSnapshotDTO(): EngineSnapshotDTO {
  absSerial = 0;
  const wallSource = makeWall();
  const wall = wallSource.slice(52, 122);
  const deadWall = wallSource.slice(122, 136);

  const state: GameStateDTO = {
    phase: "DRAW",
    phase_id: 1,
    current_player: 0,
    current_seat: "east",
    round_wind: 27,
    round_wind_label: "東",
    round_number: 1,
    round: "东1局",
    honba: 0,
    riichi_sticks: 0,
    scores: [25000, 25000, 25000, 25000],
    hands_concealed: [],
    players: [
      makePlayer("east", [0, 1, 2, 3, 4, 5, 6, 7, 8, 27, 27, 31, 33], 25000),
      makePlayer("south", [9, 10, 11, 12, 13, 14, 15, 16, 17, 28, 28, 32, 33], 25000),
      makePlayer("west", [18, 19, 20, 21, 22, 23, 24, 25, 26, 29, 29, 31, 32], 25000),
      makePlayer("north", [0, 8, 9, 17, 18, 26, 30, 30, 31, 31, 32, 33, 33], 25000)
    ],
    open_melds: [],
    discards: [],
    wall,
    dead_wall: deadWall,
    dora_indicators: [deadWall[4] ?? tile(33)],
    ura_dora_indicators: [deadWall[9] ?? tile(4, true)],
    is_riichi: [false, false, false, false],
    last_action: "Mock adapter initialized",
    last_discard: null,
    last_discard_by: -1,
    remaining_tiles: wall.length,
    next_draw: {},
    legal_actions: emptyLegalActions(),
    rewards: [0, 0, 0, 0],
    done: false,
    debug: {
      connection: "mock",
      illegal_actions: 0,
      replay_frame: 0,
      response_queue: [],
      last_scoring: {
        han: 3,
        fu: 30,
        yaku: [
          { name: "立直", han: 1 },
          { name: "宝牌", han: 2 }
        ]
      }
    }
  };

  refreshState(state);

  return {
    state,
    rules: clone(DEFAULT_RULES),
    logs: [
      makeLog("info", "Mock DTO 契约已就绪，等待调试动作"),
      makeLog("debug", "动作掩码长度 77，当前为 DRAW_STATE")
    ]
  };
}

function seatByIndex(index: number): SeatDTO {
  return SEATS[index] ?? "east";
}

function nextSeat(seat: SeatDTO): SeatDTO {
  const index = SEATS.indexOf(seat);
  return SEATS[(index + 1) % SEATS.length];
}

function getPlayer(state: GameStateDTO, seat: SeatDTO) {
  return state.players.find((player) => player.seat === seat) ?? state.players[0];
}

function tileLabel(tileValue?: TileDTO | null): string {
  return tileValue?.label ?? "未指定";
}

function refreshState(state: GameStateDTO): void {
  state.remaining_tiles = state.wall.length;
  state.scores = state.players.map((player) => player.score);
  state.hands_concealed = state.players.map((player) => {
    player.hand_counts = countsFromTiles(player.hand);
    return [...player.hand_counts];
  });
  state.open_melds = state.players.map((player) => [...player.melds]);
  state.discards = state.players.map((player) => [...player.discards]);
  state.is_riichi = state.players.map((player) => player.is_riichi);
  state.current_player = SEAT_META[state.current_seat].index;

  for (const player of state.players) {
    const discardPressure = Math.min(2, player.discards.length % 3);
    player.shanten = player.tenpai ? 0 : Math.max(1, 3 - discardPressure);
    if (!player.tenpai) {
      const waitBase = (player.index * 8 + (state.debug?.replay_frame ?? 0) + player.hand.length) % 34;
      player.waits = [createTileDTO(waitBase), createTileDTO((waitBase + 7) % 34)];
    }
    player.legal_actions = player.index === state.current_player ? createLegalActions(state) : emptyLegalActions();
  }

  state.legal_actions = createLegalActions(state);
}

function commit(message: string): EngineSnapshot {
  currentSnapshot.state.debug = currentSnapshot.state.debug ?? {
    connection: "mock",
    illegal_actions: 0,
    replay_frame: 0,
    response_queue: []
  };
  currentSnapshot.state.debug.replay_frame += 1;
  currentSnapshot.state.last_action = message;
  refreshState(currentSnapshot.state);
  return snapshotDTOToUI(clone(currentSnapshot));
}

function actionLabel(action: ActionCommandDTO): string {
  const seat = action.seat ?? seatByIndex(action.actor ?? 0);
  const tileText = action.tile ? ` ${tileLabel(action.tile)}` : "";
  const illegal = action.illegal ? "非法" : "强制";
  const labels: Record<ActionType, string> = {
    draw: "摸牌",
    discard: "打牌",
    chi: "吃",
    pon: "碰",
    kan: "杠",
    ankan: "暗杠",
    kakan: "加杠",
    riichi: "立直",
    win: "和牌",
    pass: "通过",
    tsumo: "自摸",
    ron: "荣和",
    ryuukyoku: "流局",
    nuki: "拔北"
  };
  return `${SEAT_META[seat].label} ${illegal}${labels[action.type]}${tileText}`;
}

function applyForcedAction(action: ActionCommandDTO): EngineSnapshot {
  const state = currentSnapshot.state;
  const seat = action.seat ?? seatByIndex(action.actor ?? state.current_player);
  const player = getPlayer(state, seat);

  if (action.illegal) {
    state.debug!.illegal_actions += 1;
    pushLog("warn", actionLabel(action), action);
  } else {
    pushLog("action", actionLabel(action), action);
  }

  if (action.tile && action.type === "discard") {
    player.discards = [action.tile, ...player.discards].slice(0, 24);
    player.hand = player.hand.filter((candidate) => candidate.abs_id !== action.tile?.abs_id).slice(0, 14);
    state.last_discard = action.tile;
    state.last_discard_by = player.index;
    state.phase = "RESPONSE";
  }

  if (action.tile && action.type === "draw") {
    player.hand = [...player.hand, action.tile].slice(-14);
    state.phase = "DISCARD";
  }

  if (action.tile && ["chi", "pon", "kan"].includes(action.type)) {
    const meldType: MeldType = action.type === "kan" ? "kan" : action.type === "pon" ? "pon" : "chi";
    player.melds = [
      {
        type: meldType,
        tiles: action.meld_tiles?.length ? action.meld_tiles : [action.tile, action.tile, action.tile],
        tile_types: action.meld_tile_types?.length ? action.meld_tile_types : [action.tile.type_id, action.tile.type_id, action.tile.type_id],
        called_from: state.last_discard_by,
        source_tile: action.tile.type_id
      },
      ...player.melds
    ].slice(0, 4);
  }

  if (action.type === "riichi") {
    player.is_riichi = true;
    state.riichi_sticks += 1;
    player.score -= currentSnapshot.rules.riichi_stick_cost;
  }

  if (action.type === "win" || action.type === "tsumo" || action.type === "ron") {
    state.phase = "AGARI";
    state.debug!.last_scoring = {
      han: action.type === "tsumo" ? 4 : 3,
      fu: 30,
      yaku: [
        { name: action.type === "tsumo" ? "门前清自摸和" : "荣和", han: 1 },
        { name: "混合测试役", han: 2 }
      ]
    };
  }

  return commit(actionLabel(action));
}

function replayEncode(snapshot: EngineSnapshotDTO): string {
  const json = JSON.stringify(snapshot);
  return btoa(unescape(encodeURIComponent(json)));
}

function replayDecode(data: string): EngineSnapshotDTO {
  const json = decodeURIComponent(escape(atob(data.trim())));
  return JSON.parse(json) as EngineSnapshotDTO;
}

export const engineAPI = {
  getSnapshot(): EngineSnapshot {
    return snapshotDTOToUI(clone(currentSnapshot));
  },

  async reset(): Promise<EngineSnapshot> {
    currentSnapshot = createInitialSnapshotDTO();
    pushLog("info", "Reset 完成，状态机回到东1局 DRAW_STATE");
    return commit("Reset");
  },

  async step(action?: ActionCommand): Promise<EngineSnapshot> {
    if (action) {
      return applyForcedAction(uiActionToDTO(action));
    }

    const state = currentSnapshot.state;
    const player = getPlayer(state, state.current_seat);

    if (state.phase === "DRAW" || state.phase === "DEAL") {
      const prepared = state.next_draw?.[state.current_seat];
      const drawn = prepared ?? state.wall.shift() ?? tile((state.debug?.replay_frame ?? 0) % 34);
      if (state.next_draw) state.next_draw[state.current_seat] = undefined;
      player.hand = [...player.hand, drawn].slice(-14);
      state.phase = "DISCARD";
      pushLog("action", `${SEAT_META[player.seat].label} 摸 ${tileLabel(drawn)}`);
      return commit(`${SEAT_META[player.seat].label} draw ${drawn.code}`);
    }

    if (state.phase === "DISCARD" || state.phase === "RESPONSE") {
      const discarded = player.hand[player.hand.length - 1] ?? state.wall.shift() ?? tile((state.debug?.replay_frame ?? 0) % 34);
      player.hand = player.hand.slice(0, -1);
      player.discards = [...player.discards, discarded].slice(-24);
      state.last_discard = discarded;
      state.last_discard_by = player.index;
      state.current_seat = nextSeat(state.current_seat);
      state.phase = "DRAW";
      if (state.debug) state.debug.response_queue = [];
      pushLog("action", `${SEAT_META[player.seat].label} 打 ${tileLabel(discarded)}`);
      return commit(`${SEAT_META[player.seat].label} discard ${discarded.code}`);
    }

    if (state.phase === "AGARI" || state.phase === "RYUUKYOKU") {
      state.phase = "ROUND_END";
      pushLog("info", "结算帧推进");
      return commit("Settlement frame advanced");
    }

    state.phase = "DRAW";
    return commit("Phase normalized to DRAW");
  },

  async autoPlay(speed: number): Promise<EngineSnapshot> {
    pushLog("debug", `Auto Play tick，speed=${speed}ms`);
    return this.step();
  },

  async toggleAI(seat: Seat): Promise<EngineSnapshot> {
    const player = getPlayer(currentSnapshot.state, seat);
    player.ai_enabled = !player.ai_enabled;
    pushLog("info", `${SEAT_META[seat].label} AI ${player.ai_enabled ? "接管" : "暂停"}`);
    return commit(`Toggle AI ${seat}`);
  },

  async forceAction(action: Parameters<typeof uiActionToDTO>[0]): Promise<EngineSnapshot> {
    return applyForcedAction(uiActionToDTO(action));
  },

  async setNextDraw(seat: Seat, tileValue: Tile): Promise<EngineSnapshot> {
    currentSnapshot.state.next_draw = currentSnapshot.state.next_draw ?? {};
    currentSnapshot.state.next_draw[seat] = uiTileToDTO(tileValue);
    pushLog("debug", `${SEAT_META[seat].label} 下一摸固定为 ${formatTile(tileValue)}`);
    return commit(`Set next draw ${seat} ${tileValue.code}`);
  },

  async setHand(seat: Seat, tiles: Tile[]): Promise<EngineSnapshot> {
    const player = getPlayer(currentSnapshot.state, seat);
    player.hand = tiles.map(uiTileToDTO).slice(0, 14);
    player.hand_counts = countsFromTiles(player.hand);
    player.tenpai = player.hand.length >= 13;
    pushLog("debug", `${SEAT_META[seat].label} 手牌已替换，共 ${player.hand.length} 张`);
    return commit(`Set hand ${seat}`);
  },

  async injectDiscard(seat: Seat, tileValue: Tile): Promise<EngineSnapshot> {
    const dtoTile = uiTileToDTO(tileValue);
    const player = getPlayer(currentSnapshot.state, seat);
    player.discards = [...player.discards, dtoTile].slice(-24);
    currentSnapshot.state.last_discard = dtoTile;
    currentSnapshot.state.last_discard_by = player.index;
    pushLog("debug", `${SEAT_META[seat].label} 牌河注入 ${formatTile(tileValue)}`);
    return commit(`Inject discard ${seat} ${tileValue.code}`);
  },

  async setScore(seat: Seat, score: number): Promise<EngineSnapshot> {
    getPlayer(currentSnapshot.state, seat).score = score;
    pushLog("debug", `${SEAT_META[seat].label} 分数设为 ${score}`);
    return commit(`Set score ${seat}`);
  },

  async setRound(round: string): Promise<EngineSnapshot> {
    currentSnapshot.state.round = round;
    pushLog("debug", `局面设为 ${round}`);
    return commit("Set round");
  },

  async setHonba(honba: number): Promise<EngineSnapshot> {
    currentSnapshot.state.honba = Math.max(0, honba);
    pushLog("debug", `本场数设为 ${currentSnapshot.state.honba}`);
    return commit("Set honba");
  },

  async updateRules(rules: Parameters<typeof rulesUIPatchToDTO>[0]): Promise<EngineSnapshot> {
    const patch = rulesUIPatchToDTO(rules);
    currentSnapshot.rules = { ...currentSnapshot.rules, ...patch };
    pushLog("info", "规则参数已更新", patch);
    return commit("Update rules");
  },

  async exportReplay(): Promise<string> {
    pushLog("info", "Replay 已导出为 Base64 DTO");
    return replayEncode(currentSnapshot);
  },

  async importReplay(data: string): Promise<EngineSnapshot> {
    currentSnapshot = replayDecode(data);
    pushLog("info", "Replay 导入完成");
    return commit("Import replay");
  },

  async applyScenario(scenario: ScenarioKey): Promise<EngineSnapshot> {
    const state = currentSnapshot.state;
    const east = getPlayer(state, "east");
    const south = getPlayer(state, "south");
    const west = getPlayer(state, "west");

    if (scenario === "kokushi") {
      east.hand = [0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33].map((id) => tile(id));
      east.hand_counts = countsFromTiles(east.hand);
      east.tenpai = true;
      east.shanten = 0;
      east.waits = [tile(0), tile(33)];
      state.current_seat = "east";
      state.phase = "DRAW";
      pushLog("warn", "场景载入：国士无双听牌");
    }

    if (scenario === "suukantsu") {
      west.melds = [0, 9, 18, 27].map((id) => ({
        type: "kan",
        tiles: [tile(id), tile(id), tile(id), tile(id)],
        tile_types: [id, id, id, id]
      }));
      west.tenpai = true;
      west.shanten = 0;
      pushLog("warn", "场景载入：四杠子压力测试");
    }

    if (scenario === "tripleRon") {
      const discard = tile(33);
      state.last_discard = discard;
      state.phase = "RESPONSE";
      state.debug!.response_queue = SEATS.filter((seat) => seat !== state.current_seat).map((seat) => ({
        seat,
        actions: ["ron", "pass"]
      }));
      pushLog("warn", "场景载入：三家和了仲裁");
    }

    if (scenario === "furiten") {
      south.furiten = true;
      south.tenpai = true;
      south.waits = [tile(4, true), tile(7)];
      south.discards = [...south.discards, south.waits[0]];
      pushLog("warn", "场景载入：振听判定");
    }

    if (scenario === "illegalRiichi") {
      state.debug!.illegal_actions += 1;
      south.is_riichi = false;
      pushLog("warn", "场景载入：非法立直", { seat: "south", reason: "not tenpai / insufficient points" });
    }

    if (scenario === "callConflict") {
      const discard = tile(9);
      state.last_discard = discard;
      state.phase = "RESPONSE";
      state.debug!.response_queue = [
        { seat: "south", actions: ["chi", "pass"] },
        { seat: "west", actions: ["pon", "pass"] },
        { seat: "north", actions: ["ron", "pass"] }
      ];
      pushLog("warn", "场景载入：吃/碰/胡并发冲突");
    }

    return commit(`Apply scenario ${scenario}`);
  },

  // 暴露 DTO 便于调试和未来 WebSocket 层复用；UI 默认仍使用 getSnapshot()。
  getContractSnapshot(): EngineSnapshotDTO {
    return clone(currentSnapshot);
  }
};
