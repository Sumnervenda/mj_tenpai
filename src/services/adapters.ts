import {
  ActionCommand,
  EngineSnapshot,
  GameState,
  LogEntry,
  RuleSettings,
  SEAT_META,
  SEATS,
  Tile,
  TILE_DEFS,
  createActionMask,
  createTile
} from "../types/game";
import {
  ActionCommandDTO,
  EngineSnapshotDTO,
  GameStateDTO,
  LegalActionsDTO,
  LogEntryDTO,
  RuleSettingsDTO,
  SeatDTO,
  TileDTO
} from "../types/contract";

const WIND_TO_SEAT: Record<number, SeatDTO> = {
  27: "east",
  28: "south",
  29: "west",
  30: "north"
};

const RED_CODE_BY_TYPE: Record<number, string> = {
  4: "0m",
  13: "0p",
  22: "0s"
};

export function createTileDTO(typeId: number, options: { absId?: number | null; red?: boolean } = {}): TileDTO {
  const uiTile = createTile(typeId, { absId: options.absId ?? undefined, red: Boolean(options.red) });
  return uiTileToDTO(uiTile);
}

export function tileDTOToUI(tile: TileDTO | null | undefined): Tile | undefined {
  if (!tile) return undefined;
  return {
    typeId: tile.type_id,
    absId: tile.abs_id ?? undefined,
    code: tile.code,
    label: tile.label,
    red: tile.red
  };
}

export function uiTileToDTO(tile: Tile): TileDTO {
  return {
    type_id: tile.typeId,
    abs_id: tile.absId ?? null,
    code: tile.code,
    label: tile.label,
    red: tile.red
  };
}

export function tileCodeForType(typeId: number, red = false): string {
  if (red && RED_CODE_BY_TYPE[typeId]) return RED_CODE_BY_TYPE[typeId];
  return TILE_DEFS[typeId]?.code ?? "unknown";
}

export function countsFromTiles(tiles: TileDTO[]): number[] {
  const counts = Array.from({ length: 34 }, () => 0);
  for (const tile of tiles) {
    if (tile.type_id >= 0 && tile.type_id < counts.length) counts[tile.type_id] += 1;
  }
  return counts;
}

export function expandCounts(counts: number[]): TileDTO[] {
  return counts.flatMap((count, typeId) =>
    Array.from({ length: count }, () => createTileDTO(typeId))
  );
}

export function uiActionToDTO(action: ActionCommand): ActionCommandDTO {
  return {
    type: action.type,
    seat: action.seat,
    actor: SEAT_META[action.seat].index,
    tile: action.tile ? uiTileToDTO(action.tile) : null,
    tile_type: action.tile?.typeId,
    meld_tiles: action.meldTiles?.map(uiTileToDTO),
    meld_tile_types: action.meldTiles?.map((tile) => tile.typeId),
    illegal: action.illegal
  };
}

export function dtoActionToUI(action: ActionCommandDTO, fallbackSeat: SeatDTO): ActionCommand {
  return {
    type: action.type,
    seat: action.seat ?? fallbackSeat,
    tile: tileDTOToUI(action.tile) ?? undefined,
    meldTiles: action.meld_tiles?.map((tile) => tileDTOToUI(tile)).filter((tile): tile is Tile => Boolean(tile)),
    illegal: action.illegal
  };
}

export function rulesDTOToUI(rules: RuleSettingsDTO): RuleSettings {
  return {
    startScore: rules.start_score,
    targetScore: rules.target_score,
    riichiStickCost: rules.riichi_stick_cost,
    honbaBonus: rules.honba_bonus,
    kuitan: rules.kuitan,
    akaDora: rules.aka_dora,
    akadora: rules.akadora,
    ryanhanShibari: rules.ryanhan_shibari,
    kuikae: rules.kuikae,
    atozuke: rules.atozuke,
    openRiichi: rules.open_riichi,
    yakumanMultiple: rules.yakuman_multiple,
    eastOnly: rules.east_only,
    agariYame: rules.agari_yame,
    tenpaiRenchan: rules.tenpai_renchan,
    tobi: rules.tobi,
    wareme: rules.wareme,
    multipleRon: rules.multiple_ron,
    nagashiMangan: rules.nagashi_mangan,
    responsibility: rules.responsibility
  };
}

export function rulesUIPatchToDTO(rules: Partial<RuleSettings>): Partial<RuleSettingsDTO> {
  return {
    start_score: rules.startScore,
    target_score: rules.targetScore,
    riichi_stick_cost: rules.riichiStickCost,
    honba_bonus: rules.honbaBonus,
    kuitan: rules.kuitan,
    aka_dora: rules.akaDora,
    akadora: rules.akadora,
    ryanhan_shibari: rules.ryanhanShibari,
    kuikae: rules.kuikae,
    atozuke: rules.atozuke,
    open_riichi: rules.openRiichi,
    yakuman_multiple: rules.yakumanMultiple,
    east_only: rules.eastOnly,
    agari_yame: rules.agariYame,
    tenpai_renchan: rules.tenpaiRenchan,
    tobi: rules.tobi,
    wareme: rules.wareme,
    multiple_ron: rules.multipleRon,
    nagashi_mangan: rules.nagashiMangan,
    responsibility: rules.responsibility
  };
}

function legalActionsDTOToUI(legal: LegalActionsDTO | undefined, fallbackSeat: SeatDTO) {
  return {
    actions: legal?.actions.map((action) => dtoActionToUI(action, fallbackSeat)) ?? [],
    mask: legal?.mask ?? createActionMask()
  };
}

function logDTOToUI(log: LogEntryDTO): LogEntry {
  return {
    id: log.id,
    time: log.time,
    level: log.level,
    message: log.message,
    payload: log.payload
  };
}

export function gameStateDTOToUI(state: GameStateDTO): GameState {
  const currentSeat = state.current_seat ?? SEATS[state.current_player] ?? "east";
  return {
    players: state.players.map((player) => ({
      seat: player.seat,
      hand: player.hand.map((tile) => tileDTOToUI(tile)).filter((tile): tile is Tile => Boolean(tile)),
      melds: player.melds.map((meld) => ({
        type: meld.type,
        tiles: meld.tiles.map((tile) => tileDTOToUI(tile)).filter((tile): tile is Tile => Boolean(tile)),
        from: typeof meld.called_from === "number" && meld.called_from >= 0 ? SEATS[meld.called_from] : undefined
      })),
      discards: player.discards.map((tile) => tileDTOToUI(tile)).filter((tile): tile is Tile => Boolean(tile)),
      score: player.score,
      shanten: player.shanten ?? (player.tenpai ? 0 : 2),
      tenpai: player.tenpai,
      waits: player.waits.map((tile) => tileDTOToUI(tile)).filter((tile): tile is Tile => Boolean(tile)),
      aiEnabled: Boolean(player.ai_enabled),
      riichi: player.is_riichi,
      furiten: player.furiten
    })),
    wall: state.wall.map((tile) => tileDTOToUI(tile)).filter((tile): tile is Tile => Boolean(tile)),
    deadWall: state.dead_wall.map((tile) => tileDTOToUI(tile)).filter((tile): tile is Tile => Boolean(tile)),
    doraIndicators: state.dora_indicators.map((tile) => tileDTOToUI(tile)).filter((tile): tile is Tile => Boolean(tile)),
    uraDoraIndicators: state.ura_dora_indicators.map((tile) => tileDTOToUI(tile)).filter((tile): tile is Tile => Boolean(tile)),
    turn: state.debug?.replay_frame ?? 0,
    phase: state.phase,
    currentSeat,
    round: state.round,
    roundWind: WIND_TO_SEAT[state.round_wind] ?? "east",
    roundNumber: state.round_number,
    honba: state.honba,
    riichiSticks: state.riichi_sticks,
    lastAction: state.last_action,
    lastDiscard: tileDTOToUI(state.last_discard) ?? undefined,
    nextDraw: Object.fromEntries(
      Object.entries(state.next_draw ?? {}).map(([seat, tile]) => [seat, tileDTOToUI(tile)])
    ) as GameState["nextDraw"],
    legalActions: legalActionsDTOToUI(state.legal_actions, currentSeat),
    debug: {
      connection: state.debug?.connection ?? "mock",
      illegalActions: state.debug?.illegal_actions ?? 0,
      replayFrame: state.debug?.replay_frame ?? 0,
      responseQueue: state.debug?.response_queue ?? [],
      paused: state.debug?.paused,
      lastScoring: state.debug?.last_scoring
    }
  };
}

export function snapshotDTOToUI(snapshot: EngineSnapshotDTO): EngineSnapshot {
  return {
    state: gameStateDTOToUI(snapshot.state),
    rules: rulesDTOToUI(snapshot.rules),
    logs: snapshot.logs.map(logDTOToUI)
  };
}
