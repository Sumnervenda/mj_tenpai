export type Seat = "east" | "south" | "west" | "north";
export type TileTypeId = number;
export type TileId = number;

export const SEATS: Seat[] = ["east", "south", "west", "north"];

export const SEAT_META: Record<Seat, { label: string; short: string; index: number }> = {
  east: { label: "东家", short: "东", index: 0 },
  south: { label: "南家", short: "南", index: 1 },
  west: { label: "西家", short: "西", index: 2 },
  north: { label: "北家", short: "北", index: 3 }
};

export type GamePhase =
  | "DEAL"
  | "DRAW"
  | "DISCARD"
  | "RESPONSE"
  | "AGARI"
  | "RYUUKYOKU"
  | "ROUND_END"
  | "GAME_END";

export type ActionType =
  | "draw"
  | "discard"
  | "chi"
  | "pon"
  | "kan"
  | "ankan"
  | "kakan"
  | "riichi"
  | "win"
  | "pass"
  | "tsumo"
  | "ron"
  | "ryuukyoku"
  | "nuki";

export type MeldType = "chi" | "pon" | "kan" | "ankan" | "kakan";

export interface Tile {
  typeId: TileTypeId;
  absId?: TileId;
  code: string;
  label: string;
  red?: boolean;
}

export interface Meld {
  type: MeldType;
  tiles: Tile[];
  from?: Seat;
}

export interface PlayerState {
  seat: Seat;
  hand: Tile[];
  melds: Meld[];
  discards: Tile[];
  score: number;
  shanten: number;
  tenpai: boolean;
  waits: Tile[];
  aiEnabled: boolean;
  riichi: boolean;
  furiten: boolean;
}

export interface LegalActions {
  actions: ActionCommand[];
  mask: number[];
}

export interface ScoringDetail {
  han: number;
  fu: number;
  yaku: Array<{ name: string; han: number }>;
}

export interface DebugState {
  connection: "mock" | "websocket" | "sse";
  illegalActions: number;
  replayFrame: number;
  responseQueue: Array<{ seat: Seat; actions: ActionType[] }>;
  paused?: boolean;
  lastScoring?: ScoringDetail;
}

export interface GameState {
  players: PlayerState[];
  wall: Tile[];
  deadWall: Tile[];
  doraIndicators: Tile[];
  uraDoraIndicators: Tile[];
  turn: number;
  phase: GamePhase;
  currentSeat: Seat;
  round: string;
  roundWind: Seat;
  roundNumber: number;
  honba: number;
  riichiSticks: number;
  lastAction?: string;
  lastDiscard?: Tile;
  nextDraw: Partial<Record<Seat, Tile>>;
  legalActions: LegalActions;
  debug: DebugState;
}

export interface RuleSettings {
  startScore: number;
  targetScore: number;
  riichiStickCost: number;
  honbaBonus: number;
  kuitan: boolean;
  akaDora: boolean;
  akadora: number;
  ryanhanShibari: boolean;
  kuikae: boolean;
  atozuke: boolean;
  openRiichi: boolean;
  yakumanMultiple: boolean;
  eastOnly: boolean;
  agariYame: boolean;
  tenpaiRenchan: boolean;
  tobi: boolean;
  wareme: boolean;
  multipleRon: boolean;
  nagashiMangan: boolean;
  responsibility: boolean;
}

export interface ActionCommand {
  type: ActionType;
  seat: Seat;
  tile?: Tile;
  meldTiles?: Tile[];
  illegal?: boolean;
}

export type LogLevel = "info" | "action" | "warn" | "error" | "debug";

export interface LogEntry {
  id: string;
  time: string;
  level: LogLevel;
  message: string;
  payload?: unknown;
}

export interface EngineSnapshot {
  state: GameState;
  rules: RuleSettings;
  logs: LogEntry[];
}

export type ScenarioKey =
  | "kokushi"
  | "suukantsu"
  | "tripleRon"
  | "furiten"
  | "illegalRiichi"
  | "callConflict";

export const TILE_DEFS: Array<{ typeId: TileTypeId; code: string; label: string; suit: string }> = [
  ...Array.from({ length: 9 }, (_, i) => ({ typeId: i, code: `${i + 1}m`, label: `${i + 1}万`, suit: "万" })),
  ...Array.from({ length: 9 }, (_, i) => ({ typeId: i + 9, code: `${i + 1}p`, label: `${i + 1}筒`, suit: "筒" })),
  ...Array.from({ length: 9 }, (_, i) => ({ typeId: i + 18, code: `${i + 1}s`, label: `${i + 1}索`, suit: "索" })),
  { typeId: 27, code: "E", label: "东", suit: "字" },
  { typeId: 28, code: "S", label: "南", suit: "字" },
  { typeId: 29, code: "W", label: "西", suit: "字" },
  { typeId: 30, code: "N", label: "北", suit: "字" },
  { typeId: 31, code: "white", label: "白", suit: "字" },
  { typeId: 32, code: "haku", label: "发", suit: "字" },
  { typeId: 33, code: "zhong", label: "中", suit: "字" }
];

const RED_CODE_BY_TYPE: Record<number, string> = {
  4: "0m",
  13: "0p",
  22: "0s"
};

export function createTile(typeId: TileTypeId, options: { absId?: TileId; red?: boolean } = {}): Tile {
  const def = TILE_DEFS[typeId] ?? TILE_DEFS[0];
  const red = Boolean(options.red && RED_CODE_BY_TYPE[typeId]);

  return {
    typeId,
    absId: options.absId,
    code: red ? RED_CODE_BY_TYPE[typeId] : def.code,
    label: red ? `赤${def.label}` : def.label,
    red
  };
}

export function tileFromCode(code: string): Tile | undefined {
  const normalized = code.trim();
  if (!normalized) return undefined;

  if (normalized === "0m") return createTile(4, { red: true });
  if (normalized === "0p") return createTile(13, { red: true });
  if (normalized === "0s") return createTile(22, { red: true });

  const def = TILE_DEFS.find((tile) => tile.code.toLowerCase() === normalized.toLowerCase());
  return def ? createTile(def.typeId) : undefined;
}

export function parseTileInput(input: string): Tile[] {
  return input
    .split(/[\s,，]+/)
    .map(tileFromCode)
    .filter((tile): tile is Tile => Boolean(tile));
}

export function formatTile(tile?: Tile): string {
  return tile ? tile.label : "未指定";
}

export function createActionMask(enabled: number[] = []): number[] {
  const mask = Array.from({ length: 77 }, () => 0);
  for (const index of enabled) {
    if (index >= 0 && index < mask.length) mask[index] = 1;
  }
  return mask;
}
