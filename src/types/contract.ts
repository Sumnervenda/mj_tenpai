import { ActionType, LogLevel, MeldType, ScenarioKey, Seat } from "./game";

export type SeatDTO = Seat;
export type GamePhaseDTO =
  | "DEAL"
  | "DRAW"
  | "DISCARD"
  | "RESPONSE"
  | "AGARI"
  | "RYUUKYOKU"
  | "ROUND_END"
  | "GAME_END";

export interface TileDTO {
  type_id: number;
  abs_id?: number | null;
  code: string;
  label: string;
  red?: boolean;
}

export interface MeldDTO {
  type: MeldType;
  tiles: TileDTO[];
  tile_types?: number[];
  called_from?: number;
  source_tile?: number;
}

export interface ActionCommandDTO {
  type: ActionType;
  action_type?: number;
  seat?: SeatDTO;
  actor?: number;
  tile?: TileDTO | null;
  tile_type?: number;
  meld_tiles?: TileDTO[];
  meld_tile_types?: number[];
  illegal?: boolean;
  label?: string;
}

export interface LegalActionsDTO {
  actions: ActionCommandDTO[];
  mask: number[];
}

export interface PlayerStateDTO {
  index: number;
  seat: SeatDTO;
  seat_label: string;
  seat_wind: number;
  score: number;
  hand_counts: number[];
  hand: TileDTO[];
  melds: MeldDTO[];
  discards: TileDTO[];
  is_riichi: boolean;
  is_double_riichi?: boolean;
  is_ippatsu?: boolean;
  has_won?: boolean;
  tenpai: boolean;
  shanten?: number;
  waits: TileDTO[];
  furiten: boolean;
  ai_enabled?: boolean;
  is_tenpai_at_ryuukyoku?: boolean;
  legal_actions?: LegalActionsDTO;
}

export interface ScoringDetailDTO {
  han: number;
  fu: number;
  yaku: Array<{ name: string; han: number }>;
}

export interface DebugStateDTO {
  connection: "mock" | "websocket" | "sse";
  illegal_actions: number;
  replay_frame: number;
  response_queue: Array<{ seat: SeatDTO; actions: ActionType[] }>;
  paused?: boolean;
  last_scoring?: ScoringDetailDTO;
}

export interface GameStateDTO {
  phase: GamePhaseDTO;
  phase_id?: number;
  current_player: number;
  current_seat: SeatDTO;
  round_wind: number;
  round_wind_label?: string;
  round_number: number;
  round: string;
  honba: number;
  riichi_sticks: number;
  scores: number[];
  hands_concealed: number[][];
  players: PlayerStateDTO[];
  open_melds: MeldDTO[][];
  discards: TileDTO[][];
  wall: TileDTO[];
  dead_wall: TileDTO[];
  dora_indicators: TileDTO[];
  ura_dora_indicators: TileDTO[];
  is_riichi: boolean[];
  last_action?: string;
  last_discard?: TileDTO | null;
  last_discard_by: number;
  remaining_tiles: number;
  next_draw?: Partial<Record<SeatDTO, TileDTO | undefined>>;
  legal_actions: LegalActionsDTO;
  rewards: number[];
  done: boolean;
  debug?: DebugStateDTO;
}

export interface RuleSettingsDTO {
  start_score: number;
  target_score: number;
  riichi_stick_cost: number;
  honba_bonus: number;
  kuitan: boolean;
  aka_dora: boolean;
  akadora: number;
  ryanhan_shibari: boolean;
  kuikae: boolean;
  atozuke: boolean;
  open_riichi: boolean;
  uma: [number, number, number, number] | number[];
  oka: number;
  yakuman_multiple: boolean;
  rounds: number;
  east_only: boolean;
  agari_yame: boolean;
  tenpai_renchan: boolean;
  tobi: boolean;
  wareme: boolean;
  multiple_ron: boolean;
  use_red_dora: boolean;
  nagashi_mangan: boolean;
  responsibility: boolean;
}

export interface LogEntryDTO {
  id: string;
  time: string;
  level: LogLevel;
  message: string;
  payload?: unknown;
}

export interface EngineSnapshotDTO {
  state: GameStateDTO;
  rules: RuleSettingsDTO;
  logs: LogEntryDTO[];
}

export type ScenarioKeyDTO = ScenarioKey;
