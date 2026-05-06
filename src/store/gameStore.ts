import { create } from "zustand";
import { engineAPI } from "../services/engineAPI";
import { liveAPI } from "../services/liveAPI";
import { snapshotDTOToUI } from "../services/adapters";
import { EngineSnapshotDTO } from "../types/contract";
import {
  ActionCommand,
  EngineSnapshot,
  LogEntry,
  RuleSettings,
  ScenarioKey,
  Seat,
  Tile,
  createTile
} from "../types/game";

type SourceMode = "mock" | "live";

interface GameStore extends EngineSnapshot {
  sourceMode: SourceMode;
  liveUrl: string;
  liveConnected: boolean;
  livePaused: boolean;
  speed: number;
  autoPlaying: boolean;
  selectedSeat: Seat;
  selectedTile: Tile;
  replayBuffer: string;
  reset: () => Promise<void>;
  step: (action?: ActionCommand) => Promise<void>;
  toggleAuto: () => void;
  setSpeed: (speed: number) => void;
  toggleAI: (seat: Seat) => Promise<void>;
  forceAction: (action: ActionCommand) => Promise<void>;
  setNextDraw: (seat: Seat, tile: Tile) => Promise<void>;
  setHand: (seat: Seat, tiles: Tile[]) => Promise<void>;
  injectDiscard: (seat: Seat, tile: Tile) => Promise<void>;
  setScore: (seat: Seat, score: number) => Promise<void>;
  setRound: (round: string) => Promise<void>;
  setHonba: (honba: number) => Promise<void>;
  updateRules: (rules: Partial<RuleSettings>) => Promise<void>;
  exportReplay: () => Promise<void>;
  importReplay: () => Promise<void>;
  applyScenario: (scenario: ScenarioKey) => Promise<void>;
  connectLive: (url?: string) => void;
  disconnectLive: () => void;
  pauseLive: () => Promise<void>;
  resumeLive: () => Promise<void>;
  toggleLivePause: () => Promise<void>;
  applyContractSnapshot: (snapshot: EngineSnapshotDTO) => void;
  setLiveUrl: (url: string) => void;
  setSelectedSeat: (seat: Seat) => void;
  setSelectedTile: (tile: Tile) => void;
  setReplayBuffer: (value: string) => void;
}

const initialSnapshot = engineAPI.getSnapshot();

function snapshotPatch(snapshot: EngineSnapshot): Pick<GameStore, "state" | "rules" | "logs"> {
  return {
    state: snapshot.state,
    rules: snapshot.rules,
    logs: snapshot.logs
  };
}

function errorLog(error: unknown): LogEntry {
  return {
    id: `client-error-${Date.now()}`,
    time: new Date().toLocaleTimeString("zh-CN", { hour12: false }),
    level: "error",
    message: error instanceof Error ? error.message : "未知前端错误",
    payload: error
  };
}

function clientLog(level: LogEntry["level"], message: string, payload?: unknown): LogEntry {
  return {
    id: `client-${Date.now()}-${Math.random().toString(16).slice(2)}`,
    time: new Date().toLocaleTimeString("zh-CN", { hour12: false }),
    level,
    message,
    payload
  };
}

function prependLog(logs: LogEntry[], log: LogEntry): LogEntry[] {
  return [log, ...logs].slice(0, 80);
}

export const useGameStore = create<GameStore>((set, get) => {
  const blockLiveMutation = (actionName: string): boolean => {
    if (get().sourceMode !== "live") return false;
    set((state) => ({
      logs: prependLog(
        state.logs,
        clientLog("warn", `Live 只读模式：${actionName} 仅在 Mock 模式可用，请先断开 Live。`)
      )
    }));
    return true;
  };

  return ({
    ...initialSnapshot,
    sourceMode: "mock",
    liveUrl: "http://127.0.0.1:8765/stream",
    liveConnected: false,
    livePaused: false,
    speed: 520,
    autoPlaying: false,
    selectedSeat: "east",
    selectedTile: createTile(0),
    replayBuffer: "",

  reset: async () => {
    if (blockLiveMutation("Reset")) return;
    const snapshot = await engineAPI.reset();
    set(snapshotPatch(snapshot));
  },

  step: async (action) => {
    if (blockLiveMutation("Step")) return;
    const snapshot = await engineAPI.step(action);
    set(snapshotPatch(snapshot));
  },

  toggleAuto: () => {
    if (blockLiveMutation("Auto Play")) return;
    set((state) => ({ autoPlaying: !state.autoPlaying }));
  },

  setSpeed: (speed) => set({ speed }),

  toggleAI: async (seat) => {
    if (blockLiveMutation("Toggle AI")) return;
    const snapshot = await engineAPI.toggleAI(seat);
    set(snapshotPatch(snapshot));
  },

  forceAction: async (action) => {
    if (blockLiveMutation("Action Injection")) return;
    const snapshot = await engineAPI.forceAction(action);
    set(snapshotPatch(snapshot));
  },

  setNextDraw: async (seat, tile) => {
    if (blockLiveMutation("Set Next Draw")) return;
    const snapshot = await engineAPI.setNextDraw(seat, tile);
    set(snapshotPatch(snapshot));
  },

  setHand: async (seat, tiles) => {
    if (blockLiveMutation("Set Hand")) return;
    const snapshot = await engineAPI.setHand(seat, tiles);
    set(snapshotPatch(snapshot));
  },

  injectDiscard: async (seat, tile) => {
    if (blockLiveMutation("Inject Discard")) return;
    const snapshot = await engineAPI.injectDiscard(seat, tile);
    set(snapshotPatch(snapshot));
  },

  setScore: async (seat, score) => {
    if (blockLiveMutation("Set Score")) return;
    const snapshot = await engineAPI.setScore(seat, score);
    set(snapshotPatch(snapshot));
  },

  setRound: async (round) => {
    if (blockLiveMutation("Set Round")) return;
    const snapshot = await engineAPI.setRound(round);
    set(snapshotPatch(snapshot));
  },

  setHonba: async (honba) => {
    if (blockLiveMutation("Set Honba")) return;
    const snapshot = await engineAPI.setHonba(honba);
    set(snapshotPatch(snapshot));
  },

  updateRules: async (rules) => {
    if (blockLiveMutation("Rule Settings")) return;
    const snapshot = await engineAPI.updateRules(rules);
    set(snapshotPatch(snapshot));
  },

  exportReplay: async () => {
    if (blockLiveMutation("Export Replay")) return;
    const replay = await engineAPI.exportReplay();
    const snapshot = engineAPI.getSnapshot();
    set({ ...snapshotPatch(snapshot), replayBuffer: replay });
  },

  importReplay: async () => {
    if (blockLiveMutation("Import Replay")) return;
    try {
      const snapshot = await engineAPI.importReplay(get().replayBuffer);
      set(snapshotPatch(snapshot));
    } catch (error) {
      set((state) => ({ logs: [errorLog(error), ...state.logs] }));
    }
  },

  applyScenario: async (scenario) => {
    if (blockLiveMutation("Scenario Load")) return;
    const snapshot = await engineAPI.applyScenario(scenario);
    set(snapshotPatch(snapshot));
  },

  connectLive: (url) => {
    const liveUrl = (url ?? get().liveUrl).trim() || "http://127.0.0.1:8765/stream";
    set((state) => ({
      sourceMode: "live",
      liveUrl,
      liveConnected: false,
      livePaused: false,
      autoPlaying: false,
      logs: prependLog(state.logs, clientLog("info", `连接 Live SSE：${liveUrl}`))
    }));
    liveAPI.connect(
      liveUrl,
      (snapshot) => get().applyContractSnapshot(snapshot),
      (error) => set((state) => ({
        liveConnected: false,
        logs: prependLog(state.logs, errorLog(error))
      }))
    );
  },

  disconnectLive: () => {
    liveAPI.disconnect();
    set((state) => ({
      sourceMode: "mock",
      liveConnected: false,
      livePaused: false,
      autoPlaying: false,
      logs: prependLog(state.logs, clientLog("info", "已断开 Live SSE，Mock 控制恢复可写。"))
    }));
  },

  pauseLive: async () => {
    if (get().sourceMode !== "live") return;
    try {
      const status = await liveAPI.pause(get().liveUrl);
      set((state) => ({
        livePaused: status.paused,
        logs: prependLog(state.logs, clientLog("info", "已暂停 Python Live 对局。"))
      }));
    } catch (error) {
      set((state) => ({ logs: prependLog(state.logs, errorLog(error)) }));
    }
  },

  resumeLive: async () => {
    if (get().sourceMode !== "live") return;
    try {
      const status = await liveAPI.resume(get().liveUrl);
      set((state) => ({
        livePaused: status.paused,
        logs: prependLog(state.logs, clientLog("info", "已继续 Python Live 对局。"))
      }));
    } catch (error) {
      set((state) => ({ logs: prependLog(state.logs, errorLog(error)) }));
    }
  },

  toggleLivePause: async () => {
    if (get().sourceMode !== "live") return;
    try {
      const status = await liveAPI.togglePause(get().liveUrl);
      set((state) => ({
        livePaused: status.paused,
        logs: prependLog(
          state.logs,
          clientLog("info", status.paused ? "已暂停 Python Live 对局。" : "已继续 Python Live 对局。")
        )
      }));
    } catch (error) {
      set((state) => ({ logs: prependLog(state.logs, errorLog(error)) }));
    }
  },

  applyContractSnapshot: (snapshot) => {
    const uiSnapshot = snapshotDTOToUI(snapshot);
    set({
      ...snapshotPatch(uiSnapshot),
      sourceMode: "live",
      liveConnected: true,
      livePaused: Boolean(snapshot.state.debug?.paused),
      autoPlaying: false
    });
  },

  setLiveUrl: (url) => set({ liveUrl: url }),
  setSelectedSeat: (seat) => set({ selectedSeat: seat }),
  setSelectedTile: (tile) => set({ selectedTile: tile }),
  setReplayBuffer: (value) => set({ replayBuffer: value })
  });
});
