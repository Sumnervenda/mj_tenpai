import { EngineSnapshotDTO } from "../types/contract";

type SnapshotHandler = (snapshot: EngineSnapshotDTO) => void;
type ErrorHandler = (error: unknown) => void;

export interface LiveControlStatus {
  ok: boolean;
  version: number;
  done: boolean;
  paused: boolean;
}

let source: EventSource | null = null;

function normalizeError(error: unknown): Error {
  if (error instanceof Error) return error;
  const normalized = new Error("Live SSE 连接异常，请确认 python main.py --live-console 正在运行。");
  normalized.name = "LiveSSEError";
  return normalized;
}

function controlUrl(streamUrl: string, endpoint: "pause" | "resume" | "toggle-pause"): string {
  const url = new URL(streamUrl, window.location.href);
  url.pathname = `/${endpoint}`;
  url.search = "";
  url.hash = "";
  return url.toString();
}

async function requestControl(streamUrl: string, endpoint: "pause" | "resume" | "toggle-pause"): Promise<LiveControlStatus> {
  const response = await fetch(controlUrl(streamUrl, endpoint), { method: "POST" });
  if (!response.ok) {
    throw new Error(`Live 控制请求失败：HTTP ${response.status}`);
  }
  return response.json() as Promise<LiveControlStatus>;
}

export const liveAPI = {
  connect(url: string, onSnapshot: SnapshotHandler, onError: ErrorHandler): void {
    this.disconnect();

    if (typeof EventSource === "undefined") {
      onError(new Error("当前浏览器不支持 EventSource，无法连接 Live SSE。"));
      return;
    }

    const eventSource = new EventSource(url);
    source = eventSource;

    eventSource.addEventListener("snapshot", (event) => {
      try {
        const payload = JSON.parse((event as MessageEvent<string>).data) as EngineSnapshotDTO;
        onSnapshot(payload);
      } catch (error) {
        onError(normalizeError(error));
      }
    });

    eventSource.onerror = (event) => {
      if (source === eventSource) {
        eventSource.close();
        source = null;
      }
      onError(normalizeError(event));
    };
  },

  disconnect(): void {
    if (!source) return;
    source.close();
    source = null;
  },

  pause(streamUrl: string): Promise<LiveControlStatus> {
    return requestControl(streamUrl, "pause");
  },

  resume(streamUrl: string): Promise<LiveControlStatus> {
    return requestControl(streamUrl, "resume");
  },

  togglePause(streamUrl: string): Promise<LiveControlStatus> {
    return requestControl(streamUrl, "toggle-pause");
  }
};
