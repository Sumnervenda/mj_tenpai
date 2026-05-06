import { Bot, Pause, Play, Radio, RotateCcw, StepForward, Unplug } from "lucide-react";
import { useEffect } from "react";
import { Panel } from "../common/Panel";
import { SEAT_META, SEATS } from "../../types/game";
import { engineAPI } from "../../services/engineAPI";
import { useGameStore } from "../../store/gameStore";

export function FlowControl() {
  const state = useGameStore((store) => store.state);
  const speed = useGameStore((store) => store.speed);
  const autoPlaying = useGameStore((store) => store.autoPlaying);
  const sourceMode = useGameStore((store) => store.sourceMode);
  const liveUrl = useGameStore((store) => store.liveUrl);
  const liveConnected = useGameStore((store) => store.liveConnected);
  const livePaused = useGameStore((store) => store.livePaused);
  const reset = useGameStore((store) => store.reset);
  const step = useGameStore((store) => store.step);
  const toggleAuto = useGameStore((store) => store.toggleAuto);
  const setSpeed = useGameStore((store) => store.setSpeed);
  const toggleAI = useGameStore((store) => store.toggleAI);
  const setLiveUrl = useGameStore((store) => store.setLiveUrl);
  const connectLive = useGameStore((store) => store.connectLive);
  const disconnectLive = useGameStore((store) => store.disconnectLive);
  const toggleLivePause = useGameStore((store) => store.toggleLivePause);
  const liveLocked = sourceMode === "live";

  useEffect(() => {
    if (!autoPlaying || liveLocked) return undefined;
    const timer = window.setInterval(async () => {
      const snapshot = await engineAPI.autoPlay(speed);
      useGameStore.setState({ state: snapshot.state, rules: snapshot.rules, logs: snapshot.logs });
    }, speed);
    return () => window.clearInterval(timer);
  }, [autoPlaying, speed]);

  return (
    <Panel title="Flow Control" eyebrow="STATE MACHINE">
      <div className="control-cluster">
        <button className="tool-button secondary" onClick={reset} disabled={liveLocked}>
          <RotateCcw size={16} /> Reset
        </button>
        <button className="tool-button primary" onClick={() => step()} disabled={liveLocked}>
          <StepForward size={16} /> Step
        </button>
        <button className={`tool-button ${autoPlaying ? "danger" : "secondary"}`} onClick={toggleAuto} disabled={liveLocked}>
          {autoPlaying ? <Pause size={16} /> : <Play size={16} />}
          {autoPlaying ? "Pause" : "Auto"}
        </button>
      </div>

      <label className="field-label" htmlFor="speed">
        Speed <span>{speed} ms</span>
      </label>
      <input
        id="speed"
        className="range"
        min={120}
        max={1200}
        step={40}
        value={speed}
        type="range"
        disabled={liveLocked}
        onChange={(event) => setSpeed(Number(event.target.value))}
      />

      <div className="live-console">
        <label className="field-label" htmlFor="live-url">
          Live SSE
          <span className={`status-pill ${livePaused ? "paused" : liveConnected ? "online" : liveLocked ? "pending" : ""}`}>
            {livePaused ? "Paused" : liveConnected ? "Live" : liveLocked ? "Connecting" : "Mock"}
          </span>
        </label>
        <input
          id="live-url"
          className="field"
          value={liveUrl}
          disabled={liveLocked}
          onChange={(event) => setLiveUrl(event.target.value)}
        />
        <div className="split-actions">
          {liveLocked ? (
            <>
              <button className="tool-button secondary" onClick={toggleLivePause} disabled={!liveConnected}>
                {livePaused ? <Play size={15} /> : <Pause size={15} />}
                {livePaused ? "Resume" : "Pause"}
              </button>
              <button className="tool-button danger" onClick={disconnectLive}>
                <Unplug size={15} /> Disconnect
              </button>
            </>
          ) : (
            <button className="tool-button secondary wide" onClick={() => connectLive(liveUrl)}>
              <Radio size={15} /> Connect Live
            </button>
          )}
        </div>
      </div>

      <div className="state-readout">
        <span>Source</span>
        <strong>{liveLocked ? "LIVE / SSE" : "MOCK"}</strong>
        <span>Run</span>
        <strong>{liveLocked ? livePaused ? "PAUSED" : "RUNNING" : autoPlaying ? "AUTO" : "MANUAL"}</strong>
        <span>Phase</span>
        <strong>{state.phase}</strong>
        <span>Turn</span>
        <strong>{state.turn}</strong>
      </div>

      <div className="seat-toggle-grid">
        {SEATS.map((seat) => {
          const player = state.players.find((entry) => entry.seat === seat);
          return (
            <button
              key={seat}
              className={`seat-toggle ${player?.aiEnabled ? "active" : ""}`}
              onClick={() => toggleAI(seat)}
              disabled={liveLocked}
            >
              <Bot size={15} />
              <span>{SEAT_META[seat].short}</span>
            </button>
          );
        })}
      </div>
    </Panel>
  );
}
