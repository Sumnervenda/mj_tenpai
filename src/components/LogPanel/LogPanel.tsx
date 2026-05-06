import { Download, Upload } from "lucide-react";
import { Panel } from "../common/Panel";
import { useGameStore } from "../../store/gameStore";

export function LogPanel() {
  const state = useGameStore((store) => store.state);
  const logs = useGameStore((store) => store.logs);
  const replayBuffer = useGameStore((store) => store.replayBuffer);
  const setReplayBuffer = useGameStore((store) => store.setReplayBuffer);
  const exportReplay = useGameStore((store) => store.exportReplay);
  const importReplay = useGameStore((store) => store.importReplay);
  const readonly = useGameStore((store) => store.sourceMode === "live");
  const rawJson = JSON.stringify(state, null, 2);

  return (
    <Panel title="Logs & Debug" eyebrow="TRACE">
      <div className="scoring-box">
        <span>算番详情</span>
        <strong>{state.debug.lastScoring?.han ?? 0} han / {state.debug.lastScoring?.fu ?? 0} fu</strong>
        <div>
          {state.debug.lastScoring?.yaku.map((entry) => (
            <mark key={entry.name}>{entry.name} +{entry.han}</mark>
          ))}
        </div>
      </div>

      <div className="log-list">
        {logs.map((log) => (
          <article key={log.id} className={`log-row level-${log.level}`}>
            <time>{log.time}</time>
            <span>{log.level}</span>
            <p>{log.message}</p>
          </article>
        ))}
      </div>

      <div className="replay-tools">
        <textarea
          className="field textarea"
          value={replayBuffer}
          onChange={(event) => setReplayBuffer(event.target.value)}
          placeholder="Replay Base64"
          disabled={readonly}
        />
        <div className="split-actions">
          <button className="tool-button secondary" disabled={readonly} onClick={exportReplay}>
            <Download size={15} /> Export
          </button>
          <button className="tool-button secondary" disabled={readonly} onClick={importReplay}>
            <Upload size={15} /> Import
          </button>
        </div>
      </div>

      <details className="json-details" open>
        <summary>Raw State JSON</summary>
        <pre>{rawJson}</pre>
      </details>
    </Panel>
  );
}
