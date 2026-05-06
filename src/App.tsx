import { Activity, Database, Radio, ShieldCheck } from "lucide-react";
import { ActionPanel } from "./components/ActionPanel/ActionPanel";
import { CheatPanel } from "./components/CheatPanel/CheatPanel";
import { FlowControl } from "./components/FlowControl/FlowControl";
import { LogPanel } from "./components/LogPanel/LogPanel";
import { MahjongTable } from "./components/MahjongTable/MahjongTable";
import { RuleSettings } from "./components/RuleSettings/RuleSettings";
import { StateInspector } from "./components/StateInspector/StateInspector";
import { useGameStore } from "./store/gameStore";

export default function App() {
  const state = useGameStore((store) => store.state);
  const sourceMode = useGameStore((store) => store.sourceMode);
  const liveConnected = useGameStore((store) => store.liveConnected);
  const livePaused = useGameStore((store) => store.livePaused);

  return (
    <main className="console-shell">
      <header className="topbar">
        <div className="brand-block">
          <span className="brand-mark">MJ</span>
          <div>
            <h1>Mahjong Engine Console</h1>
            <p>状态驱动 / 可复现 / 可破坏的日麻引擎调试台</p>
          </div>
        </div>

        <div className="status-strip" aria-label="engine status">
          <span><Radio size={15} /> {sourceMode === "live" ? livePaused ? "Paused" : liveConnected ? "Live SSE" : "Live..." : "Mock"}</span>
          <span><Activity size={15} /> {state.phase}</span>
          <span><Database size={15} /> Wall {state.wall.length}</span>
          <span><ShieldCheck size={15} /> Illegal {state.debug.illegalActions}</span>
        </div>
      </header>

      <section className="console-grid">
        <aside className="rail rail-left">
          <FlowControl />
          <RuleSettings />
          <ActionPanel />
        </aside>

        <section className="stage">
          <MahjongTable />
          <StateInspector />
        </section>

        <aside className="rail rail-right">
          <CheatPanel />
          <LogPanel />
        </aside>
      </section>
    </main>
  );
}
