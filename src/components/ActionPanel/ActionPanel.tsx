import { FlaskConical, Send } from "lucide-react";
import { useState } from "react";
import { Panel } from "../common/Panel";
import { TileSelect } from "../common/TileView";
import { ActionType, SEAT_META, SEATS, ScenarioKey } from "../../types/game";
import { useGameStore } from "../../store/gameStore";

const actionTypes: Array<{ type: ActionType; label: string }> = [
  { type: "draw", label: "摸牌" },
  { type: "discard", label: "打牌" },
  { type: "chi", label: "吃" },
  { type: "pon", label: "碰" },
  { type: "kan", label: "杠" },
  { type: "riichi", label: "立直" },
  { type: "win", label: "和牌" }
];

const scenarios: Array<{ key: ScenarioKey; label: string }> = [
  { key: "kokushi", label: "国士无双" },
  { key: "suukantsu", label: "四杠子" },
  { key: "tripleRon", label: "三家和了" },
  { key: "furiten", label: "振听" },
  { key: "illegalRiichi", label: "非法立直" },
  { key: "callConflict", label: "吃碰胡冲突" }
];

export function ActionPanel() {
  const selectedSeat = useGameStore((store) => store.selectedSeat);
  const selectedTile = useGameStore((store) => store.selectedTile);
  const setSelectedSeat = useGameStore((store) => store.setSelectedSeat);
  const setSelectedTile = useGameStore((store) => store.setSelectedTile);
  const forceAction = useGameStore((store) => store.forceAction);
  const applyScenario = useGameStore((store) => store.applyScenario);
  const readonly = useGameStore((store) => store.sourceMode === "live");
  const [type, setType] = useState<ActionType>("discard");
  const [illegal, setIllegal] = useState(false);

  return (
    <Panel title="Action Injection" eyebrow="FORCE API">
      <div className="form-stack">
        <label>
          Seat
          <select className="field" value={selectedSeat} disabled={readonly} onChange={(event) => setSelectedSeat(event.target.value as typeof selectedSeat)}>
            {SEATS.map((seat) => (
              <option key={seat} value={seat}>{SEAT_META[seat].label}</option>
            ))}
          </select>
        </label>

        <label>
          Action
          <select className="field" value={type} disabled={readonly} onChange={(event) => setType(event.target.value as ActionType)}>
            {actionTypes.map((item) => (
              <option key={item.type} value={item.type}>{item.label}</option>
            ))}
          </select>
        </label>

        <label>
          Tile
          <TileSelect value={selectedTile} onChange={setSelectedTile} />
        </label>

        <label className="switch-row compact">
          <span>非法测试</span>
          <input type="checkbox" checked={illegal} disabled={readonly} onChange={(event) => setIllegal(event.target.checked)} />
        </label>

        <button className="tool-button primary wide" disabled={readonly} onClick={() => forceAction({ type, seat: selectedSeat, tile: selectedTile, illegal })}>
          <Send size={16} /> Inject
        </button>
      </div>

      <div className="scenario-grid">
        {scenarios.map((scenario) => (
          <button key={scenario.key} className="scenario-button" disabled={readonly} onClick={() => applyScenario(scenario.key)}>
            <FlaskConical size={14} /> {scenario.label}
          </button>
        ))}
      </div>
    </Panel>
  );
}
