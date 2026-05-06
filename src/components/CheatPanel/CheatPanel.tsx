import { Crosshair, PencilRuler } from "lucide-react";
import { useMemo, useState } from "react";
import { Panel } from "../common/Panel";
import { TileSelect } from "../common/TileView";
import { SEAT_META, SEATS, parseTileInput } from "../../types/game";
import { useGameStore } from "../../store/gameStore";

export function CheatPanel() {
  const state = useGameStore((store) => store.state);
  const selectedSeat = useGameStore((store) => store.selectedSeat);
  const selectedTile = useGameStore((store) => store.selectedTile);
  const setSelectedSeat = useGameStore((store) => store.setSelectedSeat);
  const setSelectedTile = useGameStore((store) => store.setSelectedTile);
  const setNextDraw = useGameStore((store) => store.setNextDraw);
  const injectDiscard = useGameStore((store) => store.injectDiscard);
  const setHand = useGameStore((store) => store.setHand);
  const setScore = useGameStore((store) => store.setScore);
  const setRound = useGameStore((store) => store.setRound);
  const setHonba = useGameStore((store) => store.setHonba);
  const readonly = useGameStore((store) => store.sourceMode === "live");
  const player = useMemo(() => state.players.find((entry) => entry.seat === selectedSeat), [selectedSeat, state.players]);
  const [score, setScoreDraft] = useState(player?.score ?? 25000);
  const [round, setRoundDraft] = useState(state.round);
  const [honba, setHonbaDraft] = useState(state.honba);
  const [handText, setHandText] = useState("1m 9m 1p 9p 1s 9s E S W N white haku zhong");

  return (
    <Panel title="Cheat Panel" eyebrow="GOD MODE">
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
          Tile
          <TileSelect value={selectedTile} onChange={setSelectedTile} />
        </label>

        <div className="split-actions">
          <button className="tool-button secondary" disabled={readonly} onClick={() => setNextDraw(selectedSeat, selectedTile)}>
            <Crosshair size={15} /> Next Draw
          </button>
          <button className="tool-button secondary" disabled={readonly} onClick={() => injectDiscard(selectedSeat, selectedTile)}>
            <PencilRuler size={15} /> Discard
          </button>
        </div>

        <label>
          Hand Editor
          <textarea className="field textarea compact-textarea" value={handText} disabled={readonly} onChange={(event) => setHandText(event.target.value)} />
        </label>
        <button className="tool-button primary wide" disabled={readonly} onClick={() => setHand(selectedSeat, parseTileInput(handText))}>
          Set Hand
        </button>

        <div className="numeric-grid">
          <label>
            Score
            <input className="field" type="number" value={score} disabled={readonly} onChange={(event) => setScoreDraft(Number(event.target.value))} />
          </label>
          <label>
            Honba
            <input className="field" type="number" value={honba} disabled={readonly} onChange={(event) => setHonbaDraft(Number(event.target.value))} />
          </label>
        </div>

        <label>
          Round
          <input className="field" value={round} disabled={readonly} onChange={(event) => setRoundDraft(event.target.value)} />
        </label>

        <div className="split-actions">
          <button className="tool-button secondary" disabled={readonly} onClick={() => setScore(selectedSeat, score)}>Set Score</button>
          <button className="tool-button secondary" disabled={readonly} onClick={() => { setRound(round); setHonba(honba); }}>Set Round</button>
        </div>
      </div>
    </Panel>
  );
}
