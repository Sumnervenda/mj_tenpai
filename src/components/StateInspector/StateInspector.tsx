import { Panel } from "../common/Panel";
import { TileView } from "../common/TileView";
import { SEAT_META } from "../../types/game";
import { useGameStore } from "../../store/gameStore";

export function StateInspector() {
  const state = useGameStore((store) => store.state);

  return (
    <Panel title="State Inspector" eyebrow="GOD VIEW" className="state-inspector-panel">
      <div className="inspector-grid">
        <section className="inspector-section">
          <div className="section-title">
            <span>Wall</span>
            <strong>{state.wall.length}</strong>
          </div>
          <div className="tile-strip dense">
            {state.wall.slice(0, 42).map((tile) => (
              <TileView key={`${tile.absId}-${tile.code}`} tile={tile} small />
            ))}
          </div>
        </section>

        <section className="inspector-section">
          <div className="section-title">
            <span>Dead Wall / Dora</span>
            <strong>{state.deadWall.length}</strong>
          </div>
          <div className="tile-strip dense">
            {state.deadWall.slice(0, 14).map((tile, index) => (
              <TileView key={`${tile.absId}-${index}`} tile={tile} small title={index === 4 ? "宝牌指示牌" : tile.label} />
            ))}
          </div>
          <div className="dora-row">
            {state.doraIndicators.map((tile) => <TileView key={`dora-${tile.absId}`} tile={tile} small />)}
            {state.uraDoraIndicators.map((tile) => <TileView key={`ura-${tile.absId}`} tile={tile} small />)}
          </div>
        </section>

        <section className="inspector-section player-matrix">
          {state.players.map((player) => (
            <article key={player.seat} className="player-row">
              <div>
                <strong>{SEAT_META[player.seat].label}</strong>
                <span>{player.tenpai ? "听牌" : `${player.shanten} 向听`}</span>
              </div>
              <div className="tile-strip">
                {player.hand.slice(0, 14).map((tile) => (
                  <TileView key={`${player.seat}-hand-${tile.absId}-${tile.code}`} tile={tile} small />
                ))}
              </div>
              <div className="waits">
                {player.waits.map((tile) => <TileView key={`${player.seat}-wait-${tile.code}-${tile.absId}`} tile={tile} small />)}
              </div>
            </article>
          ))}
        </section>
      </div>
    </Panel>
  );
}
