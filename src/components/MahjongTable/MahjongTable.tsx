import { TileView } from "../common/TileView";
import { SEAT_META, Seat, formatTile } from "../../types/game";
import { useGameStore } from "../../store/gameStore";

const tableSeats: Seat[] = ["east", "south", "west", "north"];

export function MahjongTable() {
  const state = useGameStore((store) => store.state);

  return (
    <section className="table-shell" aria-label="mahjong table state">
      <div className="engine-core">
        <span>{state.round} / {state.honba} 本场</span>
        <strong>{state.phase}</strong>
        <span>Current {SEAT_META[state.currentSeat].short}</span>
        <span>Dora {formatTile(state.doraIndicators[0])}</span>
      </div>

      {tableSeats.map((seat) => {
        const player = state.players.find((entry) => entry.seat === seat);
        if (!player) return null;

        return (
          <article key={seat} className={`table-player player-${seat} ${state.currentSeat === seat ? "current" : ""}`}>
            <header>
              <span>{SEAT_META[seat].label}</span>
              <strong>{player.score.toLocaleString()}</strong>
            </header>
            <div className="mini-status">
              <span>{player.tenpai ? "TENPAI" : `${player.shanten} 向听`}</span>
              <span>{player.riichi ? "RIICHI" : player.aiEnabled ? "AI" : "MANUAL"}</span>
            </div>
            <div className="table-hand">
              {player.hand.slice(0, 14).map((tile) => (
                <TileView key={`${seat}-table-hand-${tile.absId}-${tile.code}`} tile={tile} small={seat !== "east"} />
              ))}
            </div>
            <div className="discard-grid">
              {player.discards.slice(-12).map((tile, index) => (
                <TileView key={`${seat}-discard-${tile.absId}-${index}`} tile={tile} small />
              ))}
            </div>
          </article>
        );
      })}

      <div className="wall-meter" style={{ "--wall-left": `${Math.max(0, Math.min(100, state.wall.length))}%` } as React.CSSProperties}>
        <span>Wall</span>
        <i />
        <strong>{state.wall.length}</strong>
      </div>
    </section>
  );
}
