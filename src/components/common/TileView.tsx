import { Tile, TILE_DEFS, createTile } from "../../types/game";

const tileImages = import.meta.glob("../../../figures/*.png", {
  eager: true,
  query: "?url",
  import: "default"
}) as Record<string, string>;

function tileSrc(tile: Tile): string | undefined {
  const key = Object.keys(tileImages).find((path) => path.endsWith(`/${tile.code}.png`));
  return key ? tileImages[key] : undefined;
}

interface TileViewProps {
  tile?: Tile;
  small?: boolean;
  sideways?: boolean;
  ghost?: boolean;
  title?: string;
}

export function TileView({ tile, small = false, sideways = false, ghost = false, title }: TileViewProps) {
  if (!tile || ghost) {
    return <span className={`tile-view tile-back ${small ? "tile-small" : ""} ${sideways ? "tile-sideways" : ""}`} />;
  }

  const src = tileSrc(tile);

  return (
    <span
      className={`tile-view ${small ? "tile-small" : ""} ${sideways ? "tile-sideways" : ""} ${tile.red ? "tile-red" : ""}`}
      title={title ?? tile.label}
    >
      {src ? <img src={src} alt={tile.label} draggable={false} /> : <span>{tile.label}</span>}
    </span>
  );
}

interface TileSelectProps {
  value: Tile;
  onChange: (tile: Tile) => void;
  id?: string;
}

export function TileSelect({ value, onChange, id }: TileSelectProps) {
  return (
    <select
      id={id}
      className="field"
      value={value.typeId}
      onChange={(event) => onChange(createTile(Number(event.target.value)))}
    >
      {TILE_DEFS.map((tile) => (
        <option key={tile.typeId} value={tile.typeId}>
          {tile.code} / {tile.label}
        </option>
      ))}
    </select>
  );
}
