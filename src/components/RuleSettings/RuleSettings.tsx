import { Panel } from "../common/Panel";
import { RuleSettings as RuleSettingsType } from "../../types/game";
import { useGameStore } from "../../store/gameStore";

const toggles: Array<{ key: keyof RuleSettingsType; label: string }> = [
  { key: "kuitan", label: "食断" },
  { key: "akaDora", label: "赤宝牌" },
  { key: "multipleRon", label: "多家荣和" },
  { key: "nagashiMangan", label: "流局满贯" },
  { key: "responsibility", label: "包牌责任" },
  { key: "ryanhanShibari", label: "二番缚" },
  { key: "kuikae", label: "食替禁止" },
  { key: "openRiichi", label: "开立直" },
  { key: "eastOnly", label: "东风战" },
  { key: "tobi", label: "飞人终局" }
];

export function RuleSettings() {
  const rules = useGameStore((store) => store.rules);
  const updateRules = useGameStore((store) => store.updateRules);
  const readonly = useGameStore((store) => store.sourceMode === "live");

  return (
    <Panel title="Rule Settings" eyebrow="PARAMETERS">
      <div className="numeric-grid">
        <label>
          起始点
          <input
            className="field"
            type="number"
            value={rules.startScore}
            disabled={readonly}
            onChange={(event) => updateRules({ startScore: Number(event.target.value) })}
          />
        </label>
        <label>
          本场
          <input
            className="field"
            type="number"
            value={rules.honbaBonus}
            disabled={readonly}
            onChange={(event) => updateRules({ honbaBonus: Number(event.target.value) })}
          />
        </label>
        <label>
          赤牌数
          <input
            className="field"
            type="number"
            min={0}
            max={3}
            value={rules.akadora}
            disabled={readonly}
            onChange={(event) => updateRules({ akadora: Number(event.target.value) })}
          />
        </label>
      </div>

      <div className="toggle-list">
        {toggles.map(({ key, label }) => (
          <label className="switch-row" key={String(key)}>
            <span>{label}</span>
            <input
              type="checkbox"
              checked={Boolean(rules[key])}
              disabled={readonly}
              onChange={(event) => updateRules({ [key]: event.target.checked } as Partial<RuleSettingsType>)}
            />
          </label>
        ))}
      </div>
    </Panel>
  );
}
