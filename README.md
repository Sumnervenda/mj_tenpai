# Mahjong DL-Engine + Frontend Console

本项目包含两部分：

- Python 日麻规则引擎：负责牌编码、牌山、手牌、胡牌判定、役种、计分、动作掩码和完整状态机。
- React 前端调试控制台：面向开发者的暗色 DevTool，默认用 Mock DTO 数据源驱动控制台状态，并支持通过 SSE 只读观察真实 Python 随机对局。

当前写操作仍停留在前端 Mock 控制台；真实 Python Live 模式只负责推送快照，不接收前端动作注入。`engine/interface.py` 提供 JSON-friendly 契约层，`src/services/engineAPI.ts` 维护 Mock DTO，`src/services/liveAPI.ts` 连接 `main.py --live-console` 的 SSE 流。

---

## 项目结构

```text
mj_tenpai/
├── engine/                         # Python 日麻引擎核心
│   ├── tile.py                     # 牌编码：绝对 ID / 类型 ID
│   ├── wall.py                     # 牌山、王牌、宝牌指示牌
│   ├── hand.py                     # 手牌直方图与副露
│   ├── agari.py                    # O(1) 查表胡牌 / 听牌
│   ├── yaku.py                     # 役种判定
│   ├── scoring.py                  # 符翻与点数结算
│   ├── actions.py                  # 动作定义与 77 维动作掩码
│   ├── rules.py                    # 可配置规则
│   ├── game.py                     # 事件驱动状态机
│   └── interface.py                # JSON-friendly 前后端统一契约层
├── src/                            # React/Vite 前端控制台
│   ├── App.tsx                     # 三栏 DevTool 主布局
│   ├── main.tsx                    # React 入口
│   ├── styles.css                  # 暗色控制台样式
│   ├── types/game.ts               # UI 视图模型与牌图映射
│   ├── types/contract.ts           # 与 Python 对齐的 DTO 契约类型
│   ├── store/gameStore.ts          # Zustand 状态管理
│   ├── services/adapters.ts        # DTO ↔ UI 状态转换
│   ├── services/engineAPI.ts       # Mock DTO 数据源 / 未来通信边界
│   ├── services/liveAPI.ts         # SSE Live 只读数据源
│   └── components/                 # Flow、规则、动作、状态、作弊、日志面板
├── figures/                        # 麻将牌 PNG 素材
├── tests/                          # Python 单元测试（含契约层与 Live SSE 测试）
├── main.py                         # Python 随机对局演示、benchmark、Live Console SSE
├── console.html                    # Vite 控制台入口
├── index.html                      # 旧静态麻将桌参考原型
├── Engine_control_readme.md        # 前端控制台需求说明
├── CHINESE_GUIDE.md                # 详细中文开发手册
├── package.json                    # 前端依赖与 npm scripts
└── vite.config.ts                  # Vite 构建配置
```

---

## 快速开始

### Python 引擎

```bash
pip install numpy pytest

# 单局随机对局
python main.py

# 指定随机种子
python main.py --seed 12345

# 性能基准
python main.py --benchmark 100

# 批量随机对局记录：输出 JSONL + CSV
python main.py --record-random 10000 --record-output records/random_10000_seed42 --seed 42

# Live Console：运行随机对局并推送 SSE 快照
python main.py --live-console --seed 42 --delay 0.3

# 单元测试
python -m pytest tests/ -v
```

### 前端控制台

```bash
npm install

# 启动开发服务器
npm run dev -- --port 5173
```

浏览器访问：

```text
http://127.0.0.1:5173/console.html
```

若要观察真实 Python 随机对局，先启动 `python main.py --live-console`，再在 Flow Control 的 Live SSE 输入框连接：

```text
http://127.0.0.1:8765/stream
```

生产构建：

```bash
npm run build
```

---

## 前端控制台功能

控制台是开发工具，不是玩家 UI。它用于观察和破坏性测试麻将引擎状态。

| 模块 | 作用 |
| --- | --- |
| Flow Control | Reset、Step、Auto Play、Speed、Toggle AI、Live SSE 连接 |
| Rule Settings | 食断、赤宝牌、多家荣和、流局满贯、包牌责任等规则开关 |
| Action Injection | 强制摸牌、打牌、吃、碰、杠、立直、和牌，并支持非法动作测试 |
| Mahjong Table | 中央牌桌视图，展示四家手牌、牌河、分数、当前阶段 |
| State Inspector | 牌山透视、王牌、宝牌、所有玩家手牌、向听与听牌列表 |
| Cheat Panel | 指定下一摸、修改手牌、注入牌河、改分数、改局名与本场数 |
| Logs & Debug | Action Log、算番详情、Raw JSON、Replay Base64 导入导出 |

内置 Mock 场景：

- 国士无双
- 四杠子
- 三家和了
- 振听
- 非法立直
- 吃 / 碰 / 胡并发冲突

Live 模式说明：

- 启动命令：`python main.py --live-console --seed 42 --delay 0.3`
- 默认服务：`http://127.0.0.1:8765`
- 健康检查：`GET /health`
- 当前快照：`GET /snapshot`
- 实时流：`GET /stream`
- 暂停推进：`POST /pause`
- 继续推进：`POST /resume`
- 切换暂停：`POST /toggle-pause`
- 前端连接后，牌桌、日志、Raw JSON、阶段、分数、牌河会跟随 Python `GameEngine` 随机对局更新。
- 连接 Live 后可在 Flow Control 中按 `Pause` 暂停 Python 随机对局，停在当前快照上检查手牌、牌河、Raw JSON，再按 `Resume` 继续。
- Live 模式第一版只读；Reset、Step、Auto、Cheat、Action Injection、Replay 写操作需断开 Live 后回到 Mock 模式使用。

---

## 牌图映射

前端通过 `figures/` 下的 PNG 渲染麻将牌：

| 文件名 | 含义 |
| --- | --- |
| `1m` - `9m` | 一万 - 九万 |
| `1p` - `9p` | 一筒 - 九筒 |
| `1s` - `9s` | 一索 - 九索 |
| `0m` / `0p` / `0s` | 赤五万 / 赤五筒 / 赤五索 |
| `E` / `S` / `W` / `N` | 东 / 南 / 西 / 北 |
| `white` / `haku` / `zhong` | 白 / 发 / 中 |

Python 引擎仍使用类型 ID `0-33` 和绝对 ID `0-135`。`engine/interface.py` 与 `src/services/adapters.ts` 共同维护传输 DTO 与 UI 牌对象之间的映射。

---

## 引擎核心设计

- 双轨编码：
  - 绝对 ID `0-135` 表示真实物理牌，用于牌山、牌河、宝牌指示牌和牌谱复现。
  - 类型 ID `0-33` 表示 34 种逻辑牌型，用于 `int[34]` 手牌直方图与算法计算。
- 查表胡牌：
  - `engine/agari.py` 预计算面子 / 雀头表，实现标准形、七对子、国士无双判定。
- 事件驱动状态机：
  - `DRAW` 与 `DISCARD` 之间循环，切牌后按荣和、碰/杠、吃、通过进行仲裁。
- AI 训练适配：
  - 77 维合法动作掩码。
  - 354 维状态特征张量。
  - `clone()` 支持 MCTS / rollout。

---

## 批量随机对局记录

用于生成随机自对弈样本：

```bash
python main.py --record-random 10000 --record-output records/random_10000_seed42 --seed 42
```

输出文件：

- `records/random_10000_seed42.jsonl`：完整记录，每条为 `agari` 事件或 `game_summary`。
- `records/random_10000_seed42_agari.csv`：每个和牌者一行，含赢家、荣和/自摸、放铳者、和牌牌、符翻、役种、宝牌、点数变化、和牌手牌与副露。

记录器会在每次 `AGARI` 精算前抓取四家分数和手牌；终局后额外写入整场 `final_scores`、`adjusted_scores`、`ranks`。

---

## 前端通信边界

统一接口以 Python 引擎为事实来源：

- `engine/interface.py`：把 `GameEngine`、`GameState`、`GameConfig`、`Action` 序列化为 JSON-friendly DTO。
- `src/types/contract.ts`：TypeScript 侧传输契约类型，字段贴近 Python 命名，如 `current_player`、`round_wind`、`riichi_sticks`。
- `src/services/adapters.ts`：把 DTO 派生为 React UI 视图模型，如 `currentSeat`、`players`、`Tile.code`。
- `src/services/engineAPI.ts`：当前维护 Mock DTO 快照；未来接真实后端时优先替换这里。
- `src/services/liveAPI.ts`：当前连接标准库 SSE Live 通道，把 `EngineSnapshotDTO` 交给同一个 adapter 映射。

Python 契约出口：

- `serialize_game_state(engine)`
- `serialize_config(config)`
- `serialize_legal_actions(legal)`
- `deserialize_action(command)`
- `create_engine_snapshot(engine, logs=None, debug=None)`

前端控制台 API 仍保留以下能力：

- `reset()`
- `step(action?)`
- `autoPlay(speed)`
- `toggleAI(seat)`
- `forceAction(action)`
- `setNextDraw(seat, tile)`
- `setHand(seat, tiles)`
- `injectDiscard(seat, tile)`
- `updateRules(rules)`
- `exportReplay()` / `importReplay(data)`

后续接入真实 Python 服务时，优先让后端返回 `EngineSnapshotDTO`，然后复用 `src/services/adapters.ts`，保持 `src/store/gameStore.ts` 和 UI 组件不变。

---

## 参考文档

- `CHINESE_GUIDE.md`：完整中文开发手册。
- `Engine_control_readme.md`：前端控制台需求说明。
- `index.html`：旧静态麻将桌布局参考。
#   m j _ t e n p a i  
 