# 雀魂规则麻将深度学习引擎 (Mahjong DL-Engine) 中文开发手册

> 基于 README.md 项目总览维护的完整开发手册，覆盖 Python 日麻引擎与 React 前端调试控制台。
> 当前版本包含游戏规则层、AI 训练接口、Mock 驱动的前端 DevTool，以及通过 SSE 只读观察真实 Python 随机对局的 Live Console。

---

## 目录

1. [项目概览](#项目概览)
2. [快速开始](#快速开始)
3. [架构设计](#架构设计)
4. [模块详解](#模块详解)
5. [前端控制台架构](#前端控制台架构)
6. [控制台操作手册](#控制台操作手册)
7. [游戏流程](#游戏流程)
8. [前后端集成预留接口](#前后端集成预留接口)
9. [API 参考](#api-参考)
10. [AI 训练接口](#ai-训练接口)
11. [性能指标](#性能指标)
12. [已知限制与待办事项](#已知限制与待办事项)

---

## 项目概览

### 项目结构

```
mj_tenpai/
├── README.md                 # 项目总览：Python 引擎 + 前端控制台
├── CHINESE_GUIDE.md          # 本文档：中文开发手册
├── Engine_control_readme.md  # 前端控制台需求说明
├── engine/                   # 游戏引擎核心模块
│   ├── __init__.py           # 模块导出，统一 API
│   ├── tile.py               # 牌编码系统（双轨制）
│   ├── wall.py               # 牌山管理（136张）
│   ├── hand.py               # 手牌表示（int[34] 直方图）
│   ├── agari.py              # 胡牌判定（查表法 LUT）
│   ├── yaku.py               # 役种判定系统（38种役）
│   ├── scoring.py            # 点数计算（符/翻 → 得分）
│   ├── actions.py            # 动作定义与合法动作掩码
│   ├── rules.py              # 规则配置（可开关）
│   ├── game.py               # 状态机/游戏主循环
│   └── interface.py          # JSON-friendly 前后端统一契约层
├── src/                      # React/Vite 前端调试控制台
│   ├── App.tsx               # 三栏 DevTool 主布局
│   ├── main.tsx              # React 入口
│   ├── styles.css            # 暗色控制台样式
│   ├── types/game.ts         # UI 视图模型、牌图映射、动作结构
│   ├── types/contract.ts     # 与 Python 对齐的 DTO 契约类型
│   ├── store/gameStore.ts    # Zustand 状态管理
│   ├── services/adapters.ts  # DTO ↔ UI 状态转换
│   ├── services/engineAPI.ts # Mock DTO 数据源 / 未来通信边界
│   ├── services/liveAPI.ts   # SSE Live 只读数据源
│   └── components/           # 控制台各功能面板
│       ├── FlowControl/
│       ├── RuleSettings/
│       ├── ActionPanel/
│       ├── MahjongTable/
│       ├── StateInspector/
│       ├── CheatPanel/
│       ├── LogPanel/
│       └── common/
├── figures/                  # 前端麻将牌 PNG 素材
├── tests/                    # 单元测试
│   ├── test_tile.py          # 牌编码测试
│   ├── test_agari.py         # 胡牌判定测试
│   ├── test_game.py          # 游戏流程集成测试
│   └── test_live_console.py  # Live SSE 辅助逻辑测试
├── main.py                   # 演示入口（随机对局 + 性能基准 + Live SSE）
├── console.html              # Vite 前端控制台入口
├── index.html                # 旧静态麻将桌参考原型
├── package.json              # 前端依赖与 npm scripts
├── package-lock.json         # 前端依赖锁定文件
├── vite.config.ts            # Vite 构建配置
└── tsconfig*.json            # TypeScript 配置
```

### 设计哲学

1. **双轨编码机制**
   - **绝对 ID（0-135）**：每张牌的物理唯一标识，用于牌山、牌河、宝牌指示牌。可区分红宝牌等特殊牌。
   - **类型 ID（0-33）**：34 种逻辑牌型，用于手牌直方图 `int[34]`，高效计算。

2. **查表法（LUT）**
   - 将手牌的花色分别编码为五进制数（万/筒/条 9 位，字牌 7 位）
   - BFS 预计算所有合法面子组合 → O(1) 胡牌判定
   - 构建耗时 ~250ms（模块首次导入时）

3. **事件驱动状态机**
   - 非线性的日麻打断机制：碰/杠/吃/荣和按优先级仲裁
   - 支持深度复制（MCTS 模拟）

4. **AI 训练原生适配**
   - 合法动作掩码向量（77 维）
   - 游戏状态特征张量（354 维）
   - Gym-like `step()` 接口
   - 即时奖励信号（分数变动）

5. **开发者控制台**
   - 使用 React + Vite + TypeScript 构建
   - 使用 Zustand 管理前端状态
   - 默认使用 Mock DTO 数据源模拟引擎通信，便于 UI 和调试流程先行
   - 可连接 `python main.py --live-console` 的 SSE 流，只读观察真实 `GameEngine` 随机对局
   - 使用 `figures/` PNG 素材渲染麻将牌，不依赖 Unicode 牌字

6. **统一契约层**
   - Python 侧 `engine/interface.py` 输出 JSON-friendly DTO
   - TypeScript 侧 `src/types/contract.ts` 定义同名传输结构
   - `src/services/adapters.ts` 负责 DTO 到 UI 视图模型的唯一转换

---

## 快速开始

### 环境要求

- Python 3.10+
- numpy
- pytest（用于运行测试）
- Node.js 18+（用于前端控制台）
- npm 9+ / 10+ / 11+

### 安装依赖

Python 引擎依赖：

```bash
pip install numpy pytest
```

前端控制台依赖：

```bash
npm install
```

### 运行演示

```bash
# 单局随机对局（可视化输出）
python main.py

# 性能基准测试（100局静默运行）
python main.py --benchmark 100

# 指定随机种子
python main.py --seed 12345

# 批量随机对局记录：输出 JSONL + CSV
python main.py --record-random 10000 --record-output records/random_10000_seed42 --seed 42

# Live Console：运行随机对局并推送 SSE 快照
python main.py --live-console --seed 42 --delay 0.3
```

### 运行测试

```bash
python -m pytest tests/ -v
```

### 运行前端控制台

```bash
# 启动 Vite 开发服务器
npm run dev -- --port 5173
```

浏览器访问：

```text
http://127.0.0.1:5173/console.html
```

生产构建：

```bash
npm run build
```

前端入口是 `console.html`。旧的 `index.html` 保留为静态麻将桌布局参考，不作为当前控制台入口。

### 运行 Live Console

Live Console 用标准库 `ThreadingHTTPServer` 提供 SSE 流，不需要 FastAPI / WebSocket 依赖。第一版是只读观察模式：Python 随机对局持续推送 `EngineSnapshotDTO`，前端只展示，不把 Reset、Step、Cheat 或 Action Injection 发回真实引擎。

```bash
# 终端 1：启动前端
npm run dev -- --port 5173

# 终端 2：启动真实 Python 随机对局 Live 流
python main.py --live-console --seed 42 --delay 0.3
```

然后打开：

```text
http://127.0.0.1:5173/console.html
```

在 Flow Control 的 Live SSE 输入框连接：

```text
http://127.0.0.1:8765/stream
```

Live 服务默认端点：

| 端点 | 作用 |
|------|------|
| `GET /health` | 返回 `{ ok, version, done }` |
| `GET /snapshot` | 返回当前最新 `EngineSnapshotDTO` |
| `GET /stream` | 持续发送 `event: snapshot` 的 SSE 快照 |
| `POST /pause` | 暂停 Python 随机对局推进，前端停在当前快照便于检查 |
| `POST /resume` | 从暂停处继续随机对局 |
| `POST /toggle-pause` | 切换暂停 / 继续状态 |

### 最小代码示例

```python
from engine import GameEngine, GameConfig, GamePhase, Action, ActionType

# 创建游戏引擎
config = GameConfig()
engine = GameEngine(config=config, seed=42)

# 主循环
while not engine.is_game_over():
    state = engine.get_game_state()

    # 摸牌阶段：当前玩家决定
    if state.phase == GamePhase.DRAW:
        actions = engine.get_legal_actions()
        # 你的 AI 在这里选择动作
        action = actions.actions[0]  # 示例：选第一个合法动作
        engine.step(action)

    # 切牌响应阶段：其他三家决定
    elif state.phase == GamePhase.DISCARD:
        options = engine.get_response_options()
        responses = {}
        for p_idx, legal in options.items():
            responses[p_idx] = legal.actions[0]  # 示例：选第一个
        engine.resolve_responses(responses)

    # 和牌 / 流局 → 结算
    elif state.phase in (GamePhase.AGARI, GamePhase.RYUUKYOKU):
        engine.step(Action(ActionType.PASS))

# 查看结果
result = engine.get_result()
print(f"最终排名: {result.ranks}")
print(f"调整后分数: {result.adjusted_scores}")
```

---

## 架构设计

### 数据表示层（tile.py）

```
牌的类型 ID 映射表：

 0- 8: 万子 (一萬～九萬)
 9-17: 筒子 (一筒～九筒)
18-26: 条子 (一条～九条)
27-33: 字牌 (東, 南, 西, 北, 白, 発, 中)

编码公式：
  abs_id = type_id × 4 + copy_index  (copy_index ∈ {0, 1, 2, 3})

赤宝牌（红宝牌）：
  5万 = type 4, abs_id 19 (copy_index=3)
  5筒 = type 13, abs_id 55 (copy_index=3)
  5条 = type 22, abs_id 91 (copy_index=3)
```

### 查表法核心算法（agari.py）

#### 五进制编码

每一花色（如万子 9 种）的手牌计数是一个长度为 9、每位 0-4 的数组，编码为五进制整数：

```
状态空间：5^9 = 1,953,125（约 200 万）
```

#### LUT 查询

对每个五进制状态，预计算两个布尔值：

| LUT | 含义 |
|-----|------|
| `melds[i]` | 状态 `i` 能否完全分解为面子（无雀头） |
| `with_pair[i]` | 状态 `i` 能否分解为面子 + 1 个雀头 |

#### 完整胡牌判定

```
is_agari(hand):
  1. 检查特殊形式：七对子（7 对）、国士无双（13 幺九 + 1 对）
  2. 标准形式：将手牌拆为 4 花色
  3. 尝试让每个花色分别充当"雀头提供者"
     → 该花色查 with_pair 表，其余 3 花色查 melds 表
  4. 任意一种组合成功 → 胡牌
```

#### 探测算法

- `is_tenpai(hand)`：向 13 张手牌依次塞入 34 种牌，若 `is_agari` 返回 True → 听牌
- `get_waits(hand)`：记录所有使手牌成和的牌型列表（用于振听判定）

---

### 状态机设计（game.py）

```
游戏阶段流转：

    ┌──────────────────────────────────────────────────┐
    │                                                  │
    ▼                                                  │
  DEAL (发牌) → DRAW (摸牌) → DISCARD (切牌响应) ──────┤
                  ▲              │    │    │           │
                  │    荣和/自摸 ▼    │    │           │
                  │         AGARI (和牌) │              │
                  │              │   碰/杠/吃           │
                  │              ▼    ▼    ▼           │
                  │         ROUND_END ←─────────────────┤
                  │              │                      │
                  │              ▼                      │
                  └────── 下一局 / GAME_END (终局) ─────┘

切牌后响应仲裁优先级：
  Priority 1: 荣和 (Ron) — 多家荣和可并存（ダブロン/トリプルロン）
  Priority 2: 杠 (Kan) / 碰 (Pon)
  Priority 3: 吃 (Chi) — 只能吃上家
  Priority 4: 无 → 下家摸牌
```

### 役种体系（yaku.py）

共实现 **38 种役**，涵盖：

| 翻数 | 役种（门清限定） | 役种（副露可） |
|------|------------------|----------------|
| 1翻 | 立直, 一発, 门清自摸, 平和, 断幺九, 一盃口 | 役牌（场风/自风/三元） |
| 2翻 | 両立直, 七対子 | 対々和, 三暗刻, 三槓子, 小三元, 混老頭 |
| 2翻(副露1翻) | 混全帯么九, 一気通貫, 三色同順 | — |
| 3翻(副露2翻) | 混一色, 純全帯么九 | — |
| 3翻(门清) | 二盃口 | — |
| 6翻(副露5翻) | 清一色 | — |
| 役満 | 国士無双, 四暗刻, 大三元, 小四喜, 大四喜, 字一色, 緑一色, 清老頭, 九蓮宝燈, 天和, 地和 | — |

### 点数计算（scoring.py）

#### 符（フ）计算

```
基础符: 20符
+ 门清荣和: +10符
+ 自摸（平和除外）: +2符
+ 面子符: 明刻(2~4) / 暗刻(4~8) / 明槓(8~16) / 暗槓(16~32)
+ 雀头符: 役牌对子 +2符
+ 听牌形符: 嵌張/辺張/単騎 +2符
→ 向上取整到10的倍数（平和自摸固定20符）
```

#### 翻数 → 得点表

| 翻数 | 符数 | 荣和(子) | 荣和(親) | 自摸(子) | 自摸(親) |
|------|------|---------|---------|---------|---------|
| 1翻 | 30符 | 1000 | 1500 | 300/500 | 500all |
| 2翻 | 30符 | 2000 | 2900 | 500/1000 | 1000all |
| 3翻 | 30符 | 3900 | 5800 | 1000/2000 | 2000all |
| 4翻 | 30符 | 7700 | 11600 | 2000/3900 | 3900all |
| 5翻 | 満貫 | 8000 | 12000 | 2000/4000 | 4000all |
| 6-7翻 | 跳満 | 12000 | 18000 | 3000/6000 | 6000all |
| 8-10翻 | 倍満 | 16000 | 24000 | 4000/8000 | 8000all |
| 11-12翻 | 三倍満 | 24000 | 36000 | 6000/12000 | 12000all |
| 13翻+ | 数え役満 | 32000 | 48000 | 8000/16000 | 16000all |

---

## 模块详解

### tile.py — 牌编码系统

```python
from engine.tile import (
    abs_to_type,       # 绝对ID → 类型ID
    type_to_abs,       # 类型ID → 绝对ID列表
    is_aka,            # 判断是否为赤宝牌
    is_yaochuhai,      # 判断是否为幺九牌
    is_jihai,          # 判断是否为字牌
    suit_of,           # 获取花色（0=万, 1=筒, 2=条, 3=字）
    TILE_NAMES,        # 日文牌名列表 ["1m", "2m", ..., "中"]
    TILE_NAMES_CN,     # 中文牌名列表 ["一万", ..., "中"]
    YAOCHUHAI_TYPES,   # 幺九牌类型集合
    TANYAO_TYPES,      # 断幺九牌类型集合
    GREEN_TYPES,       # 绿一色牌类型集合
)
```

### wall.py — 牌山管理

```python
from engine.wall import Wall

wall = Wall(seed=42)        # 创建牌山
wall.shuffle()              # 洗牌
hands = wall.deal()         # 发牌 → 4手牌 (14, 13, 13, 13)
tile = wall.draw()          # 从牌山摸一张
rinshan = wall.draw_rinshan() # 从嶺上牌摸一张
dora = wall.flip_dora()     # 翻下一张宝牌指示牌
ura = wall.flip_ura_dora()  # 翻开里宝牌
dora_types = wall.get_dora_types()  # 获取当前宝牌类型列表
```

**牌山结构**：136 张牌中，最后 14 张为王牌（122-135）：
- 索引 122-131：宝牌指示牌（5 组：表宝牌 + 里宝牌）
- 索引 132-135：嶺上牌（杠后摸牌，从后往前摸）

### agari.py — 胡牌判定

```python
from engine.agari import (
    is_agari,           # 判断 int[34] 手牌是否胡牌
    is_tenpai,          # 判断是否听牌
    get_waits,          # 获取听牌列表（用于振听判定）
    is_agari_with_tile, # 模拟加入某张牌后是否胡牌（荣和/自摸探测）
    get_legal_discards_for_riichi,  # 获取可以立直的切牌选择
    can_riichi,         # 判断是否可以宣告立直
)
```

**支持的胡牌形式**：
- 标准形：4 面子（顺子/刻子/槓子） + 1 雀头
- 七对子：7 对不同牌型各 2 张
- 国士无双：13 种幺九牌各 1 张 + 任意幺九牌 1 张

### yaku.py — 役种判定

```python
from engine.yaku import YakuChecker, WinContext, YakuResult, Yaku

# 构建和牌上下文
ctx = WinContext(
    is_menzen=True,      # 是否门前清
    is_tsumo=True,       # 自摸还是荣和
    is_riichi=False,     # 是否立直
    bakaze=27,           # 场风（27=東, 28=南）
    jikaze=27,           # 自风
    concealed_tiles=hand,  # 手牌 int[34]
    winning_tile=23,     # 和了牌
)
checker = YakuChecker(ctx)
result = checker.check_all()

print(result.total_han)            # 总翻数
print(result.is_yakuman)           # 是否为役满
print(result.yakuman_list)         # 役满列表
print(result.yaku_list)            # [(役种, 翻数), ...]
```

### game.py — 游戏引擎

核心类 `GameEngine` 实现了完整的游戏主循环：

```python
engine = GameEngine(config=GameConfig(), seed=42)

# 状态查询
state = engine.get_game_state()    # GameState 快照
tensor = engine.get_state_tensor(0) # 玩家0视角的354维特征张量

# 动作处理
engine.step(action)                # 执行一个动作
engine.resolve_responses({...})    # 批量处理切牌响应

# MCTS 支持
clone = engine.clone()             # 深度复制（模拟用）

# 结果
result = engine.get_result()       # GameResult（含 ranking）
winner = engine.get_winner()       # 优胜者索引
```

**响应阶段批量处理**（推荐用于 AI 自对弈）：

```python
# 获取所有可响应玩家的合法动作
options = engine.get_response_options()
# → {p1: LegalActions, p2: LegalActions, p3: LegalActions}

# 每个玩家（AI）独立决策后，一次性提交
engine.resolve_responses({
    1: Action(ActionType.PASS),
    2: Action(ActionType.PON, tile=source_tile),
    3: Action(ActionType.PASS),
})
# 引擎按优先级自动仲裁：荣和 > 碰/杠 > 吃 > 通过
```

### interface.py — 前后端契约层

`interface.py` 不改变引擎核心对象，只负责把内部状态变成 JSON-friendly DTO：

```python
from engine import GameEngine, create_engine_snapshot, deserialize_action

engine = GameEngine(seed=42)
snapshot = create_engine_snapshot(engine)

# 前端命令 / WebSocket 消息可转回 Python Action
action = deserialize_action({
    "type": "discard",
    "tile": {"type_id": 4},
    "actor": 0,
})
```

该层是 Live Console 和未来真实 WebSocket / HTTP 后端的推荐出口。它保留 Python 风格字段名，例如 `current_player`、`hands_concealed`、`legal_actions`，前端 UI 的 camelCase 字段只在 `src/services/adapters.ts` 中派生。

---

## 前端控制台架构

前端控制台位于 `src/`，是一个面向开发者的暗色调试 DevTool。它不是玩家对局 UI，而是用于观察引擎内部状态、构造极端场景、注入非法动作、验证状态机流转和导入导出 Replay 的控制台。

### 入口与构建

| 文件 | 作用 |
|------|------|
| `console.html` | Vite HTML 入口，当前控制台访问地址对应 `/console.html` |
| `src/main.tsx` | React 挂载入口 |
| `src/App.tsx` | 三栏控制台布局：左侧控制、中央信息、右侧调试 |
| `src/styles.css` | 暗色高信息密度 DevTool 样式 |
| `vite.config.ts` | Vite 构建配置，指定 `console.html` 为输入 |

### 状态与通信边界

| 文件 | 作用 |
|------|------|
| `engine/interface.py` | Python 契约出口，序列化 `GameEngine`、`GameState`、`GameConfig`、`Action` |
| `src/types/contract.ts` | 传输 DTO 类型，字段贴近 Python 命名 |
| `src/types/game.ts` | React UI 视图模型，字段适合组件渲染 |
| `src/services/adapters.ts` | DTO ↔ UI 状态转换，统一处理字段命名、座位、牌图 code |
| `src/store/gameStore.ts` | Zustand store，集中管理状态、日志、速度、Replay 缓冲区、Live 连接状态、选中座位和牌 |
| `src/services/engineAPI.ts` | Mock DTO 数据源，模拟引擎接口；未来接 WebSocket / HTTP 时优先替换这里 |
| `src/services/liveAPI.ts` | 使用浏览器 `EventSource` 连接 `/stream`，接收真实 Python `EngineSnapshotDTO` |

当前 `engineAPI.ts` 内部维护一个 `EngineSnapshotDTO`，再通过 `adapters.ts` 派生 UI 状态：

```ts
interface EngineSnapshotDTO {
  state: GameStateDTO
  rules: RuleSettingsDTO
  logs: LogEntryDTO[]
}
```

UI 组件只依赖 Zustand store，不直接依赖 DTO、Mock 实现或 SSE 细节。当前 store 通过 `sourceMode: "mock" | "live"` 区分数据源；Live 模式下写操作会被置为只读，真实引擎快照仍复用同一个 `snapshotDTOToUI()` adapter。

### Python 契约出口

`engine/interface.py` 当前提供以下函数：

```python
serialize_game_state(engine)          # GameEngine -> GameStateDTO
serialize_config(config)              # GameConfig -> RuleSettingsDTO
serialize_legal_actions(legal)        # LegalActions -> LegalActionsDTO
deserialize_action(command)           # ActionCommandDTO -> Action
create_engine_snapshot(engine, logs, debug)  # EngineSnapshotDTO
```

契约层保留中文注释，重点说明牌 ID 映射、动作掩码、Mock 系统动作与真实引擎动作的差异。

### 组件分层

| 目录 | 面板 |
|------|------|
| `components/FlowControl/` | Reset、Step、Auto Play、Speed、Toggle AI、Live SSE 连接 |
| `components/RuleSettings/` | 规则开关与数值参数 |
| `components/ActionPanel/` | 强制动作注入、非法动作测试、极端场景载入 |
| `components/MahjongTable/` | 中央牌桌视图，展示四家手牌、牌河、分数、当前阶段 |
| `components/StateInspector/` | 牌山透视、王牌、宝牌、所有玩家手牌、向听与听牌列表 |
| `components/CheatPanel/` | 指定下一摸、改手牌、注入牌河、改分数、改局名与本场数 |
| `components/LogPanel/` | Action Log、算番详情、Raw JSON、Replay Base64 导入导出 |
| `components/common/` | 通用面板与麻将牌渲染组件 |

### 牌图映射

前端使用 `figures/` 中的 PNG 渲染麻将牌，映射在 `src/types/game.ts` 中维护：

| 图片 code | 含义 |
|-----------|------|
| `1m` - `9m` | 一万 - 九万 |
| `1p` - `9p` | 一筒 - 九筒 |
| `1s` - `9s` | 一索 - 九索 |
| `0m` / `0p` / `0s` | 赤五万 / 赤五筒 / 赤五索 |
| `E` / `S` / `W` / `N` | 东 / 南 / 西 / 北 |
| `white` / `haku` / `zhong` | 白 / 发 / 中 |

Python 引擎的类型 ID 仍为 `0-33`：`0-8` 万子，`9-17` 筒子，`18-26` 条子，`27-33` 字牌。前端 `Tile` 同时保存 `typeId`、`absId`、`code`、`label` 和 `red` 标记。

---

## 控制台操作手册

### Flow Control

- `Reset`：重置 Mock 引擎快照，回到东1局 `DRAW`。
- `Step`：推进一帧。`DRAW` 阶段会摸牌，`DISCARD/RESPONSE` 阶段会自动切出一张牌并轮到下家。
- `Auto`：按 Speed 间隔循环调用 `engineAPI.autoPlay(speed)`。
- `Speed`：控制 Auto Play 间隔，单位毫秒。
- `Toggle AI`：按座位切换 Mock AI 接管状态，仅影响控制台显示与日志。
- `Live SSE`：连接 `http://127.0.0.1:8765/stream` 后，控制台切换到真实 Python 随机对局只读观察模式。
- `Pause / Resume`：暂停或继续 Python 随机对局推进。暂停时 SSE 连接保持打开，前端停在当前 `EngineSnapshotDTO`，可以检查牌桌、日志和 Raw JSON。
- `Disconnect`：断开 SSE 后回到 Mock 模式，Reset、Step、Auto、Cheat、Action Injection、Replay 写操作恢复可用。

Live 模式只读：前端不会把动作注入、规则修改或作弊面板操作发送给真实 `GameEngine`。这些控件在 Live 模式下会禁用，避免把 Mock 状态和真实快照混在一起。

### Rule Settings

规则面板会调用 `updateRules(rules)` 更新前端规则快照。当前支持起始点、本场点、赤牌数，以及食断、赤宝牌、多家荣和、流局满贯、包牌责任、二番缚、食替禁止、开立直、东风战、飞人终局等开关。

### Action Injection

动作注入面板用于测试状态机边界：

- 选择座位、动作和牌。
- 勾选“非法测试”后，会记录非法动作计数与 warn 日志。
- 支持 `draw`、`discard`、`chi`、`pon`、`kan`、`riichi`、`win`。
- 内置压力场景：国士无双、四杠子、三家和了、振听、非法立直、吃/碰/胡冲突。

### Cheat Panel

作弊面板用于构造可复现局面：

- `Next Draw`：指定某家下一张摸牌。
- `Discard`：直接向某家牌河注入一张牌。
- `Hand Editor`：输入牌 code 列表并替换某家手牌，例如 `1m 9m 1p 9p 1s 9s E S W N white haku zhong`。
- `Set Score`：修改某家分数。
- `Set Round`：修改局名和本场数。

### Logs / Replay / Raw JSON

- `Action Log` 记录 Mock 引擎动作、规则更新、场景载入、非法操作，以及 Live SSE 推来的真实对局事件。
- `算番详情` 显示当前快照的 han / fu / yaku；Mock 会提供示例算番，真实 Live 快照按契约层 debug 字段展示。
- `Export` 将完整 `EngineSnapshot` 编码为 Base64。
- `Import` 从 Base64 恢复快照。
- `Raw State JSON` 展示前端当前完整状态，便于和未来后端返回值对齐。

---

## 游戏流程

### 批量随机对局记录

如需生成随机自对弈样本，可使用：

```bash
python main.py --record-random 10000 --record-output records/random_10000_seed42 --seed 42
```

输出两份文件：

- `records/random_10000_seed42.jsonl`：完整 JSONL。每条记录是一次 `agari` 事件或一场 `game_summary`，包含四家手牌、分数、局面、和牌者、支付信息。
- `records/random_10000_seed42_agari.csv`：表格化和牌结果。每个和牌者一行，包含 `winner`、`win_type`、`loser`、`winning_tile`、`han`、`fu`、`score_name`、`yaku`、`dora_count`、`payments`、`scores_after_payment`、`winner_hand`、`winner_melds`。

记录时机是 `AGARI` 精算后、进入下一局前，因此点数已经更新，手牌仍保留在和牌瞬间。`game_summary` 则记录整场结束后的 `final_scores`、`adjusted_scores` 和 `ranks`。

### 完整一局（半荘）流程

```
1. init_game()          初始化玩家分数 (25000点)
2. start_round()        洗牌 → 发牌 → 翻初始宝牌指示牌
3. ┌─ DRAW phase ───────────────────────────────┐
   │  当前玩家摸牌                               │
   │  计算合法动作: [切牌, 自摸, 立直, 暗槓, 加槓] │
   │  AI 决策 → step(action)                     │
   └────────────────────────────────────────────┘
        │
        ▼ (切牌)
4. ┌─ DISCARD phase ────────────────────────────┐
   │  其他三家计算合法响应: [荣和, 碰, 大明槓, 吃, 通过] │
   │  按优先级仲裁                                │
   │  荣和 → 结算和牌 → AGARI                     │
   │  碰/杠 → 改变当前玩家 → DRAW（跳过大牌）       │
   │  吃 → 改变当前玩家 → DRAW（跳过摸牌）          │
   │  全部通过 → 下家摸牌 → DRAW                   │
   └────────────────────────────────────────────┘
        │
5. 重复 3-4 直到：
   - 有人和牌 → AGARI settlement → 下一局 / 终局
   - 牌山摸尽 → RYUUKYOKU → 听牌/不听牌结算 → 下一局 / 终局
6. 终局判定：东4局（或南4局）结束 → GAME_END
7. get_result() → 最终排名 + 马点调整
```

### 庄家轮换规则

- **和牌连荘**（agari renchan）：庄家（東家）和牌 → 本场数 +1，不换庄
- **听牌连荘**（tenpai renchan）：流局时庄家听牌 → 本场数 +1，不换庄
- **轮庄**：其他情况 → 庄家移至下家，本场数归零

---

## 前后端集成预留接口

当前有两条数据源：

- Mock 写模式：`src/services/engineAPI.ts` 在浏览器内维护 `EngineSnapshotDTO`，支持 Reset、Step、Action Injection、Cheat Panel 和 Replay。
- Live 只读模式：`python main.py --live-console` 用标准库 SSE 推送真实 `GameEngine` 快照，`src/services/liveAPI.ts` 通过 `EventSource` 接收后交给 `src/services/adapters.ts`。

后续如果要做真实可控后端，可以新增 FastAPI / WebSocket / HTTP 服务，并让后端统一返回 `EngineSnapshotDTO`。前端应继续复用 `adapters.ts`，尽量不让 UI 组件直接依赖通信协议。

### 当前 Live SSE 接口

```bash
python main.py --live-console --seed 42 --delay 0.3 --port 8765
```

| 端点 | 说明 |
|------|------|
| `GET /health` | 返回服务状态：`{ ok, version, done }` |
| `GET /snapshot` | 返回当前最新 `EngineSnapshotDTO` |
| `GET /stream` | SSE 流，持续发送 `event: snapshot` 和 `data: <EngineSnapshotDTO JSON>` |
| `POST /pause` | 暂停随机对局推进，并推送一帧 `debug.paused: true` 的快照 |
| `POST /resume` | 继续随机对局推进，并推送一帧 `debug.paused: false` 的快照 |
| `POST /toggle-pause` | 切换暂停状态，供前端单按钮调用 |

暂停不会关闭 SSE 连接，也不会释放终局快照；只是让 Python 随机推进循环阻塞在当前状态。对局结束后，服务会保留最终快照并继续运行，直到用户在 Python 进程中按 `Ctrl+C`。

### 未来可写后端接口

| 前端方法 | 预期后端行为 |
|----------|--------------|
| `reset()` | 创建或重置一局，并返回 `EngineSnapshot` |
| `step(action?)` | 执行当前合法动作或自动推进一帧 |
| `autoPlay(speed)` | 后端可忽略 speed，仅返回下一帧；speed 由前端定时器控制 |
| `toggleAI(seat)` | 切换指定座位是否由 AI 决策 |
| `forceAction(action)` | 注入动作，允许携带 `illegal: true` 做非法测试 |
| `setNextDraw(seat, tile)` | 指定下一张摸牌 |
| `setHand(seat, tiles)` | 替换某家手牌 |
| `injectDiscard(seat, tile)` | 向某家牌河注入一张牌 |
| `updateRules(rules)` | 更新规则配置 |
| `exportReplay()` | 导出可复现牌谱 / 快照 |
| `importReplay(data)` | 导入可复现牌谱 / 快照 |

建议后端响应统一返回：

```ts
interface EngineSnapshotDTO {
  state: GameStateDTO
  rules: RuleSettingsDTO
  logs: LogEntryDTO[]
}
```

契约层字段尽量使用 Python 风格命名，例如 `current_player`、`round_wind`、`riichi_sticks`、`hands_concealed`。UI 层的 `currentSeat`、`deadWall`、`doraIndicators` 等 camelCase 字段只在 `adapters.ts` 中生成。

动作掩码仍按 77 维结构，与 `engine/actions.py` 的概念保持一致。前端只要求 `mask.length === 77`，不会假设具体后端实现语言。

---

## API 参考

### GameConfig 可配置规则

```python
from engine import GameConfig

config = GameConfig(
    # ── 基础设定 ──
    start_score=25000,         # 初始持点
    target_score=30000,        # 返点目标
    riichi_stick_cost=1000,    # 立直棒费用
    honba_bonus=300,           # 本场积点

    # ── 役种开关 ──
    kuitan=True,               # 食い断（副露断幺九）
    aka_dora=True,             # 赤宝牌
    ryanhan_shibari=False,     # 二翻縛り（5本场以上强制2翻）
    open_riichi=False,         # 开立直

    # ── 终局设定 ──
    uma=(20, 10, -10, -20),   # 马点（千点单位）
    oka=0,                     # オカ
    east_only=False,           # 東風戦（True）/ 半荘戦（False）
    agari_yame=True,           # 和了り止め
    tenpai_renchan=True,       # 听牌连荘

    # ── 规则设定 ──
    tobi=False,                # 飛び（负分即终局）
    multiple_ron=True,         # 多家荣和
)
```

### 动作掩码结构（77 维）

| 索引范围 | 含义 |
|----------|------|
| 0-33 | 切牌（Discard）每种牌型 |
| 34 | 自摸（Tsumo） |
| 35 | 荣和（Ron） |
| 36 | 立直宣言（Riichi） |
| 37-70 | 立直切牌选择（Riichi + Discard）每种牌型 |
| 71 | 碰（Pon） |
| 72 | 吃（Chi） |
| 73 | 大明槓（Daiminkan） |
| 74 | 暗槓（Ankan） |
| 75 | 加槓（Kakan） |
| 76 | 通过（Pass） |

### 状态特征张量（354 维）

| 特征维度 | 通道数 | 描述 |
|----------|--------|------|
| 自家手牌 | 34 | 每类牌的持有数（0-4） |
| 自家副露 | 34 | 每类牌在副露中的数量 |
| 自家牌河 | 34 | 每类牌的打出数 |
| 宝牌指示 | 34 | 当前宝牌类型的 one-hot |
| 对手牌河 ×3 | 102 | 每家对手每类牌的打出数 |
| 对手副露 ×3 | 102 | 每家对手每类牌在副露中的数量 |
| 全局特征 | 7 | 自家分数/1000, 本场数, 立直棒, 剩余牌数/122, 是否立直, 场风, 局数 |
| 分数差 | 3 | 自家与其他三家的分数差/1000 |
| 自风 | 4 | 自风 one-hot (東/南/西/北) |

---

## AI 训练接口

### OpenAI Gym 风格接口

```python
# 环境初始化
engine = GameEngine()
state = engine.get_game_state()
done = False

while not done:
    # 获取观测
    obs = engine.get_state_tensor(player_idx)  # np.ndarray (354,)
    mask = engine.get_legal_actions().mask      # np.ndarray (77,)

    # AI 预测动作（用 mask 屏蔽非法动作）
    action_idx = model.predict(obs, mask)

    # 执行动作
    next_state = engine.step(action_from_idx(action_idx))

    # 奖励 = 分数变动
    reward = next_state.rewards[player_idx]
    done = next_state.done

    # 存储经验 (obs, action, reward, next_obs, done)
    replay_buffer.add(obs, action_idx, reward, next_state, done)
```

### MCTS 模拟

```python
def mcts_simulate(engine: GameEngine, num_simulations: int = 100):
    """对当前局面进行 MCTS 模拟"""
    for _ in range(num_simulations):
        # 深度复制引擎状态
        sim = engine.clone()

        # 随机 rollout 到终局
        while not sim.is_game_over():
            actions = sim.get_legal_actions()
            if sim.phase == GamePhase.DRAW:
                action = random_choice(actions)
                sim.step(action)
            elif sim.phase == GamePhase.DISCARD:
                options = sim.get_response_options()
                sim.resolve_responses({
                    p: random_choice(opt)
                    for p, opt in options.items()
                })
            else:
                sim.step(Action(ActionType.PASS))

        # 收集 rollout 结果
        yield sim.get_result()
```

### 特征工程扩展

可通过继承 `GameEngine` 并覆写 `get_state_tensor()` 方法来扩展特征：

```python
class MyGameEngine(GameEngine):
    def get_state_tensor(self, player_idx: int) -> np.ndarray:
        base = super().get_state_tensor(player_idx)
        # 添加自定义特征...
        extra_features = self._compute_extra_features(player_idx)
        return np.concatenate([base, extra_features])
```

---

## 性能指标

| 指标 | 数值 | 备注 |
|------|------|------|
| LUT 构建时间 | ~250ms | 首次导入时一次性开销 |
| 单局对局速度 | ~8 局/秒 | Python 实现（非优化） |
| LUT 内存占用 | ~8 MB | 2M × 2 bool + 78K × 2 bool |
| 状态张量维度 | 354 | 每次调用实时计算 |
| 动作空间 | 77 | 离散动作空间 |
| 单次 `is_agari()` | < 1μs | 纯数组查表 O(1) |

> **注意**：README 中的 10⁴ 局/秒目标是针对 C++ / Rust 实现。Python 版本为快速原型验证，后续可将热路径（LUT 查询、游戏步进）移植到编译语言并封装 Python 接口。

---

## 已知限制与待办事项

### 当前已完成

- [x] 完整游戏流程（摸牌→切牌→响应→结算→轮回）
- [x] 136 张牌（含 3 张赤宝牌）
- [x] O(1) 胡牌判定（查表法）
- [x] 听牌 / 听牌列表探测
- [x] 38 种役种判定（含役满）
- [x] 符 / 翻计算 + 点数授受
- [x] 立直 / 一発 / 双立直
- [x] 暗槓 / 大明槓 / 加槓 + 嶺上牌 + 新宝牌
- [x] 振听判定（含立直振听）
- [x] 流局精算（听牌 / 不听牌）
- [x] 马点 / オカ / 顺位点
- [x] 可配置规则（食断、赤宝牌、二翻缚等）
- [x] 动作掩码向量（77 维）
- [x] 状态特征张量（354 维）
- [x] 深度复制（MCTS 支持）
- [x] 机器可读的牌谱输出
- [x] React/Vite 前端调试控制台（Mock DTO 数据源）
- [x] PNG 麻将牌素材渲染与开发者控制台三栏布局
- [x] 前端 Replay Base64 导入导出、Raw JSON、Action Log
- [x] Python `engine/interface.py` 统一契约层
- [x] TypeScript `contract.ts` DTO 类型与 `adapters.ts` UI 转换层
- [x] `python main.py --live-console` 标准库 SSE 实时推送真实随机对局
- [x] 前端 Live 只读模式，实时展示 Python `GameEngine` 快照
- [x] 前端可暂停 / 继续 Python Live 随机对局，便于中途检查状态

### 待完善

- [ ] **真实 Python 可写 WebSocket / HTTP 桥接层**：当前 Live SSE 只读观察真实随机对局；Reset、Step、Cheat、Action Injection 仍只作用于 Mock 数据源。
- [ ] **役种判定严格化**：当前 `_can_win()` 未实际检查手牌是否满足至少一种役（允许了无役和牌）。需要在实际和牌前进行役种预判。
- [ ] **手牌分解枚举**：`decompose_hand()` 只返回第一种分解方案。对于需要枚举所有分解的役种（如判断是否为纯全带幺九时需最优分解），应支持多方案遍历。
- [ ] **岭上开花 / 枪槓判定**：当前未在役种判定中区分岭上开花和枪槓。
- [ ] **流局满贯**：荒牌流局时未实现"流し満貫"判定。
- [ ] **和了止め逻辑**：`agari_yame` 规则的完整实现（庄家在最终局和牌后可选终局）。
- [ ] **包牌（パオ）规则**：大三元 / 大四喜 / 四槓子的包牌责任。
- [ ] **食い替え禁止**：部分规则下，副露后不能立即打出同种牌。
- [ ] **牌谱 SGF / JSON 导出**：标准牌谱格式的读写支持。
- [ ] **C++/Rust 移植**：性能优化的编译语言实现 + Python 绑定。
- [ ] **真实玩家 UI**：当前控制台是 DevTool，不是玩家对局界面；如需人机对弈 UI 应另行设计。

---

## 引用

项目总览：`README.md`

前端控制台需求说明：`Engine_control_readme.md`

旧静态麻将桌参考原型：`index.html`

---

*文档版本: 1.1 | 最后更新: 2026-05-05*
