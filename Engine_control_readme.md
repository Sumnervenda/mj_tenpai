下面是一份面向 \*\*Codex（代码生成/自动化开发）驱动开发流程\*\* 的前端控制台 README。内容强调：\*\*模块划分、数据接口约定、状态驱动、可测试性\*\*，并直接对应你给出的控制台需求。



\---



\# 🀄 Mahjong Engine Frontend Console



\## 开发者调试控制台（README）



\---



\## 1. 项目简介



本项目是一个服务于麻将引擎的\*\*前端调试控制台（Console / Dashboard）\*\*，用于：



\* 调试核心算法（算番、向听数、听牌判断）

\* 验证状态机正确性（合法/非法操作）

\* 构造极端测试场景（役满、流局、并发冲突）

\* 可视化引擎内部状态（God Mode）



> ⚠️ 本项目 \*\*不是玩家UI\*\*，而是开发工具（DevTool）



\---



\## 2. 技术栈



| 层级   | 技术                        |

| ---- | ------------------------- |

| 框架   | React / Vue3（推荐 React）    |

| 状态管理 | Zustand / Pinia           |

| UI   | TailwindCSS + Headless UI |

| 控制面板 | Tweakpane / dat.gui       |

| 通信   | WebSocket（推荐） / HTTP      |

| 类型   | TypeScript                |



\---



\## 3. 开发方式（Codex 驱动）



\### 3.1 推荐开发流程



使用 Codex（或类似 AI 编程工具）进行模块化生成：



```bash

\# 示例 Prompt

生成一个麻将控制台的 Flow Control React 组件：

\- 包含 step / auto-play / reset

\- 使用 Zustand 管理状态

\- 支持 speed 调节

```



建议按模块拆分逐步生成：



1\. Flow Control

2\. State Inspector

3\. Cheat Panel

4\. Action Panel

5\. Logs Panel

6\. Rule Settings



\---



\## 4. 项目结构



```bash

src/

├── components/

│   ├── FlowControl/

│   ├── StateInspector/

│   ├── CheatPanel/

│   ├── ActionPanel/

│   ├── LogPanel/

│   └── RuleSettings/

│

├── store/

│   └── gameStore.ts

│

├── services/

│   └── engineAPI.ts

│

├── types/

│   └── game.ts

│

└── App.tsx

```



\---



\## 5. 核心数据结构（必须统一）



```ts

interface GameState {

&#x20; players: PlayerState\[]

&#x20; wall: Tile\[]

&#x20; deadWall: Tile\[]

&#x20; turn: number

&#x20; round: string

&#x20; honba: number

&#x20; riichiSticks: number

}



interface PlayerState {

&#x20; seat: 'east' | 'south' | 'west' | 'north'

&#x20; hand: Tile\[]

&#x20; melds: Meld\[]

&#x20; discards: Tile\[]

&#x20; score: number

&#x20; shanten: number

&#x20; tenpai: boolean

}

```



\---



\## 6. 功能模块说明



\---



\## 6.1 Flow Control（流程控制）



\### 功能



\* Reset

\* Step（单步执行）

\* Auto Play（自动播放）

\* Speed Control

\* Toggle AI



\### 示例 API



```ts

engine.step()

engine.reset()

engine.autoPlay(speed)

engine.toggleAI(seat)

```



\---



\## 6.2 State Inspector（状态监控）



\### 功能



\#### 1️⃣ 牌山透视



\* 显示 wall\[]

\* 标记：



&#x20; \* 宝牌

&#x20; \* 岭上牌



\#### 2️⃣ 全手牌可见



\* 展示所有玩家 hand



\#### 3️⃣ 向听数监控



```ts

player.shanten

player.tenpai

player.waits // 听牌列表

```



\#### 4️⃣ 原始 JSON



```json

{

&#x20; "rawState": {}

}

```



\---



\## 6.3 Cheat Panel（作弊系统）



\### 功能



\#### 发牌控制



```ts

engine.setNextDraw(seat, tile)

```



\#### 手牌编辑



```ts

engine.setHand(seat, tiles)

```



\#### 牌河修改



```ts

engine.injectDiscard(seat, tile)

```



\#### 环境修改



```ts

engine.setScore(seat, score)

engine.setRound("南4局")

engine.setHonba(3)

```



\---



\## 6.4 Action Injection（动作注入）



\### 支持操作



| 操作 | API           |

| -- | ------------- |

| 摸牌 | draw()        |

| 打牌 | discard(tile) |

| 吃  | chi()         |

| 碰  | pon()         |

| 杠  | kan()         |

| 立直 | riichi()      |

| 和牌 | win()         |



\---



\### 非法测试（关键）



```ts

engine.forceAction({

&#x20; type: "riichi",

&#x20; seat: "south",

&#x20; illegal: true

})

```



\---



\## 6.5 Logs \& Debug（日志系统）



\### Action Log



```text

\[01:05] 东家 摸 🀆

\[01:06] 东家 打 🀀

\[01:07] 南家 碰 🀀

```



\---



\### 算番详情



```ts

{

&#x20; han: 3,

&#x20; fu: 30,

&#x20; yaku: \[

&#x20;   { name: "立直", han: 1 },

&#x20;   { name: "宝牌", han: 2 }

&#x20; ]

}

```



\---



\### 回放导入导出



```ts

engine.exportReplay() // Base64

engine.importReplay(data)

```



\---



\## 6.6 Rule Settings（规则系统）



\### 可调参数



```ts

{

&#x20; kuitan: true,

&#x20; akadora: 3,

&#x20; multiRon: true,

&#x20; nagashiMangan: true,

&#x20; responsibility: true

}

```



\---



\## 7. UI 布局设计



\### 推荐结构

参考index.html
麻将牌的ui替换png素材在文件夹figures中，其中1-9p、1-9s、1-9m代表了一-九索、一-九饼
、一-九万，0s、0p、0m代表了红宝牌，E、S、W、N、white、haku、zhong代表了东南西北白发中

\---



\## 8. 与引擎通信



\### WebSocket 推荐



```ts

ws.send({

&#x20; type: "ACTION",

&#x20; payload: { ... }

})

```



监听：



```ts

ws.onmessage = (msg) => {

&#x20; updateState(msg.data)

}

```



\---



\## 9. 测试建议



\### 必测场景



\* 四杠子

\* 国士无双

\* 三家和了

\* 振听

\* 非法立直

\* 并发吃/碰/胡冲突



\---



\## 10. Codex Prompt 模板



你可以直接用这些 Prompt 生成代码：



\---



\### 生成 Flow Control



```

生成一个 React Flow Control 面板：

\- step / auto / reset 按钮

\- Zustand 状态管理

\- 支持 speed slider

```



\---



\### 生成 State Inspector



```

生成一个麻将 State Inspector：

\- 显示 wall

\- 显示所有玩家手牌

\- 显示 shanten

```



\---



\### 生成 Cheat Panel



```

生成一个麻将作弊面板：

\- dropdown 选择牌

\- 指定玩家发牌

\- 修改手牌

```



\---



\## 11. 后续扩展



\* AI 决策可视化（策略树）

\* Monte Carlo 模拟显示

\* Heatmap（打牌分布）

\* Debug Timeline（时间轴回放）



\---



\## 12. 总结



该控制台的核心设计原则：



\* \*\*状态驱动（State-driven）\*\*

\* \*\*完全可控（Deterministic Debugging）\*\*

\* \*\*可复现（Replayable）\*\*

\* \*\*可破坏（Fuzz Testing）\*\*



\---



