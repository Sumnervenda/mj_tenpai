# 代码 Review 结果

审阅日期：2026-05-06  
审阅范围：后端 Python 引擎、计分、训练、数据解析与命令行入口；按要求未审阅 `src/**`、Vite/前端界面相关实现。

## 验证结果

- `python -m pytest -q -p no:cacheprovider tests`：通过，`61 passed in 0.61s`。
- `python -c "import engine, data, models, training; print('imports ok')"`：通过。
- 对 `engine/`、`data/`、`models/`、`training/`、`main.py`、`validator.py` 做内存语法编译：30 个文件，0 个语法错误。
- `python -m compileall -q engine data models training main.py validator.py`：未能完成，原因是当前工作区部分 `__pycache__` 文件写入/替换被 Windows 拒绝访问；这更像本地文件权限/锁问题。已用内存编译替代确认源码语法。

## 主要发现

### [P1] 连庄/流局后的本场数会被新局初始化清零

位置：`engine/game.py:217-240`、`engine/game.py:321-330`

`_step_agari_settlement()` 在庄家和牌时会执行 `self.honba += 1`，流局庄家听牌时也会递增本场数；但随后 `_start_round()` 无条件执行 `self.honba = 0`。因此本场积点不会带到下一局，`compute_payments(..., honba=self.honba)` 在下一局基本拿不到正确本场数。

复现：

```powershell
python -c "from engine.game import GameEngine; e=GameEngine(seed=1); e.honba=3; e._start_round(); print(e.honba)"
```

输出为 `0`。这会直接影响日麻核心结算规则，建议把本场数的增减只放在和牌/流局结算路径，不在 `_start_round()` 重置。

### [P1] 轮庄与自风模型不一致，下一局可能出现重复自风

位置：`engine/game.py:236-251`、`engine/game.py:833-862`、`engine/game.py:324-329`、`engine/game.py:686`

当前实现把 `current_player` 固定重置为 `0`，同时多处用 `winner == 0` / `is_dealer = winner == 0` 判定庄家。`_rotate_winds()` 只修改 `seat_wind` 和 `round_number`，没有维护独立庄家索引；随后 `_start_round()` 又按玩家列表位置重算自风。

更严重的是 `_seat_wind_for()` 使用 `self.players.index(player)`。`PlayerState` 是 dataclass，`list.index()` 会走值相等判断；当前面玩家被重置成同样状态后，后续玩家可能被识别成前一个玩家，导致重复自风。

复现：

```powershell
python -c "from engine.game import GameEngine; e=GameEngine(seed=1); print([p.seat_wind for p in e.players]); e._rotate_winds(); e._start_round(); print([p.seat_wind for p in e.players], e.current_player)"
```

轮庄后可得到类似 `[27, 27, 29, 29] 0`，出现两个东、两个西，且庄家仍是 P0。建议引入 `dealer_idx` 或明确玩家座次旋转模型，并用枚举索引计算自风，避免用 dataclass 值相等查找玩家位置。

### [P1] 训练自对弈的动作索引转换会丢失具体牌/面子信息

位置：`engine/actions.py:180-191`、`training/selfplay_env.py:87-128`

`compute_draw_actions()` 会把 `mask[36]` 标为可立直，但 `legal.actions` 中没有独立的 36 号动作，只有 `37-70` 的“立直并切某张牌”。同时，当某张牌可作为立直切牌时，普通切牌动作不加入 `legal.actions`，但 `mask[0-33]` 仍标为合法，导致动作列表和 mask 不是同一个契约。

`SelfPlayEnv._action_from_index()` 又直接按索引新建动作：`CHI`、`ANKAN`、`KAKAN`、`RIICHI(36)` 都会缺少具体 `tile` 或 `meld_tiles`。例如合法吃牌动作原本带有 `tile=0, meld_tiles=[0,1,2]`，转换后变成 `tile=-1, meld_tiles=[]`，传给引擎会污染副露或触发异常。

复现：

```powershell
python -c "from engine.actions import compute_response_actions; from training.selfplay_env import SelfPlayEnv; hand=[0]*34; hand[1]=1; hand[2]=1; legal=compute_response_actions(hand,0,0,1,chi_options=[[0,1,2]]); env=object.__new__(SelfPlayEnv); a=env._action_from_index(72,1,legal); print([(int(x.action_type),x.tile,x.meld_tiles) for x in legal.actions]); print(int(a.action_type),a.tile,a.meld_tiles)"
```

输出中合法动作是 `(5, 0, [0, 1, 2])`，但转换结果是 `5 -1 []`。建议让训练环境从 `legal.actions` 中反查与动作索引匹配的具体 `Action`，或把动作空间扩展到能表达具体吃牌组合/杠牌类型。

### [P2] MJSON 解析器把大量非听牌状态标为可立直

位置：`data/mjson_parser.py:335-354`、`data/mjson_parser.py:372-382`

`build_draw_mask()` 注释写的是“门清、未立直、分数 >= 1000、听牌”，但 `_can_riichi()` 实际只检查门清、未立直、分数，不检查手牌是否听牌，也不检查每个切牌后的听牌状态。因此任意闭门手牌都会把 `37-70` 中所有持有牌标为可立直切牌。

复现：

```powershell
python -c "from data.mjson_parser import MJSONGameTracker; g=MJSONGameTracker(); g.hands[0][0]=1; g.hands[0][4]=1; g.hands[0][8]=1; print(g._can_riichi(0)); print([i for i,v in enumerate(g.build_draw_mask(0)) if v and 37 <= i <= 70])"
```

输出为 `True` 和若干立直切牌索引。这个问题会污染监督学习 mask，使模型在训练时看到不合法动作也可选。建议复用 `engine.agari.can_riichi()` 与 `get_legal_discards_for_riichi()`，并保证解析器生成的 mask 与引擎动作空间一致。

### [P2] MJSON 响应阶段缺少 pass 负样本，训练数据会偏向鸣牌/和牌

位置：`data/mjson_parser.py:614-626`

代码注释说明要记录其他玩家对弃牌选择 `pass` 的决策，但当前分支只有 `pass` 语句，没有生成样本。这样 MJSON 监督数据会更偏向“发生了 chi/pon/kan/ron 的正样本”，而缺少大量真实对局中“不鸣牌”的反例，策略头容易学出过度鸣牌/过度荣和的先验。

建议至少对有响应机会的玩家生成受控比例的 pass 样本，或在采样阶段做类别均衡，避免响应动作分布失真。

## 其他备注

- 当前 Python 单测能覆盖基础胡牌、游戏流程和接口形状，但没有覆盖轮庄后的自风唯一性、本场数延续、训练动作索引反查、MJSON 合法立直 mask 等高风险路径。
- 工作区存在 `.pytest_cache` / `__pycache__` 权限异常，以及 `git status` 提示 `.git/index.lock` 无法 unlink。提交前需要单独确认 git 锁文件状态。
