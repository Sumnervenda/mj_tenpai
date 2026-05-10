# 日麻 AI 训练框架 (Riichi Mahjong AI Training Framework)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-220%20passed-brightgreen.svg)](.)

基于 PyTorch 的立直麻将 AI 训练框架，包含完整的日麻规则引擎、两种神经网络架构（ResNet1D 基线 + Transformer MTL）、监督学习与强化学习训练流水线。

## 当前训练状态

| 架构 | 流水线状态 | 备注 |
|------|-----------|------|
| **ResNet1D** | 已接入完整 SL/RL 训练流水线 | 主训练架构 |
| **Transformer MTL** | SL 训练流水线已集成，断点续训已支持 | 主训练架构 |

Transformer MTL 进度：
- 已完成：Tokenizer、Backbone、MTL Heads（6头）、Oracle 标签（shanten/ukeire）、SL 训练入口（`--model_arch transformer`）、断点续训（`--resume`）、wandb 集成、Teacher-Student 蒸馏
- 开发中：Danger/Score head 标签、RL fine-tuning

---

## 架构概览

```
mj_tenpai/
├── engine/                          # 日麻规则引擎（纯 Python）
│   ├── tile.py                      #   牌编码：绝对 ID / 类型 ID / 赤宝牌
│   ├── wall.py                      #   牌山、王牌、宝牌指示牌管理
│   ├── hand.py                      #   手牌 int[34] 直方图 + 副露
│   ├── agari.py                     #   LUT 查表法 O(1) 胡牌 / 听牌判定
│   ├── yaku.py                      #   38 种役种判定
│   ├── scoring.py                   #   符翻计算 + 点数结算
│   ├── actions.py                   #   77 维动作空间定义 + 合法动作掩码
│   ├── rules.py                     #   可配置规则（食断、赤宝牌等开关）
│   ├── game.py                      #   事件驱动状态机（DRAW / DISCARD / AGARI）
│   └── interface.py                 #   JSON 契约层（序列化/反序列化）
│
├── models/                          # 神经网络模型
│   ├── feature_encoder.py           #   354 维状态 → (10,34) 空间 + (14,) 元数据
│   ├── resnet1d.py                  #   1D ResNet 骨干（沿 34 牌型轴卷积）
│   ├── policy_value_net.py          #   ResNet 策略-价值双头网络
│   ├── tokenizer.py                 #   局面 Token 化（~120 token 序列）
│   ├── transformer_backbone.py      #   Pre-LN Transformer Encoder
│   ├── multi_task_heads.py          #   6 头 MTL（向听/牌效/危险度/打点/策略/价值）
│   ├── transformer_policy_value.py  #   Transformer + Concept Token + MTL 完整网络
│   └── model_io.py                  #   Checkpoint 存取
│
├── data/                            # 数据流水线
│   ├── mjson_parser.py              #   MJSON（雀魂/Tenhou 格式）牌谱解析
│   ├── oracle.py                    #   Oracle 标签：向听数/进张/待牌质量
│   ├── record_parser.py            #   训练样本解析 (JSONL → TrainingSample)
│   ├── dataset.py                   #   PyTorch Dataset + TensorShard
│   └── data_generator.py            #   内置启发式策略数据生成
│
├── training/                        # 训练脚本
│   ├── sl_pretrain.py               #   监督学习预训练（从牌谱学策略）
│   ├── ppo_agent.py                 #   PPO 强化学习智能体
│   ├── rl_selfplay.py               #   自对弈 RL 训练入口
│   ├── selfplay_env.py              #   引擎 → Gym 环境包装
│   ├── iterative_rl.py              #   迭代式自博弈（对抗历史快照）
│   ├── heuristic_agent.py           #   启发式基线智能体
│   ├── reward_shaper.py             #   奖励塑形（Turtle/MadDog/RiichiFund 风格）
│   └── mjson_cache.py               #   MJSON → TensorShard 缓存构建
│
├── configs/                         # YAML 配置文件
│   ├── train_default.yaml           #   默认训练超参数
│   ├── turtle.yaml                  #   Turtle 风格 RL 配置
│   ├── mad_dog.yaml                 #   MadDog 风格 RL 配置
│   └── riichi_fund.yaml            #   RiichiFundamentalist 配置
│
├── tests/                           # 单元测试（172 个，全部通过）
├── main.py                          # 命令行入口（演示/基准/Live 调试）
├── validator.py                     # 牌谱验证脚本
└── requirements.txt                 # Python 依赖
```

---

## 快速开始

### 安装

```bash
pip install -r requirements.txt
```

### 引擎演示

```bash
# 单局随机对局
python main.py

# 指定随机种子
python main.py --seed 12345

# 性能基准（100局）
python main.py --benchmark 100

# 批量随机对局记录
python main.py --record-random 10000 --record-output records/random_10000 --seed 42
```

### 运行测试

```bash
python -m pytest tests/ -v
```

---

## 两种网络架构

### ResNet1D 基线

```
354 维 state → Encoder 拆分
  → 空间 (B,10,34) → ResNet1D → (B,128)
  → 元数据 (B,14) → MLP → (B,128)
  → Concat (B,256) → 策略头 (B,77) + 价值头 (B,1)
```

局部卷积感受野，适合快速原型和基线对比。

### Transformer + Multi-Task Learning

```
Token 序列 (~120 tokens) → Token/Type/Behavior Embedding
  → + 10 Concept Tokens (Semantic Bottleneck)
  → 6× TransformerBlock (Pre-LN, 8 heads, GELU)
  → 提取 Concept Outputs → 6 任务头
```

| Head | Concept | 输出 | Loss |
|------|---------|------|------|
| Shanten | [0:2] | 7 类 (0-6 向听) | CrossEntropy |
| Efficiency | [2:4] | 3 标量 | MSE |
| Danger | [4:6] | 34 标量 | MSE |
| Score | [6:7] | 1 标量 | MSE |
| Policy | [7:8] | 77 logits | CrossEntropy (Masked) |
| Value | [8:10] | 1 标量 [-1,1] | MSE |

Self-Attention 天然捕捉长距离时序依赖（下家数巡前的手切、立直现物变化、宝牌周边危险度）。

---

## 引擎核心设计

### 双轨牌编码
- **绝对 ID (0-135)**: 每张牌物理唯一标识，用于牌山、牌河、宝牌指示牌
- **类型 ID (0-33)**: 34 种逻辑牌型，用于 `int[34]` 手牌直方图与算法计算

### LUT 查表胡牌
`engine/agari.py` BFS 预计算面子/雀头组合表，五进制编码 → O(1) 查询。支持一般形（4面子+1雀头）、七对子、国士无双。

### 事件驱动状态机
`DRAW ↔ DISCARD` 循环，切牌后按「荣和 → 杠/碰 → 吃 → PASS」优先级仲裁。

### 训练接口
- **77 维** 合法动作掩码（切牌 34 + 立直切牌 34 + 自摸/荣和/立直/碰/吃/大明杠/暗杠/加杠/PASS 各 1）
- **354 维** 状态特征张量
- `clone()` 支持 MCTS/Rollout

---

## 训练流程

### 监督学习预训练

从 Tenhou/雀魂牌谱学习策略先验：

```bash
python -m training.sl_pretrain \
    --config configs/train_default.yaml \
    --epochs 10 \
    --batch_size 2048
```

### PPO 强化学习

自对弈探索 + 奖励塑形：

```bash
python -m training.rl_selfplay \
    --base_model checkpoints/sl_best.pt \
    --personality configs/turtle.yaml \
    --total_timesteps 2000000
```

### 迭代式自博弈

Trainee 逐版对抗自身历史快照：

```bash
python -m training.iterative_rl \
    --base_model checkpoints/sl_best.pt \
    --personality configs/turtle.yaml \
    --num_versions 5
```

---

## 奖励塑形风格

| 风格 | 特点 |
|------|------|
| **Turtle** (龟) | 避免放铳，重视防守，四位惩罚 |
| **MadDog** (狂犬) | 积极进攻，一位奖励，和牌得分 |
| **RiichiFundamentalist** | 门清立直至上，副露惩罚 |

---

## 云服务器训练

### 一键配置

```bash
# 克隆仓库
git clone git@github.com:Sumnervenda/mj_tenpai.git
cd mj_tenpai

# 一键配置（自动检测 GPU、安装 PyTorch + CUDA、验证环境）
chmod +x setup_server.sh && ./setup_server.sh
```

### 手动配置

```bash
# 1. 创建虚拟环境
python3 -m venv .venv && source .venv/bin/activate

# 2. 安装 PyTorch（根据 CUDA 版本选择）
pip install torch --index-url https://download.pytorch.org/whl/cu124  # CUDA 12.4
# pip install torch --index-url https://download.pytorch.org/whl/cu118  # CUDA 11.8

# 3. 安装依赖
pip install -r requirements.txt

# 4. 验证
python -c "import torch; print(torch.cuda.is_available())"
```

### 上传数据集

数据集 (`dataset/datasets_years/`) 未包含在 git 中，需单独上传：

```bash
# 从本地上传到服务器
scp -r dataset/datasets_years/ user@server:~/mj_tenpai/dataset/

# 或使用 rsync（支持断点续传）
rsync -avz --progress dataset/datasets_years/ user@server:~/mj_tenpai/dataset/datasets_years/
```

### 训练命令

```bash
# Transformer SL 预训练（2021-2026 数据，batch_size=256 适配 16GB 显存）
python -m training.sl_pretrain \
    --data_format mjson \
    --random_split_all_mjson dataset/datasets_years \
    --mjson_years '2021,2022,2023,2024,2025,2026' \
    --stream_mjson \
    --model_arch transformer \
    --epochs 10 --batch_size 256 --lr 3e-4 \
    --save_every 2 --wandb \
    --wandb_project mahjong-dl \
    --wandb_name transformer_sl_server \
    --checkpoint_dir checkpoints/transformer_server \
    --device cuda

# 从断点续训（自动恢复 model/optimizer/scheduler/RNG 状态）
python -m training.sl_pretrain \
    --resume checkpoints/transformer_server/sl_resume.pt \
    --data_format mjson \
    --random_split_all_mjson dataset/datasets_years \
    --mjson_years '2021,2022,2023,2024,2025,2026' \
    --stream_mjson \
    --model_arch transformer \
    --epochs 10 --batch_size 256 --lr 3e-4 \
    --save_every 2 --wandb \
    --checkpoint_dir checkpoints/transformer_server \
    --device cuda
```

### tmux 后台训练（推荐）

```bash
tmux new -s train                    # 创建 session
# 执行训练命令...
# Ctrl+B, D                          # 断开（训练继续运行）
tmux attach -t train                 # 重新连接
```

### Checkpoint 说明

| 文件 | 保存时机 | 内容 |
|------|---------|------|
| `sl_resume.pt` | 每个 epoch 结束 | 完整续训状态（model + optimizer + scheduler + scaler + RNG） |
| `sl_best.pt` | val accuracy 创新高 | 完整续训状态 + best_val_acc |
| `sl_epoch_NNN.pt` | 每 `--save_every` 个 epoch | 完整续训状态 |
| `sl_final.pt` | 训练结束 | 完整续训状态 + test 评估结果 |

---

## Oracle 算法

`data/oracle.py` 为 MTL 模型生成监督标签：

- **calculate_shanten**: 标准形 + 七对子 + 国士无双组合向听数
- **compute_ukeire**: 34 维有效进张布尔掩码 + 剩余牌种计数
- **classify_wait**: 待牌类型分类（两面/坎张/边张/单骑/双碰/多面）+ 质量评分

---

## 依赖

```
torch >= 2.0 (CUDA 12.x 推荐)
numpy >= 1.24
pyyaml >= 6.0
wandb >= 0.17
orjson >= 3.10
```

开发依赖：`pytest >= 7.0`（见 `requirements-dev.txt`）

---

## License

MIT
