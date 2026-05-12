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
│   └── model_io.py                  #   Checkpoint 存取（含断点续训 save/load_resume_checkpoint）
│
├── data/                            # 数据流水线
│   ├── mjson_parser.py              #   MJSON（雀魂/Tenhou 格式）牌谱解析
│   ├── oracle.py                    #   Oracle 标签：向听数/进张/待牌质量
│   ├── record_parser.py             #   训练样本解析 (JSONL → TrainingSample)
│   ├── dataset.py                   #   PyTorch Dataset + TensorShard + 流式 IterableDataset
│   └── data_generator.py            #   内置启发式策略数据生成
│
├── training/                        # 训练脚本
│   ├── sl_pretrain.py               #   监督学习预训练（从牌谱学策略）
│   ├── benchmark.py                 #   GPU 短基准测试（batch/s、VRAM、ETA）
│   ├── agents.py                    #   ResNetAgent / TransformerAgent
│   ├── distillation.py              #   KL + value MSE 蒸馏 loss
│   ├── selfplay_recorder.py         #   Oracle 轨迹录制器
│   ├── ppo_agent.py                 #   PPO 强化学习智能体
│   ├── rl_selfplay.py               #   自对弈 RL 训练入口
│   ├── selfplay_env.py              #   引擎 → Gym 环境包装
│   ├── iterative_rl.py              #   迭代式自博弈（对抗历史快照）
│   ├── heuristic_agent.py           #   启发式基线智能体
│   ├── reward_shaper.py             #   奖励塑形（Turtle/MadDog/RiichiFund 风格）
│   ├── mjson_cache.py               #   MJSON → TensorShard 缓存构建
│   └── mjson_token_cache.py         #   MJSON → Token 序列缓存构建（Transformer 专用）
│
├── scripts/                         # 工具脚本
│   └── save_manifest.py             #   保存 train/val/test 数据 split manifest
│
├── configs/                         # YAML 配置文件
│   ├── train_default.yaml           #   默认训练超参数
│   ├── turtle.yaml                  #   Turtle 风格 RL 配置
│   ├── mad_dog.yaml                 #   MadDog 风格 RL 配置
│   └── riichi_fund.yaml             #   RiichiFundamentalist 配置
│
├── tests/                           # 单元测试（220 个，全部通过）
├── main.py                          # 命令行入口（演示/基准/Live 调试）
├── validator.py                     # 牌谱验证脚本
├── requirements.txt                 # Python 依赖
├── requirements-dev.txt             # 开发依赖（pytest）
├── setup_server.sh                  # 一键 GPU 服务器环境配置
├── run_benchmark.sh                 # 租卡后快速 GPU 基准测试
├── run_train.sh                     # 一键启动训练
└── run_resume.sh                    # 一键断点续训
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

更完整的服务器操作手册见 [`SERVER_TRAINING.md`](SERVER_TRAINING.md)。

### 快速上手（推荐流程）

```bash
# 1. 克隆 + 一键配置
git clone git@github.com:Sumnervenda/mj_tenpai.git && cd mj_tenpai
chmod +x setup_server.sh && ./setup_server.sh

# 2. 上传数据集（推荐打包后上传，避免 90 万小文件）
# 本地打包: tar -cf dataset_2021_2026.tar dataset/datasets_years/202[1-6]*
scp dataset_2021_2026.tar user@server:~/mj_tenpai/
# 服务器解压: tar -xf dataset_2021_2026.tar

# 3. 构建 Transformer token cache（训练阶段不再解析 mjson）
python -m training.mjson_token_cache build \
    --source dataset/datasets_years \
    --cache dataset/token_cache_2021-2026 \
    --years 2021,2022,2023,2024,2025,2026 \
    --num_workers 16 \
    --shard_size 10000

# 4. 先跑短基准（确认 GPU 性能，判断是否换卡）
chmod +x run_benchmark.sh && ./run_benchmark.sh

# 5. 启动训练（默认使用 token cache；step-level checkpoint 每 30 分钟自动保存）
chmod +x run_train.sh && ./run_train.sh
```

### 短基准脚本（租卡后第一步）

```bash
# 跑 1000 batches，输出 batch/s、samples/s、VRAM、预计 epoch 时长
python -m training.benchmark --checkpoint checkpoints/transformer_server/sl_best.pt

# 自定义 batch sizes 和数量
python -m training.benchmark --checkpoint sl_best.pt --sizes 128,256,512 --batches 500
```

**决策阈值**（推理速度，训练约为推理的 40%）：

| 推理 batch/s | 评价 | 预计训练耗时/epoch |
|---|---|---|
| >= 30 | Excellent | ~10h |
| 20-30 | Good | ~15h |
| < 20 | 检查配置 | >20h |

若 batch_size=512 不比 256 快，优先用 256（省显存）。若 compile 不提速或 OOM，不用 `--compile`。

### Token 缓存（消除 CPU 瓶颈）

训练时每 epoch 重复解析 mjson → 回放对局 → tokenization 420M 样本，CPU 成为瓶颈。**Token 缓存**将 tokenization 结果预先计算并压缩存储，训练时直接加载，数据读取速度提升 10-50x。

```bash
# 1. 构建 token 缓存（一次性，推荐 --num_workers 16-32 充分利用 CPU）
python -m training.mjson_token_cache build \
    --source dataset/datasets_years \
    --cache dataset/token_cache_2021-2026 \
    --years 2021,2022,2023,2024,2025,2026 \
    --num_workers 16

# 2. 查看缓存信息
python -m training.mjson_token_cache info --cache dataset/token_cache_2021-2026

# 3. 使用缓存训练（跳过流式解析）
python -m training.sl_pretrain \
    --mjson_token_cache dataset/token_cache_2021-2026 \
    --model_arch transformer \
    --epochs 10 --batch_size 256 --lr 3e-4 \
    --wandb --checkpoint_dir checkpoints/transformer_2021-2026 \
    --device cuda
```

**存储估算**：~13 GB / 420M 样本（int16 + zlib 压缩，约为原始 mjson 的 10%）。通过长度分桶（5 个桶: 1-80, 81-120, 121-160, 161-200, 201-256）最小化 padding 浪费。

**注意**：Token 缓存仅支持 `--model_arch transformer`。ResNet 使用 `--mjson_cache_dir` 和 `--data_format mjson_cache`。

### 数据 Split Manifest（确保可复现）

```bash
# 生成 manifest（train/val/test 文件列表，JSON 格式）
python scripts/save_manifest.py --root dataset/datasets_years --years 2021-2026

# 使用 manifest 训练（跨机器保证相同 split）
python -m training.sl_pretrain --manifest dataset/datasets_years/manifest_2021-2026_seed42.json ...
```

首次训练时会自动保存 `data_manifest.json` 到 checkpoint 目录。

### 上传数据集

数据集 (`dataset/datasets_years/`) 未包含在 git 中，需单独上传：

```bash
# 推荐：打包后上传（一个大文件比 90 万小文件快得多）
tar -cf dataset_2021_2026.tar dataset/datasets_years/202[1-6]*
scp dataset_2021_2026.tar user@server:~/mj_tenpai/

# 替代方案：rsync（支持断点续传）
rsync -avz --progress dataset/datasets_years/ user@server:~/mj_tenpai/dataset/datasets_years/
```

### 训练命令

```bash
# 一键启动训练（脚本内已配置所有参数，可直接修改 run_train.sh 头部配置）
./run_train.sh

# 手动启动（完整参数）
python -m training.sl_pretrain \
    --data_format mjson \
    --random_split_all_mjson dataset/datasets_years \
    --mjson_years '2021,2022,2023,2024,2025,2026' \
    --stream_mjson \
    --model_arch transformer \
    --epochs 10 --batch_size 256 --lr 3e-4 \
    --save_every 2 --save_interval_min 30 \
    --wandb --wandb_project mahjong-dl \
    --checkpoint_dir checkpoints/transformer_server \
    --device cuda
```

### 断点续训

```bash
# 一键续训（自动从 sl_resume.pt 恢复）
./run_resume.sh

# 手动续训
python -m training.sl_pretrain \
    --resume checkpoints/transformer_server/sl_resume.pt \
    --data_format mjson \
    --random_split_all_mjson dataset/datasets_years \
    --mjson_years '2021,2022,2023,2024,2025,2026' \
    --stream_mjson \
    --model_arch transformer \
    --epochs 10 --batch_size 256 --lr 3e-4 \
    --save_every 2 --save_interval_min 30 \
    --wandb --checkpoint_dir checkpoints/transformer_server \
    --device cuda
```

续训自动恢复：model weights、optimizer state、scheduler state、GradScaler state、RNG states (torch CPU/CUDA + numpy)。

### Step-level Checkpoint（Spot/抢占实例必备）

单 epoch 约 37 小时，step-level checkpoint 避免中断丢失整天进度：

```bash
# 每 30 分钟自动保存 sl_resume.pt（断了最多损失 30 分钟）
--save_interval_min 30

# 限制训练 batch 数（快速测试配置是否正确）
--max_batches 5000
```

### tmux 后台训练（推荐）

```bash
tmux new -s train                    # 创建 session
./run_train.sh                       # 执行训练
# Ctrl+B, D                          # 断开（训练继续运行）
tmux attach -t train                 # 重新连接
```

### W&B 策略

```bash
# 在线模式（默认，需联网）
export WANDB_MODE=online

# 离线模式（网络不稳时本地记录，训练后手动同步）
export WANDB_MODE=offline
# 训练结束后:
wandb sync wandb/offline-run-*
```

### Checkpoint 说明

| 文件 | 保存时机 | 内容 |
|------|---------|------|
| `sl_resume.pt` | 每个 epoch + 每 N 分钟 | 完整续训状态 + total_batches |
| `sl_best.pt` | val accuracy 创新高 | 完整续训状态 + best_val_acc |
| `sl_epoch_NNN.pt` | 每 `--save_every` 个 epoch | 完整续训状态 |
| `sl_final.pt` | 训练结束 | 完整续训状态 + test 评估结果 |
| `data_manifest.json` | 首次训练自动生成 | train/val/test 文件列表（可复现） |
| `benchmark_results.json` | 运行 benchmark 后 | 各配置吞吐量对比 |

---

## Oracle 算法

`data/oracle.py` 为 MTL 模型生成监督标签：

- **calculate_shanten**: 标准形 + 七对子 + 国士无双组合向听数
- **compute_ukeire**: 34 维有效进张布尔掩码 + 剩余牌种计数
- **classify_wait**: 待牌类型分类（两面/坎张/边张/单骑/双碰/多面）+ 质量评分

---

## sl_pretrain.py 参数速查

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_arch` | `resnet` | 模型架构：`resnet` 或 `transformer` |
| `--data_format` | `jsonl` | 数据格式：`jsonl`、`mjson`、`mjson_cache` |
| `--random_split_all_mjson` | - | 数据根目录（含年份子目录） |
| `--mjson_years` | - | 使用的年份，如 `"2021,2022,2023"` |
| `--stream_mjson` | - | 流式加载（避免 OOM） |
| `--manifest` | - | 数据 split manifest JSON 路径 |
| `--mjson_token_cache` | - | Token 缓存目录（Transformer only，替代流式解析） |
| `--mjson_cache_dir` | - | ResNet tensor 缓存目录（`data_format=mjson_cache` 时使用） |
| `--build_mjson_cache` | - | 构建 ResNet tensor 缓存 |
| `--epochs` | `10` | 训练 epoch 数 |
| `--batch_size` | `256` | batch size（256 适配 16GB 显存） |
| `--lr` | `3e-4` | 学习率 |
| `--save_every` | `5` | 每 N 个 epoch 保存编号 checkpoint |
| `--save_interval_min` | `0` | 每 N 分钟保存 step-level checkpoint（0=禁用） |
| `--max_batches` | `0` | 总 batch 数限制（0=不限制） |
| `--resume` | - | 断点续训 checkpoint 路径 |
| `--wandb` | - | 启用 Weights & Biases 日志 |
| `--wandb_project` | `mahjong-dl` | W&B 项目名 |
| `--wandb_name` | - | W&B 实验名 |
| `--device` | `cuda` | 设备：`cuda` 或 `cpu` |
| `--no_amp` | - | 禁用混合精度训练 |
| `--compile` | - | 使用 `torch.compile` 加速 |
| `--teacher_mode` | - | 启用 Teacher-Student 蒸馏 |
| `--teacher_checkpoint` | - | 冻结 Teacher 模型路径 |
| `--oracle_data` | - | Oracle 轨迹 JSONL 路径 |

---

## benchmark.py 参数速查

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--checkpoint` | - | 模型 checkpoint 路径；不传则使用随机初始化模型 |
| `--batches` | `1000` | 每个配置跑的 batch 数 |
| `--sizes` | `128,256,512` | 测试的 batch sizes |
| `--device` | `cuda` | 设备 |
| `--no_compile` | - | 跳过 `torch.compile` 测试 |
| `--output` | 自动 | JSON 结果输出路径；`none` 表示不写文件 |

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
