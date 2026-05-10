#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_train.sh — 云端启动训练（一步到位）
#
# 用法:
#   chmod +x run_train.sh && ./run_train.sh
#
# 前置条件:
#   1. 已运行 setup_server.sh 配置环境
#   2. 已上传 dataset/ 或解压 dataset_2021_2026.tar
#   3. 已设置 WANDB_API_KEY（或 WANDB_MODE=offline）
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── 配置（按需修改）───────────────────────────────────────────────────────────
DATA_DIR="dataset/datasets_years"
YEARS="2021,2022,2023,2024,2025,2026"
MODEL_ARCH="transformer"
EPOCHS=10
BATCH_SIZE=256
LR=3e-4
SAVE_EVERY=2
SAVE_INTERVAL_MIN=30          # 每 30 分钟保存 step-level checkpoint
CHECKPOINT_DIR="checkpoints/transformer_server"
WANDB_PROJECT="mahjong-dl"
WANDB_NAME="transformer_sl_$(date +%Y%m%d_%H%M)"
MANIFEST=""                   # 留空则自动生成; 或指定 manifest JSON 路径
DEVICE="cuda"

# ── 数据解压（如果需要）─────────────────────────────────────────────────────
if [ ! -d "$DATA_DIR" ] && [ -f "dataset_2021_2026.tar" ]; then
    echo "解压数据集..."
    tar -xf dataset_2021_2026.tar
    echo "解压完成"
fi

if [ ! -d "$DATA_DIR" ]; then
    echo "错误: 数据集目录不存在: $DATA_DIR"
    echo "请先上传数据集: scp -r dataset/datasets_years/ user@server:~/mj_tenpai/dataset/"
    exit 1
fi

# ── W&B 模式 ──
if [ -z "${WANDB_API_KEY:-}" ] && [ "${WANDB_MODE:-}" != "offline" ]; then
    echo "WANDB_API_KEY 未设置，切换到 offline 模式"
    export WANDB_MODE=offline
fi

# ── manifest 参数 ──
MANIFEST_ARG=""
if [ -n "$MANIFEST" ]; then
    MANIFEST_ARG="--manifest $MANIFEST"
fi

# ── 启动训练 ──
echo "=========================================="
echo "  Training Start"
echo "=========================================="
echo "  Model:      $MODEL_ARCH"
echo "  Epochs:     $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  LR:         $LR"
echo "  Save every: $SAVE_EVERY epochs + every ${SAVE_INTERVAL_MIN} min"
echo "  Checkpoint: $CHECKPOINT_DIR"
echo "  W&B:        ${WANDB_MODE:-online}"
echo "=========================================="

# tmux 检测
if [ -z "${TMUX:-}" ]; then
    echo ""
    echo "建议在 tmux 中运行以防止断连:"
    echo "  tmux new -s train"
    echo "  ./run_train.sh"
    echo ""
fi

python -m training.sl_pretrain \
    --data_format mjson \
    --random_split_all_mjson "$DATA_DIR" \
    --mjson_years "$YEARS" \
    --stream_mjson \
    --model_arch "$MODEL_ARCH" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --save_every "$SAVE_EVERY" \
    --save_interval_min "$SAVE_INTERVAL_MIN" \
    --num_workers 8 \
    --wandb \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_name "$WANDB_NAME" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --device "$DEVICE" \
    $MANIFEST_ARG
