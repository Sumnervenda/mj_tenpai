#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_resume.sh — 从断点续训
#
# 用法:
#   chmod +x run_resume.sh && ./run_resume.sh [checkpoint_path]
#
# 默认从 checkpoints/transformer_server/sl_resume.pt 续训
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

CKPT="${1:-checkpoints/transformer_server/sl_resume.pt}"

# ── 配置（与 run_train.sh 保持一致）───────────────────────────────────────────
DATA_DIR="dataset/datasets_years"
YEARS="2021,2022,2023,2024,2025,2026"
MODEL_ARCH="transformer"
EPOCHS=10
BATCH_SIZE=256
LR=3e-4
SAVE_EVERY=2
SAVE_INTERVAL_MIN=30
CHECKPOINT_DIR="checkpoints/transformer_server"
WANDB_PROJECT="mahjong-dl"
WANDB_NAME="transformer_sl_resume_$(date +%Y%m%d_%H%M)"
DEVICE="cuda"

# ── 检查 checkpoint ──
if [ ! -f "$CKPT" ]; then
    echo "错误: checkpoint 不存在: $CKPT"
    echo "可用的 checkpoint:"
    ls -lh checkpoints/transformer_server/*.pt 2>/dev/null || echo "  (无)"
    exit 1
fi

# ── W&B 模式 ──
if [ -z "${WANDB_API_KEY:-}" ] && [ "${WANDB_MODE:-}" != "offline" ]; then
    echo "WANDB_API_KEY 未设置，切换到 offline 模式"
    export WANDB_MODE=offline
fi

# ── 从 checkpoint 读取 manifest 路径（如果存在）──
MANIFEST_ARG=""
MANIFEST_FILE="$CHECKPOINT_DIR/data_manifest.json"
if [ -f "$MANIFEST_FILE" ]; then
    MANIFEST_ARG="--manifest $MANIFEST_FILE"
    echo "Using manifest: $MANIFEST_FILE"
fi

# ── 启动续训 ──
echo "=========================================="
echo "  Resume Training"
echo "=========================================="
echo "  Checkpoint: $CKPT"
echo "  Batch size: $BATCH_SIZE"
echo "  Save every: $SAVE_EVERY epochs + every ${SAVE_INTERVAL_MIN} min"
echo "  W&B:        ${WANDB_MODE:-online}"
echo "=========================================="

python -m training.sl_pretrain \
    --resume "$CKPT" \
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
