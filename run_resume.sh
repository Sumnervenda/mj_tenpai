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
DATA_DIR="${DATA_DIR:-dataset/datasets_years}"
TOKEN_CACHE_DIR="${TOKEN_CACHE_DIR:-dataset/token_cache_2021-2026}"
TOKEN_MMAP_DIR="${TOKEN_MMAP_DIR:-dataset/token_mmap_2021-2026}"
DATA_MODE="${DATA_MODE:-token_cache}"     # token_mmap | token_cache | stream_mjson
AUTO_BUILD_TOKEN_CACHE="${AUTO_BUILD_TOKEN_CACHE:-0}"
YEARS="${YEARS:-2021,2022,2023,2024,2025,2026}"
MODEL_ARCH="${MODEL_ARCH:-transformer}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LR="${LR:-3e-4}"
SAVE_EVERY="${SAVE_EVERY:-2}"
SAVE_INTERVAL_MIN="${SAVE_INTERVAL_MIN:-30}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$(dirname "$CKPT")}"
WANDB_PROJECT="${WANDB_PROJECT:-mahjong-dl}"
WANDB_NAME="${WANDB_NAME:-transformer_sl_resume_$(date +%Y%m%d_%H%M)}"
USE_WANDB="${USE_WANDB:-1}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-8}"
CACHE_WORKERS="${CACHE_WORKERS:-16}"
CACHE_SHARD_SIZE="${CACHE_SHARD_SIZE:-10000}"
MMAP_SHARD_SIZE="${MMAP_SHARD_SIZE:-200000}"
CACHE_OVERWRITE="${CACHE_OVERWRITE:-0}"

die() {
    echo "错误: $*" >&2
    exit 1
}

if [ -d ".venv" ] && [ -z "${VIRTUAL_ENV:-}" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

# ── 检查 checkpoint ──
if [ ! -f "$CKPT" ]; then
    echo "错误: checkpoint 不存在: $CKPT"
    echo "可用的 checkpoint:"
    ls -lh checkpoints/transformer_server/*.pt 2>/dev/null || echo "  (无)"
    exit 1
fi

NEEDS_DATA_DIR=0
if [ "$DATA_MODE" = "stream_mjson" ]; then
    NEEDS_DATA_DIR=1
fi

if [ "$NEEDS_DATA_DIR" = "1" ] && [ ! -d "$DATA_DIR" ] && [ -f "dataset_2021_2026.tar" ]; then
    echo "解压数据集..."
    tar -xf dataset_2021_2026.tar
    echo "解压完成"
fi

if [ "$NEEDS_DATA_DIR" = "1" ] && [ ! -d "$DATA_DIR" ]; then
    die "数据集目录不存在: $DATA_DIR"
fi

if [ "$MODEL_ARCH" != "transformer" ] && { [ "$DATA_MODE" = "token_cache" ] || [ "$DATA_MODE" = "token_mmap" ]; }; then
    die "DATA_MODE=$DATA_MODE 仅支持 MODEL_ARCH=transformer"
fi

# ── 数据模式 ────────────────────────────────────────────────────────────────
DATA_ARGS=()
if [ "$DATA_MODE" = "token_mmap" ]; then
    if [ ! -f "$TOKEN_MMAP_DIR/manifest.json" ]; then
        if [ "$AUTO_BUILD_TOKEN_CACHE" = "1" ]; then
            BUILD_ARGS=()
            if [ -f "$TOKEN_CACHE_DIR/manifest.json" ]; then
                echo "未找到 token mmap，开始从 token cache 转换: $TOKEN_MMAP_DIR"
                BUILD_ARGS=(
                    convert-mmap
                    --source_cache "$TOKEN_CACHE_DIR"
                    --cache "$TOKEN_MMAP_DIR"
                    --shard_size "$MMAP_SHARD_SIZE"
                )
            else
                if [ ! -d "$DATA_DIR" ]; then
                    die "无法从 mjson 构建 token mmap，数据集目录不存在: $DATA_DIR"
                fi
                echo "未找到 token mmap/token cache，开始从 mjson 构建 mmap: $TOKEN_MMAP_DIR"
                BUILD_ARGS=(
                    build-mmap
                    --source "$DATA_DIR"
                    --cache "$TOKEN_MMAP_DIR"
                    --years "$YEARS"
                    --num_workers "$CACHE_WORKERS"
                    --shard_size "$MMAP_SHARD_SIZE"
                )
            fi
            if [ "$CACHE_OVERWRITE" = "1" ]; then
                BUILD_ARGS+=(--overwrite)
            fi
            python -m training.mjson_token_cache "${BUILD_ARGS[@]}"
        else
            cat >&2 <<EOF
错误: token mmap cache 不存在: $TOKEN_MMAP_DIR/manifest.json

从已有 token cache 转换:
  python -m training.mjson_token_cache convert-mmap \\
    --source_cache "$TOKEN_CACHE_DIR" \\
    --cache "$TOKEN_MMAP_DIR" \\
    --shard_size "$MMAP_SHARD_SIZE"

或让脚本自动转换/构建:
  DATA_MODE=token_mmap AUTO_BUILD_TOKEN_CACHE=1 ./run_resume.sh "$CKPT"
EOF
            exit 1
        fi
    fi
    python -m training.mjson_token_cache info --cache "$TOKEN_MMAP_DIR" >/dev/null
    DATA_ARGS=(--mjson_token_mmap "$TOKEN_MMAP_DIR")
elif [ "$DATA_MODE" = "token_cache" ]; then
    if [ ! -f "$TOKEN_CACHE_DIR/manifest.json" ]; then
        if [ "$AUTO_BUILD_TOKEN_CACHE" = "1" ]; then
            if [ ! -d "$DATA_DIR" ]; then
                die "无法构建 token cache，数据集目录不存在: $DATA_DIR"
            fi
            echo "未找到 token cache manifest，开始构建: $TOKEN_CACHE_DIR"
            BUILD_ARGS=(
                build
                --source "$DATA_DIR" \
                --cache "$TOKEN_CACHE_DIR" \
                --years "$YEARS" \
                --num_workers "$CACHE_WORKERS" \
                --shard_size "$CACHE_SHARD_SIZE"
            )
            if [ "$CACHE_OVERWRITE" = "1" ]; then
                BUILD_ARGS+=(--overwrite)
            fi
            python -m training.mjson_token_cache "${BUILD_ARGS[@]}"
        else
            cat >&2 <<EOF
错误: token cache 未完成或不存在: $TOKEN_CACHE_DIR/manifest.json

续训建议使用和原训练相同的数据 split。若这是 token cache 训练，请先构建:
  python -m training.mjson_token_cache build \\
    --source "$DATA_DIR" \\
    --cache "$TOKEN_CACHE_DIR" \\
    --years "$YEARS" \\
    --num_workers "$CACHE_WORKERS" \\
    --shard_size "$CACHE_SHARD_SIZE"

如果上一次构建中断，覆盖重建:
  AUTO_BUILD_TOKEN_CACHE=1 CACHE_OVERWRITE=1 ./run_resume.sh "$CKPT"

临时回退到 mjson 流式解析:
  DATA_MODE=stream_mjson ./run_resume.sh "$CKPT"
EOF
            exit 1
        fi
    fi
    python -m training.mjson_token_cache info --cache "$TOKEN_CACHE_DIR" >/dev/null
    DATA_ARGS=(--mjson_token_cache "$TOKEN_CACHE_DIR")
elif [ "$DATA_MODE" = "stream_mjson" ]; then
    DATA_ARGS=(
        --data_format mjson
        --random_split_all_mjson "$DATA_DIR"
        --mjson_years "$YEARS"
        --stream_mjson
    )
    MANIFEST_FILE="$CHECKPOINT_DIR/data_manifest.json"
    if [ -f "$MANIFEST_FILE" ]; then
        DATA_ARGS+=(--manifest "$MANIFEST_FILE")
        echo "Using manifest: $MANIFEST_FILE"
    fi
else
    die "未知 DATA_MODE=$DATA_MODE，可选: token_mmap | token_cache | stream_mjson"
fi

# ── W&B 模式 ──
if [ "$USE_WANDB" = "1" ] && [ -z "${WANDB_API_KEY:-}" ] && [ "${WANDB_MODE:-}" != "offline" ]; then
    echo "WANDB_API_KEY 未设置，切换到 offline 模式"
    export WANDB_MODE=offline
fi

# ── 启动续训 ──
echo "=========================================="
echo "  Resume Training"
echo "=========================================="
echo "  Checkpoint: $CKPT"
echo "  Data mode:  $DATA_MODE"
if [ "$DATA_MODE" = "token_cache" ]; then
    echo "  Token cache: $TOKEN_CACHE_DIR"
elif [ "$DATA_MODE" = "token_mmap" ]; then
    echo "  Token mmap:  $TOKEN_MMAP_DIR"
else
    echo "  MJSON data: $DATA_DIR"
fi
echo "  Batch size: $BATCH_SIZE"
echo "  Save every: $SAVE_EVERY epochs + every ${SAVE_INTERVAL_MIN} min"
echo "  W&B:        $([ "$USE_WANDB" = "1" ] && echo "${WANDB_MODE:-online}" || echo "disabled")"
echo "=========================================="

TRAIN_ARGS=(
    --resume "$CKPT"
    --model_arch "$MODEL_ARCH"
    --epochs "$EPOCHS"
    --batch_size "$BATCH_SIZE"
    --lr "$LR"
    --save_every "$SAVE_EVERY"
    --save_interval_min "$SAVE_INTERVAL_MIN"
    --num_workers "$NUM_WORKERS"
    --checkpoint_dir "$CHECKPOINT_DIR"
    --device "$DEVICE"
)

if [ "$USE_WANDB" = "1" ]; then
    TRAIN_ARGS+=(
        --wandb
        --wandb_project "$WANDB_PROJECT"
        --wandb_name "$WANDB_NAME"
    )
fi

python -m training.sl_pretrain "${DATA_ARGS[@]}" "${TRAIN_ARGS[@]}"
