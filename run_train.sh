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
DATA_DIR="${DATA_DIR:-dataset/datasets_years}"
TOKEN_CACHE_DIR="${TOKEN_CACHE_DIR:-dataset/token_cache_2021-2026}"
DATA_MODE="${DATA_MODE:-token_cache}"     # token_cache | stream_mjson
AUTO_BUILD_TOKEN_CACHE="${AUTO_BUILD_TOKEN_CACHE:-0}"
YEARS="${YEARS:-2021,2022,2023,2024,2025,2026}"
MODEL_ARCH="${MODEL_ARCH:-transformer}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LR="${LR:-3e-4}"
SAVE_EVERY="${SAVE_EVERY:-2}"
SAVE_INTERVAL_MIN="${SAVE_INTERVAL_MIN:-30}"  # 每 N 分钟保存 step-level checkpoint
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/transformer_server}"
WANDB_PROJECT="${WANDB_PROJECT:-mahjong-dl}"
WANDB_NAME="${WANDB_NAME:-transformer_sl_$(date +%Y%m%d_%H%M)}"
USE_WANDB="${USE_WANDB:-1}"
MANIFEST="${MANIFEST:-}"                   # stream_mjson 模式下可指定 manifest JSON
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-8}"
CACHE_WORKERS="${CACHE_WORKERS:-16}"
CACHE_SHARD_SIZE="${CACHE_SHARD_SIZE:-10000}"
CACHE_OVERWRITE="${CACHE_OVERWRITE:-0}"

die() {
    echo "错误: $*" >&2
    exit 1
}

if [ -d ".venv" ] && [ -z "${VIRTUAL_ENV:-}" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

# ── 数据解压（如果需要）─────────────────────────────────────────────────────
if [ ! -d "$DATA_DIR" ] && [ -f "dataset_2021_2026.tar" ]; then
    echo "解压数据集..."
    tar -xf dataset_2021_2026.tar
    echo "解压完成"
fi

if [ ! -d "$DATA_DIR" ]; then
    echo "错误: 数据集目录不存在: $DATA_DIR" >&2
    echo "请先上传数据集: scp -r dataset/datasets_years/ user@server:~/mj_tenpai/dataset/"
    exit 1
fi

if [ "$MODEL_ARCH" != "transformer" ] && [ "$DATA_MODE" = "token_cache" ]; then
    die "DATA_MODE=token_cache 仅支持 MODEL_ARCH=transformer"
fi

# ── 数据模式：默认使用 token cache，避免每个 epoch 解析 mjson ───────────────
DATA_ARGS=()
if [ "$DATA_MODE" = "token_cache" ]; then
    if [ ! -f "$TOKEN_CACHE_DIR/manifest.json" ]; then
        if [ "$AUTO_BUILD_TOKEN_CACHE" = "1" ]; then
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

推荐先构建一次，之后训练将不再解析 mjson:
  python -m training.mjson_token_cache build \\
    --source "$DATA_DIR" \\
    --cache "$TOKEN_CACHE_DIR" \\
    --years "$YEARS" \\
    --num_workers "$CACHE_WORKERS" \\
    --shard_size "$CACHE_SHARD_SIZE"

或者让脚本自动先构建再训练:
  AUTO_BUILD_TOKEN_CACHE=1 ./run_train.sh

如果上一次构建中断，覆盖重建:
  AUTO_BUILD_TOKEN_CACHE=1 CACHE_OVERWRITE=1 ./run_train.sh

临时回退到每 epoch 流式解析:
  DATA_MODE=stream_mjson ./run_train.sh
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
    if [ -n "$MANIFEST" ]; then
        DATA_ARGS+=(--manifest "$MANIFEST")
    fi
else
    die "未知 DATA_MODE=$DATA_MODE，可选: token_cache | stream_mjson"
fi

# ── W&B 模式 ──
if [ "$USE_WANDB" = "1" ] && [ -z "${WANDB_API_KEY:-}" ] && [ "${WANDB_MODE:-}" != "offline" ]; then
    echo "WANDB_API_KEY 未设置，切换到 offline 模式"
    export WANDB_MODE=offline
fi

# ── 启动训练 ──
echo "=========================================="
echo "  Training Start"
echo "=========================================="
echo "  Model:      $MODEL_ARCH"
echo "  Data mode:  $DATA_MODE"
if [ "$DATA_MODE" = "token_cache" ]; then
echo "  Token cache: $TOKEN_CACHE_DIR"
else
echo "  MJSON data: $DATA_DIR"
fi
echo "  Epochs:     $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  LR:         $LR"
echo "  Save every: $SAVE_EVERY epochs + every ${SAVE_INTERVAL_MIN} min"
echo "  Checkpoint: $CHECKPOINT_DIR"
echo "  W&B:        $([ "$USE_WANDB" = "1" ] && echo "${WANDB_MODE:-online}" || echo "disabled")"
echo "=========================================="

# tmux 检测
if [ -z "${TMUX:-}" ]; then
    echo ""
    echo "建议在 tmux 中运行以防止断连:"
    echo "  tmux new -s train"
    echo "  ./run_train.sh"
    echo ""
fi

TRAIN_ARGS=(
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
