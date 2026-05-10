#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_benchmark.sh — 租卡后快速评估 GPU 性能
#
# 用法:
#   chmod +x run_benchmark.sh && ./run_benchmark.sh [checkpoint_path]
#
# 输出:
#   batch/s, samples/s, VRAM, 预计 epoch 时长
#   用于判断 4090 / A100 / H100 哪个真实性价比最高
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

CKPT="${1:-}"
BATCHES="${BENCHMARK_BATCHES:-1000}"

echo "=========================================="
echo "  GPU Benchmark"
echo "=========================================="

# GPU 信息
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "No GPU detected"
echo ""

# 构建参数
BENCH_ARGS="--batches $BATCHES --sizes 128,256,512"
if [ -n "$CKPT" ] && [ -f "$CKPT" ]; then
    echo "Checkpoint: $CKPT"
    BENCH_ARGS="$BENCH_ARGS --checkpoint $CKPT"
else
    echo "No checkpoint found, using randomly initialized model"
fi
echo ""

# 运行基准测试
python -m training.benchmark $BENCH_ARGS

echo ""
echo "=========================================="
echo "  Decision Thresholds"
echo "=========================================="
echo "  >= 30 batch/s (inference) -> training ~12+ batch/s -> Excellent"
echo "  20-30 batch/s             -> training ~8-12 batch/s -> Good"
echo "  < 20 batch/s              -> check config or try smaller batch_size"
echo ""
echo "  若 batch_size=512 不比 256 快，优先用 256（省显存）"
echo "  若 compile 不提速或 OOM，不用 --compile"
