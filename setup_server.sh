#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# setup_server.sh — 一键配置 GPU 训练服务器环境
#
# 用法:
#   chmod +x setup_server.sh && ./setup_server.sh
#
# 功能:
#   1. 检测 GPU 和 CUDA 版本
#   2. 创建 Python 虚拟环境
#   3. 安装 PyTorch (CUDA 版本自动匹配)
#   4. 安装项目依赖
#   5. 验证 CUDA 可用
#   6. 检查数据集目录
#   7. 运行冒烟测试
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

VENV_DIR=".venv"
PYTHON="${PYTHON:-python3}"
DATA_DIR="${DATA_DIR:-dataset/datasets_years}"
TOKEN_CACHE_DIR="${TOKEN_CACHE_DIR:-dataset/token_cache_2021-2026}"
YEARS="${YEARS:-2021,2022,2023,2024,2025,2026}"
TORCH_CUDA_TAG="${TORCH_CUDA_TAG:-}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"
ALLOW_CPU="${ALLOW_CPU:-0}"
RUN_TESTS="${RUN_TESTS:-1}"

# ── 颜色输出 ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
fail()  { echo -e "${RED}[FAIL]${NC}  $*"; }

# ── 0. Python 版本检查 ──────────────────────────────────────────────────────
info "检查 Python..."
$PYTHON - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit(f"Python >= 3.10 required, got {sys.version.split()[0]}")
print(f"  Python: {sys.version.split()[0]}")
PY

# ── 1. 检测 GPU ──────────────────────────────────────────────────────────────
info "检测 GPU 环境..."

CUDA_VERSION=""
GPU_EXPECTED=0
if command -v nvidia-smi &>/dev/null; then
    GPU_EXPECTED=1
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
    CUDA_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    ok "GPU: ${GPU_NAME} (${GPU_MEM}), Driver: ${CUDA_DRIVER}"

    # 从 nvcc 或 nvidia-smi 推断 CUDA 版本
    if command -v nvcc &>/dev/null; then
        CUDA_VERSION=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
        ok "CUDA Toolkit: ${CUDA_VERSION}"
    else
        # 从 driver version 推断最高支持的 CUDA 版本
        DRIVER_MAJOR=$(echo "$CUDA_DRIVER" | cut -d. -f1)
        if [ "$DRIVER_MAJOR" -ge 550 ]; then
            CUDA_VERSION="12.4"
        elif [ "$DRIVER_MAJOR" -ge 535 ]; then
            CUDA_VERSION="12.2"
        elif [ "$DRIVER_MAJOR" -ge 525 ]; then
            CUDA_VERSION="12.0"
        else
            CUDA_VERSION="11.8"
        fi
        warn "nvcc 未安装，从 driver 推断 CUDA: ${CUDA_VERSION}"
    fi
else
    warn "nvidia-smi 未找到，将安装 CPU 版 PyTorch"
    CUDA_VERSION="cpu"
fi

# ── 2. 创建虚拟环境 ──────────────────────────────────────────────────────────
info "创建 Python 虚拟环境: ${VENV_DIR}/"

if [ -d "$VENV_DIR" ]; then
    warn "虚拟环境已存在: ${VENV_DIR}/，跳过创建"
else
    $PYTHON -m venv "$VENV_DIR"
    ok "虚拟环境创建完成"
fi

# 激活虚拟环境
source "${VENV_DIR}/bin/activate"
ok "已激活虚拟环境: $(python --version)"
python -m pip install --upgrade pip setuptools wheel -q

# ── 3. 安装 PyTorch ─────────────────────────────────────────────────────────
info "安装 PyTorch..."

if [ -n "$TORCH_INDEX_URL" ]; then
    info "使用自定义 PyTorch index: ${TORCH_INDEX_URL}"
    python -m pip install torch --index-url "$TORCH_INDEX_URL" -q
elif [ "$CUDA_VERSION" = "cpu" ]; then
    python -m pip install torch --index-url https://download.pytorch.org/whl/cpu -q
else
    # 将 CUDA 版本映射到 PyTorch whl index
    CUDA_SHORT=$(echo "$CUDA_VERSION" | tr -d '.')
    # PyTorch 2.x/后续版本常见 index：cu118, cu121, cu124, cu126, cu128
    if [ -n "$TORCH_CUDA_TAG" ]; then
        TORCH_CUDA="$TORCH_CUDA_TAG"
    elif [ "$CUDA_SHORT" -ge 128 ]; then
        TORCH_CUDA="cu128"
    elif [ "$CUDA_SHORT" -ge 126 ]; then
        TORCH_CUDA="cu126"
    elif [ "$CUDA_SHORT" -ge 124 ]; then
        TORCH_CUDA="cu124"
    elif [ "$CUDA_SHORT" -ge 121 ]; then
        TORCH_CUDA="cu121"
    else
        TORCH_CUDA="cu118"
    fi
    info "PyTorch CUDA index: ${TORCH_CUDA}"
    python -m pip install torch --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}" -q
fi
ok "PyTorch 安装完成: $(python -c 'import torch; print(torch.__version__)')"

# ── 4. 安装项目依赖 ─────────────────────────────────────────────────────────
info "安装项目依赖..."
python -m pip install -r requirements.txt -q
if [ -f requirements-dev.txt ]; then
    python -m pip install -r requirements-dev.txt -q
else
    python -m pip install pytest -q
fi
ok "依赖安装完成"

# ── 5. 验证 CUDA ─────────────────────────────────────────────────────────────
info "验证 CUDA 可用性..."
python - <<PY
import torch
gpu_expected = bool(int('${GPU_EXPECTED}'))
allow_cpu = bool(int('${ALLOW_CPU}'))
cuda_ok = torch.cuda.is_available()
if cuda_ok:
    name = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f'  CUDA: {torch.version.cuda}')
    print(f'  GPU:  {name} ({mem:.1f} GB)')
    print(f'  cuDNN: {torch.backends.cudnn.version()}')
else:
    print('  WARNING: CUDA 不可用，将使用 CPU 训练（速度极慢）')
    if gpu_expected and not allow_cpu:
        raise SystemExit(
            'nvidia-smi 可见但 torch.cuda.is_available() 为 False。'
            '请检查 PyTorch CUDA wheel / Driver 版本，或设置 ALLOW_CPU=1 跳过。')
PY
ok "CUDA 验证通过"

# ── 6. 检查数据集 ────────────────────────────────────────────────────────────
info "检查数据集..."
if [ -d "$DATA_DIR" ]; then
    YEAR_COUNT=$(ls -d ${DATA_DIR}/20*/ 2>/dev/null | wc -l)
    ok "数据集目录: ${DATA_DIR}/ (${YEAR_COUNT} 个年份)"
    ls -d ${DATA_DIR}/20*/ 2>/dev/null | while read d; do
        COUNT=$(find "$d" -maxdepth 1 -type f \( -name '*.mjson' -o -name '*.mjson.gz' \) 2>/dev/null | wc -l)
        echo "    $(basename $d): ${COUNT} 场对局"
    done
else
    warn "数据集目录不存在: ${DATA_DIR}/"
    echo ""
    echo "  请上传数据集到服务器:"
    echo "    scp -r dataset/datasets_years/ user@server:~/mj_tenpai/dataset/"
    echo ""
fi

# ── 7. 检查 token cache ─────────────────────────────────────────────────────
info "检查 Transformer token cache..."
if [ -f "$TOKEN_CACHE_DIR/manifest.json" ]; then
    python -m training.mjson_token_cache info --cache "$TOKEN_CACHE_DIR" | head -40
    ok "token cache 可用: $TOKEN_CACHE_DIR"
else
    warn "token cache 未完成: $TOKEN_CACHE_DIR/manifest.json"
    echo "  推荐先构建一次，之后训练不再解析 mjson:"
    echo "    python -m training.mjson_token_cache build \\"
    echo "      --source \"$DATA_DIR\" \\"
    echo "      --cache \"$TOKEN_CACHE_DIR\" \\"
    echo "      --years \"$YEARS\" \\"
    echo "      --num_workers 16 \\"
    echo "      --shard_size 10000"
fi

# ── 8. 运行冒烟测试 ──────────────────────────────────────────────────────────
if [ "$RUN_TESTS" = "1" ]; then
    info "运行冒烟测试..."
    mkdir -p .tmp
    python -m training.benchmark \
        --device "$(python -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")')" \
        --batches 1 \
        --sizes 1 \
        --no_compile \
        --output .tmp/setup_benchmark_results.json
    python -m pytest tests/ -x -q -p no:cacheprovider --tb=short
    ok "冒烟测试完成"
else
    warn "跳过冒烟测试 (RUN_TESTS=0)"
fi

# ── 9. 打印训练命令 ─────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  环境配置完成！推荐操作流程："
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "  source .venv/bin/activate"
echo ""
echo "  # 1. 构建 token cache（只需要一次；训练阶段 0 mjson 解析）"
echo "  python -m training.mjson_token_cache build \\"
echo "    --source dataset/datasets_years \\"
echo "    --cache dataset/token_cache_2021-2026 \\"
echo "    --years 2021,2022,2023,2024,2025,2026 \\"
echo "    --num_workers 16 \\"
echo "    --shard_size 10000"
echo ""
echo "  # 2. 先跑短基准（确认 GPU 性能，判断是否换卡）"
echo "  ./run_benchmark.sh"
echo ""
echo "  # 3. 启动训练（默认使用 token cache，每 30 分钟保存 checkpoint）"
echo "  ./run_train.sh"
echo ""
echo "  # 4. 断点续训"
echo "  ./run_resume.sh"
echo ""
echo "  # 5. 如果临时不用 token cache，可回退到 mjson 流式解析"
echo "  DATA_MODE=stream_mjson ./run_train.sh"
echo ""
echo "  # 6. 生成数据 manifest（stream_mjson 模式下确保可复现）"
echo "  python scripts/save_manifest.py --root dataset/datasets_years --years 2021-2026"
echo ""
echo "  # 决策阈值（推理 batch/s，训练约为 40%）:"
echo "  #   >= 30 -> Excellent    20-30 -> Good    < 20 -> 检查配置"
echo ""
echo "════════════════════════════════════════════════════════════════"
