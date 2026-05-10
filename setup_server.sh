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
DATA_DIR="dataset/datasets_years"

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

# ── 1. 检测 GPU ──────────────────────────────────────────────────────────────
info "检测 GPU 环境..."

CUDA_VERSION=""
if command -v nvidia-smi &>/dev/null; then
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

# ── 3. 安装 PyTorch ─────────────────────────────────────────────────────────
info "安装 PyTorch..."

if [ "$CUDA_VERSION" = "cpu" ]; then
    pip install torch --index-url https://download.pytorch.org/whl/cpu -q
else
    # 将 CUDA 版本映射到 PyTorch whl index
    CUDA_SHORT=$(echo "$CUDA_VERSION" | tr -d '.')
    # PyTorch 2.x 支持 cu118, cu121, cu124, cu126
    if [ "$CUDA_SHORT" -ge 126 ]; then
        TORCH_CUDA="cu126"
    elif [ "$CUDA_SHORT" -ge 124 ]; then
        TORCH_CUDA="cu124"
    elif [ "$CUDA_SHORT" -ge 121 ]; then
        TORCH_CUDA="cu121"
    else
        TORCH_CUDA="cu118"
    fi
    info "PyTorch CUDA index: ${TORCH_CUDA}"
    pip install torch --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}" -q
fi
ok "PyTorch 安装完成: $(python -c 'import torch; print(torch.__version__)')"

# ── 4. 安装项目依赖 ─────────────────────────────────────────────────────────
info "安装项目依赖..."
pip install -r requirements.txt -q
pip install pytest -q
ok "依赖安装完成"

# ── 5. 验证 CUDA ─────────────────────────────────────────────────────────────
info "验证 CUDA 可用性..."
python -c "
import torch
cuda_ok = torch.cuda.is_available()
if cuda_ok:
    name = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
    print(f'  CUDA: {torch.version.cuda}')
    print(f'  GPU:  {name} ({mem:.1f} GB)')
    print(f'  cuDNN: {torch.backends.cudnn.version()}')
else:
    print('  WARNING: CUDA 不可用，将使用 CPU 训练（速度极慢）')
" && ok "CUDA 验证通过" || warn "CUDA 不可用"

# ── 6. 检查数据集 ────────────────────────────────────────────────────────────
info "检查数据集..."
if [ -d "$DATA_DIR" ]; then
    YEAR_COUNT=$(ls -d ${DATA_DIR}/20*/ 2>/dev/null | wc -l)
    ok "数据集目录: ${DATA_DIR}/ (${YEAR_COUNT} 个年份)"
    ls -d ${DATA_DIR}/20*/ 2>/dev/null | while read d; do
        COUNT=$(ls "$d"*.mjson 2>/dev/null | wc -l)
        echo "    $(basename $d): ${COUNT} 场对局"
    done
else
    fail "数据集目录不存在: ${DATA_DIR}/"
    echo ""
    echo "  请上传数据集到服务器:"
    echo "    scp -r dataset/datasets_years/ user@server:~/mj_tenpai/dataset/"
    echo ""
fi

# ── 7. 运行冒烟测试 ──────────────────────────────────────────────────────────
info "运行冒烟测试..."
python -m pytest tests/ -x -q --tb=short 2>&1 | tail -5
ok "冒烟测试完成"

# ── 8. 打印训练命令 ─────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  环境配置完成！以下是常用训练命令："
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "  # 激活环境"
echo "  source .venv/bin/activate"
echo ""
echo "  # Transformer SL 预训练 (2021-2026 数据)"
echo "  python -m training.sl_pretrain \\"
echo "      --data_format mjson \\"
echo "      --random_split_all_mjson dataset/datasets_years \\"
echo "      --mjson_years '2021,2022,2023,2024,2025,2026' \\"
echo "      --stream_mjson \\"
echo "      --model_arch transformer \\"
echo "      --epochs 10 --batch_size 256 --lr 3e-4 \\"
echo "      --save_every 2 --wandb \\"
echo "      --wandb_project mahjong-dl \\"
echo "      --wandb_name transformer_sl_server \\"
echo "      --checkpoint_dir checkpoints/transformer_server \\"
echo "      --device cuda"
echo ""
echo "  # 从断点续训"
echo "  python -m training.sl_pretrain \\"
echo "      --resume checkpoints/transformer_server/sl_resume.pt \\"
echo "      ... (其他参数同上)"
echo ""
echo "  # 后台训练（推荐 tmux）"
echo "  tmux new -s train"
echo "  # 执行训练命令..."
echo "  # Ctrl+B, D 断开; tmux attach -t train 重新连接"
echo ""
echo "════════════════════════════════════════════════════════════════"
