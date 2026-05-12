"""模型 checkpoint 存取工具。"""

from pathlib import Path
from typing import Any, Dict, Optional

import torch


def save_checkpoint(model: torch.nn.Module,
                    optimizer: Optional[torch.optim.Optimizer],
                    epoch: int,
                    path: str,
                    metadata: Optional[Dict[str, Any]] = None) -> None:
    """保存模型 checkpoint。

    Args:
        model: PyTorch 模型
        optimizer: 优化器（可为 None）
        epoch: 当前 epoch / step
        path: 保存路径（.pt 或 .pth）
        metadata: 额外元数据（配置、超参数等）
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'metadata': metadata or {},
    }
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    torch.save(checkpoint, path)


def load_checkpoint_metadata(path: str) -> Dict[str, Any]:
    """只读取 checkpoint metadata，不加载权重。

    用于在构造模型之前自动检测架构（ResNet / Transformer）。

    Args:
        path: checkpoint 文件路径

    Returns:
        metadata dict（包含 model_arch, d_model, n_layers 等字段）
    """
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    meta = checkpoint.get('metadata', {})
    meta.setdefault('epoch', checkpoint.get('epoch', 0))
    return meta


def infer_transformer_config_from_state_dict(
        state_dict: dict,
        metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """从 state_dict 推断 Transformer 配置，用于 metadata 不完整的 checkpoint。

    推断顺序：state_dict 形状优先；metadata 只补充形状无法可靠推断的字段。
    这样可以避免旧 checkpoint 的 metadata 与实际权重形状冲突时构造出
    无法 load 或语义错配的模型。

    Args:
        state_dict: 模型 state_dict
        metadata: 已有的 metadata（可为 None）

    Returns:
        配置 dict，包含 model_arch/d_model/n_layers/n_heads/n_concept/max_len
    """
    meta = dict(metadata or {})
    meta.setdefault('model_arch', 'transformer')

    # d_model: token_embedding.weight shape [vocab_size, d_model]
    if 'token_embedding.weight' in state_dict:
        meta['d_model'] = state_dict['token_embedding.weight'].shape[1]

    # n_concept: concept_tokens shape [n_concept, d_model]
    if 'concept_tokens' in state_dict:
        meta['n_concept'] = state_dict['concept_tokens'].shape[0]

    # max_len: pos_embedding shape [1, max_len, d_model]
    if 'backbone.pos_embedding' in state_dict:
        meta['max_len'] = state_dict['backbone.pos_embedding'].shape[1]

    # n_layers: count unique backbone.blocks.<idx> prefixes
    layer_ids = set()
    for k in state_dict:
        if k.startswith('backbone.blocks.'):
            parts = k.split('.')
            if len(parts) >= 3 and parts[2].isdigit():
                layer_ids.add(int(parts[2]))
    if layer_ids:
        meta['n_layers'] = max(layer_ids) + 1

    # n_heads 无法从权重形状可靠推断，保留为 None
    return meta


def load_checkpoint(model: torch.nn.Module,
                    path: str,
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    device: str = 'cpu') -> tuple[int, Dict[str, Any]]:
    """加载模型 checkpoint。

    Args:
        model: 待加载权重的模型（原地修改）
        path: checkpoint 文件路径
        optimizer: 优化器（可为 None，不恢复优化器状态）
        device: 加载设备

    Returns:
        (epoch, metadata): checkpoint 中的 epoch 和元数据
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint.get('epoch', 0)
    metadata = checkpoint.get('metadata', {})
    return epoch, metadata


# ── 断点续训（full resume）checkpoint ──────────────────────────────────────────

def save_resume_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    scaler: Optional[Any],
    epoch: int,
    path: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """保存完整断点续训 checkpoint（model + optimizer + scheduler + scaler + RNG）。

    Args:
        model: PyTorch 模型
        optimizer: 优化器
        scheduler: 学习率调度器（CosineAnnealingLR 等，可为 None）
        scaler: GradScaler（AMP，可为 None）
        epoch: 已完成的 epoch 数（下次从 epoch+1 开始）
        path: 保存路径（.pt）
        metadata: 额外元数据（history、best_val_acc 等）
    """
    import numpy as np
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    rng_state = {
        'torch_cpu': torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        rng_state['torch_cuda'] = torch.cuda.get_rng_state()
    rng_state['numpy'] = np.random.get_state()

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'metadata': metadata or {},
        'rng_state': rng_state,
    }
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()
    torch.save(checkpoint, path)


def load_resume_checkpoint(
    model: torch.nn.Module,
    path: str,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    device: str = 'cpu',
) -> tuple[int, Dict[str, Any]]:
    """加载断点续训 checkpoint，恢复 model / optimizer / scheduler / scaler / RNG。

    Args:
        model: 待加载权重的模型（原地修改）
        path: checkpoint 文件路径
        optimizer: 优化器（原地恢复）
        scheduler: 学习率调度器（可为 None）
        scaler: GradScaler（可为 None）
        device: 加载设备

    Returns:
        (epoch, metadata): checkpoint 中的 epoch 和元数据
    """
    import numpy as np
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    # 恢复随机数状态，确保续训后数据顺序一致
    rng_state = checkpoint.get('rng_state', {})
    if 'torch_cpu' in rng_state:
        torch.random.set_rng_state(rng_state['torch_cpu'])
    if 'torch_cuda' in rng_state and torch.cuda.is_available():
        torch.cuda.set_rng_state(rng_state['torch_cuda'])
    if 'numpy' in rng_state:
        np.random.set_state(rng_state['numpy'])

    epoch = checkpoint.get('epoch', 0)
    metadata = checkpoint.get('metadata', {})
    return epoch, metadata
# 中文注释：模型 checkpoint 的保存与加载工具，统一处理权重、优化器和元数据。
