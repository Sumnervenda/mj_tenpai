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
