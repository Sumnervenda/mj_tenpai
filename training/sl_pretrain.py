"""阶段一：监督学习预训练。

加载对局记录 → 提取状态-动作对 → 训练 1D-CNN ResNet 模型 → 保存 Base Model。

用法:
    python -m training.sl_pretrain --data records/heuristic_games.jsonl --epochs 50
    python -m training.sl_pretrain --data records/ --generate 500  # 自动生成500局训练数据
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models import MahjongPolicyValueNet, save_checkpoint, load_checkpoint
from data import (
    JSONLRecordParser,
    MJSONRecordParser,
    MahjongStateActionDataset,
    MJSONIterableDataset,
)


def parse_args():
    parser = argparse.ArgumentParser(description='SL Pretraining for Mahjong AI')
    parser.add_argument('--data', type=str, default='records/',
                        help='JSONL/MJSON file or directory containing game records')
    parser.add_argument('--train_data', type=str, default=None,
                        help='Optional explicit training dataset path')
    parser.add_argument('--val_data', type=str, default=None,
                        help='Optional explicit validation dataset path')
    parser.add_argument('--test_data', type=str, default=None,
                        help='Optional explicit test dataset path')
    parser.add_argument('--data_format', type=str, default='jsonl',
                        choices=['jsonl', 'mjson'],
                        help='Game record format (default: jsonl)')
    parser.add_argument('--max_mjson_files', type=int, default=0,
                        help='Max .mjson files to load (0 = all, for huge datasets)')
    parser.add_argument('--max_val_mjson_files', type=int, default=0,
                        help='Max validation .mjson files to load (0 = all)')
    parser.add_argument('--max_test_mjson_files', type=int, default=0,
                        help='Max test .mjson files to load (0 = all)')
    parser.add_argument('--generate', type=int, default=0,
                        help='If >0, generate N games with heuristic agent instead')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--num_workers', type=int, default=12,
                        help='Number of DataLoader worker processes')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device: cuda or cpu')
    parser.add_argument('--no_amp', action='store_true',
                        help='Disable automatic mixed precision')
    parser.add_argument('--compile', action='store_true',
                        help='Use torch.compile for faster training (PyTorch 2.0+)')
    parser.add_argument('--stream_mjson', action='store_true',
                        help='Stream MJSON samples from files instead of loading all into memory')
    parser.add_argument('--random_split_all_mjson', type=str, default=None,
                        help='Root directory of all yearly MJSON files to split randomly at file level')
    parser.add_argument('--split_seed', type=int, default=42,
                        help='Random seed for full-dataset file split')
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='mahjong-dl',
                        help='W&B project name')
    parser.add_argument('--wandb_name', type=str, default=None,
                        help='W&B run name (default: auto-generated)')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='W&B entity/team name')
    return parser.parse_args()


def load_data(data_path: str, data_format: str = 'jsonl',
              max_mjson_files: int = 0) -> MahjongStateActionDataset:
    """从 JSONL/MJSON 文件/目录加载数据。

    Args:
        data_path: 数据文件或目录路径
        data_format: 'jsonl' 或 'mjson'（默认 jsonl）
        max_mjson_files: MJSON 模式下限制读取文件数（0=全部）
    """
    p = Path(data_path)

    # 自动检测：如果路径包含 .mjson 扩展名，强制使用 mjson 格式
    if data_format == 'mjson' or (p.is_file() and p.suffix == '.mjson') or \
       (p.is_dir() and any(any(Path(dp).glob('*.mjson*')) for dp, _, _ in os.walk(data_path))):
        parser = MJSONRecordParser()
        if p.is_dir():
            samples = parser.parse_directory(data_path, max_files=max_mjson_files)
        else:
            samples = parser.parse_file(data_path)
    else:
        parser = JSONLRecordParser()
        if p.is_dir():
            samples = parser.parse_directory(data_path)
        else:
            samples = parser.parse_file(data_path)

    if not samples:
        raise ValueError(f"No training samples found in {data_path}")

    dataset = MahjongStateActionDataset(samples)
    print(f"Loaded {len(dataset)} samples from {data_path}")
    return dataset


def load_explicit_split(train_path: str, val_path: str, test_path: str,
                        data_format: str = 'jsonl',
                        max_train_mjson_files: int = 0,
                        max_val_mjson_files: int = 0,
                        max_test_mjson_files: int = 0) -> tuple[MahjongStateActionDataset, MahjongStateActionDataset, MahjongStateActionDataset]:
    train_set = load_data(train_path, data_format=data_format,
                          max_mjson_files=max_train_mjson_files)
    val_set = load_data(val_path, data_format=data_format,
                        max_mjson_files=max_val_mjson_files)
    test_set = load_data(test_path, data_format=data_format,
                         max_mjson_files=max_test_mjson_files)
    return train_set, val_set, test_set


def collect_mjson_files(root_dir: str, max_files: int = 0) -> List[str]:
    pattern = os.path.join(root_dir, '**', '*.mjson')
    files = sorted(str(p) for p in Path(root_dir).glob('**/*.mjson'))
    if max_files > 0:
        files = files[:max_files]
    if not files:
        raise ValueError(f'No .mjson files found under {root_dir}')
    return files


def split_mjson_files(file_paths: List[str], train_ratio: float,
                      val_ratio: float, test_ratio: float,
                      seed: int) -> tuple[List[str], List[str], List[str]]:
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError('train_ratio + val_ratio + test_ratio must equal 1.0')

    rng = np.random.default_rng(seed)
    shuffled = list(file_paths)
    rng.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_files = shuffled[:train_end]
    val_files = shuffled[train_end:val_end]
    test_files = shuffled[val_end:]

    if not train_files or not val_files or not test_files:
        raise ValueError('Train/val/test file splits must all be non-empty')
    return train_files, val_files, test_files


def build_streaming_dataset(file_paths: List[str], shuffle_files: bool,
                            seed: int) -> MJSONIterableDataset:
    return MJSONIterableDataset(
        file_paths=file_paths,
        shuffle_files=shuffle_files,
        seed=seed,
        parser_verbose=False,
    )


def train_epoch(model: nn.Module, dataloader: DataLoader,
                optimizer: torch.optim.Optimizer, device: str,
                scaler: Optional[GradScaler] = None) -> dict:
    """训练一个 epoch，返回指标。"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    use_amp = scaler is not None

    for states, masks, labels in dataloader:
        states = states.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast('cuda', enabled=use_amp):
            logits, _ = model(states, masks)
            loss = nn.CrossEntropyLoss()(logits, labels)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * states.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += states.size(0)

    return {
        'loss': total_loss / total,
        'accuracy': correct / total,
    }


def validate(model: nn.Module, dataloader: DataLoader, device: str) -> dict:
    """验证，返回指标。"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for states, masks, labels in dataloader:
            states = states.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            logits, _ = model(states, masks)
            loss = nn.CrossEntropyLoss()(logits, labels)

            total_loss += loss.item() * states.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += states.size(0)

    return {
        'loss': total_loss / total,
        'accuracy': correct / total,
    }


def main():
    args = parse_args()

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'

    use_cuda = device == 'cuda'
    if use_cuda:
        torch.backends.cudnn.benchmark = True
        print(f"GPU: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory // 1024**2:,} MB VRAM)")

    # 数据加载 or 生成
    if args.generate > 0:
        from .heuristic_agent import generate_training_data
        print(f"Generating {args.generate} heuristic games...")
        args.data = generate_training_data(args.generate, seed=0)
        print(f"Generated data saved to {args.data}")

    streaming_mode = False

    if args.random_split_all_mjson:
        if args.data_format != 'mjson':
            raise ValueError('random_split_all_mjson requires --data_format mjson')
        if not args.stream_mjson:
            raise ValueError('random_split_all_mjson requires --stream_mjson to avoid exhausting memory')

        all_files = collect_mjson_files(args.random_split_all_mjson,
                                        max_files=args.max_mjson_files)
        train_files, val_files, test_files = split_mjson_files(
            all_files,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.split_seed,
        )
        print(f"Random file split: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")

        streaming_mode = True
        train_set = build_streaming_dataset(train_files, shuffle_files=True,
                                            seed=args.split_seed)
        val_set = build_streaming_dataset(val_files, shuffle_files=False,
                                          seed=args.split_seed)
        test_set = build_streaming_dataset(test_files, shuffle_files=False,
                                           seed=args.split_seed)
    elif args.train_data or args.val_data or args.test_data:
        if not (args.train_data and args.test_data):
            raise ValueError('train_data and test_data must be provided together when using explicit dataset paths')

        if args.val_data:
            train_set, val_set, test_set = load_explicit_split(
                args.train_data,
                args.val_data,
                args.test_data,
                data_format=args.data_format,
                max_train_mjson_files=args.max_mjson_files,
                max_val_mjson_files=args.max_val_mjson_files,
                max_test_mjson_files=args.max_test_mjson_files,
            )
            print(f"Explicit split: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")
        else:
            train_full = load_data(args.train_data, data_format=args.data_format,
                                   max_mjson_files=args.max_mjson_files)
            train_set, val_set, _ = train_full.train_val_test_split(ratios=(0.9, 0.1, 0.0))
            test_set = load_data(args.test_data, data_format=args.data_format,
                                 max_mjson_files=args.max_test_mjson_files)
            print(f"Cross-year split: train={len(train_set)}, val={len(val_set)} (from train_data), test={len(test_set)}")
    else:
        dataset = load_data(args.data, data_format=args.data_format,
                            max_mjson_files=args.max_mjson_files)

        # 拆分
        train_set, val_set, test_set = dataset.train_val_test_split()
        print(f"Split: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")

    if not streaming_mode and (len(train_set) == 0 or len(val_set) == 0 or len(test_set) == 0):
        raise ValueError('Train/val/test datasets must all be non-empty')

    dl_kwargs = dict(
        pin_memory=use_cuda,
        num_workers=args.num_workers,
        persistent_workers=use_cuda and args.num_workers > 0,
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=not streaming_mode, drop_last=True, **dl_kwargs)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            shuffle=False, **dl_kwargs)

    # 模型
    model = MahjongPolicyValueNet()
    model = model.to(device)

    if args.compile and hasattr(torch, 'compile'):
        model = torch.compile(model, mode='reduce-overhead')
        print("Model compiled with torch.compile")

    print(f"Model: {model.count_parameters():,} parameters")

    optimizer = AdamW(model.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 混合精度
    scaler = GradScaler('cuda', enabled=use_cuda and not args.no_amp)

    start_epoch = 0
    if args.resume:
        start_epoch, meta = load_checkpoint(
            model, args.resume, optimizer=optimizer, device=device)
        print(f"Resumed from epoch {start_epoch}")

    # 训练循环
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0
    history = []

    amp_status = "AMP" if (use_cuda and not args.no_amp) else "FP32"
    print(f"\nTraining on {device} ({amp_status}) for {args.epochs} epochs...")
    print(f"{'Epoch':>6} {'Train Loss':>12} {'Train Acc':>10} "
          f"{'Val Loss':>10} {'Val Acc':>8} {'Time':>8} {'LR':>10}")

    # ── wandb 初始化 ──
    wandb_run = None
    if args.wandb:
        import wandb
        run_name = args.wandb_name or f"sl_{Path(args.checkpoint_dir).name}"
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            entity=args.wandb_entity,
            config={
                'model': model.count_parameters(),
                'dataset': args.data or args.random_split_all_mjson or args.train_data,
                'data_format': args.data_format,
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'lr': args.lr,
                'weight_decay': args.weight_decay,
                'device': device,
                'amp': use_cuda and not args.no_amp,
                'compile': args.compile,
            },
            reinit=True,
        )
        wandb.watch(model, log='gradients', log_freq=100)

    for epoch in range(start_epoch, start_epoch + args.epochs):
        t0 = time.time()

        if streaming_mode and hasattr(train_set, 'set_epoch'):
            train_set.set_epoch(epoch)

        train_metrics = train_epoch(
            model, train_loader, optimizer, device, scaler)
        val_metrics = validate(model, val_loader, device)

        scheduler.step()
        elapsed = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'train_acc': train_metrics['accuracy'],
            'val_loss': val_metrics['loss'],
            'val_acc': val_metrics['accuracy'],
        })

        print(f"{epoch + 1:>6} {train_metrics['loss']:>12.4f} "
              f"{train_metrics['accuracy']:>10.4f} "
              f"{val_metrics['loss']:>10.4f} {val_metrics['accuracy']:>8.4f} "
              f"{elapsed:>7.1f}s {current_lr:>10.2e}")

        if wandb_run is not None:
            wandb.log({
                'train/loss': train_metrics['loss'],
                'train/accuracy': train_metrics['accuracy'],
                'val/loss': val_metrics['loss'],
                'val/accuracy': val_metrics['accuracy'],
                'lr': current_lr,
                'epoch': epoch + 1,
            }, step=epoch + 1)

        # 保存最佳模型
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            save_checkpoint(
                model, optimizer, epoch + 1,
                os.path.join(args.checkpoint_dir, 'sl_best.pt'),
                metadata={'val_acc': best_val_acc, 'config': vars(args)},
            )

        # 定期保存
        if (epoch + 1) % 5 == 0:
            save_checkpoint(
                model, optimizer, epoch + 1,
                os.path.join(args.checkpoint_dir, f'sl_epoch_{epoch + 1:03d}.pt'),
            )

    # 最终保存
    save_checkpoint(
        model, optimizer, start_epoch + args.epochs,
        os.path.join(args.checkpoint_dir, 'sl_final.pt'),
        metadata={'history': history, 'best_val_acc': best_val_acc},
    )

    # 测试集评估
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                             shuffle=False, **dl_kwargs)
    test_metrics = validate(model, test_loader, device)
    print(f"\nTest: loss={test_metrics['loss']:.4f}, "
          f"accuracy={test_metrics['accuracy']:.4f}")
    print(f"Best val accuracy: {best_val_acc:.4f}")
    print(f"Model saved to {args.checkpoint_dir}")

    if wandb_run is not None:
        wandb.log({
            'test/loss': test_metrics['loss'],
            'test/accuracy': test_metrics['accuracy'],
            'best_val_accuracy': best_val_acc,
        })
        wandb.finish()


if __name__ == '__main__':
    main()
