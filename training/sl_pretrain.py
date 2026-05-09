"""阶段一：监督学习预训练。

加载对局记录 → 提取状态-动作对 → 训练 1D-CNN ResNet 模型 → 保存 Base Model。

用法:
    python -m training.sl_pretrain --data records/heuristic_games.jsonl --epochs 50
    python -m training.sl_pretrain --data records/ --generate 500  # 自动生成500局训练数据
"""

import argparse
import csv
import json
import os
import multiprocessing as mp
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

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
    TensorShardBatchDataset,
)
from training.mjson_cache import (
    build_mjson_cache,
    collect_mjson_files as collect_mjson_files_fast,
    load_mjson_cache_manifest,
    shard_paths_for_split,
    split_mjson_files as split_mjson_files_fast,
)


_MP_WORKERS_AVAILABLE: Optional[bool] = None
_MP_WORKER_WARNING_EMITTED = False

DEFAULT_METRIC_KEYS = (
    'train/loss',
    'train/accuracy',
    'val/loss',
    'val/accuracy',
    'test/loss',
    'test/accuracy',
    'best_val_accuracy',
    'lr',
    'time/epoch_sec',
)

METRIC_DESCRIPTIONS = {
    'train/loss': 'Training CrossEntropyLoss over human action labels; lower is better.',
    'train/accuracy': 'Training top-1 action accuracy against the recorded human action.',
    'val/loss': 'Validation CrossEntropyLoss on held-out records; lower is better.',
    'val/accuracy': 'Validation top-1 action accuracy on held-out records; higher is better.',
    'test/loss': 'Final test CrossEntropyLoss after training completes.',
    'test/accuracy': 'Final test top-1 action accuracy after training completes.',
    'best_val_accuracy': 'Best validation accuracy observed across epochs.',
    'lr': 'Optimizer learning rate after the scheduler step.',
    'time/epoch_sec': 'Wall-clock seconds spent on one training epoch plus validation.',
}

LEGACY_METRIC_ALIASES = {
    'train/loss': ('train_loss',),
    'train/accuracy': ('train_acc', 'train_accuracy'),
    'val/loss': ('val_loss',),
    'val/accuracy': ('val_acc', 'val_accuracy'),
    'test/loss': ('test_loss',),
    'test/accuracy': ('test_acc', 'test_accuracy'),
}


def configure_local_runtime_dirs(checkpoint_dir: str) -> None:
    """Keep training runtime artifacts in the project tree.

    Some Windows environments deny writes to AppData/Temp for subprocess
    services such as wandb-core. Pointing temp and W&B directories at the
    checkpoint tree makes the command reproducible in restricted shells.
    """
    runtime_root = Path(checkpoint_dir).resolve() / "_runtime"
    temp_dir = runtime_root / "tmp"
    wandb_dir = runtime_root / "wandb"
    wandb_cache = runtime_root / "wandb_cache"
    wandb_config = runtime_root / "wandb_config"
    wandb_data = runtime_root / "wandb_data"

    for path in (temp_dir, wandb_dir, wandb_cache, wandb_config, wandb_data):
        path.mkdir(parents=True, exist_ok=True)

    for key in ("TMP", "TEMP", "TMPDIR"):
        os.environ[key] = str(temp_dir)
    os.environ.setdefault("WANDB_DIR", str(wandb_dir))
    os.environ.setdefault("WANDB_CACHE_DIR", str(wandb_cache))
    os.environ.setdefault("WANDB_CONFIG_DIR", str(wandb_config))
    os.environ.setdefault("WANDB_DATA_DIR", str(wandb_data))
    tempfile.tempdir = str(temp_dir)


def multiprocessing_workers_available(num_workers: int) -> bool:
    """Return whether multiprocessing queues can be created in this shell."""
    if num_workers <= 0:
        return True
    global _MP_WORKERS_AVAILABLE, _MP_WORKER_WARNING_EMITTED
    if _MP_WORKERS_AVAILABLE is not None:
        if not _MP_WORKERS_AVAILABLE and not _MP_WORKER_WARNING_EMITTED:
            print(
                "WARNING: Multiprocessing workers/services are unavailable "
                "in this shell; falling back to single-process mode.",
                flush=True,
            )
            _MP_WORKER_WARNING_EMITTED = True
        return _MP_WORKERS_AVAILABLE

    try:
        queue = mp.get_context().Queue()
        queue.close()
        queue.join_thread()
        _MP_WORKERS_AVAILABLE = True
    except (PermissionError, OSError) as exc:
        _MP_WORKERS_AVAILABLE = False
        if not _MP_WORKER_WARNING_EMITTED:
            print(
                f"WARNING: Multiprocessing workers/services are unavailable "
                f"in this shell ({exc}); falling back to single-process mode.",
                flush=True,
            )
            _MP_WORKER_WARNING_EMITTED = True
    return _MP_WORKERS_AVAILABLE


def make_dataloader(dataset, batch_size: int, shuffle: bool, drop_last: bool,
                    use_cuda: bool, requested_workers: int,
                    prebatched: bool = False,
                    prefetch_factor: int = 4) -> DataLoader:
    """Create a DataLoader with safe worker fallback for Windows shells."""
    num_workers = requested_workers
    if not multiprocessing_workers_available(num_workers):
        num_workers = 0

    kwargs = {
        "pin_memory": use_cuda,
        "num_workers": num_workers,
    }
    if prebatched:
        kwargs["batch_size"] = None
        kwargs["shuffle"] = False
        kwargs["drop_last"] = False
    else:
        kwargs["batch_size"] = batch_size
        kwargs["shuffle"] = shuffle
        kwargs["drop_last"] = drop_last
    if num_workers > 0:
        kwargs["persistent_workers"] = use_cuda
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, **kwargs)


def parse_metric_keys(value: Optional[str]) -> List[str]:
    if not value:
        return list(DEFAULT_METRIC_KEYS)
    keys = [key.strip() for key in value.split(',') if key.strip()]
    return keys or list(DEFAULT_METRIC_KEYS)


def metrics_history_path(checkpoint_dir: str) -> Path:
    return Path(checkpoint_dir) / 'metrics_history.jsonl'


def metrics_summary_path(checkpoint_dir: str) -> Path:
    return Path(checkpoint_dir) / 'metrics_summary.json'


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def normalize_metric_row(row: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(row)
    for metric_key, aliases in LEGACY_METRIC_ALIASES.items():
        if metric_key in normalized:
            continue
        for alias in aliases:
            if alias in normalized:
                normalized[metric_key] = normalized[alias]
                break
    return normalized


def append_metric_history(checkpoint_dir: str, metrics: Dict[str, Any]) -> None:
    path = metrics_history_path(checkpoint_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: to_jsonable(value) for key, value in metrics.items()}
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(payload, ensure_ascii=False) + '\n')


def write_metrics_summary(checkpoint_dir: str, summary: Dict[str, Any]) -> None:
    path = metrics_summary_path(checkpoint_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: to_jsonable(value) for key, value in summary.items()}
    with path.open('w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write('\n')


def read_metric_history_file(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(normalize_metric_row(json.loads(line)))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f'Invalid metrics JSON at {path}:{line_number}'
                ) from exc
    return rows


def read_checkpoint_history(checkpoint_dir: str) -> List[Dict[str, Any]]:
    checkpoint_dir_path = Path(checkpoint_dir)
    checkpoint_path = checkpoint_dir_path / 'sl_final.pt'
    if not checkpoint_path.exists():
        checkpoint_path = checkpoint_dir_path / 'sl_best.pt'
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f'No metrics_history.jsonl or checkpoint found in {checkpoint_dir}'
        )

    checkpoint = torch.load(checkpoint_path, map_location='cpu',
                            weights_only=False)
    metadata = checkpoint.get('metadata', {})
    history = metadata.get('history', [])
    return [normalize_metric_row(row) for row in history]


def read_local_metric_rows(checkpoint_dir: str,
                           tail: int = 20) -> List[Dict[str, Any]]:
    history_file = metrics_history_path(checkpoint_dir)
    if history_file.exists():
        rows = read_metric_history_file(history_file)
    else:
        rows = read_checkpoint_history(checkpoint_dir)
    if tail > 0:
        return rows[-tail:]
    return rows


def selected_metric_columns(rows: Sequence[Dict[str, Any]],
                            metrics: Sequence[str]) -> List[str]:
    columns: List[str] = []
    for key in ('phase', '_step', 'epoch'):
        if any(key in row for row in rows):
            columns.append(key)
    for metric in metrics:
        if metric not in columns and any(metric in row for row in rows):
            columns.append(metric)
    return columns


def format_metric_value(value: Any) -> str:
    if value is None:
        return '-'
    if isinstance(value, float):
        if value == 0:
            return '0'
        if abs(value) < 1e-4 or abs(value) >= 10000:
            return f'{value:.3e}'
        return f'{value:.6f}'.rstrip('0').rstrip('.')
    return str(value)


def print_metric_table(rows: Sequence[Dict[str, Any]],
                       metrics: Sequence[str]) -> None:
    if not rows:
        print('No metric rows found.')
        return
    columns = selected_metric_columns(rows, metrics)
    table = [[format_metric_value(row.get(col)) for col in columns]
             for row in rows]
    widths = [
        max(len(col), *(len(row[idx]) for row in table))
        for idx, col in enumerate(columns)
    ]
    print('  '.join(col.ljust(widths[idx])
                    for idx, col in enumerate(columns)))
    print('  '.join('-' * width for width in widths))
    for row in table:
        print('  '.join(row[idx].ljust(widths[idx])
                        for idx in range(len(columns))))


def write_metric_csv(path: str, rows: Sequence[Dict[str, Any]],
                     metrics: Sequence[str]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = selected_metric_columns(rows, metrics)
    with output.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col) for col in columns})


def print_metric_descriptions(metrics: Iterable[str]) -> None:
    print('Metric descriptions:')
    for key in metrics:
        description = METRIC_DESCRIPTIONS.get(key, 'Custom W&B metric.')
        print(f'  {key}: {description}')


def resolve_wandb_project_path(entity: Optional[str], project: str) -> str:
    return f'{entity}/{project}' if entity else project


def resolve_wandb_run(api: Any, project: str, entity: Optional[str],
                      run_ref: Optional[str]) -> Any:
    if run_ref and run_ref.count('/') == 2:
        return api.run(run_ref)

    project_path = resolve_wandb_project_path(entity, project)

    runs = api.runs(project_path, order='-created_at', per_page=200)
    if not run_ref:
        for run in runs:
            return run
        raise ValueError(f'No W&B runs found under {project_path}')

    matches = []
    for run in runs:
        run_names = {str(run.id), str(run.name)}
        display_name = getattr(run, 'display_name', None)
        if display_name is not None:
            run_names.add(str(display_name))
        if run_ref in run_names:
            matches.append(run)

    if not matches:
        raise ValueError(
            f'No W&B run named/id "{run_ref}" found under {project_path}. '
            'Pass --wandb_run entity/project/run_id for an exact run path.'
        )
    return matches[0]


def read_wandb_metric_rows(project: str, entity: Optional[str],
                           run_ref: Optional[str],
                           metrics: Sequence[str],
                           tail: int,
                           checkpoint_dir: str) -> tuple[Any, List[Dict[str, Any]]]:
    configure_local_runtime_dirs(checkpoint_dir)
    import wandb

    api = wandb.Api(timeout=30)
    run = resolve_wandb_run(api, project, entity, run_ref)
    rows_by_step: Dict[Any, Dict[str, Any]] = {}
    step_order: List[Any] = []

    for metric in metrics:
        for row in run.scan_history(keys=['_step', 'epoch', metric],
                                    page_size=1000):
            step = row.get('_step', row.get('epoch'))
            if step is None:
                step = len(step_order)
            if step not in rows_by_step:
                rows_by_step[step] = {'_step': step}
                step_order.append(step)
            if 'epoch' in row:
                rows_by_step[step]['epoch'] = row['epoch']
            if metric in row:
                rows_by_step[step][metric] = row[metric]

    rows = [normalize_metric_row(rows_by_step[step]) for step in step_order]
    if tail > 0:
        rows = rows[-tail:]
    return run, rows


def run_metrics_query(args) -> None:
    metrics = parse_metric_keys(args.metrics)
    if args.describe_metrics:
        print_metric_descriptions(metrics)

    if args.metrics_source == 'wandb':
        run, rows = read_wandb_metric_rows(
            project=args.wandb_project,
            entity=args.wandb_entity,
            run_ref=args.wandb_run or args.wandb_name,
            metrics=metrics,
            tail=args.metrics_tail,
            checkpoint_dir=args.checkpoint_dir,
        )
        print(f'W&B run: {run.entity}/{run.project}/{run.id} ({run.name})')
        run_url = getattr(run, 'url', None)
        if run_url:
            print(f'URL: {run_url}')
    else:
        rows = read_local_metric_rows(args.checkpoint_dir,
                                      tail=args.metrics_tail)
        print(f'Local metrics: {metrics_history_path(args.checkpoint_dir)}')

    print_metric_table(rows, metrics)

    if args.metrics_csv:
        write_metric_csv(args.metrics_csv, rows, metrics)
        print(f'Wrote CSV metrics to {args.metrics_csv}')


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
                        choices=['jsonl', 'mjson', 'mjson_cache'],
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
    parser.add_argument('--prefetch_factor', type=int, default=4,
                        help='DataLoader prefetch factor when num_workers > 0')
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
    parser.add_argument('--mjson_cache_dir', type=str, default=None,
                        help='Directory containing or receiving prebuilt MJSON tensor shards')
    parser.add_argument('--build_mjson_cache', action='store_true',
                        help='Build MJSON tensor shards before training, or build and exit when data_format is not mjson_cache')
    parser.add_argument('--cache_shard_size', type=int, default=65536,
                        help='Samples per tensor cache shard')
    parser.add_argument('--cache_num_workers', type=int, default=0,
                        help='Worker processes for cache building (0 = use num_workers)')
    parser.add_argument('--cache_state_dtype', type=str, default='float16',
                        choices=['float16', 'float32'],
                        help='State tensor dtype saved in cache shards')
    parser.add_argument('--cache_overwrite', action='store_true',
                        help='Overwrite existing MJSON tensor cache shards')
    parser.add_argument('--no_cache_shuffle_samples', action='store_true',
                        help='Disable within-shard sample shuffle for cache training')
    parser.add_argument('--random_split_all_mjson', type=str, default=None,
                        help='Root directory of all yearly MJSON files to split randomly at file level')
    parser.add_argument('--mjson_years', type=str, default=None,
                        help='Comma-separated years to include when using random_split_all_mjson (e.g. "2023,2024,2025,2026")')
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
    parser.add_argument('--query_metrics', action='store_true',
                        help='Print saved local or W&B loss/accuracy metrics and exit')
    parser.add_argument('--metrics_source', type=str, default='local',
                        choices=['local', 'wandb'],
                        help='Metric source for --query_metrics')
    parser.add_argument('--metrics', type=str,
                        default=','.join(DEFAULT_METRIC_KEYS),
                        help='Comma-separated metric keys to query')
    parser.add_argument('--metrics_tail', type=int, default=20,
                        help='Rows to print/export for --query_metrics (<=0 = all)')
    parser.add_argument('--metrics_csv', type=str, default=None,
                        help='Optional CSV output path for queried metrics')
    parser.add_argument('--wandb_run', type=str, default=None,
                        help='W&B run id, name, or entity/project/run_id for --query_metrics')
    parser.add_argument('--describe_metrics', action='store_true',
                        help='Print metric meanings before query output')
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


def collect_mjson_files(root_dir: str, max_files: int = 0,
                        years: Optional[List[str]] = None) -> List[str]:
    return collect_mjson_files_fast(root_dir, max_files=max_files, years=years)


def split_mjson_files(file_paths: List[str], train_ratio: float,
                      val_ratio: float, test_ratio: float,
                      seed: int) -> tuple[List[str], List[str], List[str]]:
    return split_mjson_files_fast(
        file_paths, train_ratio=train_ratio, val_ratio=val_ratio,
        test_ratio=test_ratio, seed=seed)


def build_streaming_dataset(file_paths: List[str], shuffle_files: bool,
                            seed: int) -> MJSONIterableDataset:
    return MJSONIterableDataset(
        file_paths=file_paths,
        shuffle_files=shuffle_files,
        seed=seed,
        parser_verbose=False,
    )


def resolve_mjson_cache_dir(args) -> str:
    if args.mjson_cache_dir:
        return args.mjson_cache_dir
    if args.random_split_all_mjson:
        years_key = args.mjson_years.replace(',', '-') if args.mjson_years else 'all'
        return str(Path(args.random_split_all_mjson) /
                   f'_tensor_cache_{years_key}_seed{args.split_seed}')
    return str(Path(args.data) / '_tensor_cache')


def build_cache_if_requested(args, mjson_years: Optional[List[str]]) -> Optional[Dict[str, Any]]:
    if not args.build_mjson_cache:
        return None
    source_root = args.random_split_all_mjson or args.data
    cache_workers = args.cache_num_workers or args.num_workers
    cache_dir = resolve_mjson_cache_dir(args)
    return build_mjson_cache(
        source_root=source_root,
        cache_dir=cache_dir,
        years=mjson_years,
        max_files=args.max_mjson_files,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.split_seed,
        shard_size=args.cache_shard_size,
        state_dtype=args.cache_state_dtype,
        num_workers=cache_workers,
        overwrite=args.cache_overwrite,
    )


def build_cache_dataset(cache_dir: str, manifest: Dict[str, Any], split: str,
                        batch_size: int, shuffle: bool, drop_last: bool,
                        seed: int, shuffle_samples: bool) -> TensorShardBatchDataset:
    split_info = manifest['splits'][split]
    shard_paths = shard_paths_for_split(cache_dir, manifest, split)
    return TensorShardBatchDataset(
        shard_paths=shard_paths,
        batch_size=batch_size,
        shuffle_shards=shuffle,
        shuffle_samples=shuffle_samples,
        drop_last=drop_last,
        seed=seed,
        total_samples=int(split_info.get('num_samples', 0)),
    )


def dataset_sample_count(dataset) -> int:
    total_samples = getattr(dataset, 'total_samples', None)
    if total_samples is not None:
        return int(total_samples)
    return len(dataset)


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
        states = states.to(device, dtype=torch.float32, non_blocking=True)
        masks = masks.to(device, dtype=torch.float32, non_blocking=True)
        labels = labels.to(device, dtype=torch.long, non_blocking=True)

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
            states = states.to(device, dtype=torch.float32, non_blocking=True)
            masks = masks.to(device, dtype=torch.float32, non_blocking=True)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)

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

    if args.query_metrics:
        run_metrics_query(args)
        return

    mjson_years = ([y.strip() for y in args.mjson_years.split(',')]
                   if args.mjson_years else None)
    if args.build_mjson_cache and args.data_format != 'mjson_cache':
        build_cache_if_requested(args, mjson_years)
        print(f"MJSON tensor cache ready: {resolve_mjson_cache_dir(args)}")
        return

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
    prebatched_mode = False
    cache_manifest = build_cache_if_requested(args, mjson_years)

    if args.data_format == 'mjson_cache':
        cache_dir = resolve_mjson_cache_dir(args)
        if cache_manifest is None:
            cache_manifest = load_mjson_cache_manifest(cache_dir)
        cache_shuffle_samples = not args.no_cache_shuffle_samples
        streaming_mode = True
        prebatched_mode = True
        train_set = build_cache_dataset(
            cache_dir, cache_manifest, 'train',
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            seed=args.split_seed,
            shuffle_samples=cache_shuffle_samples,
        )
        val_set = build_cache_dataset(
            cache_dir, cache_manifest, 'val',
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            seed=args.split_seed,
            shuffle_samples=False,
        )
        test_set = build_cache_dataset(
            cache_dir, cache_manifest, 'test',
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            seed=args.split_seed,
            shuffle_samples=False,
        )
        print(
            f"Tensor cache split: train={dataset_sample_count(train_set)}, "
            f"val={dataset_sample_count(val_set)}, "
            f"test={dataset_sample_count(test_set)} samples"
        )
    elif args.random_split_all_mjson:
        if args.data_format != 'mjson':
            raise ValueError('random_split_all_mjson requires --data_format mjson')
        if not args.stream_mjson:
            raise ValueError('random_split_all_mjson requires --stream_mjson to avoid exhausting memory')

        all_files = collect_mjson_files(args.random_split_all_mjson,
                                        max_files=args.max_mjson_files,
                                        years=mjson_years)
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

    train_loader = make_dataloader(
        train_set,
        batch_size=args.batch_size,
        shuffle=not streaming_mode,
        drop_last=True,
        use_cuda=use_cuda,
        requested_workers=args.num_workers,
        prebatched=prebatched_mode,
        prefetch_factor=args.prefetch_factor,
    )
    val_loader = make_dataloader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        use_cuda=use_cuda,
        requested_workers=args.num_workers,
        prebatched=prebatched_mode,
        prefetch_factor=args.prefetch_factor,
    )

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
    resume_meta: Dict[str, Any] = {}
    if args.resume:
        start_epoch, resume_meta = load_checkpoint(
            model, args.resume, optimizer=optimizer, device=device)
        print(f"Resumed from epoch {start_epoch}")

    # 训练循环
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    best_val_acc = float(
        resume_meta.get('best_val_acc', resume_meta.get('val_acc', 0.0)) or 0.0
    )
    history = list(resume_meta.get('history', []))
    if start_epoch == 0:
        metrics_history_path(args.checkpoint_dir).write_text('', encoding='utf-8')

    amp_status = "AMP" if (use_cuda and not args.no_amp) else "FP32"
    print(f"\nTraining on {device} ({amp_status}) for {args.epochs} epochs...")
    print(f"{'Epoch':>6} {'Train Loss':>12} {'Train Acc':>10} "
          f"{'Val Loss':>10} {'Val Acc':>8} {'Time':>8} {'LR':>10}")

    # ── wandb 初始化 ──
    wandb_run = None
    if args.wandb:
        configure_local_runtime_dirs(args.checkpoint_dir)
        try:
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
                    'mjson_years': args.mjson_years,
                    'mjson_cache_dir': args.mjson_cache_dir,
                    'cache_shard_size': args.cache_shard_size,
                    'cache_state_dtype': args.cache_state_dtype,
                    'prebatched': prebatched_mode,
                    'epochs': args.epochs,
                    'batch_size': args.batch_size,
                    'num_workers': train_loader.num_workers,
                    'lr': args.lr,
                    'weight_decay': args.weight_decay,
                    'device': device,
                    'amp': use_cuda and not args.no_amp,
                    'compile': args.compile,
                },
                reinit="finish_previous",
                settings=wandb.Settings(
                    mode="online",
                    init_timeout=120,
                ),
            )
            wandb.define_metric('epoch')
            for metric_pattern in ('train/*', 'val/*', 'test/*',
                                   'time/*', 'best_val_accuracy', 'lr'):
                wandb.define_metric(metric_pattern, step_metric='epoch')
            wandb.watch(model, log='gradients', log_freq=100)
        except Exception as exc:
            raise RuntimeError(
                "W&B was requested with --wandb, but initialization failed. "
                "Run `wandb login` or check the local runtime directory permissions."
            ) from exc

    for epoch in range(start_epoch, start_epoch + args.epochs):
        t0 = time.time()

        if streaming_mode and hasattr(train_set, 'set_epoch'):
            train_set.set_epoch(epoch)

        print(f'  Starting epoch {epoch+1} training...', flush=True)
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, scaler)
        val_metrics = validate(model, val_loader, device)

        scheduler.step()
        elapsed = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]

        is_best = val_metrics['accuracy'] > best_val_acc
        if is_best:
            best_val_acc = val_metrics['accuracy']

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'train_acc': train_metrics['accuracy'],
            'val_loss': val_metrics['loss'],
            'val_acc': val_metrics['accuracy'],
        })
        epoch_metrics = {
            'phase': 'train_val',
            '_step': epoch + 1,
            'epoch': epoch + 1,
            'train/loss': train_metrics['loss'],
            'train/accuracy': train_metrics['accuracy'],
            'val/loss': val_metrics['loss'],
            'val/accuracy': val_metrics['accuracy'],
            'best_val_accuracy': best_val_acc,
            'lr': current_lr,
            'time/epoch_sec': elapsed,
            'train/num_workers': train_loader.num_workers,
            'train/batch_size': args.batch_size,
        }
        append_metric_history(args.checkpoint_dir, epoch_metrics)

        print(f"{epoch + 1:>6} {train_metrics['loss']:>12.4f} "
              f"{train_metrics['accuracy']:>10.4f} "
              f"{val_metrics['loss']:>10.4f} {val_metrics['accuracy']:>8.4f} "
              f"{elapsed:>7.1f}s {current_lr:>10.2e}")

        if wandb_run is not None:
            wandb_run.log(epoch_metrics, step=epoch + 1)

        # 保存最佳模型
        if is_best:
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
    test_loader = make_dataloader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        use_cuda=use_cuda,
        requested_workers=args.num_workers,
        prebatched=prebatched_mode,
        prefetch_factor=args.prefetch_factor,
    )
    test_metrics = validate(model, test_loader, device)
    print(f"\nTest: loss={test_metrics['loss']:.4f}, "
          f"accuracy={test_metrics['accuracy']:.4f}")
    print(f"Best val accuracy: {best_val_acc:.4f}")
    print(f"Model saved to {args.checkpoint_dir}")

    final_step = start_epoch + args.epochs
    final_metrics = {
        'phase': 'test',
        '_step': final_step,
        'epoch': final_step,
        'test/loss': test_metrics['loss'],
        'test/accuracy': test_metrics['accuracy'],
        'best_val_accuracy': best_val_acc,
    }
    append_metric_history(args.checkpoint_dir, final_metrics)
    write_metrics_summary(args.checkpoint_dir, {
        **final_metrics,
        'checkpoint_dir': args.checkpoint_dir,
        'history_file': str(metrics_history_path(args.checkpoint_dir)),
    })

    if wandb_run is not None:
        wandb_run.log(final_metrics, step=final_step)
        wandb.finish()


if __name__ == '__main__':
    main()
# 中文注释：监督学习预训练入口，负责数据载入、训练循环、W&B/本地指标和模型保存。
