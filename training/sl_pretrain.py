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
from models.model_io import save_resume_checkpoint, load_resume_checkpoint
from models.transformer_policy_value import TransformerPolicyValueNet
from data import (
    JSONLRecordParser,
    MJSONRecordParser,
    MahjongStateActionDataset,
    MJSONIterableDataset,
    MJSONTokenIterableDataset,
    MJSONPublicPrivateTokenIterableDataset,
    TensorShardBatchDataset,
    TokenDataset,
    collate_transformer_batch,
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


def _make_checkpoint_metadata(args, best_val_acc: float) -> dict:
    """Build checkpoint metadata including model architecture fields."""
    meta = {
        'model_arch': args.model_arch,
        'training_stage': getattr(args, '_training_stage', 'public_sl'),
        'val_acc': best_val_acc,
    }
    if args.model_arch == 'transformer':
        meta.update({
            'd_model': args.transformer_d_model,
            'n_layers': args.transformer_n_layers,
            'n_heads': args.transformer_n_heads,
            'n_concept': args.transformer_n_concept,
            'max_len': args.max_len,
        })
    if getattr(args, 'teacher_mode', False):
        meta.update({
            'teacher_mode': True,
            'teacher_checkpoint': args.teacher_checkpoint,
            'private_visibility': args.private_visibility,
            'distill_temperature': args.distill_temperature,
            'distill_alpha': args.distill_alpha,
            'distill_value_alpha': args.distill_value_alpha,
        })
    return meta


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
                    prefetch_factor: int = 4,
                    collate_fn=None) -> DataLoader:
    """Create a DataLoader with safe worker fallback for Windows shells."""
    num_workers = requested_workers
    if not multiprocessing_workers_available(num_workers):
        num_workers = 0

    kwargs: Dict[str, Any] = {
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
    if collate_fn is not None:
        kwargs["collate_fn"] = collate_fn
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
                        help='Resume from full checkpoint (model+optimizer+scheduler+scaler+RNG)')
    parser.add_argument('--save_every', type=int, default=5,
                        help='Save full resume checkpoint every N epochs (default: 5)')
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
    parser.add_argument('--model_arch', type=str, default='resnet',
                        choices=['resnet', 'transformer'],
                        help='Model architecture (default: resnet)')
    parser.add_argument('--mtl_alpha', type=float, default=0.3,
                        help='Weight for shanten auxiliary loss (transformer only)')
    parser.add_argument('--mtl_beta', type=float, default=0.2,
                        help='Weight for ukeire auxiliary loss (transformer only)')
    parser.add_argument('--transformer_d_model', type=int, default=256,
                        help='Transformer hidden dimension')
    parser.add_argument('--transformer_n_layers', type=int, default=6,
                        help='Transformer encoder layers')
    parser.add_argument('--transformer_n_heads', type=int, default=8,
                        help='Transformer attention heads')
    parser.add_argument('--transformer_n_concept', type=int, default=10,
                        help='Transformer concept tokens (min 10)')
    parser.add_argument('--max_len', type=int, default=256,
                        help='Max token sequence length for Transformer')

    # ── God's-eye Teacher-Student Training ────────────────────────────
    parser.add_argument('--teacher_mode', action='store_true',
                        help='Enable God\'s-eye teacher-student distillation')
    parser.add_argument('--oracle_data', type=str, default=None,
                        help='Path to oracle trajectory JSONL file or directory '
                             '(from selfplay_recorder). For Transformer teacher/student training.')
    parser.add_argument('--oracle_teacher_train', action='store_true',
                        help='Train Oracle Teacher with teacher-mode forward '
                             '(public+private) on oracle trajectory data. '
                             'Requires --oracle_data and --model_arch transformer.')
    parser.add_argument('--private_visibility', type=float, default=1.0,
                        help='Fraction of private info teacher sees (0.0-1.0)')
    parser.add_argument('--visibility_schedule', type=str, default=None,
                        help='Comma-separated visibility values per epoch, '
                             'e.g. "1.0,0.75,0.5,0.25,0.0"')
    parser.add_argument('--distill_temperature', type=float, default=2.0,
                        help='Temperature for KL distillation')
    parser.add_argument('--distill_alpha', type=float, default=1.0,
                        help='Weight for policy KL distillation loss')
    parser.add_argument('--distill_value_alpha', type=float, default=0.5,
                        help='Weight for value distillation loss')
    parser.add_argument('--teacher_checkpoint', type=str, default=None,
                        help='Path to frozen teacher checkpoint for distillation')

    # ── Step-level checkpoint / profiling / benchmark ──────────────────
    parser.add_argument('--save_interval_min', type=float, default=0,
                        help='Save resume checkpoint every N minutes (0=disabled, '
                             'recommended: 30 for spot instances)')
    parser.add_argument('--max_batches', type=int, default=0,
                        help='Stop after N total batches across all epochs (0=no limit)')
    parser.add_argument('--resume_batch', type=int, default=0,
                        help='Number of batches to fast-forward on resume (saved in checkpoint metadata)')
    parser.add_argument('--manifest', type=str, default=None,
                        help='Path to data split manifest JSON (replaces random split)')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run benchmark mode: measure throughput for --benchmark_batches then exit')
    parser.add_argument('--benchmark_batches', type=int, default=1000,
                        help='Number of batches per benchmark run (default: 1000)')
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


def build_streaming_token_dataset(file_paths: List[str], shuffle_files: bool,
                                   seed: int) -> MJSONTokenIterableDataset:
    return MJSONTokenIterableDataset(
        file_paths=file_paths,
        shuffle_files=shuffle_files,
        seed=seed,
        parser_verbose=False,
    )


def build_streaming_public_private_token_dataset(
        file_paths: List[str], shuffle_files: bool,
        seed: int) -> MJSONPublicPrivateTokenIterableDataset:
    return MJSONPublicPrivateTokenIterableDataset(
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


def train_epoch_transformer(model: nn.Module, dataloader: DataLoader,
                           optimizer: torch.optim.Optimizer, device: str,
                           scaler: Optional[GradScaler] = None,
                           alpha: float = 0.3, beta: float = 0.2,
                           save_callback=None,
                           skip_batches: int = 0,
                           max_batches: int = 0,
                           global_batch_offset: int = 0,
                           save_interval_batches: int = 0) -> dict:
    """训练一个 epoch（Transformer MTL），返回指标。

    Args:
        save_callback: 可选的回调函数 save_callback(total_batches_processed)
                       在 save_interval_batches 到达时调用
        skip_batches: 快速前进跳过的 batch 数（断点续训用）
        max_batches: 本 epoch 最多训练的 batch 数（0=不限制）
        global_batch_offset: 全局 batch 计数偏移（用于日志显示）
        save_interval_batches: 每隔多少 batch 调用 save_callback（0=禁用）

    Returns:
        dict: 包含 loss, accuracy 等指标，以及 '_batches_trained' 和 '_stopped_early'
    """
    model.train()
    total_loss = 0.0
    total_policy_loss = 0.0
    total_shanten_loss = 0.0
    total_ukeire_loss = 0.0
    correct = 0
    total = 0
    use_amp = scaler is not None
    batch_count = 0
    skipped = 0
    t_epoch_start = time.time()
    t_last_log = t_epoch_start
    stopped_early = False

    # 计算 profiling 用的 data_wait 累计
    t_data_wait_total = 0.0
    t_gpu_step_total = 0.0

    for batch in dataloader:
        batch_count += 1

        # ── Fast-forward: 跳过已训练的 batches ──
        if skipped < skip_batches:
            skipped += 1
            if skipped % 5000 == 0:
                print(f"    [fast-forward] skipped {skipped}/{skip_batches} batches",
                      flush=True)
            continue

        t_data_end = time.time()

        token_ids = batch['token_ids'].to(device, dtype=torch.long)
        token_types = batch['token_types'].to(device, dtype=torch.long)
        behavior_ids = batch['behavior_ids'].to(device, dtype=torch.long)
        attention_mask = batch['attention_mask'].to(device, dtype=torch.bool)
        action_mask = batch['action_mask'].to(device, dtype=torch.float32)
        labels = batch['labels'].to(device, dtype=torch.long)

        optimizer.zero_grad(set_to_none=True)

        t_gpu_start = time.time()
        with autocast('cuda', enabled=use_amp):
            outputs = model(token_ids, token_types, behavior_ids,
                           attention_mask, action_mask)
            policy_logits = outputs['policy_logits']
            policy_loss = nn.CrossEntropyLoss()(policy_logits, labels)

            shanten_loss = torch.tensor(0.0, device=device)
            if 'oracle_shanten' in batch and alpha > 0:
                shanten_targets = batch['oracle_shanten'].to(
                    device, dtype=torch.long)
                shanten_loss = nn.CrossEntropyLoss()(
                    outputs['shanten'], shanten_targets)

            ukeire_loss = torch.tensor(0.0, device=device)
            if 'oracle_ukeire_mask' in batch and beta > 0:
                ukeire_targets = batch['oracle_ukeire_mask'].to(
                    device, dtype=torch.float32)
                ukeire_logits = outputs.get('ukeire', None)
                if ukeire_logits is not None and ukeire_logits.shape[-1] == 34:
                    ukeire_loss = nn.BCEWithLogitsLoss()(
                        ukeire_logits, ukeire_targets)

            div_loss = model.compute_diversity_loss() * 0.01
            loss = policy_loss + alpha * shanten_loss + beta * ukeire_loss + div_loss

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

        t_gpu_end = time.time()

        B = token_ids.size(0)
        total_loss += loss.item() * B
        total_policy_loss += policy_loss.item() * B
        total_shanten_loss += shanten_loss.item() * B
        total_ukeire_loss += ukeire_loss.item() * B
        preds = policy_logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += B

        # 训练的 batch 数（不含 fast-forward）
        trained = batch_count - skipped
        effective_global = global_batch_offset + trained

        # profiling 累计
        t_data_wait_total += (t_data_end - t_last_log if trained > 1 else 0)
        t_gpu_step_total += (t_gpu_end - t_gpu_start)
        t_last_log = t_gpu_end

        # ── Profiling 日志 ──
        if trained % 500 == 0:
            elapsed = time.time() - t_epoch_start
            avg_loss = total_loss / max(total, 1)
            avg_acc = correct / max(total, 1)
            batch_s = trained / max(elapsed, 1e-6)
            samples_s = total / max(elapsed, 1e-6)
            max_mem = (torch.cuda.max_memory_allocated() / 1024**3
                       if torch.cuda.is_available() else 0)
            seq_len = token_ids.size(1)
            print(f"    batch {effective_global}: loss={avg_loss:.4f} "
                  f"acc={avg_acc:.4f} | "
                  f"{batch_s:.1f} batch/s {samples_s:.0f} samples/s | "
                  f"seq_len={seq_len} GPU_mem={max_mem:.1f}GB | "
                  f"{elapsed:.0f}s", flush=True)

        # ── Step-level checkpoint 保存 ──
        if save_interval_batches > 0 and save_callback is not None:
            if trained % save_interval_batches == 0 and trained > 0:
                save_callback(effective_global)

        # ── max_batches 限制 ──
        if max_batches > 0 and trained >= max_batches:
            print(f"    [max_batches] reached {max_batches}, stopping epoch early",
                  flush=True)
            stopped_early = True
            break

    return {
        'loss': total_loss / max(total, 1),
        'policy_loss': total_policy_loss / max(total, 1),
        'shanten_loss': total_shanten_loss / max(total, 1),
        'ukeire_loss': total_ukeire_loss / max(total, 1),
        'accuracy': correct / max(total, 1),
        '_batches_trained': batch_count - skipped,
        '_stopped_early': stopped_early,
    }


def train_epoch_transformer_distill(
        model: nn.Module, dataloader: DataLoader,
        optimizer: torch.optim.Optimizer, device: str,
        scaler=None, alpha: float = 0.3, beta: float = 0.2,
        distill_alpha: float = 1.0, distill_value_alpha: float = 0.5,
        distill_temperature: float = 2.0,
        private_visibility: float = 1.0,
        teacher_model: Optional[nn.Module] = None) -> dict:
    """训练一个 epoch（Teacher-Student 蒸馏模式）。

    Student 只看 public tokens；Teacher 看 public + private tokens。
    Loss = policy_ce + alpha * shanten_ce + beta * ukeire_bce
           + distill_alpha * masked_kl(teacher, student)
           + distill_value_alpha * value_mse

    Args:
        teacher_model: 独立的冻结 Teacher 模型。若为 None，使用 model 自身
            做 teacher forward（teacher logits/value 会 detach 以阻断梯度）。
        private_visibility: private tokens 可见比例（0.0~1.0），用于 curriculum。
    """
    from training.distillation import masked_kl_loss

    model.train()
    # 若有独立 teacher，设为 eval 模式并冻结
    teacher = teacher_model if teacher_model is not None else model
    if teacher_model is not None:
        teacher_model.eval()

    total_loss = 0.0
    total_policy_loss = 0.0
    total_distill_kl = 0.0
    total_value_mse = 0.0
    correct = 0
    total = 0
    use_amp = scaler is not None

    has_private = True  # set to False when private fields are not in batch

    for batch in dataloader:
        if not isinstance(batch, dict):
            continue  # skip non-dict batches (misconfigured dataloader)

        token_ids = batch['token_ids'].to(device, dtype=torch.long)
        token_types = batch['token_types'].to(device, dtype=torch.long)
        behavior_ids = batch['behavior_ids'].to(device, dtype=torch.long)
        attention_mask = batch['attention_mask'].to(device, dtype=torch.bool)
        action_mask = batch['action_mask'].to(device, dtype=torch.float32)
        labels = batch['labels'].to(device, dtype=torch.long)

        # Private tokens (optional, skip if not present)
        priv_ids = batch.get('private_token_ids')
        if priv_ids is not None and priv_ids.shape[1] > 0:
            priv_ids = priv_ids.to(device, dtype=torch.long)
            priv_types = batch['private_token_types'].to(device, dtype=torch.long)
            priv_behavior = batch.get('private_behavior_ids')
            if priv_behavior is not None:
                priv_behavior = priv_behavior.to(device, dtype=torch.long)
            priv_attn = batch.get('private_attention_mask')
            if priv_attn is not None:
                priv_attn = priv_attn.to(device, dtype=torch.bool)
            has_private = True
        else:
            has_private = False

        # ── visibility curriculum：Bernoulli dropout on private tokens ──
        if has_private and private_visibility < 1.0:
            B_priv = priv_ids.size(0)
            n_priv = priv_ids.size(1)
            if n_priv > 0:
                keep_mask = (torch.rand(B_priv, n_priv, device=device)
                             < private_visibility)
                # visibility=0 → 零化所有 private token，但仍走 Teacher forward
                # （Teacher cross-attention 在全 PAD private 下等价于 public-only）
                if not keep_mask.any():
                    priv_ids = priv_ids.clone()
                    priv_ids[:] = 0
                    priv_types = priv_types.clone()
                    priv_types[:] = 0
                    if priv_attn is not None:
                        priv_attn = priv_attn.clone()
                        priv_attn[:] = True
                    if priv_behavior is not None:
                        priv_behavior = priv_behavior.clone()
                        priv_behavior[:] = 0
                else:
                    # 每个样本至少保留 1 个 token（visibility > 0 时）
                    any_kept = keep_mask.any(dim=1)
                    if not any_kept.all():
                        fix = (~any_kept).nonzero(as_tuple=True)[0]
                        keep_mask[fix, 0] = True
                    # PAD 掉未保留的 private token
                    priv_ids = priv_ids.clone()
                    priv_ids[~keep_mask] = 0
                    priv_types = priv_types.clone()
                    priv_types[~keep_mask] = 0
                    if priv_attn is not None:
                        priv_attn = priv_attn.clone()
                        priv_attn = priv_attn | ~keep_mask
                    if priv_behavior is not None:
                        priv_behavior = priv_behavior.clone()
                        priv_behavior[~keep_mask] = 0

        optimizer.zero_grad(set_to_none=True)

        with autocast('cuda', enabled=use_amp):
            # Student: public-only forward
            outputs_s = model(token_ids, token_types, behavior_ids,
                            attention_mask, action_mask, mode='student')
            policy_loss = nn.CrossEntropyLoss()(
                outputs_s['policy_logits'], labels)

            # MTL auxiliary losses
            shanten_loss = torch.tensor(0.0, device=device)
            if 'oracle_shanten' in batch and alpha > 0:
                shanten_loss = nn.CrossEntropyLoss()(
                    outputs_s['shanten'],
                    batch['oracle_shanten'].to(device, dtype=torch.long))

            ukeire_loss = torch.tensor(0.0, device=device)
            if 'oracle_ukeire_mask' in batch and beta > 0:
                ukeire_targets = batch['oracle_ukeire_mask'].to(
                    device, dtype=torch.float32)
                ukeire_logits = outputs_s.get('ukeire', None)
                if ukeire_logits is not None and ukeire_logits.shape[-1] == 34:
                    ukeire_loss = nn.BCEWithLogitsLoss()(
                        ukeire_logits, ukeire_targets)

            distill_kl = torch.tensor(0.0, device=device)
            value_mse = torch.tensor(0.0, device=device)

            if has_private and (distill_alpha > 0 or distill_value_alpha > 0):
                # Teacher: public + private forward（不回传梯度到 teacher）
                with torch.no_grad():
                    outputs_t = teacher(
                        token_ids, token_types, behavior_ids,
                        attention_mask, action_mask,
                        private_token_ids=priv_ids,
                        private_token_types=priv_types,
                        private_behavior_ids=priv_behavior,
                        private_attention_mask=priv_attn,
                        mode='teacher')

                # Detach teacher logits/value 以阻断梯度回传
                if distill_alpha > 0:
                    teacher_logits = outputs_t['policy_logits'].detach()
                    distill_kl = masked_kl_loss(
                        teacher_logits,
                        outputs_s['policy_logits'],
                        action_mask,
                        temperature=distill_temperature,
                    )

                if distill_value_alpha > 0:
                    oracle_v = outputs_t.get('oracle_value', outputs_t['value'])
                    oracle_v = oracle_v.detach()
                    value_mse = nn.MSELoss()(outputs_s['value'], oracle_v)

            div_loss = model.compute_diversity_loss() * 0.01

            loss = (policy_loss + alpha * shanten_loss + beta * ukeire_loss
                    + distill_alpha * distill_kl
                    + distill_value_alpha * value_mse
                    + div_loss)

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

        B = token_ids.size(0)
        total_loss += loss.item() * B
        total_policy_loss += policy_loss.item() * B
        total_distill_kl += distill_kl.item() * B
        total_value_mse += value_mse.item() * B
        preds = outputs_s['policy_logits'].argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += B

    result = {
        'loss': total_loss / max(total, 1),
        'policy_loss': total_policy_loss / max(total, 1),
        'accuracy': correct / max(total, 1),
    }
    if has_private:
        result['distill/kl'] = total_distill_kl / max(total, 1)
        result['distill/value_mse'] = total_value_mse / max(total, 1)
    if total == 0:
        raise RuntimeError(
            'train_epoch_transformer_distill processed 0 valid samples. '
            'Check that --oracle_data or MJSON data contains enough samples '
            'for the given --batch_size. If using small data, use --batch_size 1 '
            'or record more games.')
    return result


def train_epoch_teacher(
        model: nn.Module, dataloader: DataLoader,
        optimizer: torch.optim.Optimizer, device: str,
        scaler=None, alpha: float = 0.3, beta: float = 0.2,
        value_loss_coef: float = 0.5,
        require_private: bool = False) -> dict:
    """训练一个 epoch（Oracle Teacher，mode='teacher'）。

    Teacher 看 public + private tokens，直接从 oracle 轨迹学习。
    Loss = policy_ce + value_mse + alpha * shanten_ce + beta * ukeire_bce + diversity

    Args:
        value_loss_coef: oracle_value 与 reward 的 MSE 权重
        require_private: 若为 True，遇到无 private tokens 的 batch 时 raise 而非 skip
    """
    model.train()
    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    correct = 0
    total = 0
    use_amp = scaler is not None

    for batch in dataloader:
        if not isinstance(batch, dict):
            continue

        token_ids = batch['token_ids'].to(device, dtype=torch.long)
        token_types = batch['token_types'].to(device, dtype=torch.long)
        behavior_ids = batch['behavior_ids'].to(device, dtype=torch.long)
        attention_mask = batch['attention_mask'].to(device, dtype=torch.bool)
        action_mask = batch['action_mask'].to(device, dtype=torch.float32)
        labels = batch['labels'].to(device, dtype=torch.long)

        # Private tokens (required for teacher training)
        priv_ids = batch.get('private_token_ids')
        if priv_ids is None or priv_ids.shape[1] == 0:
            if require_private:
                raise RuntimeError(
                    'train_epoch_teacher: batch has no private tokens. '
                    'Oracle training requires non-empty private_token_ids in every batch. '
                    'Check that oracle_data contains valid samples.')
            continue  # skip batches without private tokens
        priv_ids = priv_ids.to(device, dtype=torch.long)
        priv_types = batch['private_token_types'].to(device, dtype=torch.long)
        priv_behavior = batch.get('private_behavior_ids')
        if priv_behavior is not None:
            priv_behavior = priv_behavior.to(device, dtype=torch.long)
        priv_attn = batch.get('private_attention_mask')
        if priv_attn is not None:
            priv_attn = priv_attn.to(device, dtype=torch.bool)

        # Reward / outcome targets (optional)
        rewards = batch.get('rewards')

        optimizer.zero_grad(set_to_none=True)

        with autocast('cuda', enabled=use_amp):
            outputs = model(
                token_ids, token_types, behavior_ids,
                attention_mask, action_mask,
                private_token_ids=priv_ids,
                private_token_types=priv_types,
                private_behavior_ids=priv_behavior,
                private_attention_mask=priv_attn,
                mode='teacher')

            policy_logits = outputs['policy_logits']
            policy_loss = nn.CrossEntropyLoss()(policy_logits, labels)

            # Value loss: oracle_value vs actual reward
            value_loss = torch.tensor(0.0, device=device)
            oracle_v = outputs.get('oracle_value', outputs.get('value'))
            if oracle_v is not None and rewards is not None:
                rewards_t = rewards.to(device, dtype=torch.float32)
                value_loss = nn.MSELoss()(oracle_v.squeeze(-1), rewards_t)

            # MTL auxiliary losses
            shanten_loss = torch.tensor(0.0, device=device)
            if 'oracle_shanten' in batch and alpha > 0:
                shanten_loss = nn.CrossEntropyLoss()(
                    outputs['shanten'],
                    batch['oracle_shanten'].to(device, dtype=torch.long))

            ukeire_loss = torch.tensor(0.0, device=device)
            if 'oracle_ukeire_mask' in batch and beta > 0:
                ukeire_logits = outputs.get('ukeire', None)
                if ukeire_logits is not None and ukeire_logits.shape[-1] == 34:
                    ukeire_loss = nn.BCEWithLogitsLoss()(
                        ukeire_logits,
                        batch['oracle_ukeire_mask'].to(device, dtype=torch.float32))

            div_loss = model.compute_diversity_loss() * 0.01

            loss = (policy_loss + value_loss_coef * value_loss
                    + alpha * shanten_loss + beta * ukeire_loss + div_loss)

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

        B = token_ids.size(0)
        total_loss += loss.item() * B
        total_policy_loss += policy_loss.item() * B
        total_value_loss += value_loss.item() * B
        preds = policy_logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += B

    if total == 0:
        raise RuntimeError(
            'train_epoch_teacher processed 0 valid samples. '
            'Check that --oracle_data contains non-empty token samples '
            '(ResNet/heuristic recorder with --model_arch heuristic now '
            'creates tokens automatically). If using small data, ensure '
            'batch_size <= sample count or use --oracle_data with enough games.')

    return {
        'loss': total_loss / total,
        'policy_loss': total_policy_loss / total,
        'value_loss': total_value_loss / total,
        'accuracy': correct / total,
    }


def validate_transformer(model: nn.Module, dataloader: DataLoader,
                         device: str, alpha: float = 0.3,
                         beta: float = 0.2) -> dict:
    """验证 Transformer 模型，返回指标。"""
    model.eval()
    total_loss = 0.0
    total_policy_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            token_ids = batch['token_ids'].to(device, dtype=torch.long)
            token_types = batch['token_types'].to(device, dtype=torch.long)
            behavior_ids = batch['behavior_ids'].to(device, dtype=torch.long)
            attention_mask = batch['attention_mask'].to(device, dtype=torch.bool)
            action_mask = batch['action_mask'].to(device, dtype=torch.float32)
            labels = batch['labels'].to(device, dtype=torch.long)

            outputs = model(token_ids, token_types, behavior_ids,
                           attention_mask, action_mask)
            policy_logits = outputs['policy_logits']
            policy_loss = nn.CrossEntropyLoss()(policy_logits, labels)

            B = token_ids.size(0)
            total_loss += policy_loss.item() * B
            total_policy_loss += policy_loss.item() * B
            preds = policy_logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += B

    if total == 0:
        raise RuntimeError(
            'validate_transformer processed 0 valid samples. '
            'Check that validation data is non-empty and batch_size is '
            'smaller than sample count.')

    return {
        'loss': total_loss / total,
        'policy_loss': total_policy_loss / total,
        'accuracy': correct / total,
    }


def validate_teacher(model: nn.Module, dataloader: DataLoader,
                     device: str, alpha: float = 0.3,
                     beta: float = 0.2,
                     value_loss_coef: float = 0.5,
                     require_private: bool = False) -> dict:
    """验证 Oracle Teacher 模型（mode='teacher' + private tokens）。

    Args:
        require_private: 若为 True，遇到无 private tokens 的 batch 时 raise 而非 skip
    """
    model.eval()
    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            if not isinstance(batch, dict):
                continue
            token_ids = batch['token_ids'].to(device, dtype=torch.long)
            token_types = batch['token_types'].to(device, dtype=torch.long)
            behavior_ids = batch['behavior_ids'].to(device, dtype=torch.long)
            attention_mask = batch['attention_mask'].to(device, dtype=torch.bool)
            action_mask = batch['action_mask'].to(device, dtype=torch.float32)
            labels = batch['labels'].to(device, dtype=torch.long)

            priv_ids = batch.get('private_token_ids')
            if priv_ids is None or priv_ids.shape[1] == 0:
                if require_private:
                    raise RuntimeError(
                        'validate_teacher: batch has no private tokens. '
                        'Oracle training requires non-empty private_token_ids '
                        'in every batch.')
                continue
            priv_ids = priv_ids.to(device, dtype=torch.long)
            priv_types = batch['private_token_types'].to(device, dtype=torch.long)
            priv_behavior = batch.get('private_behavior_ids')
            if priv_behavior is not None:
                priv_behavior = priv_behavior.to(device, dtype=torch.long)
            priv_attn = batch.get('private_attention_mask')
            if priv_attn is not None:
                priv_attn = priv_attn.to(device, dtype=torch.bool)
            rewards = batch.get('rewards')

            outputs = model(
                token_ids, token_types, behavior_ids,
                attention_mask, action_mask,
                private_token_ids=priv_ids,
                private_token_types=priv_types,
                private_behavior_ids=priv_behavior,
                private_attention_mask=priv_attn,
                mode='teacher')

            policy_logits = outputs['policy_logits']
            policy_loss = nn.CrossEntropyLoss()(policy_logits, labels)

            value_loss = torch.tensor(0.0, device=device)
            oracle_v = outputs.get('oracle_value', outputs.get('value'))
            if oracle_v is not None and rewards is not None:
                rewards_t = rewards.to(device, dtype=torch.float32)
                value_loss = nn.MSELoss()(oracle_v.squeeze(-1), rewards_t)

            B = token_ids.size(0)
            total_loss += (policy_loss.item() + value_loss_coef * value_loss.item()) * B
            total_policy_loss += policy_loss.item() * B
            total_value_loss += value_loss.item() * B
            preds = policy_logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += B

    if total == 0:
        raise RuntimeError(
            'validate_teacher processed 0 valid samples. '
            'Check that oracle data contains non-empty private tokens '
            'and valid action_mask / chosen_action fields.')

    return {
        'loss': total_loss / total,
        'policy_loss': total_policy_loss / total,
        'value_loss': total_value_loss / total,
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

    # 模型架构标记（需在数据加载前确定，用于选择 dataset 类型）
    is_transformer = args.model_arch == 'transformer'

    # ── 从 checkpoint 提前读取架构元数据（必须在数据加载前） ──
    resume_meta: Dict[str, Any] = {}
    if args.resume:
        _ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        resume_meta = _ckpt.get('metadata', {})
        ckpt_arch = resume_meta.get('model_arch', args.model_arch)
        if ckpt_arch != args.model_arch:
            print(f"Auto-detected model_arch={ckpt_arch} from checkpoint "
                  f"(overriding --model_arch={args.model_arch})")
            args.model_arch = ckpt_arch
        # 恢复 Transformer 超参数
        for key in ('d_model', 'n_layers', 'n_heads', 'n_concept', 'max_len'):
            ckpt_val = resume_meta.get(key)
            arg_key = f'transformer_{key}' if key != 'max_len' else 'max_len'
            if ckpt_val is not None and hasattr(args, arg_key):
                setattr(args, arg_key, ckpt_val)
        # 更新架构标记（checkpoint 可能覆盖了 args.model_arch）
        is_transformer = (args.model_arch == 'transformer')

    # 数据加载 or 生成
    if args.generate > 0:
        from .heuristic_agent import generate_training_data
        print(f"Generating {args.generate} heuristic games...")
        args.data = generate_training_data(args.generate, seed=0)
        print(f"Generated data saved to {args.data}")

    streaming_mode = False
    prebatched_mode = False
    cache_manifest = build_cache_if_requested(args, mjson_years)

    if args.teacher_mode and is_transformer and args.data_format != 'mjson' and not args.oracle_data:
        raise ValueError(
            '--teacher_mode requires --data_format mjson with '
            '--random_split_all_mjson and --stream_mjson, '
            'or --oracle_data (recorder JSONL with public+private tokens).')

    if args.teacher_mode and is_transformer and not args.teacher_checkpoint:
        raise ValueError(
            '--teacher_mode requires --teacher_checkpoint. '
            'Student distillation must load a pre-trained frozen Teacher; '
            'self-distillation (same model as Teacher) is not supported. '
            'Train a Teacher first with standard SL, then pass its checkpoint.')

    # Oracle teacher training validation
    if args.oracle_teacher_train:
        if not is_transformer:
            raise ValueError(
                '--oracle_teacher_train requires --model_arch transformer.')
        if not args.oracle_data:
            raise ValueError(
                '--oracle_teacher_train requires --oracle_data '
                '(path to recorder JSONL from selfplay_recorder).')
        if args.teacher_mode:
            raise ValueError(
                '--oracle_teacher_train and --teacher_mode are mutually exclusive. '
                '--oracle_teacher_train trains the Teacher itself; '
                '--teacher_mode trains a Student via distillation from a frozen Teacher.')

    # 确定训练阶段：teacher_train / student_distill / public_sl
    if is_transformer:
        if args.oracle_teacher_train:
            args._training_stage = 'teacher_train'
        elif args.teacher_mode:
            args._training_stage = 'student_distill'
        else:
            args._training_stage = 'public_sl'

        # Teacher 模式下 public+private 拼接后序列可能远超 public-only 长度，
        # 自动提升 max_len 以避免位置编码溢出
        _needs_teacher_len = (args._training_stage in ('teacher_train', 'student_distill')
                              and args.oracle_data)
        if _needs_teacher_len and args.max_len < 512:
            print(f"Auto-bumping --max_len from {args.max_len} to 512 "
                  f"for teacher/student training with oracle data "
                  f"(public+private padded length can exceed 256).")
            args.max_len = 512

    # Oracle trajectory data loading (from selfplay_recorder JSONL)
    if args.oracle_data and is_transformer:
        from data.dataset import OracleTrajectoryIterableDataset
        oracle_path = Path(args.oracle_data)
        if oracle_path.is_file():
            oracle_files = [str(oracle_path)]
        elif oracle_path.is_dir():
            oracle_files = sorted(str(p) for p in oracle_path.glob('*.jsonl'))
        else:
            raise ValueError(f'--oracle_data path not found: {args.oracle_data}')
        if not oracle_files:
            raise ValueError(f'No .jsonl files found in {args.oracle_data}')

        # 文件级 split：小样本时复用文件保证 val/test 非空
        rng = np.random.default_rng(args.split_seed)
        n_files = len(oracle_files)
        if n_files >= 3:
            indices = rng.permutation(n_files)
            n_train = max(1, int(n_files * 0.8))
            n_val = max(1, int(n_files * 0.1))
            train_files = [oracle_files[i] for i in indices[:n_train]]
            val_files = [oracle_files[i] for i in indices[n_train:n_train + n_val]]
            test_files = [oracle_files[i] for i in indices[n_train + n_val:]]
            # 保证 test 非空
            if not test_files:
                test_files = val_files
        else:
            # 1-2 个文件：全部复用，避免空 val/test 导致除零
            train_files = oracle_files
            val_files = oracle_files
            test_files = oracle_files
        print(f"Oracle trajectory file split: train={len(train_files)}, "
              f"val={len(val_files)}, test={len(test_files)}")

        streaming_mode = True
        train_set = OracleTrajectoryIterableDataset(
            train_files, shuffle_files=True, seed=args.split_seed)
        val_set = OracleTrajectoryIterableDataset(
            val_files, shuffle_files=False, seed=args.split_seed)
        test_set = OracleTrajectoryIterableDataset(
            test_files, shuffle_files=False, seed=args.split_seed)

    elif args.data_format == 'mjson_cache':
        if is_transformer:
            raise ValueError(
                'mjson_cache does not support Transformer yet. '
                'Use --data_format mjson with --random_split_all_mjson and '
                '--stream_mjson instead.')
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

        # ── 数据划分：manifest 或随机 split ──
        if args.manifest:
            import json as _json
            with open(args.manifest, 'r', encoding='utf-8') as _mf:
                _manifest = _json.load(_mf)
            train_files = _manifest['train_files']
            val_files = _manifest['val_files']
            test_files = _manifest['test_files']
            print(f"Loaded manifest: train={len(train_files)}, "
                  f"val={len(val_files)}, test={len(test_files)}")
        else:
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
            # 自动保存 manifest 以便复现
            _manifest_path = os.path.join(args.checkpoint_dir, 'data_manifest.json')
            Path(_manifest_path).parent.mkdir(parents=True, exist_ok=True)
            import json as _json
            with open(_manifest_path, 'w', encoding='utf-8') as _mf:
                _json.dump({
                    'train_files': train_files,
                    'val_files': val_files,
                    'test_files': test_files,
                    'seed': args.split_seed,
                    'train_ratio': args.train_ratio,
                    'val_ratio': args.val_ratio,
                    'test_ratio': args.test_ratio,
                    'years': mjson_years,
                }, _mf, ensure_ascii=False, indent=2)
            print(f"Random file split: train={len(train_files)}, "
                  f"val={len(val_files)}, test={len(test_files)}")
            print(f"Manifest saved to {_manifest_path}")

        streaming_mode = True
        if is_transformer:
            if args.teacher_mode:
                train_set = build_streaming_public_private_token_dataset(
                    train_files, shuffle_files=True, seed=args.split_seed)
                val_set = build_streaming_public_private_token_dataset(
                    val_files, shuffle_files=False, seed=args.split_seed)
                test_set = build_streaming_public_private_token_dataset(
                    test_files, shuffle_files=False, seed=args.split_seed)
            else:
                train_set = build_streaming_token_dataset(
                    train_files, shuffle_files=True, seed=args.split_seed)
                val_set = build_streaming_token_dataset(
                    val_files, shuffle_files=False, seed=args.split_seed)
                test_set = build_streaming_token_dataset(
                    test_files, shuffle_files=False, seed=args.split_seed)
        else:
            train_set = build_streaming_dataset(
                train_files, shuffle_files=True, seed=args.split_seed)
            val_set = build_streaming_dataset(
                val_files, shuffle_files=False, seed=args.split_seed)
            test_set = build_streaming_dataset(
                test_files, shuffle_files=False, seed=args.split_seed)
    elif args.train_data or args.val_data or args.test_data:
        if is_transformer:
            raise ValueError(
                'Explicit data paths with --model_arch transformer require '
                '--random_split_all_mjson and --stream_mjson. '
                'Direct state-tensor datasets are not compatible with Transformer.')
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
        if is_transformer:
            raise ValueError(
                'Direct data loading with --model_arch transformer requires '
                '--random_split_all_mjson and --stream_mjson. '
                'Direct state-tensor datasets are not compatible with Transformer.')
        dataset = load_data(args.data, data_format=args.data_format,
                            max_mjson_files=args.max_mjson_files)

        # 拆分
        train_set, val_set, test_set = dataset.train_val_test_split()
        print(f"Split: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")

    if not streaming_mode and (len(train_set) == 0 or len(val_set) == 0 or len(test_set) == 0):
        raise ValueError('Train/val/test datasets must all be non-empty')

    train_collate = collate_transformer_batch if is_transformer else None
    # Oracle/teacher 数据默认 drop_last=False，避免小样本被整批丢弃
    train_drop_last = not args.oracle_data
    train_loader = make_dataloader(
        train_set,
        batch_size=args.batch_size,
        shuffle=not streaming_mode,
        drop_last=train_drop_last,
        use_cuda=use_cuda,
        requested_workers=args.num_workers,
        prebatched=prebatched_mode,
        prefetch_factor=args.prefetch_factor,
        collate_fn=train_collate,
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
        collate_fn=train_collate,
    )

    # resume_meta 已在数据加载前读取（见上方 checkpoint 元数据提取）

    # 模型
    if is_transformer:
        model = TransformerPolicyValueNet(
            d_model=args.transformer_d_model,
            n_layers=args.transformer_n_layers,
            n_heads=args.transformer_n_heads,
            n_concept=args.transformer_n_concept,
            max_len=args.max_len,
        )
    else:
        model = MahjongPolicyValueNet()
    model = model.to(device)

    if args.compile and hasattr(torch, 'compile'):
        model = torch.compile(model, mode='reduce-overhead')
        print("Model compiled with torch.compile")

    print(f"Model ({args.model_arch}): {model.count_parameters():,} parameters")

    optimizer = AdamW(model.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 混合精度
    scaler = GradScaler('cuda', enabled=use_cuda and not args.no_amp)

    start_epoch = 0
    skip_batches = 0
    total_batches_trained = 0
    if args.resume:
        start_epoch, resume_meta = load_resume_checkpoint(
            model, args.resume, optimizer=optimizer,
            scheduler=scheduler, scaler=scaler, device=device)
        # 恢复 skip_batches：优先用 checkpoint 中的值，其次用命令行参数
        skip_batches = resume_meta.get('resume_batch', args.resume_batch)
        total_batches_trained = resume_meta.get('total_batches', 0)
        print(f"Resumed from epoch {start_epoch} "
              f"(scheduler/scaler/RNG restored, "
              f"skip_batches={skip_batches}, "
              f"total_batches={total_batches_trained})")

    # 训练循环
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    best_val_acc = float(
        resume_meta.get('best_val_acc', resume_meta.get('val_acc', 0.0)) or 0.0
    )
    history = list(resume_meta.get('history', []))

    # 加载独立 Teacher checkpoint（teacher_mode 下必须提供）
    teacher_model = None
    if args.teacher_checkpoint and is_transformer and args.teacher_mode:
        from models.model_io import infer_transformer_config_from_state_dict
        _teacher_ckpt = torch.load(args.teacher_checkpoint, map_location=device,
                                   weights_only=False)
        _teacher_meta = _teacher_ckpt.get('metadata', {})
        _teacher_arch = _teacher_meta.get('model_arch', 'transformer')
        if _teacher_arch != 'transformer':
            raise ValueError(
                f'--teacher_checkpoint must be a Transformer model, '
                f'but got model_arch={_teacher_arch!r}. '
                f'ResNet teacher is not supported for Transformer student distillation '
                f'(ResNet.forward() does not accept token inputs or mode="teacher").')
        _sd = _teacher_ckpt['model_state_dict']
        # 从 metadata + state_dict 推断完整配置（metadata 缺失时从权重形状补充）
        _teacher_cfg = infer_transformer_config_from_state_dict(_sd, _teacher_meta)
        _teacher_d_model = _teacher_cfg.get('d_model', args.transformer_d_model)
        _teacher_n_layers = _teacher_cfg.get('n_layers', args.transformer_n_layers)
        _teacher_n_concept = _teacher_cfg.get('n_concept', args.transformer_n_concept)
        _teacher_max_len = _teacher_cfg.get('max_len', args.max_len)
        # n_heads 无法从权重可靠推断，必须在 metadata 中明确指定
        _teacher_n_heads = _teacher_meta.get('n_heads')
        if _teacher_n_heads is None:
            raise ValueError(
                'Teacher checkpoint metadata is missing n_heads. '
                'n_heads cannot be reliably inferred from state_dict. '
                'Please re-save the teacher checkpoint with n_heads in metadata, '
                'or pass --transformer_n_heads explicitly when training the teacher.')

        teacher_model = TransformerPolicyValueNet(
            d_model=_teacher_d_model,
            n_layers=_teacher_n_layers,
            n_heads=_teacher_n_heads,
            n_concept=_teacher_n_concept,
            max_len=_teacher_max_len,
        )
        teacher_model.load_state_dict(_sd)
        teacher_model = teacher_model.to(device)
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad_(False)
        print(f"Loaded frozen Transformer teacher from {args.teacher_checkpoint} "
              f"(d_model={_teacher_d_model}, n_layers={_teacher_n_layers}, "
              f"n_concept={_teacher_n_concept}, max_len={_teacher_max_len})")
        # 校验 Teacher checkpoint 的 max_len 是否足够
        if args.oracle_data and _teacher_max_len < args.max_len:
            raise ValueError(
                f"Frozen Teacher checkpoint max_len={_teacher_max_len} is smaller "
                f"than student max_len={args.max_len}. "
                f"Oracle data with public+private tokens may produce sequences "
                f"longer than {_teacher_max_len} in teacher forward. "
                f"Please re-train the Teacher with --max_len {args.max_len} "
                f"or higher before distilling the Student.")
    if start_epoch == 0:
        metrics_history_path(args.checkpoint_dir).write_text('', encoding='utf-8')

    # ── MTL 标签可用性状态报告 ──
    if is_transformer:
        print("\n[MTL Head Status]")
        print(f"  policy:     always trained (human action labels)")
        print(f"  shanten:    {'weight={:.2f}, expects oracle_shanten labels'.format(args.mtl_alpha) if args.mtl_alpha > 0 else 'weight=0.0, skipped'}")
        print(f"  ukeire:     {'weight={:.2f}, expects oracle_ukeire_mask labels'.format(args.mtl_beta) if args.mtl_beta > 0 else 'weight=0.0, skipped'}")
        print("  danger:     PLACEHOLDER (label always 0.0 — no training signal)")
        print("  score:      PLACEHOLDER (label always 0.0 — no training signal)")
        print("  efficiency: PLACEHOLDER (label always 0.0 — no training signal)")
        if args.teacher_mode:
            print(f"  distill KL: weight={args.distill_alpha}, "
                  f"temperature={args.distill_temperature}")
            print(f"  value MSE:  weight={args.distill_value_alpha}")
            print(f"  visibility: {args.private_visibility} "
                  f"(schedule={args.visibility_schedule or 'N/A'})")
        if streaming_mode:
            print("  [Note] Streaming MJSON pipeline provides public/private tokens")
            print("         but NOT oracle_shanten/ukeire labels (collate has no")
            print("         'oracle_shanten' or 'oracle_ukeire_mask' keys).")
            print("         MTL auxiliary heads receive no training signal in streaming mode.")
            print("         Use mjson_cache (TensorShardBatchDataset) to train MTL heads.")
        print()

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
                    'model_arch': args.model_arch,
                    'model': model.count_parameters(),
                    'dataset': args.oracle_data or args.data or args.random_split_all_mjson or args.train_data,
                    'oracle_data': args.oracle_data,
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
                    'training_stage': getattr(args, '_training_stage', 'public_sl'),
                    'mtl_alpha': args.mtl_alpha if is_transformer else None,
                    'mtl_beta': args.mtl_beta if is_transformer else None,
                    'teacher_mode': args.teacher_mode,
                    'private_visibility': args.private_visibility if args.teacher_mode else None,
                    'distill_temperature': args.distill_temperature if args.teacher_mode else None,
                    'distill_alpha': args.distill_alpha if args.teacher_mode else None,
                    'distill_value_alpha': args.distill_value_alpha if args.teacher_mode else None,
                    'mtl_policy': 'always_trained',
                    'mtl_shanten': 'active' if (is_transformer and args.mtl_alpha > 0) else 'off',
                    'mtl_ukeire': 'active' if (is_transformer and args.mtl_beta > 0) else 'off',
                    'mtl_danger': 'PLACEHOLDER_no_label',
                    'mtl_score': 'PLACEHOLDER_no_label',
                    'mtl_efficiency': 'PLACEHOLDER_no_label',
                },
                reinit="finish_previous",
                settings=wandb.Settings(
                    mode="online",
                    init_timeout=120,
                ),
            )
            wandb.define_metric('epoch')
            for metric_pattern in ('train/*', 'val/*', 'test/*',
                                   'time/*', 'best_val_accuracy', 'lr',
                                   'distill/*', 'teacher/*', 'student/*'):
                wandb.define_metric(metric_pattern, step_metric='epoch')
            wandb.watch(model, log='gradients', log_freq=100)
        except Exception as exc:
            raise RuntimeError(
                "W&B was requested with --wandb, but initialization failed. "
                "Run `wandb login` or check the local runtime directory permissions."
            ) from exc

    # ── Step-level checkpoint 配置 ──
    # 将 save_interval_min 转换为 batch 数（估算：~3150 samples/sec, batch_size=256 → ~12.3 batch/s）
    save_interval_batches = 0
    if args.save_interval_min > 0:
        # 每秒约 12 batch（保守估计），转换为 batch 数
        est_batches_per_sec = 12.0
        save_interval_batches = max(100, int(args.save_interval_min * 60 * est_batches_per_sec))
        print(f"Step-level checkpoint: every {save_interval_batches} batches "
              f"(~{args.save_interval_min} min)")

    def _make_step_checkpoint(global_batches):
        """保存 step-level resume checkpoint"""
        _ckpt_path = os.path.join(args.checkpoint_dir, 'sl_resume.pt')
        _meta = {
            **_make_checkpoint_metadata(args, best_val_acc),
            'best_val_acc': best_val_acc,
            'history': history,
            'total_batches': global_batches,
            'resume_batch': 0,  # 续训时从 epoch 开头 fast-forward
        }
        save_resume_checkpoint(model, optimizer, scheduler, scaler, epoch,
                               _ckpt_path, metadata=_meta)
        print(f"    [checkpoint] saved at global_batch={global_batches} "
              f"-> {_ckpt_path}", flush=True)

    for epoch in range(start_epoch, start_epoch + args.epochs):
        t0 = time.time()

        if streaming_mode and hasattr(train_set, 'set_epoch'):
            train_set.set_epoch(epoch)

        # 只在 resume 后的第一个 epoch fast-forward
        epoch_skip = skip_batches if epoch == start_epoch else 0
        if epoch_skip > 0:
            print(f"  Fast-forwarding {epoch_skip} batches in epoch {epoch+1}...",
                  flush=True)

        print(f'  Starting epoch {epoch+1} training '
              f'(global_batches={total_batches_trained})...', flush=True)

        # 解析 visibility_schedule（每个 epoch 使用不同的 private_visibility）
        if is_transformer and args.teacher_mode and args.visibility_schedule:
            _vis_list = [float(v) for v in args.visibility_schedule.split(',')]
            _vis_idx = min(epoch - start_epoch, len(_vis_list) - 1)
            current_visibility = _vis_list[_vis_idx]
        else:
            current_visibility = args.private_visibility

        if is_transformer:
            if args.oracle_teacher_train:
                train_metrics = train_epoch_teacher(
                    model, train_loader, optimizer, device, scaler,
                    alpha=args.mtl_alpha, beta=args.mtl_beta,
                    require_private=True)
            elif args.teacher_mode:
                train_metrics = train_epoch_transformer_distill(
                    model, train_loader, optimizer, device, scaler,
                    alpha=args.mtl_alpha, beta=args.mtl_beta,
                    distill_alpha=args.distill_alpha,
                    distill_value_alpha=args.distill_value_alpha,
                    distill_temperature=args.distill_temperature,
                    private_visibility=current_visibility,
                    teacher_model=teacher_model)
            else:
                train_metrics = train_epoch_transformer(
                    model, train_loader, optimizer, device, scaler,
                    alpha=args.mtl_alpha, beta=args.mtl_beta,
                    save_callback=_make_step_checkpoint,
                    skip_batches=epoch_skip,
                    max_batches=args.max_batches,
                    global_batch_offset=total_batches_trained,
                    save_interval_batches=save_interval_batches)
            if args.oracle_teacher_train:
                val_metrics = validate_teacher(
                    model, val_loader, device,
                    alpha=args.mtl_alpha, beta=args.mtl_beta,
                    value_loss_coef=0.5,
                    require_private=True)
            else:
                val_metrics = validate_transformer(
                    model, val_loader, device,
                    alpha=args.mtl_alpha, beta=args.mtl_beta)
        else:
            train_metrics = train_epoch(
                model, train_loader, optimizer, device, scaler)
            val_metrics = validate(model, val_loader, device)

        scheduler.step()
        elapsed = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]

        # 更新全局 batch 计数
        epoch_batches = train_metrics.get('_batches_trained', 0)
        total_batches_trained += epoch_batches
        stopped_early = train_metrics.get('_stopped_early', False)

        is_best = val_metrics['accuracy'] > best_val_acc
        if is_best:
            best_val_acc = val_metrics['accuracy']

        history_record = {
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'train_acc': train_metrics['accuracy'],
            'val_loss': val_metrics['loss'],
            'val_acc': val_metrics['accuracy'],
        }
        if is_transformer and 'policy_loss' in train_metrics:
            history_record['train_policy_loss'] = train_metrics['policy_loss']
            history_record['train_shanten_loss'] = train_metrics.get(
                'shanten_loss', 0.0)
            history_record['train_ukeire_loss'] = train_metrics.get(
                'ukeire_loss', 0.0)
        history.append(history_record)
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
            'model_arch': args.model_arch,
        }
        if is_transformer:
            epoch_metrics['train/policy_loss'] = train_metrics.get(
                'policy_loss', 0.0)
            epoch_metrics['train/shanten_loss'] = train_metrics.get(
                'shanten_loss', 0.0)
            epoch_metrics['train/ukeire_loss'] = train_metrics.get(
                'ukeire_loss', 0.0)
        if is_transformer and args.teacher_mode:
            epoch_metrics['train/private_visibility'] = current_visibility
            epoch_metrics['distill/kl'] = train_metrics.get(
                'distill/kl', 0.0)
            epoch_metrics['distill/value_mse'] = train_metrics.get(
                'distill/value_mse', 0.0)
        if is_transformer and args.oracle_teacher_train:
            epoch_metrics['teacher/train_value_loss'] = train_metrics.get(
                'value_loss', 0.0)
            epoch_metrics['teacher/val_value_loss'] = val_metrics.get(
                'value_loss', 0.0)
        append_metric_history(args.checkpoint_dir, epoch_metrics)

        if is_transformer:
            print(f"{epoch + 1:>6} loss={train_metrics['loss']:.4f} "
                  f"policy={train_metrics.get('policy_loss', 0):.4f} "
                  f"acc={train_metrics['accuracy']:.4f} "
                  f"val_loss={val_metrics['loss']:.4f} "
                  f"val_acc={val_metrics['accuracy']:.4f} "
                  f"{elapsed:.1f}s lr={current_lr:.2e}")
        else:
            print(f"{epoch + 1:>6} {train_metrics['loss']:>12.4f} "
                  f"{train_metrics['accuracy']:>10.4f} "
                  f"{val_metrics['loss']:>10.4f} {val_metrics['accuracy']:>8.4f} "
                  f"{elapsed:>7.1f}s {current_lr:>10.2e}")

        if wandb_run is not None:
            wandb_run.log(epoch_metrics, step=epoch + 1)

        # 元数据：包含 total_batches 以便 step-level 续训
        _ckpt_meta = {
            **_make_checkpoint_metadata(args, best_val_acc),
            'best_val_acc': best_val_acc,
            'history': history,
            'total_batches': total_batches_trained,
            'resume_batch': 0,
        }

        # 保存最佳模型（带完整续训状态）
        if is_best:
            save_resume_checkpoint(
                model, optimizer, scheduler, scaler, epoch + 1,
                os.path.join(args.checkpoint_dir, 'sl_best.pt'),
                metadata=_ckpt_meta,
            )

        # 每个 epoch 结束都保存 sl_resume.pt（完整续训状态）
        save_resume_checkpoint(
            model, optimizer, scheduler, scaler, epoch + 1,
            os.path.join(args.checkpoint_dir, 'sl_resume.pt'),
            metadata=_ckpt_meta,
        )

        # 定期保存带编号的 checkpoint（完整续训状态）
        if (epoch + 1) % args.save_every == 0:
            save_resume_checkpoint(
                model, optimizer, scheduler, scaler, epoch + 1,
                os.path.join(args.checkpoint_dir, f'sl_epoch_{epoch + 1:03d}.pt'),
                metadata=_ckpt_meta,
            )

        # max_batches 到达后提前结束
        if stopped_early:
            print(f"  Training stopped early after {total_batches_trained} total batches",
                  flush=True)
            break

    # 最终保存（完整续训状态）
    save_resume_checkpoint(
        model, optimizer, scheduler, scaler, start_epoch + args.epochs,
        os.path.join(args.checkpoint_dir, 'sl_final.pt'),
        metadata={'history': history, 'best_val_acc': best_val_acc,
                  **_make_checkpoint_metadata(args, best_val_acc)},
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
        collate_fn=train_collate,
    )
    if is_transformer:
        if args.oracle_teacher_train:
            test_metrics = validate_teacher(model, test_loader, device,
                                            alpha=args.mtl_alpha, beta=args.mtl_beta,
                                            value_loss_coef=0.5,
                                            require_private=True)
        else:
            test_metrics = validate_transformer(model, test_loader, device)
    else:
        test_metrics = validate(model, test_loader, device)
    test_value_loss = test_metrics.get('value_loss')
    print(f"\nTest: loss={test_metrics['loss']:.4f}, "
          f"accuracy={test_metrics['accuracy']:.4f}"
          + (f", value_loss={test_value_loss:.4f}" if test_value_loss is not None else ""))
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
        'model_arch': args.model_arch,
    }
    if test_value_loss is not None:
        final_metrics['test/value_loss'] = test_value_loss
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
