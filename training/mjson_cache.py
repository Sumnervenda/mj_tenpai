"""MJSON 牌谱 → PyTorch Tensor Shard 缓存构建与加载。

将 MJSON 牌谱批量解析为 .npz shard 文件，支持：
  - 多进程并行解析
  - 训练/验证/测试集分割
  - FP16 压缩存储
  - 分片随机读取 Dataset
"""

from __future__ import annotations

import concurrent.futures as futures
import json
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from data.mjson_parser import MJSONRecordParser


CACHE_VERSION = 1
STATE_DIM = 354
ACTION_DIM = 77
SPLITS = ("train", "val", "test")


def _is_mjson_filename(name: str) -> bool:
    return name.endswith(".mjson") or name.endswith(".mjson.gz")


def _scan_mjson_files(root: Path, limit: int = 0) -> List[str]:
    if not root.exists():
        return []

    files: List[str] = []
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            entries = list(os.scandir(current))
        except OSError:
            continue

        dirs = []
        for entry in entries:
            try:
                if entry.is_dir(follow_symlinks=False):
                    dirs.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False) and _is_mjson_filename(entry.name):
                    files.append(entry.path)
                    if limit > 0 and len(files) >= limit:
                        return sorted(files)
            except OSError:
                continue
        stack.extend(reversed(sorted(dirs, key=lambda p: str(p))))

    return sorted(files)


def collect_mjson_files(root_dir: str, max_files: int = 0,
                        years: Optional[Sequence[str]] = None) -> List[str]:
    root = Path(root_dir)
    if years:
        years = [str(year) for year in years]
        if max_files > 0:
            per_year = max_files // len(years)
            remainder = max_files % len(years)
        else:
            per_year = 0
            remainder = 0

        files: List[str] = []
        for idx, year in enumerate(years):
            limit = 0
            if max_files > 0:
                limit = per_year + (1 if idx < remainder else 0)
            files.extend(_scan_mjson_files(root / year, limit=limit))

        if max_files > 0 and len(files) < max_files:
            selected = set(files)
            for fp in _scan_mjson_files(root):
                if fp not in selected:
                    files.append(fp)
                if len(files) >= max_files:
                    break
        files = sorted(files[:max_files] if max_files > 0 else files)
    else:
        files = _scan_mjson_files(root, limit=max_files)

    if not files:
        raise ValueError(f"No .mjson files found under {root_dir}")
    return files


def split_mjson_files(file_paths: Sequence[str], train_ratio: float,
                      val_ratio: float, test_ratio: float,
                      seed: int) -> Tuple[List[str], List[str], List[str]]:
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

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
        raise ValueError("Train/val/test file splits must all be non-empty")
    return train_files, val_files, test_files


def _empty_arrays(state_dtype: str):
    return (
        np.empty((0, STATE_DIM), dtype=np.dtype(state_dtype)),
        np.empty((0, ACTION_DIM), dtype=np.uint8),
        np.empty((0,), dtype=np.uint8),
    )


def _parse_mjson_file_to_arrays(filepath: str, state_dtype: str):
    try:
        parser = MJSONRecordParser(verbose=False)
        samples = parser.parse_file(filepath)
        if not samples:
            states, masks, labels = _empty_arrays(state_dtype)
            return filepath, states, masks, labels, None

        states = np.stack(
            [sample.state_tensor for sample in samples]
        ).astype(np.dtype(state_dtype), copy=False)
        masks = np.stack(
            [sample.action_mask for sample in samples]
        ).astype(np.uint8, copy=False)
        labels = np.fromiter(
            (sample.chosen_action for sample in samples),
            dtype=np.uint8,
            count=len(samples),
        )
        return filepath, states, masks, labels, None
    except Exception as exc:  # pragma: no cover - defensive worker boundary
        states, masks, labels = _empty_arrays(state_dtype)
        return filepath, states, masks, labels, f"{type(exc).__name__}: {exc}"


def _iter_parse_results(file_paths: Sequence[str], state_dtype: str,
                        num_workers: int):
    if num_workers <= 1:
        for fp in file_paths:
            yield _parse_mjson_file_to_arrays(fp, state_dtype)
        return

    pending: Dict[futures.Future, str] = {}
    path_iter = iter(file_paths)

    def submit_next(executor) -> bool:
        try:
            fp = next(path_iter)
        except StopIteration:
            return False
        future = executor.submit(_parse_mjson_file_to_arrays, fp, state_dtype)
        pending[future] = fp
        return True

    with futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        for _ in range(max(1, num_workers * 3)):
            if not submit_next(executor):
                break

        while pending:
            done, _ = futures.wait(
                pending, return_when=futures.FIRST_COMPLETED)
            for future in done:
                pending.pop(future, None)
                yield future.result()
                submit_next(executor)


class _ShardWriter:
    def __init__(self, cache_root: Path, split: str, shard_size: int):
        self.cache_root = cache_root
        self.split = split
        self.shard_size = shard_size
        self.split_dir = cache_root / split
        self.split_dir.mkdir(parents=True, exist_ok=True)
        self.states: List[np.ndarray] = []
        self.masks: List[np.ndarray] = []
        self.labels: List[np.ndarray] = []
        self.buffered = 0
        self.total_samples = 0
        self.shards: List[Dict[str, object]] = []

    def add(self, states: np.ndarray, masks: np.ndarray,
            labels: np.ndarray) -> None:
        offset = 0
        n = int(labels.shape[0])
        while offset < n:
            room = self.shard_size - self.buffered
            take = min(room, n - offset)
            end = offset + take
            self.states.append(states[offset:end])
            self.masks.append(masks[offset:end])
            self.labels.append(labels[offset:end])
            self.buffered += take
            offset = end
            if self.buffered >= self.shard_size:
                self.flush()

    def flush(self) -> None:
        if self.buffered <= 0:
            return
        shard_index = len(self.shards)
        shard_path = self.split_dir / f"shard_{shard_index:06d}.npz"
        states = np.concatenate(self.states, axis=0)
        masks = np.concatenate(self.masks, axis=0)
        labels = np.concatenate(self.labels, axis=0)
        np.savez(shard_path, states=states, masks=masks, labels=labels)

        n = int(labels.shape[0])
        rel_path = shard_path.relative_to(self.cache_root).as_posix()
        self.shards.append({"path": rel_path, "samples": n})
        self.total_samples += n
        self.states.clear()
        self.masks.clear()
        self.labels.clear()
        self.buffered = 0


def _remove_old_cache_files(cache_dir: Path) -> None:
    for split in SPLITS:
        split_dir = cache_dir / split
        if not split_dir.exists():
            continue
        for shard in split_dir.glob("*.npz"):
            shard.unlink()
    manifest = cache_dir / "manifest.json"
    if manifest.exists():
        manifest.unlink()


def _write_manifest(cache_dir: Path, manifest: Dict[str, object]) -> None:
    manifest_path = cache_dir / "manifest.json"
    tmp_path = cache_dir / "manifest.json.tmp"
    tmp_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(manifest_path)


def _write_split_cache(cache_dir: Path, split: str, file_paths: Sequence[str],
                       shard_size: int, state_dtype: str,
                       num_workers: int) -> Dict[str, object]:
    writer = _ShardWriter(cache_dir, split, shard_size)
    started = time.time()
    last_report = started
    errors: List[Dict[str, str]] = []
    processed = 0

    for fp, states, masks, labels, error in _iter_parse_results(
            file_paths, state_dtype, num_workers):
        processed += 1
        if error:
            errors.append({"file": fp, "error": error})
        elif labels.shape[0] > 0:
            writer.add(states, masks, labels)

        now = time.time()
        if processed == len(file_paths) or now - last_report >= 30:
            elapsed = max(now - started, 1e-6)
            rate = processed / elapsed
            print(
                f"  cache {split}: {processed}/{len(file_paths)} files, "
                f"{writer.total_samples + writer.buffered} samples, "
                f"{rate:.1f} files/s",
                flush=True,
            )
            last_report = now

    writer.flush()
    return {
        "num_files": len(file_paths),
        "num_samples": writer.total_samples,
        "num_shards": len(writer.shards),
        "shards": writer.shards,
        "num_errors": len(errors),
        "errors": errors[:50],
    }


def build_mjson_cache(source_root: str, cache_dir: str,
                      years: Optional[Sequence[str]] = None,
                      max_files: int = 0,
                      train_ratio: float = 0.8,
                      val_ratio: float = 0.1,
                      test_ratio: float = 0.1,
                      seed: int = 42,
                      shard_size: int = 65536,
                      state_dtype: str = "float16",
                      num_workers: int = 1,
                      overwrite: bool = False) -> Dict[str, object]:
    if state_dtype not in ("float16", "float32"):
        raise ValueError("state_dtype must be float16 or float32")
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_path / "manifest.json"
    if manifest_path.exists() and not overwrite:
        print(f"Reusing existing MJSON tensor cache: {manifest_path}")
        return load_mjson_cache_manifest(str(cache_path))
    if overwrite:
        _remove_old_cache_files(cache_path)

    print("Collecting MJSON files...", flush=True)
    all_files = collect_mjson_files(source_root, max_files=max_files,
                                    years=years)
    train_files, val_files, test_files = split_mjson_files(
        all_files, train_ratio=train_ratio, val_ratio=val_ratio,
        test_ratio=test_ratio, seed=seed)
    split_files = {
        "train": train_files,
        "val": val_files,
        "test": test_files,
    }
    print(
        f"Cache file split: train={len(train_files)}, "
        f"val={len(val_files)}, test={len(test_files)}",
        flush=True,
    )

    manifest: Dict[str, object] = {
        "version": CACHE_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_root": str(Path(source_root).resolve()),
        "years": list(years) if years else None,
        "max_files": max_files,
        "split_seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "state_dtype": state_dtype,
        "mask_dtype": "uint8",
        "label_dtype": "uint8",
        "shard_size": shard_size,
        "splits": {},
    }

    for split in SPLITS:
        print(f"Building {split} cache shards...", flush=True)
        split_info = _write_split_cache(
            cache_path, split, split_files[split], shard_size,
            state_dtype, max(1, num_workers))
        manifest["splits"][split] = split_info
        _write_manifest(cache_path, manifest)

    return manifest


def load_mjson_cache_manifest(cache_dir: str) -> Dict[str, object]:
    manifest_path = Path(cache_dir) / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"MJSON tensor cache manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("version") != CACHE_VERSION:
        raise ValueError(
            f"Unsupported MJSON cache version: {manifest.get('version')}")
    return manifest


def shard_paths_for_split(cache_dir: str, manifest: Dict[str, object],
                          split: str) -> List[str]:
    split_info = manifest.get("splits", {}).get(split)
    if not split_info:
        raise ValueError(f"MJSON tensor cache has no split named {split!r}")
    cache_path = Path(cache_dir)
    paths = [str(cache_path / shard["path"])
             for shard in split_info.get("shards", [])]
    missing = [path for path in paths if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Missing tensor cache shard: {missing[0]}")
    return paths
# 中文注释：MJSON → tensor shard 缓存构建和加载工具，支持分布式训练的数据高速读取。

