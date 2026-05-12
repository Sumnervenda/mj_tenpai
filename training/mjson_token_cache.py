"""MJSON → Token Shard 缓存构建与加载。

将 MJSON 牌谱批量解析为 token 序列 .npz shard 文件，消除训练时的
每样本 tokenization 开销。构建后可替代流式 MJSONTokenIterableDataset。

用法:
    python -m training.mjson_token_cache build \\
        --source dataset/datasets_years --cache dataset/token_cache_2021-2026 \\
        --years 2021,2022,2023,2024,2025,2026 --num_workers 8

    python -m training.mjson_token_cache info --cache dataset/token_cache_2021-2026
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from training.mjson_cache import (
    collect_mjson_files,
    split_mjson_files,
)

CACHE_VERSION = 1
SPLITS = ("train", "val", "test")

# Sequence length buckets: (min_len, max_len_pad)
# Each bucket pads to its max to minimize waste
LEN_BUCKETS = [
    (1, 80),
    (81, 120),
    (121, 160),
    (161, 200),
    (201, 256),
]


def _bucket_for_length(seq_len: int) -> int:
    for i, (_lo, hi) in enumerate(LEN_BUCKETS):
        if seq_len <= hi:
            return i
    return len(LEN_BUCKETS) - 1


def _bucket_max_len(bucket_idx: int) -> int:
    return LEN_BUCKETS[bucket_idx][1]


# ── Per-worker mjson → token samples ────────────────────────────────────────

def _parse_mjson_to_token_arrays(filepath: str
                                 ) -> Tuple[str, List[Tuple[np.ndarray,
                                                            np.ndarray,
                                                            np.ndarray,
                                                            np.ndarray,
                                                            int]],
                                            Optional[str]]:
    """Parse one mjson file into token samples. Returns (filepath, samples, error).

    Each sample is (token_ids, token_types, behavior_ids, action_mask, label).
    """
    try:
        from data.mjson_parser import MJSONRecordParser
        parser = MJSONRecordParser(verbose=False)
        raw = parser.parse_file_token_samples(filepath)
        samples = []
        for token_ids, token_types, behavior_ids, action_mask, label in raw:
            samples.append((
                np.array(token_ids, dtype=np.int16),
                np.array(token_types, dtype=np.int8),
                np.array(behavior_ids, dtype=np.int16),
                action_mask.astype(np.float16, copy=False),
                int(label),
            ))
        return filepath, samples, None
    except Exception as exc:
        return filepath, [], f"{type(exc).__name__}: {exc}"


# ── Multi-process dispatch ──────────────────────────────────────────────────

def _iter_token_results(file_paths: Sequence[str], num_workers: int):
    if num_workers <= 1:
        for fp in file_paths:
            yield _parse_mjson_to_token_arrays(fp)
        return

    pending: Dict[futures.Future, str] = {}
    path_iter = iter(file_paths)

    def _submit(executor) -> bool:
        try:
            fp = next(path_iter)
        except StopIteration:
            return False
        pending[executor.submit(_parse_mjson_to_token_arrays, fp)] = fp
        return True

    with futures.ProcessPoolExecutor(max_workers=num_workers) as ex:
        for _ in range(max(1, num_workers * 3)):
            if not _submit(ex):
                break
        while pending:
            done, _ = futures.wait(pending, return_when=futures.FIRST_COMPLETED)
            for future in done:
                pending.pop(future, None)
                yield future.result()
                _submit(ex)


# ── Shard writer with bucketing ─────────────────────────────────────────────

class _TokenShardWriter:
    """Accumulates token samples and flushes to compressed npz shards.

    Uses length bucketing to minimize padding waste.
    """

    def __init__(self, cache_root: Path, split: str, shard_size: int):
        self.cache_root = cache_root
        self.split = split
        self.shard_size = shard_size
        self.split_dir = cache_root / split
        self.split_dir.mkdir(parents=True, exist_ok=True)

        # Per-bucket buffers: bucket_idx -> list of sample dicts
        self._buckets: Dict[int, List[Tuple[np.ndarray, np.ndarray,
                                            np.ndarray, np.ndarray, int,
                                            int]]] = defaultdict(list)
        self._bucket_counts: Dict[int, int] = defaultdict(int)  # total samples
        self._shard_indices: Dict[int, int] = defaultdict(int)  # next shard index
        self._total_samples = 0
        self._shards: List[Dict[str, object]] = []

    def add(self, token_ids: np.ndarray, token_types: np.ndarray,
            behavior_ids: np.ndarray, action_mask: np.ndarray,
            label: int) -> None:
        seq_len = token_ids.shape[0]
        bucket = _bucket_for_length(seq_len)
        self._buckets[bucket].append(
            (token_ids, token_types, behavior_ids, action_mask, label, seq_len))
        self._bucket_counts[bucket] += 1

        # Check if bucket should flush
        if len(self._buckets[bucket]) >= self.shard_size:
            self._flush_bucket(bucket)

    def _flush_bucket(self, bucket: int) -> None:
        samples = self._buckets[bucket]
        if not samples:
            return

        max_len = _bucket_max_len(bucket)
        n = len(samples)
        shard_idx = self._shard_indices[bucket]

        ids = np.zeros((n, max_len), dtype=np.int16)
        types = np.zeros((n, max_len), dtype=np.int8)
        bids = np.zeros((n, max_len), dtype=np.int16)
        attn = np.ones((n, max_len), dtype=bool)  # True = padding
        masks = np.zeros((n, 77), dtype=np.float16)
        labels = np.zeros(n, dtype=np.int16)
        lengths = np.zeros(n, dtype=np.int16)

        for i, (tid, ttype, tbid, mask, label, seq_len) in enumerate(samples):
            L = min(seq_len, max_len)
            ids[i, :L] = tid[:L]
            types[i, :L] = ttype[:L]
            bids[i, :L] = tbid[:L]
            attn[i, :L] = False  # real tokens
            masks[i] = mask
            labels[i] = label
            lengths[i] = L

        shard_path = self.split_dir / f"shard_b{bucket}_{shard_idx:06d}.npz"
        np.savez_compressed(
            shard_path,
            token_ids=ids,
            token_types=types,
            behavior_ids=bids,
            attention_mask=attn,
            action_mask=masks,
            labels=labels,
            lengths=lengths,
        )

        rel = shard_path.relative_to(self.cache_root).as_posix()
        self._shards.append({"path": rel, "samples": n, "bucket": bucket,
                             "max_len": max_len})
        self._total_samples += n
        self._bucket_counts[bucket] -= n  # subtract flushed samples
        self._buckets[bucket].clear()
        self._shard_indices[bucket] = shard_idx + 1

    def flush_all(self) -> None:
        for bucket in list(self._buckets.keys()):
            self._flush_bucket(bucket)


# ── Build entry point ───────────────────────────────────────────────────────

def _write_manifest(cache_dir: Path, manifest: Dict[str, object]) -> None:
    manifest_path = cache_dir / "manifest.json"
    tmp = cache_dir / "manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    tmp.replace(manifest_path)


def validate_token_cache_manifest(manifest: Dict[str, object]) -> None:
    """Validate manifest structure before a training run uses the cache."""
    if manifest.get("version") != CACHE_VERSION:
        raise ValueError(f"Unsupported token cache version: {manifest.get('version')}")
    if manifest.get("format") != "token":
        raise ValueError(
            "This cache is not a token cache. Use --mjson_cache for ResNet format.")

    splits = manifest.get("splits")
    if not isinstance(splits, dict):
        raise ValueError("Token cache manifest is missing the 'splits' object")

    missing = [split for split in SPLITS if split not in splits]
    if missing:
        raise ValueError(
            "Token cache manifest is incomplete; missing split(s): "
            f"{', '.join(missing)}. The previous build may have been interrupted. "
            "Rebuild with `python -m training.mjson_token_cache build --overwrite ...`.")

    for split in SPLITS:
        split_info = splits[split]
        if not isinstance(split_info, dict):
            raise ValueError(f"Token cache split {split!r} is not an object")
        shards = split_info.get("shards")
        if not isinstance(shards, list):
            raise ValueError(f"Token cache split {split!r} is missing shard metadata")
        if split_info.get("num_shards", len(shards)) != len(shards):
            raise ValueError(
                f"Token cache split {split!r} num_shards does not match shards list")


def assert_token_cache_ready(cache_dir: str) -> Dict[str, object]:
    """Load a token cache and verify all split shard files exist."""
    manifest = load_token_cache_manifest(cache_dir)
    for split in SPLITS:
        token_shard_paths_for_split(cache_dir, manifest, split)
    return manifest


def build_token_cache(source_root: str, cache_dir: str,
                      years: Optional[Sequence[str]] = None,
                      max_files: int = 0,
                      train_ratio: float = 0.8,
                      val_ratio: float = 0.1,
                      test_ratio: float = 0.1,
                      seed: int = 42,
                      shard_size: int = 10000,
                      num_workers: int = 1,
                      overwrite: bool = False) -> Dict[str, object]:
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_path / "manifest.json"

    if manifest_path.exists() and not overwrite:
        print(f"Reusing existing token cache: {manifest_path}")
        try:
            return assert_token_cache_ready(str(cache_path))
        except Exception as exc:
            raise RuntimeError(
                f"Existing token cache is incomplete or invalid: {manifest_path}. "
                "Re-run the build with --overwrite, or remove the cache directory."
            ) from exc
    if overwrite:
        import shutil
        for d in cache_path.iterdir():
            if d.is_dir():
                shutil.rmtree(d)
            else:
                d.unlink()

    print("Collecting MJSON files...", flush=True)
    all_files = collect_mjson_files(source_root, max_files=max_files,
                                    years=years)
    train_files, val_files, test_files = split_mjson_files(
        all_files, train_ratio, val_ratio, test_ratio, seed)
    split_files = {"train": train_files, "val": val_files, "test": test_files}
    print(f"Token cache split: train={len(train_files)}, "
          f"val={len(val_files)}, test={len(test_files)}", flush=True)

    manifest: Dict[str, object] = {
        "version": CACHE_VERSION,
        "format": "token",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_root": str(Path(source_root).resolve()),
        "years": list(years) if years else None,
        "max_files": max_files,
        "split_seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "vocab_size": 128,
        "action_dim": 77,
        "shard_size": shard_size,
        "len_buckets": LEN_BUCKETS,
        "splits": {},
    }

    for split in SPLITS:
        print(f"Building {split} token cache shards...", flush=True)
        writer = _TokenShardWriter(cache_path, split, shard_size)
        started = time.time()
        last_report = started
        errors: List[Dict[str, str]] = []
        processed = 0
        total_samples = 0

        for fp, samples, error in _iter_token_results(
                split_files[split], max(1, num_workers)):
            processed += 1
            if error:
                errors.append({"file": fp, "error": error})
            else:
                for tok_ids, tok_types, tok_bids, mask, label in samples:
                    writer.add(tok_ids, tok_types, tok_bids, mask, label)
                total_samples += len(samples)

            now = time.time()
            if processed == len(split_files[split]) or now - last_report >= 30:
                elapsed = max(now - started, 1e-6)
                rate = processed / elapsed
                print(f"  token_cache {split}: {processed}/{len(split_files[split])} "
                      f"files, {total_samples} samples, {rate:.1f} files/s",
                      flush=True)
                last_report = now

        writer.flush_all()
        split_info = {
            "num_files": len(split_files[split]),
            "num_samples": writer._total_samples,
            "num_shards": len(writer._shards),
            "shards": writer._shards,
            "num_errors": len(errors),
            "errors": errors[:50],
        }
        manifest["splits"][split] = split_info
        _write_manifest(cache_path, manifest)

    print(f"Token cache built: {manifest['splits']['train']['num_samples']} "
          f"train samples, {len(manifest['splits']['train']['shards'])} shards")
    return manifest


# ── Load helpers ────────────────────────────────────────────────────────────

def load_token_cache_manifest(cache_dir: str) -> Dict[str, object]:
    manifest_path = Path(cache_dir) / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Token cache manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_token_cache_manifest(manifest)
    return manifest


def token_shard_paths_for_split(cache_dir: str, manifest: Dict[str, object],
                                split: str) -> List[str]:
    split_info = manifest.get("splits", {}).get(split)
    if not split_info:
        raise ValueError(f"Token cache has no split named {split!r}")
    cache_path = Path(cache_dir)
    paths = [str(cache_path / shard["path"])
             for shard in split_info.get("shards", [])]
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"Missing token cache shard: {missing[0]}")
    return paths


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='MJSON → Token shard cache builder')
    sub = parser.add_subparsers(dest='command', required=True)

    build_p = sub.add_parser('build', help='Build token cache from mjson files')
    build_p.add_argument('--source', type=str, required=True,
                         help='Root directory with year subdirectories')
    build_p.add_argument('--cache', type=str, required=True,
                         help='Output cache directory')
    build_p.add_argument('--years', type=str, default=None,
                         help='Comma-separated years (e.g. "2021,2022,2023")')
    build_p.add_argument('--num_workers', type=int, default=1,
                         help='Parallel workers for parsing')
    build_p.add_argument('--shard_size', type=int, default=10000,
                         help='Samples per shard')
    build_p.add_argument('--max_files', type=int, default=0,
                         help='Max files to process (0=all)')
    build_p.add_argument('--seed', type=int, default=42)
    build_p.add_argument('--train_ratio', type=float, default=0.8)
    build_p.add_argument('--val_ratio', type=float, default=0.1)
    build_p.add_argument('--test_ratio', type=float, default=0.1)
    build_p.add_argument('--overwrite', action='store_true')

    info_p = sub.add_parser('info', help='Show token cache manifest info')
    info_p.add_argument('--cache', type=str, required=True)

    args = parser.parse_args()

    try:
        if args.command == 'build':
            years = None
            if args.years:
                years = [y.strip() for y in args.years.split(',')]
            build_token_cache(
                source_root=args.source,
                cache_dir=args.cache,
                years=years,
                max_files=args.max_files,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                seed=args.seed,
                shard_size=args.shard_size,
                num_workers=args.num_workers,
                overwrite=args.overwrite,
            )
        elif args.command == 'info':
            manifest = assert_token_cache_ready(args.cache)
            print(json.dumps({k: v for k, v in manifest.items() if k != 'splits'},
                             indent=2, ensure_ascii=False))
            for split in SPLITS:
                s = manifest['splits'][split]
                print(f"{split}: {s['num_files']} files, "
                      f"{s['num_samples']} samples, {s['num_shards']} shards")
    except Exception as exc:
        raise SystemExit(f"Error: {exc}") from None


if __name__ == '__main__':
    main()
