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
MMAP_CACHE_VERSION = 1
SPLITS = ("train", "val", "test")
ACTION_DIM = 77
ACTION_MASK_BYTES = (ACTION_DIM + 7) // 8

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


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def _pack_action_masks(action_mask: np.ndarray) -> np.ndarray:
    """Pack (N, 77) 0/1 masks into (N, 10) uint8 bitsets."""
    mask = np.asarray(action_mask) > 0
    packed = np.packbits(mask, axis=1, bitorder="little")
    return packed[:, :ACTION_MASK_BYTES].astype(np.uint8, copy=False)


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


class _TokenMmapShardWriter:
    """Write compact ragged token shards as raw mmap-friendly binary files.

    Layout per shard:
      token_ids.u1 / token_types.u1 / behavior_ids.u1: flat token arrays
      offsets.u4: sample offsets into the flat arrays, length N+1
      action_mask.u1: bit-packed (N, 10) action masks for 77 actions
      labels.u1: target action id per sample

    This format avoids padding and stores masks as bits. It is meant for
    local NVMe / cloud disk training after upload or after building on server.
    """

    def __init__(self, cache_root: Path, split: str, shard_size: int):
        if shard_size <= 0:
            raise ValueError("mmap shard_size must be positive")
        self.cache_root = cache_root
        self.split = split
        self.shard_size = shard_size
        self.split_dir = cache_root / split
        self.split_dir.mkdir(parents=True, exist_ok=True)

        self._samples: List[Tuple[np.ndarray, np.ndarray, np.ndarray,
                                  np.ndarray, int]] = []
        self._padded_chunks: List[Tuple[np.ndarray, np.ndarray, np.ndarray,
                                        np.ndarray, np.ndarray, np.ndarray]] = []
        self._pending_samples = 0
        self._shard_index = 0
        self._total_samples = 0
        self._total_tokens = 0
        self._total_size_bytes = 0
        self._shards: List[Dict[str, object]] = []

    def add(self, token_ids: np.ndarray, token_types: np.ndarray,
            behavior_ids: np.ndarray, action_mask: np.ndarray,
            label: int) -> None:
        if self._pending_samples >= self.shard_size:
            self._flush()
        packed = _pack_action_masks(np.asarray(action_mask)[None, :])[0]
        self._samples.append((
            np.asarray(token_ids, dtype=np.uint8),
            np.asarray(token_types, dtype=np.uint8),
            np.asarray(behavior_ids, dtype=np.uint8),
            packed,
            int(label),
        ))
        self._pending_samples += 1

    def add_padded_batch(self, token_ids: np.ndarray, token_types: np.ndarray,
                         behavior_ids: np.ndarray, action_mask: np.ndarray,
                         labels: np.ndarray, lengths: np.ndarray) -> None:
        n = int(labels.shape[0])
        start = 0
        while start < n:
            if self._pending_samples >= self.shard_size:
                self._flush()
            room = self.shard_size - self._pending_samples
            end = min(n, start + room)
            sl = slice(start, end)
            self._padded_chunks.append((
                np.asarray(token_ids[sl], dtype=np.uint8).copy(),
                np.asarray(token_types[sl], dtype=np.uint8).copy(),
                np.asarray(behavior_ids[sl], dtype=np.uint8).copy(),
                _pack_action_masks(action_mask[sl]).copy(),
                np.asarray(labels[sl], dtype=np.uint8).copy(),
                np.asarray(lengths[sl], dtype=np.uint16).copy(),
            ))
            self._pending_samples += end - start
            start = end

    def flush_all(self) -> None:
        self._flush()

    def _flush(self) -> None:
        n = self._pending_samples
        if n == 0:
            return

        lengths = np.empty(n, dtype=np.uint16)
        labels = np.empty(n, dtype=np.uint8)
        packed_masks = np.empty((n, ACTION_MASK_BYTES), dtype=np.uint8)

        row = 0
        total_tokens = 0
        max_len = 0
        for tid, _ttype, _tbid, _mask, label in self._samples:
            seq_len = int(tid.shape[0])
            lengths[row] = seq_len
            labels[row] = label
            packed_masks[row] = _mask
            total_tokens += seq_len
            max_len = max(max_len, seq_len)
            row += 1
        for _ids, _types, _bids, masks, chunk_labels, chunk_lengths in self._padded_chunks:
            c_n = int(chunk_labels.shape[0])
            lengths[row:row + c_n] = chunk_lengths
            labels[row:row + c_n] = chunk_labels
            packed_masks[row:row + c_n] = masks
            total_tokens += int(chunk_lengths.astype(np.uint64).sum())
            max_len = max(max_len, int(chunk_lengths.max(initial=0)))
            row += c_n

        offsets = np.empty(n + 1, dtype=np.uint32)
        offsets[0] = 0
        np.cumsum(lengths.astype(np.uint32), out=offsets[1:])
        if int(offsets[-1]) != total_tokens:
            raise RuntimeError("Internal mmap offset calculation mismatch")

        flat_ids = np.empty(total_tokens, dtype=np.uint8)
        flat_types = np.empty(total_tokens, dtype=np.uint8)
        flat_bids = np.empty(total_tokens, dtype=np.uint8)

        cursor = 0
        for tid, ttype, tbid, _mask, _label in self._samples:
            seq_len = int(tid.shape[0])
            flat_ids[cursor:cursor + seq_len] = tid
            flat_types[cursor:cursor + seq_len] = ttype
            flat_bids[cursor:cursor + seq_len] = tbid
            cursor += seq_len
        for ids, types, bids, _masks, _labels, chunk_lengths in self._padded_chunks:
            width = ids.shape[1]
            valid = np.arange(width, dtype=np.uint16)[None, :] < chunk_lengths[:, None]
            token_count = int(valid.sum())
            flat_ids[cursor:cursor + token_count] = ids[valid]
            flat_types[cursor:cursor + token_count] = types[valid]
            flat_bids[cursor:cursor + token_count] = bids[valid]
            cursor += token_count

        shard_dir = self.split_dir / f"shard_{self._shard_index:06d}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        files = {
            "token_ids": "token_ids.u1",
            "token_types": "token_types.u1",
            "behavior_ids": "behavior_ids.u1",
            "offsets": "offsets.u4",
            "action_mask": "action_mask.u1",
            "labels": "labels.u1",
        }
        flat_ids.tofile(shard_dir / files["token_ids"])
        flat_types.tofile(shard_dir / files["token_types"])
        flat_bids.tofile(shard_dir / files["behavior_ids"])
        offsets.tofile(shard_dir / files["offsets"])
        packed_masks.tofile(shard_dir / files["action_mask"])
        labels.tofile(shard_dir / files["labels"])

        rel = shard_dir.relative_to(self.cache_root).as_posix()
        size_bytes = _dir_size_bytes(shard_dir)
        self._shards.append({
            "path": rel,
            "samples": n,
            "tokens": total_tokens,
            "max_len": max_len,
            "size_bytes": size_bytes,
            "files": files,
        })
        self._total_samples += n
        self._total_tokens += total_tokens
        self._total_size_bytes += size_bytes
        self._shard_index += 1

        self._samples.clear()
        self._padded_chunks.clear()
        self._pending_samples = 0


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


def _init_mmap_manifest(source_root: Optional[str],
                        years: Optional[Sequence[str]],
                        max_files: int,
                        seed: int,
                        train_ratio: float,
                        val_ratio: float,
                        test_ratio: float,
                        shard_size: int,
                        source_cache: Optional[str] = None) -> Dict[str, object]:
    manifest: Dict[str, object] = {
        "version": MMAP_CACHE_VERSION,
        "format": "token_mmap",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_root": str(Path(source_root).resolve()) if source_root else None,
        "source_cache": str(Path(source_cache).resolve()) if source_cache else None,
        "years": list(years) if years else None,
        "max_files": max_files,
        "split_seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "vocab_size": 128,
        "action_dim": ACTION_DIM,
        "action_mask_bytes": ACTION_MASK_BYTES,
        "shard_size": shard_size,
        "storage": {
            "layout": "ragged_flat_mmap",
            "token_dtype": "uint8",
            "offset_dtype": "uint32",
            "label_dtype": "uint8",
            "action_mask": "bitpack_little",
        },
        "splits": {},
    }
    return manifest


def build_token_mmap_cache(source_root: str, cache_dir: str,
                           years: Optional[Sequence[str]] = None,
                           max_files: int = 0,
                           train_ratio: float = 0.8,
                           val_ratio: float = 0.1,
                           test_ratio: float = 0.1,
                           seed: int = 42,
                           shard_size: int = 200000,
                           num_workers: int = 1,
                           overwrite: bool = False) -> Dict[str, object]:
    cache_path = Path(cache_dir)
    manifest_path = cache_path / "manifest.json"
    if manifest_path.exists() and not overwrite:
        print(f"Reusing existing token mmap cache: {manifest_path}")
        return assert_token_mmap_cache_ready(str(cache_path))
    if overwrite and cache_path.exists():
        import shutil
        shutil.rmtree(cache_path)
    cache_path.mkdir(parents=True, exist_ok=True)

    print("Collecting MJSON files...", flush=True)
    all_files = collect_mjson_files(source_root, max_files=max_files,
                                    years=years)
    train_files, val_files, test_files = split_mjson_files(
        all_files, train_ratio, val_ratio, test_ratio, seed)
    split_files = {"train": train_files, "val": val_files, "test": test_files}
    print(f"Token mmap split: train={len(train_files)}, "
          f"val={len(val_files)}, test={len(test_files)}", flush=True)

    manifest = _init_mmap_manifest(
        source_root=source_root,
        years=years,
        max_files=max_files,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        shard_size=shard_size,
    )

    for split in SPLITS:
        print(f"Building {split} token mmap shards...", flush=True)
        writer = _TokenMmapShardWriter(cache_path, split, shard_size)
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
                print(f"  token_mmap {split}: {processed}/{len(split_files[split])} "
                      f"files, {total_samples} samples, {rate:.1f} files/s",
                      flush=True)
                last_report = now

        writer.flush_all()
        manifest["splits"][split] = {
            "num_files": len(split_files[split]),
            "num_samples": writer._total_samples,
            "num_tokens": writer._total_tokens,
            "num_shards": len(writer._shards),
            "size_bytes": writer._total_size_bytes,
            "shards": writer._shards,
            "num_errors": len(errors),
            "errors": errors[:50],
        }
        _write_manifest(cache_path, manifest)

    print(f"Token mmap cache built: "
          f"{manifest['splits']['train']['num_samples']} train samples, "
          f"{manifest['splits']['train']['num_shards']} train shards, "
          f"{_dir_size_bytes(cache_path) / 1024**3:.2f} GiB")
    return manifest


def convert_token_cache_to_mmap(source_cache: str, cache_dir: str,
                                shard_size: int = 200000,
                                overwrite: bool = False) -> Dict[str, object]:
    """Convert existing compressed .npz token cache to compact mmap cache."""
    src_manifest = assert_token_cache_ready(source_cache)
    cache_path = Path(cache_dir)
    manifest_path = cache_path / "manifest.json"
    if manifest_path.exists() and not overwrite:
        print(f"Reusing existing token mmap cache: {manifest_path}")
        return assert_token_mmap_cache_ready(str(cache_path))
    if overwrite and cache_path.exists():
        import shutil
        shutil.rmtree(cache_path)
    cache_path.mkdir(parents=True, exist_ok=True)

    manifest = _init_mmap_manifest(
        source_root=src_manifest.get("source_root"),
        source_cache=source_cache,
        years=src_manifest.get("years"),
        max_files=int(src_manifest.get("max_files", 0) or 0),
        seed=int(src_manifest.get("split_seed", 42)),
        train_ratio=float(src_manifest.get("train_ratio", 0.8)),
        val_ratio=float(src_manifest.get("val_ratio", 0.1)),
        test_ratio=float(src_manifest.get("test_ratio", 0.1)),
        shard_size=shard_size,
    )

    for split in SPLITS:
        print(f"Converting {split} token shards to mmap...", flush=True)
        writer = _TokenMmapShardWriter(cache_path, split, shard_size)
        started = time.time()
        last_report = started
        old_paths = token_shard_paths_for_split(source_cache, src_manifest, split)
        for idx, path in enumerate(old_paths, start=1):
            with np.load(path, allow_pickle=False) as shard:
                writer.add_padded_batch(
                    shard["token_ids"],
                    shard["token_types"],
                    shard["behavior_ids"],
                    shard["action_mask"],
                    shard["labels"],
                    shard["lengths"],
                )

            now = time.time()
            if idx == len(old_paths) or now - last_report >= 30:
                elapsed = max(now - started, 1e-6)
                rate = idx / elapsed
                print(f"  convert {split}: {idx}/{len(old_paths)} shards, "
                      f"{writer._total_samples + writer._pending_samples} samples, "
                      f"{rate:.1f} shards/s", flush=True)
                last_report = now

        writer.flush_all()
        src_split = src_manifest["splits"][split]
        manifest["splits"][split] = {
            "num_files": src_split.get("num_files", 0),
            "num_samples": writer._total_samples,
            "num_tokens": writer._total_tokens,
            "num_shards": len(writer._shards),
            "size_bytes": writer._total_size_bytes,
            "shards": writer._shards,
            "num_errors": src_split.get("num_errors", 0),
            "errors": src_split.get("errors", [])[:50],
        }
        _write_manifest(cache_path, manifest)

    print(f"Token mmap cache ready: {_dir_size_bytes(cache_path) / 1024**3:.2f} GiB "
          f"at {cache_path}")
    return manifest


# ── Load helpers ────────────────────────────────────────────────────────────

def load_token_cache_manifest(cache_dir: str) -> Dict[str, object]:
    manifest_path = Path(cache_dir) / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Token cache manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_token_cache_manifest(manifest)
    return manifest


def validate_token_mmap_manifest(manifest: Dict[str, object]) -> None:
    if manifest.get("version") != MMAP_CACHE_VERSION:
        raise ValueError(
            f"Unsupported token mmap cache version: {manifest.get('version')}")
    if manifest.get("format") != "token_mmap":
        raise ValueError("This cache is not a token mmap cache.")
    if int(manifest.get("action_dim", 0)) != ACTION_DIM:
        raise ValueError(
            f"Unsupported action_dim={manifest.get('action_dim')}, expected {ACTION_DIM}")
    splits = manifest.get("splits")
    if not isinstance(splits, dict):
        raise ValueError("Token mmap manifest is missing the 'splits' object")
    missing = [split for split in SPLITS if split not in splits]
    if missing:
        raise ValueError(
            "Token mmap manifest is incomplete; missing split(s): "
            f"{', '.join(missing)}")
    for split in SPLITS:
        split_info = splits[split]
        shards = split_info.get("shards")
        if not isinstance(shards, list):
            raise ValueError(f"Token mmap split {split!r} is missing shards")
        if split_info.get("num_shards", len(shards)) != len(shards):
            raise ValueError(
                f"Token mmap split {split!r} num_shards does not match shards list")


def load_token_mmap_cache_manifest(cache_dir: str) -> Dict[str, object]:
    manifest_path = Path(cache_dir) / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Token mmap cache manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_token_mmap_manifest(manifest)
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


def token_mmap_shards_for_split(cache_dir: str, manifest: Dict[str, object],
                                split: str) -> List[Dict[str, object]]:
    split_info = manifest.get("splits", {}).get(split)
    if not split_info:
        raise ValueError(f"Token mmap cache has no split named {split!r}")
    cache_path = Path(cache_dir)
    checked = []
    for shard in split_info.get("shards", []):
        shard_dir = cache_path / shard["path"]
        files = shard.get("files", {})
        required = ["token_ids", "token_types", "behavior_ids",
                    "offsets", "action_mask", "labels"]
        for key in required:
            path = shard_dir / files.get(key, "")
            if not path.exists():
                raise FileNotFoundError(f"Missing token mmap shard file: {path}")
        checked.append(dict(shard))
    return checked


def assert_token_mmap_cache_ready(cache_dir: str) -> Dict[str, object]:
    manifest = load_token_mmap_cache_manifest(cache_dir)
    for split in SPLITS:
        token_mmap_shards_for_split(cache_dir, manifest, split)
    return manifest


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

    mmap_p = sub.add_parser('build-mmap',
                            help='Build compact mmap token cache from mjson files')
    mmap_p.add_argument('--source', type=str, required=True,
                        help='Root directory with year subdirectories')
    mmap_p.add_argument('--cache', type=str, required=True,
                        help='Output mmap cache directory')
    mmap_p.add_argument('--years', type=str, default=None,
                        help='Comma-separated years (e.g. "2021,2022,2023")')
    mmap_p.add_argument('--num_workers', type=int, default=1,
                        help='Parallel workers for parsing')
    mmap_p.add_argument('--shard_size', type=int, default=200000,
                        help='Samples per mmap shard')
    mmap_p.add_argument('--max_files', type=int, default=0,
                        help='Max files to process (0=all)')
    mmap_p.add_argument('--seed', type=int, default=42)
    mmap_p.add_argument('--train_ratio', type=float, default=0.8)
    mmap_p.add_argument('--val_ratio', type=float, default=0.1)
    mmap_p.add_argument('--test_ratio', type=float, default=0.1)
    mmap_p.add_argument('--overwrite', action='store_true')

    convert_p = sub.add_parser(
        'convert-mmap',
        help='Convert an existing compressed token cache to compact mmap format')
    convert_p.add_argument('--source_cache', type=str, required=True,
                           help='Existing .npz token cache directory')
    convert_p.add_argument('--cache', type=str, required=True,
                           help='Output mmap cache directory')
    convert_p.add_argument('--shard_size', type=int, default=200000,
                           help='Samples per mmap shard')
    convert_p.add_argument('--overwrite', action='store_true')

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
        elif args.command == 'build-mmap':
            years = None
            if args.years:
                years = [y.strip() for y in args.years.split(',')]
            build_token_mmap_cache(
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
        elif args.command == 'convert-mmap':
            convert_token_cache_to_mmap(
                source_cache=args.source_cache,
                cache_dir=args.cache,
                shard_size=args.shard_size,
                overwrite=args.overwrite,
            )
        elif args.command == 'info':
            raw_manifest_path = Path(args.cache) / "manifest.json"
            if not raw_manifest_path.exists():
                raise FileNotFoundError(
                    f"Token cache manifest not found: {raw_manifest_path}")
            raw_manifest = json.loads(raw_manifest_path.read_text(encoding="utf-8"))
            if raw_manifest.get("format") == "token_mmap":
                manifest = assert_token_mmap_cache_ready(args.cache)
            else:
                manifest = assert_token_cache_ready(args.cache)
            print(json.dumps({k: v for k, v in manifest.items() if k != 'splits'},
                             indent=2, ensure_ascii=False))
            for split in SPLITS:
                s = manifest['splits'][split]
                extra = ""
                if manifest.get("format") == "token_mmap":
                    size_gib = float(s.get("size_bytes", 0)) / 1024**3
                    extra = f", {s.get('num_tokens', 0)} tokens, {size_gib:.2f} GiB"
                print(f"{split}: {s.get('num_files', 0)} files, "
                      f"{s['num_samples']} samples, {s['num_shards']} shards"
                      f"{extra}")
    except Exception as exc:
        raise SystemExit(f"Error: {exc}") from None


if __name__ == '__main__':
    main()
