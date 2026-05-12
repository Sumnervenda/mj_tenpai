"""PyTorch Dataset —— 包装训练样本供 DataLoader 使用。"""

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from .record_parser import TrainingSample


class MahjongStateActionDataset(Dataset):
    """日麻状态-动作对数据集（监督学习用）。

    Args:
        samples: TrainingSample 列表
        states: (N, 354) 状态张量（如直接传入则跳过 samples）
        masks: (N, 77) 动作掩码（如直接传入则跳过 samples）
        labels: (N,) 目标动作索引（如直接传入则跳过 samples）
    """

    def __init__(self,
                 samples: Optional[List[TrainingSample]] = None,
                 states: Optional[torch.Tensor] = None,
                 masks: Optional[torch.Tensor] = None,
                 labels: Optional[torch.Tensor] = None):
        if samples is not None:
            self.states = torch.FloatTensor(
                np.stack([s.state_tensor for s in samples]))
            self.masks = torch.FloatTensor(
                np.stack([s.action_mask.astype(np.float32) for s in samples]))
            self.labels = torch.LongTensor([s.chosen_action for s in samples])
        else:
            self.states = states
            self.masks = masks
            self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return (self.states[idx], self.masks[idx], self.labels[idx])

    def train_val_test_split(self, ratios=(0.8, 0.1, 0.1), seed: int = 42):
        """将数据集拆分为 train/val/test 三个子集。"""
        n = len(self)
        indices = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
        train_end = int(n * ratios[0])
        val_end = train_end + int(n * ratios[1])

        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]

        return (
            MahjongStateActionDataset(
                states=self.states[train_idx],
                masks=self.masks[train_idx],
                labels=self.labels[train_idx],
            ),
            MahjongStateActionDataset(
                states=self.states[val_idx],
                masks=self.masks[val_idx],
                labels=self.labels[val_idx],
            ),
            MahjongStateActionDataset(
                states=self.states[test_idx],
                masks=self.masks[test_idx],
                labels=self.labels[test_idx],
            ),
        )


class MJSONIterableDataset(IterableDataset):
    """按文件流式解析 MJSON 训练样本，避免一次性载入全部样本。"""

    def __init__(self,
                 file_paths: Sequence[str],
                 shuffle_files: bool = False,
                 seed: int = 42,
                 parser_verbose: bool = False):
        super().__init__()
        self.file_paths = [str(p) for p in file_paths]
        self.shuffle_files = shuffle_files
        self.seed = seed
        self.epoch = 0
        self.parser_verbose = parser_verbose

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        from .mjson_parser import MJSONRecordParser
        parser = MJSONRecordParser(verbose=self.parser_verbose)
        file_paths = list(self.file_paths)

        if self.shuffle_files:
            rng = np.random.default_rng(self.seed + self.epoch)
            rng.shuffle(file_paths)

        worker = get_worker_info()
        if worker is not None:
            file_paths = file_paths[worker.id::worker.num_workers]

        for fp in file_paths:
            for sample in parser.parse_file(fp):
                yield (
                    sample.state_tensor.astype(np.float32, copy=False),
                    sample.action_mask.astype(np.float32, copy=False),
                    np.int64(sample.chosen_action),
                )

    def __len__(self) -> int:
        return len(self.file_paths)


class MJSONTokenIterableDataset(IterableDataset):
    """按文件流式解析 MJSON，输出 token 序列样本（Transformer 训练用）。

    每个样本为 (token_ids, token_types, behavior_ids, action_mask, label)，
    配合 collate_transformer_batch 使用。
    """

    def __init__(self,
                 file_paths: Sequence[str],
                 shuffle_files: bool = False,
                 seed: int = 42,
                 parser_verbose: bool = False):
        super().__init__()
        self.file_paths = [str(p) for p in file_paths]
        self.shuffle_files = shuffle_files
        self.seed = seed
        self.epoch = 0
        self.parser_verbose = parser_verbose

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        from .mjson_parser import MJSONRecordParser
        parser = MJSONRecordParser(verbose=self.parser_verbose)
        file_paths = list(self.file_paths)

        if self.shuffle_files:
            rng = np.random.default_rng(self.seed + self.epoch)
            rng.shuffle(file_paths)

        worker = get_worker_info()
        if worker is not None:
            file_paths = file_paths[worker.id::worker.num_workers]

        for fp in file_paths:
            for token_ids, token_types, behavior_ids, \
                    action_mask, label in parser.parse_file_token_samples(fp):
                yield (
                    np.array(token_ids, dtype=np.int64),
                    np.array(token_types, dtype=np.int64),
                    np.array(behavior_ids, dtype=np.int64),
                    action_mask,
                    np.int64(label),
                )

    def __len__(self) -> int:
        return len(self.file_paths)


class MJSONPublicPrivateTokenIterableDataset(IterableDataset):
    """按文件流式解析 MJSON，输出 public + private token 序列样本（Teacher 训练用）。

    每个样本为 8-tuple:
    (token_ids, token_types, behavior_ids, action_mask, label,
     priv_token_ids, priv_token_types, priv_behavior_ids)
    配合 collate_transformer_batch 使用。
    """

    def __init__(self,
                 file_paths: Sequence[str],
                 shuffle_files: bool = False,
                 seed: int = 42,
                 parser_verbose: bool = False):
        super().__init__()
        self.file_paths = [str(p) for p in file_paths]
        self.shuffle_files = shuffle_files
        self.seed = seed
        self.epoch = 0
        self.parser_verbose = parser_verbose

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        from .mjson_parser import MJSONRecordParser
        parser = MJSONRecordParser(verbose=self.parser_verbose)
        file_paths = list(self.file_paths)

        if self.shuffle_files:
            rng = np.random.default_rng(self.seed + self.epoch)
            rng.shuffle(file_paths)

        worker = get_worker_info()
        if worker is not None:
            file_paths = file_paths[worker.id::worker.num_workers]

        for fp in file_paths:
            for (token_ids, token_types, behavior_ids,
                 action_mask, label,
                 priv_ids, priv_types, priv_bids) in \
                    parser.parse_file_public_private_token_samples(fp):
                yield (
                    np.array(token_ids, dtype=np.int64),
                    np.array(token_types, dtype=np.int64),
                    np.array(behavior_ids, dtype=np.int64),
                    action_mask,
                    np.int64(label),
                    np.array(priv_ids, dtype=np.int64),
                    np.array(priv_types, dtype=np.int64),
                    np.array(priv_bids, dtype=np.int64),
                )

    def __len__(self) -> int:
        return len(self.file_paths)


class OracleTrajectoryIterableDataset(IterableDataset):
    """流式读取 selfplay_recorder 输出的 Oracle 轨迹 JSONL。

    每个样本为 8-tuple（public + private token fields），
    配合 collate_transformer_batch 使用。
    """

    def __init__(self,
                 file_paths: Sequence[str],
                 shuffle_files: bool = False,
                 seed: int = 42):
        super().__init__()
        self.file_paths = [str(p) for p in file_paths]
        self.shuffle_files = shuffle_files
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        from .record_parser import OracleTrajectoryJSONLParser
        parser = OracleTrajectoryJSONLParser()
        file_paths = list(self.file_paths)

        if self.shuffle_files:
            rng = np.random.default_rng(self.seed + self.epoch)
            rng.shuffle(file_paths)

        worker = get_worker_info()
        if worker is not None:
            file_paths = file_paths[worker.id::worker.num_workers]

        for fp in file_paths:
            yield from parser.parse_file(fp)

    def __len__(self) -> int:
        return len(self.file_paths)


class TensorShardBatchDataset(IterableDataset):
    """Stream pre-batched tensor shards produced by training.mjson_cache."""

    def __init__(self,
                 shard_paths: Sequence[str],
                 batch_size: int,
                 shuffle_shards: bool = False,
                 shuffle_samples: bool = False,
                 drop_last: bool = False,
                 seed: int = 42,
                 total_samples: Optional[int] = None):
        super().__init__()
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.shard_paths = [str(p) for p in shard_paths]
        self.batch_size = batch_size
        self.shuffle_shards = shuffle_shards
        self.shuffle_samples = shuffle_samples
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        self.total_samples = total_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        paths = list(self.shard_paths)
        rng = np.random.default_rng(self.seed + self.epoch)
        if self.shuffle_shards:
            rng.shuffle(paths)

        worker = get_worker_info()
        if worker is not None:
            paths = paths[worker.id::worker.num_workers]

        for path in paths:
            with np.load(path, allow_pickle=False) as shard:
                states = shard["states"]
                masks = shard["masks"]
                labels = shard["labels"]
                n = int(labels.shape[0])
                if n == 0:
                    continue

                order = None
                if self.shuffle_samples:
                    order = rng.permutation(n)

                for start in range(0, n, self.batch_size):
                    end = min(start + self.batch_size, n)
                    if self.drop_last and end - start < self.batch_size:
                        continue
                    if order is None:
                        index = slice(start, end)
                    else:
                        index = order[start:end]

                    batch_states = states[index].astype(np.float32, copy=False)
                    batch_masks = masks[index].astype(np.float32, copy=False)
                    batch_labels = labels[index].astype(np.int64, copy=False)
                    yield batch_states, batch_masks, batch_labels

    def __len__(self) -> int:
        if self.total_samples is None:
            return len(self.shard_paths)
        if self.drop_last:
            return self.total_samples // self.batch_size
        return math.ceil(self.total_samples / self.batch_size)


class TokenShardBatchDataset(IterableDataset):
    """Stream pre-tokenized batches from mjson_token_cache shards.

    Each shard contains pre-padded token sequences at a fixed max_len.
    The dataset yields pre-built batch dicts directly — use batch_size=None
    in DataLoader (prebatched mode).

    Args:
        shard_paths: List of .npz shard file paths
        batch_size: Number of samples per batch
        shuffle_shards: Shuffle shard order each epoch
        shuffle_samples: Shuffle sample order within each shard
        drop_last: Drop incomplete final batch
        seed: Random seed
    """

    def __init__(self,
                 shard_paths: Sequence[str],
                 batch_size: int,
                 shuffle_shards: bool = False,
                 shuffle_samples: bool = False,
                 drop_last: bool = False,
                 seed: int = 42):
        super().__init__()
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.shard_paths = [str(p) for p in shard_paths]
        self.batch_size = batch_size
        self.shuffle_shards = shuffle_shards
        self.shuffle_samples = shuffle_samples
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        paths = list(self.shard_paths)
        rng = np.random.default_rng(self.seed + self.epoch)
        if self.shuffle_shards:
            rng.shuffle(paths)

        worker = get_worker_info()
        if worker is not None:
            paths = paths[worker.id::worker.num_workers]

        for path in paths:
            with np.load(path, allow_pickle=False) as shard:
                token_ids = shard["token_ids"]
                token_types = shard["token_types"]
                behavior_ids = shard["behavior_ids"]
                attention_mask = shard["attention_mask"]
                action_mask = shard["action_mask"]
                labels = shard["labels"]
                n = int(labels.shape[0])
                if n == 0:
                    continue

                order = None
                if self.shuffle_samples:
                    order = rng.permutation(n)

                for start in range(0, n, self.batch_size):
                    end = min(start + self.batch_size, n)
                    if self.drop_last and end - start < self.batch_size:
                        continue
                    if order is None:
                        index = slice(start, end)
                    else:
                        index = order[start:end]

                    batch = {
                        'token_ids': torch.from_numpy(
                            token_ids[index].astype(np.int64, copy=False)),
                        'token_types': torch.from_numpy(
                            token_types[index].astype(np.int64, copy=False)),
                        'behavior_ids': torch.from_numpy(
                            behavior_ids[index].astype(np.int64, copy=False)),
                        'attention_mask': torch.from_numpy(
                            attention_mask[index].copy()),
                        'action_mask': torch.from_numpy(
                            action_mask[index].astype(np.float32, copy=False)),
                        'labels': torch.from_numpy(
                            labels[index].astype(np.int64, copy=False)),
                    }
                    yield batch

    def __len__(self) -> int:
        # Approximate
        total = 0
        for path in self.shard_paths:
            try:
                with np.load(path, allow_pickle=False) as shard:
                    total += int(shard["labels"].shape[0])
            except Exception:
                continue
        if self.drop_last:
            return total // self.batch_size
        return math.ceil(total / self.batch_size)
# 中文注释：PyTorch 数据集封装，连接牌谱解析结果和监督学习 DataLoader。


# ── Transformer Token Dataset ─────────────────────────────────────────────────

class TokenDataset(Dataset):
    """Transformer 训练用的 Token 序列数据集。

    每个样本包含已 tokenize 的序列和对应的标签。

    Args:
        token_ids: List[List[int]] 变长 token ID 序列
        token_types: List[List[int]] 变长 token 类型序列
        behavior_ids: List[List[int]] 变长 behavior ID 序列
        action_mask: np.ndarray (N, 77) 合法动作掩码
        labels: np.ndarray (N,) 目标动作索引
        oracle_shanten: Optional[np.ndarray] (N,) 向听标签
        oracle_ukeire_mask: Optional[np.ndarray] (N, 34) 进张掩码标签
    """

    def __init__(self,
                 token_ids: List[List[int]],
                 token_types: List[List[int]],
                 behavior_ids: List[List[int]],
                 action_mask: np.ndarray,
                 labels: np.ndarray,
                 oracle_shanten: Optional[np.ndarray] = None,
                 oracle_ukeire_mask: Optional[np.ndarray] = None):
        self.token_ids = token_ids
        self.token_types = token_types
        self.behavior_ids = behavior_ids
        self.action_mask = torch.FloatTensor(
            action_mask.astype(np.float32))
        self.labels = torch.LongTensor(labels.astype(np.int64))
        self.oracle_shanten = (torch.LongTensor(oracle_shanten.astype(np.int64))
                               if oracle_shanten is not None else None)
        self.oracle_ukeire_mask = (torch.FloatTensor(
            oracle_ukeire_mask.astype(np.float32))
            if oracle_ukeire_mask is not None else None)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        item = (
            torch.LongTensor(self.token_ids[idx]),
            torch.LongTensor(self.token_types[idx]),
            torch.LongTensor(self.behavior_ids[idx]),
            self.action_mask[idx],
            self.labels[idx],
        )
        if self.oracle_shanten is not None:
            item += (self.oracle_shanten[idx],)
        if self.oracle_ukeire_mask is not None:
            item += (self.oracle_ukeire_mask[idx],)
        return item


def collate_transformer_batch(
        batch: List[Tuple[torch.Tensor, ...]]) -> Dict[str, torch.Tensor]:
    """将变长 Token 序列列表组装为 padded batch。

    序列用 PAD=0 填充，attention_mask 中 True=padding。

    支持以下 tuple 长度：
    - 5: (token_ids, token_types, behavior_ids, action_mask, label)
    - 6: + oracle_shanten
    - 7: + oracle_ukeire_mask
    - 8: + priv_token_ids, priv_token_types, priv_behavior_ids  (teacher mode)

    Returns:
        dict with keys: token_ids, token_types, behavior_ids,
            attention_mask, action_mask, labels,
            and optionally oracle_shanten, oracle_ukeire_mask,
            private_token_ids, private_token_types, private_behavior_ids,
            private_attention_mask
    """
    tuple_len = len(batch[0])
    has_private = tuple_len >= 8
    has_reward = tuple_len >= 9
    has_shanten = tuple_len >= 6 and not has_private
    has_ukeire = tuple_len >= 7 and not has_private

    B = len(batch)
    max_len = max(item[0].shape[0] for item in batch)

    token_ids = torch.zeros(B, max_len, dtype=torch.long)
    token_types = torch.zeros(B, max_len, dtype=torch.long)
    behavior_ids = torch.zeros(B, max_len, dtype=torch.long)
    attention_mask = torch.ones(B, max_len, dtype=torch.bool)

    action_mask = torch.stack([torch.as_tensor(item[3]) for item in batch])
    labels = torch.stack([torch.as_tensor(item[4]) for item in batch])

    for i, item in enumerate(batch):
        L = item[0].shape[0]
        token_ids[i, :L] = torch.as_tensor(item[0])
        token_types[i, :L] = torch.as_tensor(item[1])
        behavior_ids[i, :L] = torch.as_tensor(item[2])
        attention_mask[i, :L] = False

    result = {
        'token_ids': token_ids,
        'token_types': token_types,
        'behavior_ids': behavior_ids,
        'attention_mask': attention_mask,
        'action_mask': action_mask,
        'labels': labels,
    }

    if has_shanten:
        result['oracle_shanten'] = torch.stack(
            [torch.as_tensor(item[5]) for item in batch])
    if has_ukeire:
        idx = 6 if has_shanten else 5
        result['oracle_ukeire_mask'] = torch.stack(
            [torch.as_tensor(item[idx]) for item in batch])

    # Private tokens for teacher mode (8-tuple)
    if has_private:
        max_priv_len = max(item[5].shape[0] for item in batch)
        priv_ids = torch.zeros(B, max_priv_len, dtype=torch.long)
        priv_types = torch.zeros(B, max_priv_len, dtype=torch.long)
        priv_bids = torch.zeros(B, max_priv_len, dtype=torch.long)
        priv_attn = torch.ones(B, max_priv_len, dtype=torch.bool)
        for i, item in enumerate(batch):
            L = item[5].shape[0]
            priv_ids[i, :L] = torch.as_tensor(item[5])
            priv_types[i, :L] = torch.as_tensor(item[6])
            priv_bids[i, :L] = torch.as_tensor(item[7])
            priv_attn[i, :L] = False
        result['private_token_ids'] = priv_ids
        result['private_token_types'] = priv_types
        result['private_behavior_ids'] = priv_bids
        result['private_attention_mask'] = priv_attn

    if has_reward:
        result['rewards'] = torch.stack(
            [torch.as_tensor(item[8]) for item in batch])

    return result
