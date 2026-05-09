"""PyTorch Dataset —— 包装训练样本供 DataLoader 使用。"""

import math
from typing import Callable, List, Optional, Sequence

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
# 中文注释：PyTorch 数据集封装，连接牌谱解析结果和监督学习 DataLoader。
