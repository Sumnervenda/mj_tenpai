"""PyTorch Dataset —— 包装训练样本供 DataLoader 使用。"""

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
                    torch.tensor(sample.state_tensor, dtype=torch.float32),
                    torch.tensor(sample.action_mask, dtype=torch.float32),
                    torch.tensor(sample.chosen_action, dtype=torch.long),
                )

    def __len__(self) -> int:
        return len(self.file_paths)

