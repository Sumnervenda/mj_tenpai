import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import TensorShardBatchDataset
from training.mjson_cache import split_mjson_files


def test_tensor_shard_batch_dataset_streams_prebatched_arrays(tmp_path):
    shard_dir = tmp_path / "train"
    shard_dir.mkdir()
    shard_path = shard_dir / "shard_000000.npz"
    np.savez(
        shard_path,
        states=np.arange(5 * 354, dtype=np.float16).reshape(5, 354),
        masks=np.ones((5, 77), dtype=np.uint8),
        labels=np.arange(5, dtype=np.uint8),
    )

    dataset = TensorShardBatchDataset(
        [str(shard_path)],
        batch_size=2,
        drop_last=False,
        total_samples=5,
    )
    loader = DataLoader(dataset, batch_size=None, num_workers=0)
    batches = list(loader)

    assert len(dataset) == 3
    assert len(batches) == 3
    states, masks, labels = batches[0]
    assert states.shape == (2, 354)
    assert masks.shape == (2, 77)
    assert labels.tolist() == [0, 1]
    assert states.dtype == torch.float32
    assert masks.dtype == torch.float32
    assert labels.dtype == torch.int64


def test_split_mjson_files_keeps_all_splits_non_empty():
    files = [f"game_{i}.mjson" for i in range(30)]

    train, val, test = split_mjson_files(
        files,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=123,
    )

    assert len(train) == 24
    assert len(val) == 3
    assert len(test) == 3
    assert sorted(train + val + test) == sorted(files)
# 中文注释：验证 TensorShard 分片数据集的分批加载和 MJSON 文件拆分逻辑。

