import json

import numpy as np

from data.dataset import TokenMmapShardBatchDataset
from training.mjson_token_cache import (
    assert_token_mmap_cache_ready,
    convert_token_cache_to_mmap,
    token_mmap_shards_for_split,
)


def _write_npz_token_cache(root):
    lengths = np.array([3, 5, 2], dtype=np.int16)
    token_ids = np.array([
        [1, 2, 3, 0, 0],
        [4, 5, 6, 7, 8],
        [9, 10, 0, 0, 0],
    ], dtype=np.int16)
    token_types = np.array([
        [0, 0, 1, 0, 0],
        [2, 2, 2, 5, 5],
        [3, 3, 0, 0, 0],
    ], dtype=np.int8)
    behavior_ids = np.array([
        [0, 1, 2, 0, 0],
        [3, 4, 5, 6, 7],
        [8, 9, 0, 0, 0],
    ], dtype=np.int16)
    attention_mask = np.ones((3, 5), dtype=bool)
    for row, length in enumerate(lengths):
        attention_mask[row, :length] = False
    action_mask = np.zeros((3, 77), dtype=np.float16)
    action_mask[0, [1, 7, 76]] = 1
    action_mask[1, [2, 3]] = 1
    action_mask[2, [0, 76]] = 1
    labels = np.array([1, 2, 0], dtype=np.int16)

    manifest = {
        "version": 1,
        "format": "token",
        "created_at": "test",
        "source_root": "test",
        "years": None,
        "max_files": 0,
        "split_seed": 42,
        "train_ratio": 0.8,
        "val_ratio": 0.1,
        "test_ratio": 0.1,
        "vocab_size": 128,
        "action_dim": 77,
        "shard_size": 3,
        "splits": {},
    }
    for split in ("train", "val", "test"):
        split_dir = root / split
        split_dir.mkdir(parents=True)
        shard_path = split_dir / "shard_b0_000000.npz"
        np.savez_compressed(
            shard_path,
            token_ids=token_ids,
            token_types=token_types,
            behavior_ids=behavior_ids,
            attention_mask=attention_mask,
            action_mask=action_mask,
            labels=labels,
            lengths=lengths,
        )
        manifest["splits"][split] = {
            "num_files": 1,
            "num_samples": 3,
            "num_shards": 1,
            "shards": [{
                "path": f"{split}/shard_b0_000000.npz",
                "samples": 3,
                "bucket": 0,
                "max_len": 5,
            }],
            "num_errors": 0,
            "errors": [],
        }
    (root / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8")


def test_convert_token_cache_to_mmap_and_read_batch(tmp_path):
    source = tmp_path / "token_cache"
    target = tmp_path / "token_mmap"
    source.mkdir()
    _write_npz_token_cache(source)

    manifest = convert_token_cache_to_mmap(
        str(source), str(target), shard_size=2)
    assert manifest["format"] == "token_mmap"
    assert manifest["splits"]["train"]["num_samples"] == 3
    assert manifest["splits"]["train"]["num_shards"] == 2

    loaded = assert_token_mmap_cache_ready(str(target))
    dataset = TokenMmapShardBatchDataset(
        str(target),
        token_mmap_shards_for_split(str(target), loaded, "train"),
        batch_size=2,
        shuffle_shards=False,
        shuffle_samples=False,
        drop_last=False,
        seed=42,
    )
    batch = next(iter(dataset))

    assert batch["token_ids"].shape == (2, 5)
    assert batch["token_ids"][0].tolist() == [1, 2, 3, 0, 0]
    assert batch["token_ids"][1].tolist() == [4, 5, 6, 7, 8]
    assert batch["attention_mask"][0].tolist() == [False, False, False, True, True]
    assert batch["action_mask"][0, 1].item() == 1.0
    assert batch["action_mask"][0, 7].item() == 1.0
    assert batch["action_mask"][0, 76].item() == 1.0
    assert batch["labels"].tolist() == [1, 2]
