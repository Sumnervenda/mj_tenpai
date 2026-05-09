from pathlib import Path

from training.sl_pretrain import (
    normalize_metric_row,
    read_local_metric_rows,
    resolve_wandb_run,
)


class FakeRun:
    def __init__(self, run_id, name, display_name=None):
        self.id = run_id
        self.name = name
        self.display_name = display_name


class FakeApi:
    def __init__(self, runs):
        self._runs = runs
        self.exact_paths = []

    def run(self, path):
        self.exact_paths.append(path)
        return FakeRun('exact', path)

    def runs(self, path, order=None, per_page=None):
        self.path = path
        self.order = order
        self.per_page = per_page
        return list(self._runs)


def test_metric_history_round_trip():
    checkpoint_dir = Path('tests/fixtures')

    rows = read_local_metric_rows(checkpoint_dir)

    assert len(rows) == 1
    assert rows[0]['phase'] == 'train_val'
    assert rows[0]['train/loss'] == 1.25
    assert rows[0]['train/accuracy'] == 0.5


def test_legacy_history_keys_are_queryable():
    row = normalize_metric_row({
        'epoch': 3,
        'train_loss': 0.9,
        'train_acc': 0.4,
        'val_loss': 1.1,
        'val_acc': 0.3,
    })

    assert row['train/loss'] == 0.9
    assert row['train/accuracy'] == 0.4
    assert row['val/loss'] == 1.1
    assert row['val/accuracy'] == 0.3


def test_resolve_wandb_run_matches_name_without_network():
    api = FakeApi([
        FakeRun('abc123', 'first'),
        FakeRun('def456', 'sl_pretrain_2023-2026', 'display-name'),
    ])

    run = resolve_wandb_run(
        api,
        project='mahjong-dl',
        entity='team',
        run_ref='sl_pretrain_2023-2026',
    )

    assert run.id == 'def456'
    assert api.path == 'team/mahjong-dl'
# 中文注释：验证训练指标历史文件、旧字段兼容和 W&B run 查询定位逻辑。
