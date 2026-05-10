"""保存数据 split manifest，确保跨机器训练使用相同的 train/val/test 划分。

用法:
    python scripts/save_manifest.py --root dataset/datasets_years --years 2021-2026
    python scripts/save_manifest.py --root dataset/datasets_years --years 2021-2026 --output my_manifest.json

输出 JSON 格式:
    {
        "train_files": [...],
        "val_files": [...],
        "test_files": [...],
        "seed": 42,
        "train_ratio": 0.8,
        "val_ratio": 0.1,
        "test_ratio": 0.1,
        "years": [2021, 2022, ...],
        "total_files": 899217,
        "created": "2026-05-10T15:30:00"
    }
"""

import argparse
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from training.sl_pretrain import collect_mjson_files, split_mjson_files


def main():
    parser = argparse.ArgumentParser(
        description='Save data split manifest for reproducible training')
    parser.add_argument('--root', type=str, required=True,
                        help='Root directory containing year subdirectories')
    parser.add_argument('--years', type=str, default=None,
                        help='Comma-separated years or range (e.g. "2021,2022,2023" or "2021-2026")')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON path (default: <root>/manifest_<years>_seed<N>.json)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--max_files', type=int, default=0,
                        help='Max total files (0 = all)')
    args = parser.parse_args()

    # 解析年份
    years = None
    if args.years:
        if '-' in args.years and ',' not in args.years:
            parts = args.years.split('-')
            years = list(range(int(parts[0]), int(parts[1]) + 1))
        else:
            years = [int(y) for y in args.years.split(',')]

    # 收集文件
    all_files = collect_mjson_files(args.root, max_files=args.max_files, years=years)
    print(f"Found {len(all_files)} mjson files")

    # 划分
    train_files, val_files, test_files = split_mjson_files(
        all_files,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    print(f"Split: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")

    # 构建 manifest
    manifest = {
        'train_files': train_files,
        'val_files': val_files,
        'test_files': test_files,
        'seed': args.seed,
        'train_ratio': args.train_ratio,
        'val_ratio': args.val_ratio,
        'test_ratio': args.test_ratio,
        'years': years,
        'total_files': len(all_files),
        'created': datetime.now().isoformat(),
    }

    # 输出路径
    if args.output:
        out_path = args.output
    else:
        year_str = args.years or 'all'
        out_path = os.path.join(args.root, f'manifest_{year_str}_seed{args.seed}.json')

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Manifest saved to {out_path}")
    print(f"  Use with: python -m training.sl_pretrain --manifest {out_path} ...")


if __name__ == '__main__':
    main()
