"""短基准脚本 — 租卡前快速评估 GPU 吞吐量。

用法:
    python -m training.benchmark --checkpoint checkpoints/transformer_2021-2026/sl_best.pt
    python -m training.benchmark --checkpoint sl_best.pt --batches 500
    python -m training.benchmark --checkpoint sl_best.pt --sizes 128,256,512

输出:
    每个配置的 batch/s, samples/s, VRAM, 预计 epoch 时长。
    用于判断 4090 / A100 / H100 哪个真实性价比最高。
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.model_io import load_checkpoint_metadata, infer_transformer_config_from_state_dict
from models.transformer_policy_value import TransformerPolicyValueNet


def run_benchmark(model, device, batch_size, num_batches, use_amp=True,
                  use_compile=False, max_len=256, d_model=256):
    """运行短基准测试，返回性能指标。"""
    model.eval()
    scaler = GradScaler('cuda', enabled=use_amp and device.startswith('cuda'))

    # 生成随机输入（减掉 concept tokens，避免超过 backbone max_len）
    seq_len = max_len - 10  # 10 public concept tokens
    token_ids = torch.randint(0, 192, (batch_size, seq_len), device=device)
    token_types = torch.randint(0, 6, (batch_size, seq_len), device=device)
    behavior_ids = torch.randint(0, 64, (batch_size, seq_len), device=device)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    action_mask = torch.ones(batch_size, 77, dtype=torch.float32, device=device)

    if use_compile and hasattr(torch, 'compile'):
        model = torch.compile(model, mode='reduce-overhead')

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            with autocast('cuda', enabled=use_amp):
                model(token_ids, token_types, behavior_ids,
                      attention_mask, action_mask)
    if device.startswith('cuda'):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # 正式测试
    t_start = time.time()
    with torch.no_grad():
        for i in range(num_batches):
            with autocast('cuda', enabled=use_amp):
                outputs = model(token_ids, token_types, behavior_ids,
                                attention_mask, action_mask)
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    t_end = time.time()

    elapsed = t_end - t_start
    total_samples = num_batches * batch_size
    batch_s = num_batches / elapsed
    samples_s = total_samples / elapsed
    max_mem_gb = (torch.cuda.max_memory_allocated() / 1024**3
                  if device.startswith('cuda') else 0)

    # 估算 epoch 时长（基于 900K 文件 × 670 样本/文件 = ~6 亿样本）
    # 训练比推理慢约 2-3x（有 backward），用 2.5x 估算
    est_train_samples_per_epoch = 420_000_000
    est_epoch_hours = est_train_samples_per_epoch / (samples_s / 2.5) / 3600

    return {
        'batch_size': batch_size,
        'num_batches': num_batches,
        'elapsed_sec': round(elapsed, 2),
        'batch_s': round(batch_s, 2),
        'samples_s': round(samples_s, 0),
        'max_vram_gb': round(max_mem_gb, 2),
        'est_epoch_hours': round(est_epoch_hours, 1),
        'use_amp': use_amp,
        'use_compile': use_compile,
        'seq_len': seq_len,
    }


def main():
    parser = argparse.ArgumentParser(description='GPU benchmark for Transformer training')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (.pt). '
                             'If omitted, uses randomly initialized model.')
    parser.add_argument('--batches', type=int, default=1000,
                        help='Number of batches per config (default: 1000)')
    parser.add_argument('--sizes', type=str, default='128,256,512',
                        help='Comma-separated batch sizes to test (default: 128,256,512)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (default: cuda)')
    parser.add_argument('--no_compile', action='store_true',
                        help='Skip torch.compile test')
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_concept', type=int, default=10)
    parser.add_argument('--max_len', type=int, default=256)
    args = parser.parse_args()

    # 加载或使用默认模型配置
    if args.checkpoint:
        meta = load_checkpoint_metadata(args.checkpoint)
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        sd = ckpt['model_state_dict']
        cfg = infer_transformer_config_from_state_dict(sd, meta)
        d_model = cfg.get('d_model', 256)
        n_layers = cfg.get('n_layers', 6)
        n_heads = cfg.get('n_heads', 8)
        n_concept = cfg.get('n_concept', 10)
        max_len = cfg.get('max_len', 256)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        sd = None
        d_model = args.d_model
        n_layers = args.n_layers
        n_heads = args.n_heads
        n_concept = args.n_concept
        max_len = args.max_len
        print("No checkpoint provided, using randomly initialized model")

    print(f"Model: d_model={d_model}, n_layers={n_layers}, n_heads={n_heads}, "
          f"n_concept={n_concept}, max_len={max_len}")

    # GPU 信息
    if args.device.startswith('cuda') and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB), CUDA {torch.version.cuda}")
    else:
        print("GPU: CPU only")

    print(f"PyTorch: {torch.__version__}")
    print()

    batch_sizes = [int(x) for x in args.sizes.split(',')]
    results = []

    for bs in batch_sizes:
        print(f"--- batch_size={bs}, batches={args.batches} ---", flush=True)
        # 为每个配置重新加载模型（避免 compile 污染）
        model = TransformerPolicyValueNet(
            d_model=d_model, n_layers=n_layers, n_heads=n_heads,
            n_concept=n_concept, max_len=max_len)
        if sd is not None:
            model.load_state_dict(sd)
        model = model.to(args.device)

        try:
            r = run_benchmark(model, args.device, bs, args.batches,
                              use_amp=True, use_compile=False, max_len=max_len)
            results.append(r)
            print(f"  {r['batch_s']:.1f} batch/s | {r['samples_s']:.0f} samples/s | "
                  f"VRAM {r['max_vram_gb']:.1f} GB | "
                  f"ETA epoch ~{r['est_epoch_hours']:.1f}h")
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"  OOM at batch_size={bs}, skipping")
                torch.cuda.empty_cache()
            else:
                raise

        # 测试 compile（仅 Linux + Triton 可用）
        if not args.no_compile:
            _compile_ok = True
            _is_windows = sys.platform == 'win32'
            if _is_windows:
                print(f"  +compile: skipped (Windows does not support Triton)")
                _compile_ok = False
            else:
                # 检测 Triton 是否安装
                try:
                    import triton  # noqa: F401
                except ImportError:
                    print(f"  +compile: skipped (Triton not installed, run: pip install triton)")
                    _compile_ok = False

            if _compile_ok:
                model2 = TransformerPolicyValueNet(
                    d_model=d_model, n_layers=n_layers, n_heads=n_heads,
                    n_concept=n_concept, max_len=max_len)
                if sd is not None:
                    model2.load_state_dict(sd)
                model2 = model2.to(args.device)
                try:
                    r2 = run_benchmark(model2, args.device, bs, min(args.batches, 200),
                                       use_amp=True, use_compile=True, max_len=max_len)
                    r2['config'] = f'bs{bs}_compile'
                    results.append(r2)
                    print(f"  +compile: {r2['batch_s']:.1f} batch/s | "
                          f"{r2['samples_s']:.0f} samples/s | "
                          f"VRAM {r2['max_vram_gb']:.1f} GB")
                except RuntimeError as e:
                    if 'out of memory' in str(e).lower():
                        print(f"  +compile: OOM at batch_size={bs}")
                        torch.cuda.empty_cache()
                    else:
                        raise
        print()

    # 汇总
    print("=" * 70)
    print(f"{'Config':<25} {'batch/s':>10} {'samples/s':>12} {'VRAM GB':>10} {'ETA epoch':>12}")
    print("-" * 70)
    for r in results:
        cfg_name = f"bs{r['batch_size']}"
        if r.get('use_compile'):
            cfg_name += "+compile"
        print(f"{cfg_name:<25} {r['batch_s']:>10.1f} {r['samples_s']:>12.0f} "
              f"{r['max_vram_gb']:>10.2f} {r['est_epoch_hours']:>10.1f}h")
    print("=" * 70)

    # 决策建议
    print("\n--- Decision Thresholds ---")
    print(">= 30 batch/s (inference): Excellent, training ~12+ batch/s")
    print("20-30 batch/s: Good, training ~8-12 batch/s")
    print("< 20 batch/s: Check config or try smaller batch_size")
    print("\nNote: Training throughput = inference / 2.5 (approx)")
    print(f"      Estimated samples_per_epoch = 420M (900K files x 670 samples)")

    # 保存结果
    _base_dir = os.path.dirname(args.checkpoint) if args.checkpoint else '.'
    out_path = os.path.join(_base_dir, 'benchmark_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
