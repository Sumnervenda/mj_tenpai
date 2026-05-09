"""迭代式 PPO 强化学习：Trainee 对抗上一版自身的 Frozen 快照。

每版 trainee 使用 TurtleShaper 训练，frozen 对手使用上一版已训练的 trainee 模型。
通过不断对抗"过去的自己"实现持续自我提升。

用法:
    python -m training.iterative_rl --base_model checkpoints/sl_best.pt \
        --personality configs/turtle.yaml --num_versions 5
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models import MahjongPolicyValueNet, load_checkpoint, save_checkpoint
from .selfplay_env import SelfPlayEnv
from .ppo_agent import PPOAgent
from .reward_shaper import load_shaper_from_config
from .rl_selfplay import load_training_config, run_eval


def parse_args():
    parser = argparse.ArgumentParser(
        description='Iterative RL Self-Play with versioned opponents')
    parser.add_argument('--base_model', type=str, required=True,
                        help='Path to SL pretrained base model (v0 starting point)')
    parser.add_argument('--personality', type=str,
                        default='configs/turtle.yaml',
                        help='Path to personality YAML config')
    parser.add_argument('--num_versions', type=int, default=5,
                        help='Number of iterative training versions')
    parser.add_argument('--total_steps', type=int, default=200000,
                        help='Total environment steps per version')
    parser.add_argument('--rollout_games', type=int, default=32,
                        help='Games per rollout collection')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/',
                        help='Directory for saving checkpoints')
    parser.add_argument('--eval_every', type=int, default=10,
                        help='Run evaluation every N iterations')
    parser.add_argument('--eval_games', type=int, default=20,
                        help='Number of evaluation games')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device: cuda or cpu')
    parser.add_argument('--no_amp', action='store_true',
                        help='Disable automatic mixed precision')
    parser.add_argument('--compile', action='store_true',
                        help='Use torch.compile (PyTorch 2.0+)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='mahjong-dl',
                        help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='W&B entity/team name')
    parser.add_argument('--resume_from', type=int, default=1,
                        help='Resume training from this version (1-indexed)')
    return parser.parse_args()


def _new_model(device: str) -> MahjongPolicyValueNet:
    m = MahjongPolicyValueNet()
    return m.to(device)


def _load_frozen(path: str, device: str) -> MahjongPolicyValueNet:
    m = _new_model(device)
    _, _ = load_checkpoint(m, path, device=device)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def train_one_version(
    version: int,
    trainee_path: str,
    baseline_path: str,
    personality_cfg: str,
    train_cfg: dict,
    args,
) -> tuple[str, dict]:
    """训练一个版本。

    Args:
        version: 版本号 (1-indexed)
        trainee_path: trainee 初始权重路径
        baseline_path: frozen 对手权重路径
        personality_cfg: 人格 YAML 路径
        train_cfg: 训练超参字典
        args: 命令行参数

    Returns:
        (output_path, final_eval_metrics)
    """
    device = args.device

    # ── 加载 trainee ──
    trainee = _new_model(device)
    epoch, meta = load_checkpoint(trainee, trainee_path, device=device)
    print(f"\n{'='*60}")
    print(f"Version {version} — Trainee: {trainee_path} (epoch {epoch})")
    if meta:
        print(f"  Metadata: {meta.get('val_acc', meta.get('eval_avg_rank', 'N/A'))}")

    if args.compile and hasattr(torch, 'compile'):
        trainee = torch.compile(trainee, mode='reduce-overhead')
        print("  Model compiled with torch.compile")

    # ── 加载 frozen baseline ──
    baseline = _load_frozen(baseline_path, device)
    print(f"  Baseline: {baseline_path}")

    # ── 奖励塑形器 ──
    reward_shaper = None
    try:
        reward_shaper = load_shaper_from_config(personality_cfg)
        print(f"  Reward shaper: {type(reward_shaper).__name__}")
    except Exception as e:
        print(f"  Warning: Could not load reward shaper: {e}")

    # ── PPO Agent ──
    use_cuda = device == 'cuda'
    agent = PPOAgent(
        model=trainee,
        device=device,
        lr=train_cfg.get('lr', 3e-4),
        clip_epsilon=train_cfg.get('clip_epsilon', 0.2),
        gamma=train_cfg.get('gamma', 0.99),
        gae_lambda=train_cfg.get('gae_lambda', 0.95),
        entropy_coef=train_cfg.get('entropy_coef', 0.01),
        value_loss_coef=train_cfg.get('value_loss_coef', 0.5),
        max_grad_norm=train_cfg.get('max_grad_norm', 1.0),
        use_amp=use_cuda and not args.no_amp,
        kl_coef=train_cfg.get('kl_coef', 0.01),
    )

    # ── 自对弈环境 ──
    env = SelfPlayEnv(trainee, device=device, deterministic=False,
                      reward_shaper=reward_shaper,
                      trainee_idx=0, baseline_model=baseline)

    trainee_idx = 0
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    total_env_steps = 0
    iteration = 0
    best_avg_rank = float('inf')
    history = []

    print(f"\n  Total steps target: {args.total_steps}")
    print(f"  Rollout games/iter: {args.rollout_games}")
    print(f"  PPO epochs: {train_cfg.get('ppo_epochs', 10)}")
    print(f"  KL coef: {train_cfg.get('kl_coef', 0.01)}")
    print(f"\n  {'Iter':>4} {'Steps':>8} {'PolicyLoss':>10} "
          f"{'ValueLoss':>8} {'Entropy':>8} {'AvgRank':>8} {'4th%':>6} {'Time':>7}")

    # ── wandb ──
    wandb_run = None
    if args.wandb:
        import wandb
        run_name = f"iter_v{version}_turtle"
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            entity=args.wandb_entity,
            config={
                'version': version,
                'trainee_path': trainee_path,
                'baseline_path': baseline_path,
                'personality': personality_cfg,
                'total_steps': args.total_steps,
                'rollout_games': args.rollout_games,
                'model_params': trainee.count_parameters(),
                'device': device,
                'amp': use_cuda and not args.no_amp,
                **train_cfg,
            },
            reinit=True,
        )

    # ── 训练循环 ──
    while total_env_steps < args.total_steps:
        t0 = time.time()
        iteration += 1

        agent.clear_buffer()
        for g in range(args.rollout_games):
            seed = args.seed + version * 100000 + iteration * 1000 + g
            traj = env.run_game(seed=seed)
            agent.collect_trajectories([traj])
            total_env_steps += traj.total_steps

        metrics = agent.update(
            ppo_epochs=train_cfg.get('ppo_epochs', 10),
            mini_batch_size=train_cfg.get('mini_batch_size', 1024),
        )

        elapsed = time.time() - t0

        eval_metrics = {}
        if iteration % args.eval_every == 0:
            eval_metrics = run_eval(trainee, args.eval_games, device,
                                    baseline_model=baseline,
                                    trainee_idx=trainee_idx)

        avg_rank = eval_metrics.get('avg_rank', float('inf'))
        fourth_rate = eval_metrics.get('fourth_rate', 0)

        print(f"  {iteration:>4} {total_env_steps:>8} "
              f"{metrics['policy_loss']:>10.4f} "
              f"{metrics['value_loss']:>8.4f} "
              f"{metrics['entropy']:>8.4f} "
              f"{avg_rank:>8.3f} "
              f"{fourth_rate:>5.0%} {elapsed:>6.0f}s")

        if wandb_run is not None:
            log_data = {
                'version': version,
                'policy_loss': metrics['policy_loss'],
                'value_loss': metrics['value_loss'],
                'entropy': metrics['entropy'],
                'total_loss': metrics['total_loss'],
                'total_steps': total_env_steps,
                'iteration': iteration,
            }
            if eval_metrics:
                log_data.update({
                    'eval/avg_rank': avg_rank,
                    'eval/fourth_rate': fourth_rate,
                    'eval/win_rate': eval_metrics['win_rate'],
                    'eval/avg_score': eval_metrics['avg_score'],
                })
            wandb.log(log_data, step=total_env_steps)

        history.append({
            'iteration': iteration,
            'total_steps': total_env_steps,
            **metrics,
            'eval_avg_rank': avg_rank,
            'eval_fourth_rate': fourth_rate,
        })

        if avg_rank < best_avg_rank:
            best_avg_rank = avg_rank
            agent.save_checkpoint(
                os.path.join(args.checkpoint_dir, f'turtle_v{version}_best.pt'),
                iteration,
                metadata={'version': version, 'eval_avg_rank': avg_rank,
                          'eval_fourth_rate': fourth_rate},
            )

    # ── 版本结束 ──
    output_path = os.path.join(args.checkpoint_dir, f'turtle_v{version}_final.pt')
    agent.save_checkpoint(
        output_path, iteration,
        metadata={'version': version, 'history': history,
                  'total_steps': total_env_steps},
    )

    final_eval = run_eval(trainee, args.eval_games, device,
                          baseline_model=baseline,
                          trainee_idx=trainee_idx)
    print(f"\n  Version {version} final eval ({args.eval_games} games):")
    print(f"    WR={final_eval['win_rate']:.1%} "
          f"AvgRank={final_eval['avg_rank']:.2f} "
          f"4th={final_eval['fourth_rate']:.0%} "
          f"AvgScore={final_eval['avg_score']:.0f}")
    rd = final_eval.get('rank_distribution', {})
    print(f"    Rank dist: {dict(sorted(rd.items()))}")
    print(f"    Best avg rank: {best_avg_rank:.3f}")
    print(f"  Saved: {output_path}")

    if wandb_run is not None:
        wandb.log({
            f'v{version}/final_avg_rank': final_eval['avg_rank'],
            f'v{version}/final_fourth_rate': final_eval['fourth_rate'],
            f'v{version}/final_win_rate': final_eval['win_rate'],
            f'v{version}/final_avg_score': final_eval['avg_score'],
            f'v{version}/best_avg_rank': best_avg_rank,
        })
        wandb.finish()

    return output_path, final_eval


def main():
    args = parse_args()

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    args.device = device

    use_cuda = device == 'cuda'
    if use_cuda:
        torch.backends.cudnn.benchmark = True
        print(f"GPU: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory // 1024**2:,} MB VRAM)")

    train_cfg = load_training_config(args.personality)

    # 版本路径规划
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 确定起始版本和基线路径
    # v1 的 trainee 和 baseline 都从 SL base 开始
    # v2+ 的 trainee 从上一版 final 开始，baseline 用上一版 final

    baseline_path = args.base_model  # 初始 baseline = SL 基座
    trainee_path = args.base_model   # 初始 trainee = SL 基座

    # 如果从中间版本恢复，找到上一版的 checkpoint
    if args.resume_from > 1:
        prev_final = str(checkpoint_dir / f'turtle_v{args.resume_from - 1}_final.pt')
        if os.path.exists(prev_final):
            trainee_path = prev_final
            baseline_path = prev_final
            print(f"Resuming from version {args.resume_from}, "
                  f"using {prev_final} as base")
        else:
            print(f"Warning: {prev_final} not found, starting from base model")

    results = {}

    for version in range(args.resume_from, args.num_versions + 1):
        t_start = time.time()

        output_path, eval_result = train_one_version(
            version=version,
            trainee_path=trainee_path,
            baseline_path=baseline_path,
            personality_cfg=args.personality,
            train_cfg=train_cfg,
            args=args,
        )

        results[version] = {
            'path': output_path,
            'eval': eval_result,
            'time': time.time() - t_start,
        }

        # 下一版：trainee 从本版 final 继续，baseline 用本版 final (快照)
        trainee_path = output_path
        baseline_path = output_path

    # ── 汇总 ──
    print(f"\n{'='*60}")
    print("Iterative Training Summary")
    print(f"{'='*60}")
    print(f"{'Ver':>4} {'WR':>6} {'AvgRank':>8} {'4th%':>6} {'AvgScore':>10} {'Time':>8}")
    for v, r in results.items():
        e = r['eval']
        print(f"{v:>4} {e['win_rate']:>5.0%} {e['avg_rank']:>8.2f} "
              f"{e['fourth_rate']:>5.0%} {e['avg_score']:>10.0f} "
              f"{r['time']:>7.0f}s")

    print(f"\nModels saved in {checkpoint_dir}/")
    for v, r in results.items():
        print(f"  v{v}: {r['path']}")


if __name__ == '__main__':
    main()
# 中文注释：迭代式 PPO 自博弈训练，trainee 逐版对抗自身历史快照实现持续提升。
