"""阶段二：PPO 强化学习自对弈训练。

加载 SL 预训练 Base Model → 自对弈收集轨迹 → PPO 更新 → 定期保存。

用法:
    python -m training.rl_selfplay --base_model checkpoints/sl_best.pt
    python -m training.rl_selfplay --base_model checkpoints/sl_best.pt \\
        --personality configs/turtle.yaml --total_steps 100000
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models import MahjongPolicyValueNet, load_checkpoint, save_checkpoint
from .selfplay_env import SelfPlayEnv, run_games
from .ppo_agent import PPOAgent
from .reward_shaper import load_shaper_from_config


def parse_args():
    parser = argparse.ArgumentParser(
        description='RL Self-Play Training for Mahjong AI')
    parser.add_argument('--base_model', type=str, required=True,
                        help='Path to SL pretrained base model checkpoint')
    parser.add_argument('--personality', type=str,
                        default='configs/train_default.yaml',
                        help='Path to personality YAML config')
    parser.add_argument('--total_steps', type=int, default=100000,
                        help='Total environment steps to train for')
    parser.add_argument('--num_envs', type=int, default=64,
                        help='Number of parallel self-play games per iteration')
    parser.add_argument('--rollout_games', type=int, default=4,
                        help='Games per rollout collection')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/')
    parser.add_argument('--checkpoint_every', type=int, default=25,
                        help='Save checkpoint every N iterations')
    parser.add_argument('--eval_every', type=int, default=10,
                        help='Run evaluation every N iterations')
    parser.add_argument('--eval_games', type=int, default=10,
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
    parser.add_argument('--wandb_name', type=str, default=None,
                        help='W&B run name (default: auto-generated)')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='W&B entity/team name')
    return parser.parse_args()


def load_training_config(config_path: str) -> dict:
    """从 YAML 配置加载训练超参数。"""
    cfg = {}
    try:
        import yaml
        with open(config_path, 'r', encoding='utf-8') as f:
            raw = yaml.safe_load(f)
        training = raw.get('rl_training') or raw.get('training', {})
        cfg['lr'] = training.get('lr', 3e-4)
        cfg['clip_epsilon'] = training.get('clip_epsilon', 0.2)
        cfg['gamma'] = training.get('gamma', 0.99)
        cfg['gae_lambda'] = training.get('gae_lambda', 0.95)
        cfg['entropy_coef'] = training.get('entropy_coef', 0.01)
        cfg['value_loss_coef'] = training.get('value_loss_coef', 0.5)
        cfg['max_grad_norm'] = training.get('max_grad_norm', 1.0)
        cfg['ppo_epochs'] = training.get('ppo_epochs', 10)
        cfg['mini_batch_size'] = training.get('mini_batch_size', 256)
        cfg['kl_coef'] = training.get('kl_coef', 0.01)
    except Exception:
        pass
    return cfg


def run_eval(model: MahjongPolicyValueNet, num_games: int,
             device: str,
             baseline_model: MahjongPolicyValueNet | None = None,
             trainee_idx: int = 0) -> dict:
    """运行评估：trainee vs frozen baseline，只统计 trainee 的指标。

    当 baseline_model 为 None 时回退到旧行为（同一模型四家），
    返回所有座位的多玩家指标。
    """
    env = SelfPlayEnv(model, device=device, deterministic=True,
                      baseline_model=baseline_model,
                      trainee_idx=trainee_idx)
    model.eval()
    if baseline_model is not None:
        baseline_model.eval()
    all_ranks = {p: [] for p in range(4)}
    all_scores = {p: [] for p in range(4)}
    total_steps = 0

    for i in range(num_games):
        traj = env.run_game(seed=10000 + i)
        total_steps += traj.total_steps

        scores = traj.final_scores
        for p in range(4):
            rank = sum(1 for q in range(4) if scores[q] > scores[p])
            all_ranks[p].append(rank)
            all_scores[p].append(scores[p])

    # 单玩家指标（trainee 视角）
    trainee_ranks = all_ranks[trainee_idx]
    trainee_scores = all_scores[trainee_idx]
    n = num_games
    fourth_count = sum(1 for r in trainee_ranks if r == 3)
    first_count = sum(1 for r in trainee_ranks if r == 0)

    return {
        'win_rate': first_count / n,
        'avg_rank': sum(trainee_ranks) / n,
        'avg_score': sum(trainee_scores) / n,
        'fourth_rate': fourth_count / n,
        'avg_steps': total_steps / n,
        'rank_distribution': {
            str(r): trainee_ranks.count(r) for r in range(4)
        },
    }


def main():
    args = parse_args()

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'

    use_cuda = device == 'cuda'
    if use_cuda:
        torch.backends.cudnn.benchmark = True
        print(f"GPU: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory // 1024**2:,} MB VRAM)")

    # 加载训练配置
    train_cfg = load_training_config(args.personality)

    # 加载奖励塑形器
    reward_shaper = None
    if args.personality != 'configs/train_default.yaml':
        try:
            reward_shaper = load_shaper_from_config(args.personality)
            print(f"Loaded reward shaper: {type(reward_shaper).__name__}")
        except Exception as e:
            print(f"Warning: Could not load reward shaper: {e}")

    # 加载 SL 预训练模型
    model = MahjongPolicyValueNet()
    epoch, meta = load_checkpoint(model, args.base_model, device=device)
    print(f"Loaded base model from {args.base_model} (epoch {epoch})")
    if meta:
        print(f"  Checkpoint metadata: {meta.get('val_acc', 'N/A')}")

    if args.compile and hasattr(torch, 'compile'):
        model = torch.compile(model, mode='reduce-overhead')
        print("Model compiled with torch.compile")

    # 加载 frozen SL baseline（对手模型，不参与训练）
    baseline_model = MahjongPolicyValueNet()
    _, _ = load_checkpoint(baseline_model, args.base_model, device=device)
    baseline_model = baseline_model.to(device)
    baseline_model.eval()
    for p in baseline_model.parameters():
        p.requires_grad_(False)
    print(f"Loaded frozen baseline from {args.base_model}")

    trainee_idx = 0  # trainee 固定坐 P0

    # 初始化 PPO Agent
    agent = PPOAgent(
        model=model,
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

    # 自对弈环境：trainee 用训练模型 + reward_shaper，对手用 frozen baseline
    env = SelfPlayEnv(model, device=device, deterministic=False,
                      reward_shaper=reward_shaper,
                      trainee_idx=trainee_idx,
                      baseline_model=baseline_model)

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    total_env_steps = 0
    iteration = 0
    best_avg_rank = float('inf')  # lower is better
    history = []

    amp_status = "AMP" if (use_cuda and not args.no_amp) else "FP32"
    print(f"\nRL Self-Play Training on {device} ({amp_status})")
    print(f"  Total steps target: {args.total_steps}")
    print(f"  Rollout games/iter: {args.rollout_games}")
    print(f"  PPO epochs: {train_cfg.get('ppo_epochs', 10)}")
    print(f"  Mini-batch size: {train_cfg.get('mini_batch_size', 1024)}")
    print(f"  Reward shaper: {type(reward_shaper).__name__ if reward_shaper else 'base'}")
    print(f"  Trainee: P{trainee_idx}, Opponents: frozen SL baseline")
    print(f"\n{'Iter':>6} {'Steps':>8} {'PolicyLoss':>12} "
          f"{'ValueLoss':>10} {'Entropy':>10} {'AvgRank':>8} {'4th%':>6} {'Time':>8}")

    # ── wandb 初始化 ──
    wandb_run = None
    if args.wandb:
        import wandb
        run_name = args.wandb_name or f"rl_{Path(args.base_model).stem}"
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            entity=args.wandb_entity,
            config={
                'base_model': args.base_model,
                'personality': args.personality,
                'total_steps': args.total_steps,
                'rollout_games': args.rollout_games,
                'model_params': model.count_parameters(),
                'device': device,
                'amp': use_cuda and not args.no_amp,
                'compile': args.compile,
                **train_cfg,
            },
            reinit=True,
        )
        wandb.watch(model, log='gradients', log_freq=100)

    while total_env_steps < args.total_steps:
        t0 = time.time()
        iteration += 1

        # 收集 rollout
        agent.clear_buffer()
        for g in range(args.rollout_games):
            seed = args.seed + iteration * 1000 + g
            traj = env.run_game(seed=seed)

            agent.collect_trajectories([traj])
            total_env_steps += traj.total_steps

        # PPO 更新
        metrics = agent.update(
            ppo_epochs=train_cfg.get('ppo_epochs', 10),
            mini_batch_size=train_cfg.get('mini_batch_size', 1024),
        )

        elapsed = time.time() - t0

        # 评估：trainee vs baseline
        eval_metrics = {}
        if iteration % args.eval_every == 0:
            eval_metrics = run_eval(model, args.eval_games, device,
                                    baseline_model=baseline_model,
                                    trainee_idx=trainee_idx)

        avg_rank = eval_metrics.get('avg_rank', float('inf'))
        fourth_rate = eval_metrics.get('fourth_rate', 0)

        print(f"{iteration:>6} {total_env_steps:>8} "
              f"{metrics['policy_loss']:>12.4f} "
              f"{metrics['value_loss']:>10.4f} "
              f"{metrics['entropy']:>10.4f} "
              f"{avg_rank:>8.3f} "
              f"{fourth_rate:>5.0%} {elapsed:>7.1f}s")

        if wandb_run is not None:
            log_data = {
                'policy_loss': metrics['policy_loss'],
                'value_loss': metrics['value_loss'],
                'entropy': metrics['entropy'],
                'total_loss': metrics['total_loss'],
                'total_steps': total_env_steps,
                'iteration': iteration,
            }
            if eval_metrics:
                log_data['eval/avg_rank'] = avg_rank
                log_data['eval/fourth_rate'] = fourth_rate
                log_data['eval/win_rate'] = eval_metrics['win_rate']
                log_data['eval/avg_score'] = eval_metrics['avg_score']
                log_data['eval/avg_steps'] = eval_metrics['avg_steps']
            wandb.log(log_data, step=total_env_steps)

        history.append({
            'iteration': iteration,
            'total_steps': total_env_steps,
            **metrics,
            'eval_avg_rank': avg_rank,
            'eval_fourth_rate': fourth_rate,
        })

        # 保存最佳模型（按 avg_rank，越低越好）
        if avg_rank < best_avg_rank:
            best_avg_rank = avg_rank
            agent.save_checkpoint(
                os.path.join(args.checkpoint_dir, 'rl_best.pt'),
                iteration,
                metadata={'eval_avg_rank': avg_rank,
                          'eval_fourth_rate': fourth_rate,
                          'personality': args.personality},
            )

        # 定期保存
        if iteration % args.checkpoint_every == 0:
            agent.save_checkpoint(
                os.path.join(args.checkpoint_dir, f'rl_iter_{iteration:04d}.pt'),
                iteration,
                metadata={'iteration': iteration, 'total_steps': total_env_steps},
            )

    # 最终保存
    agent.save_checkpoint(
        os.path.join(args.checkpoint_dir, 'rl_final.pt'),
        iteration,
        metadata={'history': history, 'total_steps': total_env_steps},
    )

    # 最终评估
    final_eval = run_eval(model, args.eval_games * 2, device,
                          baseline_model=baseline_model,
                          trainee_idx=trainee_idx)
    print(f"\nFinal evaluation ({args.eval_games * 2} games):")
    print(f"  Trainee P{trainee_idx}: WR={final_eval['win_rate']:.1%} "
          f"AvgRank={final_eval['avg_rank']:.2f} "
          f"4th={final_eval['fourth_rate']:.0%} "
          f"AvgScore={final_eval['avg_score']:.0f}")
    rd = final_eval.get('rank_distribution', {})
    print(f"  Rank dist: {dict(sorted(rd.items()))}")
    print(f"  Avg steps/game: {final_eval['avg_steps']:.1f}")
    print(f"  Best avg rank: {best_avg_rank:.3f}")
    print(f"Model saved to {args.checkpoint_dir}")

    if wandb_run is not None:
        wandb.log({'final/avg_rank': final_eval['avg_rank'],
                   'final/fourth_rate': final_eval['fourth_rate'],
                   'final/win_rate': final_eval['win_rate'],
                   'final/avg_score': final_eval['avg_score'],
                   'final/avg_steps': final_eval['avg_steps'],
                   'final/best_avg_rank': best_avg_rank})
        wandb.finish()


if __name__ == '__main__':
    main()
# 中文注释：强化学习自博弈训练入口，组织环境交互、PPO 更新和 checkpoint 保存。
