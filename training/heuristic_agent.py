"""启发式对局数据生成器 —— 使用 HeuristicAgent 自对弈生成训练数据。"""

import json

from engine import GameEngine, GameConfig, GamePhase, Action, ActionType
from data.data_generator import HeuristicAgent
from main import capture_agari_record, game_summary_record


def generate_training_data(num_games: int,
                           seed: int = 0,
                           output_prefix: str = "records/heuristic_games",
                           progress_every: int = 10) -> str:
    """使用启发式 Agent 自对弈生成 JSONL 训练数据。

    Args:
        num_games: 对局数
        seed: 随机种子
        output_prefix: 输出文件前缀
        progress_every: 进度打印间隔

    Returns:
        JSONL 文件路径
    """
    import time
    from pathlib import Path

    jsonl_path = Path(output_prefix).with_suffix('.jsonl')
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    start = time.time()
    total_agari = 0

    with jsonl_path.open('w', encoding='utf-8') as f:
        for game_index in range(num_games):
            game_seed = seed + game_index
            rng = __import__('random').Random(game_seed)
            agent = HeuristicAgent(seed=game_seed)
            engine = GameEngine(config=GameConfig(), seed=game_seed)
            event_index = 0
            agari_count = 0
            step_count = 0

            try:
                while not engine.is_game_over() and step_count < 10000:
                    if engine.phase == GamePhase.AGARI:
                        record = capture_agari_record(
                            engine, game_index, event_index,
                            game_seed, step_count)
                        f.write(json.dumps(record, ensure_ascii=False) + '\n')
                        agari_count += max(1, len(record.get('winners', [])))
                        total_agari += agari_count
                        event_index += 1

                    if engine.phase == GamePhase.DRAW:
                        p = engine.current_player
                        action = agent.select_action(engine, p)
                        engine.step(action)
                    elif engine.phase == GamePhase.DISCARD:
                        options = engine.get_response_options()
                        responses = {}
                        for p_idx, legal in options.items():
                            responses[p_idx] = agent.select_response(engine, p_idx)
                        engine.resolve_responses(responses)
                    elif engine.phase in (GamePhase.AGARI, GamePhase.RYUUKYOKU,
                                          GamePhase.ROUND_END, GamePhase.GAME_END):
                        engine.step(Action(ActionType.PASS))
                    else:
                        engine.step(Action(ActionType.PASS))
                    step_count += 1
            except Exception as e:
                print(f"  Game {game_index} error: {e}")

            summary = game_summary_record(engine, game_index, game_seed,
                                          step_count, agari_count)
            f.write(json.dumps(summary, ensure_ascii=False) + '\n')

            if progress_every > 0 and (game_index + 1) % progress_every == 0:
                elapsed = time.time() - start
                speed = (game_index + 1) / elapsed if elapsed > 0 else 0.0
                print(f"  Generated {game_index + 1}/{num_games} games, "
                      f"agari={total_agari}, {speed:.1f} games/sec")

    elapsed = time.time() - start
    print(f"Done: {num_games} games, {total_agari} agari, "
          f"{elapsed:.1f}s ({num_games / elapsed:.1f} games/sec)")
    return str(jsonl_path)
# 中文注释：启发式基线智能体，用于生成样本、做对照实验和快速自测。
