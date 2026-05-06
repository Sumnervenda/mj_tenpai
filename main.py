#!/usr/bin/env python3
"""演示程序：使用随机动作模拟完整的麻将对局。

验证游戏引擎运行正确性，输出对局进度信息。
"""

import random
import sys
import time
import json
import threading
import csv
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

sys.path.insert(0, ".")

from engine import (
    GameEngine, GameConfig, GamePhase, GameState,
    Action, ActionType, LegalActions,
    TILE_NAMES,
    create_engine_snapshot,
)


def tile_counts_to_codes(counts: List[int]) -> List[str]:
    """把 int[34] 手牌直方图展开为牌 code 列表，便于 JSONL/CSV 分析。"""
    tiles: List[str] = []
    for type_id, count in enumerate(counts):
        tiles.extend([TILE_NAMES[type_id]] * int(count))
    return tiles


def serialize_record_meld(meld) -> Dict[str, Any]:
    """记录副露信息；保持中文注释但输出字段使用机器友好的英文。"""
    return {
        "type": getattr(meld.meld_type, "name", str(meld.meld_type)),
        "tiles": [TILE_NAMES[t] for t in meld.tiles],
        "called_from": meld.called_from,
        "source_tile": TILE_NAMES[meld.source_tile] if meld.source_tile >= 0 else None,
    }


def serialize_player_record(engine: GameEngine, player_idx: int) -> Dict[str, Any]:
    """序列化某家当前手牌、分数、立直和副露状态。"""
    player = engine.players[player_idx]
    return {
        "player": player_idx,
        "score": player.score,
        "hand_counts": list(player.hand.tiles),
        "hand": tile_counts_to_codes(player.hand.tiles),
        "melds": [serialize_record_meld(meld) for meld in player.hand.melds],
        "discards": [TILE_NAMES[aid // 4] for aid in player.discards],
        "riichi": player.is_riichi,
        "menzen": player.hand.is_menzen,
        "has_won": player.has_won,
    }


def capture_agari_record(
    engine: GameEngine,
    game_index: int,
    event_index: int,
    seed: int,
    step_count: int,
) -> Dict[str, Any]:
    """在 AGARI 精算前捕获和牌结果、点数变化和四家手牌。"""
    payments = list(getattr(engine, "_last_agari_payments", []))
    win_type = "tsumo" if getattr(engine, "_last_win_is_tsumo", False) else "ron"
    win_tile = getattr(engine, "_last_win_tile", -1)

    winners = []
    for payment in payments:
        winner_hand = serialize_player_record(engine, payment.winner)
        winners.append({
            "winner": payment.winner,
            "win_type": win_type,
            "loser": payment.loser,
            "winning_tile": TILE_NAMES[win_tile] if 0 <= win_tile < len(TILE_NAMES) else None,
            "han": payment.han,
            "fu": payment.fu,
            "score_name": payment.score_name,
            "yaku": list(payment.yaku_names),
            "dora_count": payment.dora_count,
            "payments": list(payment.payments),
            "total_win": payment.total_win,
            "winner_hand": winner_hand,
        })

    return {
        "type": "agari",
        "game_index": game_index,
        "event_index": event_index,
        "seed": seed,
        "step_count": step_count,
        "round": f"{TILE_NAMES[engine.round_wind]}{engine.round_number}",
        "honba": engine.honba,
        "riichi_sticks": engine.riichi_sticks,
        "scores_after_payment": [player.score for player in engine.players],
        "winners": winners,
        "players": [serialize_player_record(engine, idx) for idx in range(4)],
    }


def game_summary_record(
    engine: GameEngine,
    game_index: int,
    seed: int,
    step_count: int,
    agari_count: int,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """记录一场随机对局结束后的总结果。"""
    result = engine.get_result()
    return {
        "type": "game_summary",
        "game_index": game_index,
        "seed": seed,
        "step_count": step_count,
        "agari_count": agari_count,
        "error": error,
        "final_scores": list(result.final_scores),
        "adjusted_scores": list(result.adjusted_scores),
        "ranks": list(result.ranks),
        "top_player": engine.get_winner(),
        "players": [serialize_player_record(engine, idx) for idx in range(4)],
    }


class LiveSnapshotHub:
    """SSE 实时快照中心。

    随机对局线程每推进一步就 publish 一次；HTTP 线程中的 /stream 客户端
    通过 condition 等待下一帧。这里仅保存最新快照，避免慢客户端拖垮对局。
    """

    def __init__(self):
        self._condition = threading.Condition()
        self._snapshot_json = "{}"
        self._version = 0
        self.done = False
        self.paused = False

    def publish(self, snapshot: Dict[str, Any]) -> None:
        """发布最新 EngineSnapshotDTO。"""
        with self._condition:
            self._version += 1
            self._snapshot_json = json.dumps(snapshot, ensure_ascii=False)
            self.done = bool(snapshot.get("state", {}).get("done", False))
            self._condition.notify_all()

    def latest(self) -> Tuple[int, str]:
        """返回当前版本号和最新快照 JSON。"""
        with self._condition:
            return self._version, self._snapshot_json

    def status(self) -> Dict[str, Any]:
        """返回 Live 服务状态，供 /health 和控制按钮使用。"""
        with self._condition:
            return {
                "ok": True,
                "version": self._version,
                "done": self.done,
                "paused": self.paused,
            }

    def set_paused(self, paused: bool) -> bool:
        """设置暂停状态，并唤醒对局循环重新检查。"""
        with self._condition:
            changed = self.paused != paused
            self.paused = paused
            self._condition.notify_all()
            return changed

    def toggle_paused(self) -> bool:
        """切换暂停状态，返回切换后的值。"""
        with self._condition:
            self.paused = not self.paused
            self._condition.notify_all()
            return self.paused

    def is_paused(self) -> bool:
        """当前是否处于暂停状态。"""
        with self._condition:
            return self.paused

    def wait_while_paused(self) -> None:
        """暂停时阻塞随机对局推进；HTTP/SSE 线程不受影响。"""
        with self._condition:
            while self.paused:
                self._condition.wait(timeout=0.5)

    def wait_delay_or_pause(self, delay: float) -> None:
        """等待下一步 delay；如果前端发出暂停，立刻返回让主循环进入暂停等待。"""
        if delay <= 0:
            return
        deadline = time.monotonic() + delay
        with self._condition:
            while not self.paused:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return
                self._condition.wait(timeout=remaining)

    def wait_next(self, version: int, timeout: float = 15.0) -> Tuple[int, str]:
        """等待比 version 更新的快照；超时则返回当前快照用于心跳。"""
        with self._condition:
            self._condition.wait_for(lambda: self._version > version, timeout=timeout)
            return self._version, self._snapshot_json


def format_sse_event(snapshot_json: str, version: int) -> bytes:
    """把快照 JSON 包装成标准 SSE snapshot 事件。"""
    return f"id: {version}\nevent: snapshot\ndata: {snapshot_json}\n\n".encode("utf-8")


def make_live_log(log_id: int, level: str, message: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """创建前端控制台可显示的日志对象。"""
    return {
        "id": f"live-{log_id}",
        "time": time.strftime("%H:%M:%S"),
        "level": level,
        "message": message,
        "payload": payload,
    }


def make_live_snapshot(
    engine: GameEngine,
    logs: List[Dict[str, Any]],
    frame: int,
    last_action: str,
    done: bool = False,
    paused: bool = False,
) -> Dict[str, Any]:
    """创建 Live 模式下推给前端的 EngineSnapshotDTO。"""
    snapshot = create_engine_snapshot(
        engine,
        logs=logs,
        debug={
            "connection": "sse",
            "illegal_actions": 0,
            "replay_frame": frame,
            "response_queue": [],
            "last_action": last_action,
            "paused": paused,
        },
    )
    snapshot["state"]["done"] = bool(done or engine.is_game_over())
    return snapshot


def make_live_handler(hub: LiveSnapshotHub, on_control=None):
    """创建绑定指定 hub 的 HTTP Handler 类。"""

    class LiveConsoleHandler(BaseHTTPRequestHandler):
        server_version = "MahjongLiveConsole/0.1"

        def _send_cors(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

        def _send_json(self, payload: Dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self._send_cors()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self._send_cors()
            self.end_headers()

        def _handle_control(self, path: str) -> bool:
            if path == "/pause":
                changed = hub.set_paused(True)
                if on_control and changed:
                    on_control(True)
                self._send_json(hub.status())
                return True

            if path == "/resume":
                changed = hub.set_paused(False)
                if on_control and changed:
                    on_control(False)
                self._send_json(hub.status())
                return True

            if path == "/toggle-pause":
                paused = hub.toggle_paused()
                if on_control:
                    on_control(paused)
                self._send_json(hub.status())
                return True

            return False

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            if self._handle_control(path):
                return
            self.send_error(404, "Not Found")

        def do_GET(self) -> None:
            path = urlparse(self.path).path

            if path == "/health":
                self._send_json(hub.status())
                return

            if self._handle_control(path):
                return

            if path == "/snapshot":
                _, snapshot_json = hub.latest()
                self.send_response(200)
                self._send_cors()
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(snapshot_json.encode("utf-8"))
                return

            if path == "/stream":
                self.send_response(200)
                self._send_cors()
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()

                version, snapshot_json = hub.latest()
                try:
                    self.wfile.write(format_sse_event(snapshot_json, version))
                    self.wfile.flush()
                    while True:
                        next_version, next_json = hub.wait_next(version)
                        if next_version == version:
                            self.wfile.write(b": keep-alive\n\n")
                        else:
                            version = next_version
                            self.wfile.write(format_sse_event(next_json, version))
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return
                return

            self.send_error(404, "Not Found")

        def log_message(self, fmt: str, *args: Any) -> None:
            """保持控制台输出干净，仅打印关键启动信息。"""
            return

    return LiveConsoleHandler


def random_action(legal_actions: LegalActions, rng: random.Random) -> Action:
    """从合法动作中随机选择一个。偏好非パス动作。"""
    if not legal_actions.actions:
        return Action(ActionType.PASS)

    # 优先选择非パス动作（パス通常在最末尾）
    non_pass = [a for a in legal_actions.actions
                if a.action_type != ActionType.PASS]
    if non_pass:
        return rng.choice(non_pass)
    return legal_actions.actions[0]


def advance_random_game(engine: GameEngine, rng: random.Random, verbose: bool = True) -> Tuple[str, bool]:
    """随机推进游戏状态机一步。

    Returns:
        (日志消息, 是否进入终局)
    """
    state = engine.get_game_state()

    if verbose and state.phase == GamePhase.DRAW:
        print(engine)
        print(f"   Remaining: {state.remaining_tiles} tiles")
        print()

    if state.phase == GamePhase.DRAW:
        actions = engine.get_legal_actions()
        action = random_action(actions, rng)
        if verbose:
            print(f"  P{state.current_player} → {action}")
        engine.step(action)
        return f"P{state.current_player} → {action}", engine.is_game_over()

    if state.phase == GamePhase.DISCARD:
        options = engine.get_response_options()
        responses = {}
        messages = []
        for p_idx, legal in options.items():
            responses[p_idx] = random_action(legal, rng)
        if verbose:
            for p_idx, act in responses.items():
                if act.action_type != ActionType.PASS:
                    print(f"  P{p_idx} responds: {act}")
                    messages.append(f"P{p_idx} responds: {act}")
        engine.resolve_responses(responses)
        return " / ".join(messages) if messages else "All responses pass", engine.is_game_over()

    if state.phase == GamePhase.AGARI:
        if verbose:
            print(f"  *** AGARI! Player {state.current_player} wins! ***")
            print(f"  Scores: {state.scores}")
        engine.step(Action(ActionType.PASS))
        return f"AGARI: Player {state.current_player}", engine.is_game_over()

    if state.phase == GamePhase.RYUUKYOKU:
        if verbose:
            print("  *** RYUUKYOKU (exhaustive draw) ***")
            for i in range(4):
                tenpai = engine.players[i].is_tenpai_at_ryuukyoku
                print(f"  P{i}: {'TENPAI' if tenpai else 'NOTEN'}")
        engine.step(Action(ActionType.PASS))
        return "RYUUKYOKU", engine.is_game_over()

    if state.phase in (GamePhase.ROUND_END, GamePhase.GAME_END):
        return f"Phase {state.phase.name}", engine.is_game_over()

    return f"Unhandled phase {state.phase.name}", engine.is_game_over()


def run_random_game(config: Optional[GameConfig] = None, seed: Optional[int] = None,
                    verbose: bool = True) -> GameEngine:
    """运行一局完整的随机对局。

    Args:
        config: 规则配置（默认雀魂标准规则）
        seed: 随机种子（保证可复现）
        verbose: 是否输出对局日志

    Returns:
        游戏结束后的引擎实例。
    """
    rng = random.Random(seed)
    config = config or GameConfig()
    engine = GameEngine(config=config, seed=seed)

    round_count = 0
    max_rounds = 50  # 安全上限（防止无限循环）

    while not engine.is_game_over() and round_count < max_rounds:
        message, _ = advance_random_game(engine, rng, verbose=verbose)
        state = engine.get_game_state()
        if state.phase in (GamePhase.ROUND_END, GamePhase.GAME_END):
            round_count += 1

    result = engine.get_result()
    if verbose:
        print(f"\n=== Game Over ===")
        print(f"Final scores: {result.final_scores}")
        print(f"Adjusted: {result.adjusted_scores}")
        print(f"Ranks: {result.ranks}")
        print(f"Winner: Player {engine.get_winner()} (rank {result.ranks[engine.get_winner()] + 1})")

    return engine


def run_live_console(
    config: Optional[GameConfig] = None,
    seed: Optional[int] = None,
    host: str = "127.0.0.1",
    port: int = 8765,
    delay: float = 0.3,
    verbose: bool = True,
) -> GameEngine:
    """运行随机对局，并通过 SSE 实时推送到前端控制台。"""
    rng = random.Random(seed)
    config = config or GameConfig()
    engine = GameEngine(config=config, seed=seed)
    hub = LiveSnapshotHub()
    logs: List[Dict[str, Any]] = []
    frame = 0
    engine_lock = threading.Lock()

    def publish_locked(level: str, message: str, done: bool = False) -> None:
        nonlocal frame
        frame += 1
        logs.insert(0, make_live_log(frame, level, message))
        del logs[80:]
        hub.publish(make_live_snapshot(
            engine,
            logs,
            frame,
            message,
            done=done,
            paused=hub.is_paused(),
        ))

    def publish(level: str, message: str, done: bool = False) -> None:
        with engine_lock:
            publish_locked(level, message, done=done)

    def handle_control(paused: bool) -> None:
        message = "Live paused by frontend" if paused else "Live resumed by frontend"
        publish("info", message)

    publish("info", f"Live console started (seed={seed})")

    server = ThreadingHTTPServer((host, port), make_live_handler(hub, on_control=handle_control))
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    print(f"Live console SSE server: http://{host}:{port}")
    print(f"  health:   http://{host}:{port}/health")
    print(f"  snapshot: http://{host}:{port}/snapshot")
    print(f"  stream:   http://{host}:{port}/stream")
    print(f"  pause:    http://{host}:{port}/pause")
    print(f"  resume:   http://{host}:{port}/resume")
    print("Open frontend console and connect Live mode. Press Ctrl+C to stop.")

    try:
        max_steps = 10000
        steps = 0
        while not engine.is_game_over() and steps < max_steps:
            hub.wait_while_paused()
            with engine_lock:
                message, done = advance_random_game(engine, rng, verbose=verbose)
                publish_locked("action", message, done=done)
            steps += 1
            hub.wait_delay_or_pause(max(0.0, delay))

        result = engine.get_result()
        final_message = (
            f"Game Over: winner=P{engine.get_winner()}, "
            f"scores={result.final_scores}, ranks={result.ranks}"
        )
        publish("info", final_message, done=True)
        print(final_message)

        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping live console...")
    finally:
        server.shutdown()
        server.server_close()

    return engine


def benchmark(num_games: int = 100) -> float:
    """性能基准测试：运行 N 局随机对局并计算速度。

    Returns:
        每秒对局数。
    """
    print(f"Running {num_games} random games...", end=" ", flush=True)
    start = time.time()

    for i in range(num_games):
        run_random_game(seed=i, verbose=False)

    elapsed = time.time() - start
    games_per_sec = num_games / elapsed
    print(f"Done in {elapsed:.2f}s ({games_per_sec:.1f} games/sec)")
    return games_per_sec


def _record_output_paths(output_prefix: str) -> Tuple[Path, Path]:
    """根据输出前缀生成 JSONL 与 CSV 路径。"""
    prefix = Path(output_prefix)
    if prefix.suffix:
        jsonl_path = prefix
        csv_path = prefix.with_name(f"{prefix.stem}_agari.csv")
    else:
        jsonl_path = prefix.with_suffix(".jsonl")
        csv_path = prefix.with_name(f"{prefix.name}_agari.csv")
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    return jsonl_path, csv_path


def _write_agari_csv_rows(writer: csv.DictWriter, record: Dict[str, Any]) -> None:
    """把一条 agari JSON 记录展开为 CSV 行。"""
    for winner in record["winners"]:
        writer.writerow({
            "game_index": record["game_index"],
            "event_index": record["event_index"],
            "seed": record["seed"],
            "round": record["round"],
            "honba": record["honba"],
            "step_count": record["step_count"],
            "winner": winner["winner"],
            "win_type": winner["win_type"],
            "loser": winner["loser"],
            "winning_tile": winner["winning_tile"],
            "han": winner["han"],
            "fu": winner["fu"],
            "score_name": winner["score_name"],
            "yaku": "|".join(winner["yaku"]),
            "dora_count": winner["dora_count"],
            "payments": json.dumps(winner["payments"], ensure_ascii=False),
            "scores_after_payment": json.dumps(record["scores_after_payment"], ensure_ascii=False),
            "winner_hand": " ".join(winner["winner_hand"]["hand"]),
            "winner_melds": json.dumps(winner["winner_hand"]["melds"], ensure_ascii=False),
        })


def run_random_records(
    num_games: int,
    seed: int = 42,
    output_prefix: str = "records/random_games",
    max_steps_per_game: int = 10000,
    progress_every: int = 100,
) -> Dict[str, Any]:
    """随机跑多场对局，并记录和牌结果、点数和手牌。

    输出：
      - JSONL：逐条保存 agari 事件和 game_summary，信息最完整。
      - CSV：每个和牌者一行，便于 Excel / pandas 快速统计。
    """
    jsonl_path, csv_path = _record_output_paths(output_prefix)
    start = time.time()
    total_agari = 0
    total_errors = 0

    csv_fields = [
        "game_index", "event_index", "seed", "round", "honba", "step_count",
        "winner", "win_type", "loser", "winning_tile", "han", "fu",
        "score_name", "yaku", "dora_count", "payments",
        "scores_after_payment", "winner_hand", "winner_melds",
    ]

    with jsonl_path.open("w", encoding="utf-8") as jsonl_file, csv_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
        writer.writeheader()

        for game_index in range(num_games):
            game_seed = seed + game_index
            rng = random.Random(game_seed)
            engine = GameEngine(config=GameConfig(), seed=game_seed)
            event_index = 0
            agari_count = 0
            step_count = 0
            error: Optional[str] = None

            try:
                while not engine.is_game_over() and step_count < max_steps_per_game:
                    if engine.phase == GamePhase.AGARI:
                        record = capture_agari_record(
                            engine=engine,
                            game_index=game_index,
                            event_index=event_index,
                            seed=game_seed,
                            step_count=step_count,
                        )
                        jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                        _write_agari_csv_rows(writer, record)
                        winners = max(1, len(record["winners"]))
                        agari_count += winners
                        total_agari += winners
                        event_index += 1

                    advance_random_game(engine, rng, verbose=False)
                    step_count += 1

                if step_count >= max_steps_per_game and not engine.is_game_over():
                    error = f"max_steps_per_game reached: {max_steps_per_game}"
                    total_errors += 1
            except Exception as exc:  # 批量记录不能因单局异常中断后续样本。
                error = f"{type(exc).__name__}: {exc}"
                total_errors += 1

            summary = game_summary_record(
                engine=engine,
                game_index=game_index,
                seed=game_seed,
                step_count=step_count,
                agari_count=agari_count,
                error=error,
            )
            jsonl_file.write(json.dumps(summary, ensure_ascii=False) + "\n")

            if progress_every > 0 and (game_index + 1) % progress_every == 0:
                elapsed = time.time() - start
                speed = (game_index + 1) / elapsed if elapsed > 0 else 0.0
                print(
                    f"Recorded {game_index + 1}/{num_games} games, "
                    f"agari={total_agari}, errors={total_errors}, "
                    f"{speed:.1f} games/sec",
                    flush=True,
                )

    elapsed = time.time() - start
    result = {
        "games": num_games,
        "agari": total_agari,
        "errors": total_errors,
        "elapsed_sec": elapsed,
        "games_per_sec": num_games / elapsed if elapsed > 0 else 0.0,
        "jsonl": str(jsonl_path),
        "csv": str(csv_path),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main():
    """主入口：单局演示或批量基准测试。"""
    import argparse
    parser = argparse.ArgumentParser(description="Mahjong DL-Engine Demo")
    parser.add_argument("--benchmark", type=int, default=0,
                        help="Run benchmark for N games")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--live-console", action="store_true",
                        help="Run one random game and publish live snapshots for the frontend console")
    parser.add_argument("--record-random", type=int, default=0,
                        help="Run N random games and record agari results, scores, and hands")
    parser.add_argument("--record-output", type=str, default="records/random_games",
                        help="Output prefix for --record-random JSONL/CSV files")
    parser.add_argument("--record-progress", type=int, default=100,
                        help="Progress print interval for --record-random")
    parser.add_argument("--record-max-steps", type=int, default=10000,
                        help="Per-game safety step limit for --record-random")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="Live console host")
    parser.add_argument("--port", type=int, default=8765,
                        help="Live console port")
    parser.add_argument("--delay", type=float, default=0.3,
                        help="Delay between live random-game steps in seconds")
    args = parser.parse_args()

    if args.record_random > 0:
        run_random_records(
            num_games=args.record_random,
            seed=args.seed,
            output_prefix=args.record_output,
            max_steps_per_game=args.record_max_steps,
            progress_every=args.record_progress,
        )
    elif args.live_console:
        run_live_console(seed=args.seed, host=args.host, port=args.port, delay=args.delay, verbose=True)
    elif args.benchmark > 0:
        benchmark(args.benchmark)
    else:
        print("=" * 60)
        print("Mahjong DL-Engine — Random Game Simulation")
        print("=" * 60)
        run_random_game(seed=args.seed, verbose=True)


if __name__ == "__main__":
    main()
