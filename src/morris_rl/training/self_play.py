"""Parallel self-play data generation.

Architecture
------------
Each :class:`SelfPlayManager` spawns ``num_workers`` independent processes.
Every worker maintains its own local copy of the network on CPU and plays
complete games using :class:`~morris_rl.mcts.search.MorrisSearch`.  Completed
:class:`GameRecord` objects are sent to the manager via a shared results queue.

Weight updates are broadcast through per-worker queues.  Workers poll for
updates at the start of each new game, so they always play with weights that
are at most one game stale.

The self-play loop follows the AlphaZero temperature schedule:
  - moves 0 … temperature_threshold-1 : temperature = 1.0  (exploratory)
  - moves temperature_threshold …      : temperature = 1e-6 (near-argmax)

Dirichlet exploration noise is always added at the MCTS root during training.
"""

from __future__ import annotations

import multiprocessing as mp
import threading
from dataclasses import dataclass, field
from typing import Any

from morris_rl.utils.logging import logger

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn

from morris_rl.env.rules import (
    Outcome,
    apply_action,
    initial_state,
    is_terminal,
)
from morris_rl.network.resnet import MorrisResNet
from morris_rl.training.replay_buffer import SampleRecord

_NUM_PLANES = 7
_ARGMAX_TEMPERATURE = 1e-6


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class GameRecord:
    """Training data produced by one complete self-play game."""

    samples: list[SampleRecord]
    game_length: int            # total half-moves played
    outcome: int                # 1=player1, 2=player2, -1=draw


@dataclass
class WorkerError:
    """Sent through the results queue when a worker process crashes."""

    exception: Exception
    worker_id: int


# ---------------------------------------------------------------------------
# Game-play helpers
# ---------------------------------------------------------------------------


def _temperature_for_move(move_number: int, threshold: int) -> float:
    return 1.0 if move_number < threshold else _ARGMAX_TEMPERATURE


def _assign_value_targets(
    steps: list[tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], int]],
    outcome: Outcome | None,
) -> list[SampleRecord]:
    """Convert (encoded, policy, current_player) triples into SampleRecords."""
    records: list[SampleRecord] = []
    for encoded, policy, player in steps:
        if outcome is None or outcome == Outcome.DRAW:
            v = 0.0
        elif int(outcome) == player:
            v = 1.0
        else:
            v = -1.0
        records.append(SampleRecord(encoded_state=encoded, policy_target=policy, value_target=v))
    return records


def _play_game(search: "MorrisSearch", temperature_threshold: int = 10) -> GameRecord:
    """Play one complete self-play game and return its training data."""
    from morris_rl.mcts.search import encode_state  # cached after worker import

    state = initial_state()
    steps: list[tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], int]] = []
    move_count = 0

    while True:
        done, _ = is_terminal(state)
        if done:
            break

        temp = _temperature_for_move(move_count, temperature_threshold)
        encoded = encode_state(state).squeeze(0).numpy().copy()
        action, visit_probs = search.run(state, temperature=temp, add_noise=True)
        steps.append((encoded, visit_probs, state.current_player))
        state = apply_action(state, action)
        move_count += 1

    _, outcome = is_terminal(state)
    samples = _assign_value_targets(steps, outcome)
    outcome_int = -1 if (outcome is None or outcome == Outcome.DRAW) else int(outcome)
    return GameRecord(samples=samples, game_length=move_count, outcome=outcome_int)


# ---------------------------------------------------------------------------
# Network reconstruction (used inside worker processes)
# ---------------------------------------------------------------------------


def _build_worker_network(cfg: dict[str, Any]) -> MorrisResNet:
    return MorrisResNet(
        num_blocks=cfg["num_blocks"],
        num_channels=cfg["num_channels"],
        num_planes=cfg.get("num_planes", _NUM_PLANES),
        policy_head_hidden=cfg["policy_head_hidden"],
        value_head_hidden=cfg["value_head_hidden"],
    )


# ---------------------------------------------------------------------------
# Worker process entry point
# ---------------------------------------------------------------------------


def _should_recycle(
    worker_id: int,
    games_played: int,
    proc: Any,
    max_rss_mb: int,
    recycle_games: int,
) -> bool:
    """Return True when the worker should exit cleanly so the manager respawns
    it from a fresh Python interpreter — caps the impact of any growing RSS
    (suspected ctree leaf accumulation in early training).
    """
    if recycle_games > 0 and games_played >= recycle_games:
        from loguru import logger as _log
        _log.info(f"worker {worker_id}: recycling after {games_played} games")
        return True
    if max_rss_mb > 0 and games_played > 0 and games_played % 5 == 0:
        rss_mb = proc.memory_info().rss / 1e6
        if rss_mb > max_rss_mb:
            from loguru import logger as _log
            _log.warning(
                f"worker {worker_id}: rss={rss_mb:.0f}MB > {max_rss_mb}MB "
                f"after {games_played} games, recycling"
            )
            return True
    return False


def _worker_fn(
    worker_id: int,
    network_cfg: dict[str, Any],
    weights_queue: mp.Queue,  # type: ignore[type-arg]
    results_queue: mp.Queue,  # type: ignore[type-arg]
    num_simulations: int,
    temperature_threshold: int,
    dirichlet_alpha: float,
    dirichlet_epsilon: float,
    seed: int,
    worker_max_rss_mb: int = 0,
    worker_recycle_games: int = 0,
) -> None:
    """Worker process: play self-play games until a None sentinel is received."""
    import random
    import sys
    import warnings

    # Silence third-party startup noise before importing lzero:
    #   - ding loguru warnings (numba, pyecharts): filtered by our loguru handler
    #   - gym "unmaintained" message: a raw print() to sys.stderr — must redirect
    #     the *name* sys.stderr to /dev/null (loguru stores the file object and is
    #     unaffected, so our filtered handler still works on the real fd).
    import os
    warnings.filterwarnings("ignore")
    _old_stderr = sys.stderr
    from loguru import logger as _log
    _log.remove()
    _log.add(_old_stderr, level="INFO", filter=lambda r: r["name"].startswith("morris_rl"))

    _devnull = open(os.devnull, "w")
    sys.stderr = _devnull
    try:
        from morris_rl.mcts.search import MorrisSearch
    finally:
        sys.stderr = _old_stderr
        _devnull.close()

    # Each worker is one MCTS pipeline; using torch's default (= all CPU cores)
    # means N workers fight over N×cores threads. One thread per worker keeps the
    # CPU cleanly partitioned and is faster for small networks (1M params).
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # already set or torch already used parallel ops

    worker_seed = seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

    network = _build_worker_network(network_cfg)

    # Block until initial weights arrive.
    weights = weights_queue.get()
    if weights is None:
        return
    network.load_state_dict(weights)
    network.eval()

    search = MorrisSearch(
        network,
        torch.device("cpu"),
        num_simulations=num_simulations,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_epsilon=dirichlet_epsilon,
    )

    import psutil
    proc = psutil.Process()
    games_played = 0

    while True:
        # Non-blocking check for updated weights or shutdown.
        try:
            update = weights_queue.get_nowait()
            if update is None:
                return
            network.load_state_dict(update)
            network.eval()
        except Exception:
            pass

        try:
            game = _play_game(search, temperature_threshold=temperature_threshold)
            results_queue.put(game)
            games_played += 1
        except Exception as exc:
            results_queue.put(WorkerError(exception=exc, worker_id=worker_id))
            return  # Worker shuts down; main process will see the error immediately.

        if _should_recycle(
            worker_id, games_played, proc, worker_max_rss_mb, worker_recycle_games
        ):
            return


# ---------------------------------------------------------------------------
# Remote-eval worker (shared GPU inference server)
# ---------------------------------------------------------------------------


def _worker_fn_remote(
    worker_id: int,
    req_send_conn: Any,     # mpc.Connection — worker's send end of req pipe
    reply_recv_conn: Any,   # mpc.Connection — worker's recv end of reply pipe
    results_queue: mp.Queue,  # type: ignore[type-arg]
    shutdown_event: Any,    # mp.Event — set when manager.stop() is called
    shm_names: Any,
    num_workers: int,
    num_simulations: int,
    temperature_threshold: int,
    dirichlet_alpha: float,
    dirichlet_epsilon: float,
    seed: int,
    worker_max_rss_mb: int = 0,
    worker_recycle_games: int = 0,
) -> None:
    """Worker process: delegates leaf evaluation to the inference server.

    Same self-play loop as :func:`_worker_fn` but evaluation goes through a
    request/reply queue pair connected to a centralized GPU server instead of
    running torch in-process.
    """
    import random
    import sys
    import warnings

    import os
    warnings.filterwarnings("ignore")
    _old_stderr = sys.stderr
    from loguru import logger as _log
    _log.remove()
    _log.add(_old_stderr, level="INFO", filter=lambda r: r["name"].startswith("morris_rl"))

    _devnull = open(os.devnull, "w")
    sys.stderr = _devnull
    try:
        from morris_rl.env.rules import get_legal_actions
        from morris_rl.mcts.search import MorrisSearch, encode_state
        from morris_rl.training.inference_server import make_remote_eval_fn
    finally:
        sys.stderr = _old_stderr
        _devnull.close()

    # Workers don't run torch in-process here, but other libs may still query
    # CPU thread defaults (e.g. numpy/BLAS); pin to one to avoid contention
    # with the inference server's host-side dispatch threads.
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    worker_seed = seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

    eval_fn = make_remote_eval_fn(
        worker_id=worker_id,
        req_send_conn=req_send_conn,
        reply_recv_conn=reply_recv_conn,
        shm_names=shm_names,
        num_workers=num_workers,
        encode_state=encode_state,
        get_legal_actions=get_legal_actions,
    )
    search = MorrisSearch(
        eval_fn=eval_fn,
        num_simulations=num_simulations,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_epsilon=dirichlet_epsilon,
    )

    import psutil
    proc = psutil.Process()
    games_played = 0

    while not shutdown_event.is_set():
        try:
            game = _play_game(search, temperature_threshold=temperature_threshold)
            results_queue.put(game)
            games_played += 1
        except Exception as exc:
            results_queue.put(WorkerError(exception=exc, worker_id=worker_id))
            return

        if _should_recycle(
            worker_id, games_played, proc, worker_max_rss_mb, worker_recycle_games
        ):
            return


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class SelfPlayManager:
    """Manages a pool of self-play worker processes.

    Workers produce :class:`GameRecord` objects which the caller collects via
    :meth:`collect_game`.  After each training step, call
    :meth:`update_network` to broadcast fresh weights.

    Args:
        network: The current policy/value network (used to seed worker weights).
        network_cfg: Plain-dict description of the network architecture, passed
            to workers for reconstruction (e.g.
            ``{"num_blocks": 10, "num_channels": 128, ...}``).
        num_workers: Number of parallel worker processes.
        num_simulations: MCTS simulations per move.
        temperature_threshold: Use temperature=1.0 for the first this many
            moves, then switch to near-argmax.
        dirichlet_alpha: Dirichlet concentration for root exploration noise.
        dirichlet_epsilon: Weight of Dirichlet noise mixed into root priors.
        seed: Base random seed; worker i uses seed + i.
    """

    def __init__(
        self,
        network: nn.Module,
        network_cfg: dict[str, Any],
        num_workers: int = 12,
        num_simulations: int = 200,
        temperature_threshold: int = 10,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        seed: int = 42,
        inference_mode: str = "per_worker_cpu",
        inference_device: str = "cuda",
        max_batch_size: int = 32,
        max_wait_ms: float = 5.0,
        log_file: str | None = None,
        worker_max_rss_mb: int = 0,
        worker_recycle_games: int = 0,
        watcher_interval_s: float = 5.0,
    ) -> None:
        if inference_mode not in ("per_worker_cpu", "shared_gpu"):
            raise ValueError(f"unknown inference_mode {inference_mode!r}")
        self._network = network
        self._network_cfg = network_cfg
        self._num_workers = num_workers
        self._num_simulations = num_simulations
        self._temperature_threshold = temperature_threshold
        self._dirichlet_alpha = dirichlet_alpha
        self._dirichlet_epsilon = dirichlet_epsilon
        self._seed = seed
        self._inference_mode = inference_mode
        self._inference_device = inference_device
        self._max_batch_size = max_batch_size
        self._max_wait_ms = max_wait_ms
        self._log_file = log_file
        self._worker_max_rss_mb = worker_max_rss_mb
        self._worker_recycle_games = worker_recycle_games
        self._watcher_interval_s = watcher_interval_s

        self._ctx = mp.get_context("spawn")
        self._results_queue: mp.Queue = self._ctx.Queue()  # type: ignore[type-arg]
        # Per-worker weights queue is only used in per_worker_cpu mode; in
        # shared_gpu mode the inference server owns the network and gets a
        # single dedicated weights channel via InferenceServer.update_weights.
        self._weights_queues: list[mp.Queue] = [  # type: ignore[type-arg]
            self._ctx.Queue(maxsize=1) for _ in range(num_workers)
        ]
        self._shutdown_event: Any = self._ctx.Event()
        self._inference_server: Any = None
        # Slot per worker — None until first start, then the live Process. The
        # watcher thread mutates this list when respawning, hence the lock.
        self._processes: list[Any] = [None] * num_workers
        self._processes_lock = threading.Lock()
        self._respawn_count: list[int] = [0] * num_workers
        self._watcher_thread: threading.Thread | None = None
        self._watcher_stop = threading.Event()
        self._running = False

    def start(self) -> None:
        """Spawn worker processes (and inference server if shared_gpu)."""
        if self._running:
            return
        if self._inference_mode == "shared_gpu":
            from morris_rl.training.inference_server import InferenceServer

            self._inference_server = InferenceServer(
                network=self._network,
                network_cfg=self._network_cfg,
                num_workers=self._num_workers,
                device=self._inference_device,
                max_batch=self._max_batch_size,
                max_wait_ms=self._max_wait_ms,
                log_file=self._log_file,
            )
            self._inference_server.start()

        for i in range(self._num_workers):
            self._spawn_worker(i)

        self._running = True
        # Recycling watcher only spins up if at least one trigger is enabled.
        if self._worker_max_rss_mb > 0 or self._worker_recycle_games > 0:
            self._watcher_thread = threading.Thread(
                target=self._watch_workers, daemon=True, name="self-play-watcher"
            )
            self._watcher_thread.start()

    def _spawn_worker(self, i: int) -> None:
        """(Re)spawn worker *i*. Used at start and on respawn after exit.

        Pipes (shared_gpu) and weights queues (per_worker_cpu) are owned by the
        manager, so they survive a worker restart and are reused as-is.
        """
        if self._inference_mode == "shared_gpu":
            assert self._inference_server is not None
            req_send, reply_recv = self._inference_server.worker_pipes(i)
            p = self._ctx.Process(
                target=_worker_fn_remote,
                args=(
                    i,
                    req_send,
                    reply_recv,
                    self._results_queue,
                    self._shutdown_event,
                    self._inference_server.shm_names,
                    self._inference_server.num_workers,
                    self._num_simulations,
                    self._temperature_threshold,
                    self._dirichlet_alpha,
                    self._dirichlet_epsilon,
                    # Reseed each respawn so we don't replay the same sequence.
                    self._seed + 1000 * (self._respawn_count[i] + 1),
                    self._worker_max_rss_mb,
                    self._worker_recycle_games,
                ),
                daemon=True,
            )
            p.start()
        else:
            p = self._ctx.Process(
                target=_worker_fn,
                args=(
                    i,
                    self._network_cfg,
                    self._weights_queues[i],
                    self._results_queue,
                    self._num_simulations,
                    self._temperature_threshold,
                    self._dirichlet_alpha,
                    self._dirichlet_epsilon,
                    self._seed + 1000 * (self._respawn_count[i] + 1),
                    self._worker_max_rss_mb,
                    self._worker_recycle_games,
                ),
                daemon=True,
            )
            p.start()
            # Per-worker-cpu workers block on weights_queue.get() at startup,
            # so push the latest weights right after spawn.
            cpu_weights = {k: v.cpu() for k, v in self._network.state_dict().items()}
            try:
                # Drain any stale entry first (1-slot queue).
                self._weights_queues[i].get_nowait()
            except Exception:
                pass
            self._weights_queues[i].put(cpu_weights)

        with self._processes_lock:
            self._processes[i] = p

    def _watch_workers(self) -> None:
        """Background thread: detect dead workers and respawn them.

        Runs every ``watcher_interval_s``. Workers exit cleanly on RSS or
        game-count triggers (see :func:`_should_recycle`); this thread reacts
        to that exit by starting a fresh process on the same slot — same pipes,
        same shared-memory slot, just a brand-new Python interpreter (no leaked
        state from ctree, torch, numpy, etc.).
        """
        while not self._watcher_stop.is_set():
            with self._processes_lock:
                snapshot = list(enumerate(self._processes))
            for i, p in snapshot:
                if self._watcher_stop.is_set():
                    return
                if p is None or p.is_alive():
                    continue
                exit_code = p.exitcode
                self._respawn_count[i] += 1
                logger.info(
                    "worker {} exited (code={}); respawning (#{}, total respawns={})",
                    i,
                    exit_code,
                    self._respawn_count[i],
                    sum(self._respawn_count),
                )
                try:
                    self._spawn_worker(i)
                except Exception as exc:
                    logger.exception(f"failed to respawn worker {i}: {exc}")
            self._watcher_stop.wait(self._watcher_interval_s)

    def collect_game(self, timeout: float = 300.0) -> GameRecord:
        """Block until one completed game is available and return it.

        Args:
            timeout: Seconds to wait before raising queue.Empty.

        Raises:
            RuntimeError: If a worker process crashed (surfaces the original exception).
            queue.Empty:  If no game arrives within *timeout* seconds.
        """
        result = self._results_queue.get(timeout=timeout)
        if isinstance(result, WorkerError):
            raise RuntimeError(
                f"Self-play worker {result.worker_id} crashed: {result.exception}"
            ) from result.exception
        return result  # type: ignore[return-value]

    def update_network(self, state_dict: dict[str, Any]) -> None:
        """Broadcast updated weights to whoever needs them.

        - per_worker_cpu : push to each worker's bounded weights queue.
        - shared_gpu     : push only to the inference server; workers don't
                           own a network in this mode.
        """
        if self._inference_mode == "shared_gpu":
            if self._inference_server is not None:
                self._inference_server.update_weights(state_dict)
            return
        cpu_weights = {k: v.cpu() for k, v in state_dict.items()}
        for q in self._weights_queues:
            try:
                q.get_nowait()
            except Exception:
                pass
            try:
                q.put_nowait(cpu_weights)
            except Exception:
                pass

    def results_qsize(self) -> int:
        """Approximate size of the results queue (best-effort on macOS/Linux)."""
        try:
            return self._results_queue.qsize()
        except NotImplementedError:
            return -1

    def weights_qsize_max(self) -> int:
        """Largest weight queue across workers (should stay ≤ 1)."""
        sizes = []
        for q in self._weights_queues:
            try:
                sizes.append(q.qsize())
            except NotImplementedError:
                return -1
        return max(sizes) if sizes else 0

    def stop(self) -> None:
        """Send shutdown sentinels and join all worker processes."""
        if not self._running:
            return
        # Stop the watcher first — otherwise it would race with shutdown and
        # cheerfully respawn any worker we just told to exit.
        self._watcher_stop.set()
        if self._watcher_thread is not None:
            self._watcher_thread.join(timeout=2)
            self._watcher_thread = None
        if self._inference_mode == "shared_gpu":
            self._shutdown_event.set()
            if self._inference_server is not None:
                self._inference_server.stop()
        else:
            for q in self._weights_queues:
                q.put(None)
        with self._processes_lock:
            procs = [p for p in self._processes if p is not None]
            self._processes = [None] * self._num_workers
        for p in procs:
            p.join(timeout=15)
            if p.is_alive():
                p.terminate()
        self._running = False

    def __enter__(self) -> SelfPlayManager:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()
