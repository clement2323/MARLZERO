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
from dataclasses import dataclass, field
from typing import Any

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

_NUM_PLANES = 8
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
        except Exception as exc:
            results_queue.put(WorkerError(exception=exc, worker_id=worker_id))
            return  # Worker shuts down; main process will see the error immediately.


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

    while not shutdown_event.is_set():
        try:
            game = _play_game(search, temperature_threshold=temperature_threshold)
            results_queue.put(game)
        except Exception as exc:
            results_queue.put(WorkerError(exception=exc, worker_id=worker_id))
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
        self._processes: list[Any] = []
        self._running = False

    def start(self) -> None:
        """Spawn worker processes (and inference server if shared_gpu)."""
        if self._running:
            return
        if self._inference_mode == "shared_gpu":
            self._start_shared_gpu()
        else:
            self._start_per_worker_cpu()
        self._running = True

    def _start_per_worker_cpu(self) -> None:
        initial_weights = {k: v.cpu() for k, v in self._network.state_dict().items()}
        for i in range(self._num_workers):
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
                    self._seed,
                ),
                daemon=True,
            )
            p.start()
            self._processes.append(p)
            self._weights_queues[i].put(initial_weights)

    def _start_shared_gpu(self) -> None:
        # Lazy import: InferenceServer pulls torch + the network module, no need
        # to load it when only per_worker_cpu mode is used (e.g. CI smoke tests).
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
                    self._seed,
                ),
                daemon=True,
            )
            p.start()
            self._processes.append(p)

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
        if self._inference_mode == "shared_gpu":
            self._shutdown_event.set()
            if self._inference_server is not None:
                # InferenceServer.stop() sends None on each worker reply pipe,
                # waking blocked workers, then signals the server's shutdown
                # pipe before joining.
                self._inference_server.stop()
        else:
            for q in self._weights_queues:
                q.put(None)
        for p in self._processes:
            p.join(timeout=15)
            if p.is_alive():
                p.terminate()
        self._running = False
        self._processes.clear()

    def __enter__(self) -> SelfPlayManager:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()
