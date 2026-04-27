"""Centralized GPU inference server for self-play workers.

Architecture
------------
A dedicated process holds the network on GPU and answers leaf-evaluation
requests from many CPU MCTS workers.  Each request from a worker carries an
encoded state + the list of legal actions; the server batches up to N requests
(or waits at most ``max_wait_ms``), runs one forward pass, and dispatches the
per-worker replies on dedicated reply queues.

This decouples MCTS branching from torch forward latency: instead of N workers
each running ~3 ms forwards in series on CPU, we get ~5 ms forwards on GPU
that serve a batch of up to N requests at once.

Why per-worker reply queues
---------------------------
A single shared reply queue would let one worker pop another worker's reply
(no shared event loop, no per-worker filtering). Per-worker queues route
responses unambiguously and are cheap (one ``mp.Queue(maxsize=4)`` each).

Weight updates
--------------
The trainer pushes fresh weights through ``weights_queue`` (maxsize=1).  The
server polls non-blockingly between batches and reloads when an update is
present.  This is lockstep AlphaZero — see CLAUDE.md.
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from morris_rl.env.board import ACTION_SPACE_SIZE
from morris_rl.network.resnet import MorrisResNet
from morris_rl.utils.logging import logger

_NUM_PLANES = 8


@dataclass(frozen=True)
class InferenceRequest:
    """One leaf evaluation request from a self-play worker."""

    worker_id: int
    req_id: int
    encoded_state: npt.NDArray[np.float32]   # shape (8, 24)
    legal_actions: tuple[int, ...]


@dataclass(frozen=True)
class InferenceReply:
    """Response to a single :class:`InferenceRequest`."""

    req_id: int
    # Dense numpy array of shape (legal_actions_count,) with probabilities
    # in the same order as the request's legal_actions tuple.
    legal_probs: npt.NDArray[np.float32]
    value: float


@dataclass(frozen=True)
class ServerError:
    """Sentinel sent on a worker's reply queue when the server crashed."""

    req_id: int
    message: str


def _build_server_network(network_cfg: dict[str, Any]) -> MorrisResNet:
    return MorrisResNet(
        num_blocks=network_cfg["num_blocks"],
        num_channels=network_cfg["num_channels"],
        num_planes=network_cfg.get("num_planes", _NUM_PLANES),
        policy_head_hidden=network_cfg["policy_head_hidden"],
        value_head_hidden=network_cfg["value_head_hidden"],
    )


def _drain_weight_update(weights_queue: mp.Queue, network: torch.nn.Module) -> bool:  # type: ignore[type-arg]
    """Apply a pending weight update if any. Returns True if applied."""
    try:
        update = weights_queue.get_nowait()
    except queue.Empty:
        return False
    if update is None:  # ignore None here; shutdown is signalled via req_queue
        return False
    network.load_state_dict(update)
    network.eval()
    return True


def _gather_batch(
    req_queue: mp.Queue,  # type: ignore[type-arg]
    max_batch: int,
    max_wait_ms: float,
) -> list[InferenceRequest] | None:
    """Block on the first request, then drain greedily up to *max_batch*.

    Returns:
        A non-empty list of requests, or ``None`` if the shutdown sentinel
        (a ``None`` value) was received.
    """
    first = req_queue.get()  # blocking
    if first is None:
        return None
    batch: list[InferenceRequest] = [first]
    deadline = time.perf_counter() + max_wait_ms / 1000.0
    while len(batch) < max_batch and time.perf_counter() < deadline:
        try:
            item = req_queue.get_nowait()
        except queue.Empty:
            time.sleep(0)  # yield to scheduler, then re-check the deadline
            continue
        if item is None:
            # Shutdown landed mid-batch: process what we have, then exit on
            # the next call (we re-push the sentinel for the outer loop).
            req_queue.put(None)
            break
        batch.append(item)
    return batch


def _server_loop(
    req_queue: mp.Queue,  # type: ignore[type-arg]
    reply_queues: list[mp.Queue],  # type: ignore[type-arg]
    weights_queue: mp.Queue,  # type: ignore[type-arg]
    network_cfg: dict[str, Any],
    device_str: str,
    max_batch: int,
    max_wait_ms: float,
    log_interval_batches: int,
) -> None:
    """Inference server main loop. Runs until a None lands on req_queue."""
    device = torch.device(device_str)
    network = _build_server_network(network_cfg)
    network.to(device).eval()

    # Block until the trainer hands us the initial weights.
    initial = weights_queue.get()
    if initial is None:
        return
    network.load_state_dict(initial)
    network.eval()

    # Stats for periodic logging — useful to verify we're actually batching.
    batch_count = 0
    total_in_batches = 0
    total_forward_s = 0.0

    while True:
        batch = _gather_batch(req_queue, max_batch, max_wait_ms)
        if batch is None:
            return

        _drain_weight_update(weights_queue, network)

        try:
            states = np.stack([req.encoded_state for req in batch], axis=0)
            states_t = torch.from_numpy(states).to(device)
            mask = torch.zeros(len(batch), ACTION_SPACE_SIZE, dtype=torch.bool, device=device)
            for i, req in enumerate(batch):
                if req.legal_actions:
                    mask[i, list(req.legal_actions)] = True

            t0 = time.perf_counter()
            with torch.no_grad():
                log_policy, value = network(states_t, mask)
            probs = log_policy.exp().cpu().numpy()
            values = value.cpu().numpy()
            total_forward_s += time.perf_counter() - t0
        except Exception as exc:
            for req in batch:
                reply_queues[req.worker_id].put(
                    ServerError(req_id=req.req_id, message=f"{type(exc).__name__}: {exc}")
                )
            logger.exception("Inference server forward pass failed")
            continue

        for i, req in enumerate(batch):
            legal_probs = np.array(
                [probs[i, a] for a in req.legal_actions], dtype=np.float32
            )
            reply_queues[req.worker_id].put(
                InferenceReply(req_id=req.req_id, legal_probs=legal_probs, value=float(values[i]))
            )

        batch_count += 1
        total_in_batches += len(batch)
        if batch_count % log_interval_batches == 0:
            mean_batch = total_in_batches / batch_count
            mean_forward_ms = (total_forward_s / batch_count) * 1000
            logger.info(
                "inference server: {} batches, mean batch={:.1f}, mean forward={:.2f} ms",
                batch_count,
                mean_batch,
                mean_forward_ms,
            )


# ---------------------------------------------------------------------------
# Worker-side eval_fn
# ---------------------------------------------------------------------------


def make_remote_eval_fn(
    worker_id: int,
    req_queue: mp.Queue,  # type: ignore[type-arg]
    reply_queue: mp.Queue,  # type: ignore[type-arg]
    encode_state: Any,  # avoid circular import; pass in encode_state callable
    get_legal_actions: Any,
    request_timeout_s: float = 60.0,
) -> Any:
    """Build an EvalFn that round-trips through the inference server."""
    counter = {"n": 0}

    def evaluate(state: Any) -> tuple[dict[int, float], float]:
        legal = get_legal_actions(state)
        encoded = encode_state(state).squeeze(0).numpy()
        counter["n"] += 1
        req_id = counter["n"]
        req_queue.put(
            InferenceRequest(
                worker_id=worker_id,
                req_id=req_id,
                encoded_state=encoded,
                legal_actions=tuple(legal),
            )
        )
        reply = reply_queue.get(timeout=request_timeout_s)
        # None signals shutdown — manager.stop() injects it to wake up workers
        # blocked on get(). SystemExit isn't caught by `except Exception`,
        # so the worker process exits cleanly without surfacing a WorkerError.
        if reply is None:
            raise SystemExit("inference server shutdown")
        if isinstance(reply, ServerError):
            raise RuntimeError(
                f"inference server error on req {reply.req_id}: {reply.message}"
            )
        if reply.req_id != req_id:
            raise RuntimeError(
                f"reply mismatch: expected {req_id}, got {reply.req_id} "
                "(worker reply queue out of sync)"
            )
        probs_dict = {a: float(p) for a, p in zip(legal, reply.legal_probs)}
        return probs_dict, float(reply.value)

    return evaluate


# ---------------------------------------------------------------------------
# Public manager class
# ---------------------------------------------------------------------------


class InferenceServer:
    """Owns the inference subprocess and the queues connecting it to workers.

    Args:
        network:        Reference network used to seed the server's initial
                        weights (state_dict is copied to CPU and shipped).
        network_cfg:    Architecture description for the server-side rebuild.
        num_workers:    Number of self-play workers that will connect.
        device:         "cuda" or "cpu" — where the server runs forwards.
        max_batch:      Upper bound on per-forward batch size.
        max_wait_ms:    Max time the batcher waits for more requests after the
                        first arrives. Trade-off: lower = lower latency,
                        smaller batches; higher = fewer forwards, larger batches.
    """

    def __init__(
        self,
        network: torch.nn.Module,
        network_cfg: dict[str, Any],
        num_workers: int,
        device: str = "cuda",
        max_batch: int = 32,
        max_wait_ms: float = 5.0,
        log_interval_batches: int = 200,
    ) -> None:
        self._ctx = mp.get_context("spawn")
        # Reasonable upper bound on in-flight requests per worker — workers send
        # one at a time and block on the reply, so depth >= 2 is plenty.
        self._req_queue: mp.Queue = self._ctx.Queue()  # type: ignore[type-arg]
        self._reply_queues: list[mp.Queue] = [  # type: ignore[type-arg]
            self._ctx.Queue(maxsize=4) for _ in range(num_workers)
        ]
        self._weights_queue: mp.Queue = self._ctx.Queue(maxsize=1)  # type: ignore[type-arg]
        self._network = network
        self._network_cfg = network_cfg
        self._device_str = device
        self._max_batch = max_batch
        self._max_wait_ms = max_wait_ms
        self._log_interval_batches = log_interval_batches
        self._process: Any = None
        self._running = False

    def start(self) -> None:
        if self._running:
            return
        initial_weights = {k: v.cpu() for k, v in self._network.state_dict().items()}
        self._weights_queue.put(initial_weights)
        self._process = self._ctx.Process(
            target=_server_loop,
            args=(
                self._req_queue,
                self._reply_queues,
                self._weights_queue,
                self._network_cfg,
                self._device_str,
                self._max_batch,
                self._max_wait_ms,
                self._log_interval_batches,
            ),
            daemon=True,
        )
        self._process.start()
        self._running = True

    def update_weights(self, state_dict: dict[str, Any]) -> None:
        """Push fresh weights to the server. Drops a stale pending update."""
        cpu_weights = {k: v.cpu() for k, v in state_dict.items()}
        try:
            self._weights_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._weights_queue.put_nowait(cpu_weights)
        except queue.Full:
            pass

    def stop(self) -> None:
        if not self._running:
            return
        self._req_queue.put(None)  # shutdown sentinel
        if self._process is not None:
            self._process.join(timeout=15)
            if self._process.is_alive():
                self._process.terminate()
        self._running = False

    @property
    def request_queue(self) -> mp.Queue:  # type: ignore[type-arg]
        return self._req_queue

    @property
    def reply_queues(self) -> list[mp.Queue]:  # type: ignore[type-arg]
        return self._reply_queues
