"""Centralized GPU inference server for self-play workers — v3 (per-worker pipes).

Architecture
------------
A dedicated process holds the network on GPU and answers leaf-evaluation
requests from many CPU MCTS workers via a fan-in/fan-out of point-to-point
pipes. Three regions of POSIX shared memory carry the bulky tensors:

  - request_states[num_workers, 8, 24] float32  — encoded states
  - reply_probs   [num_workers, ACTION_SPACE_SIZE] float32  — softmaxed probs
  - reply_values  [num_workers] float32                     — value heads

Workers write their state into their own slot, send a tiny control message
through their own ``mp.Pipe``, then block on a per-worker reply pipe. The
server uses ``multiprocessing.connection.wait`` to discover which workers
have a request ready, drains them up to ``max_batch_size``, runs one batched
forward, and writes probs/values back into each requester's slot.

Why pipes per worker (vs a single mp.Queue)
-------------------------------------------
v2 used one shared ``mp.Queue`` for requests. Under 8-way contention, the
queue's internal lock dominated: ~10–15 ms per round-trip even with shared
memory. v3 replaces that with N independent pipes, cutting send/recv to
~50 µs (no shared lock, OS-level pipe is single-producer/single-consumer
per direction).

Why shared memory for the bulky tensors
---------------------------------------
Even tiny pickled tensors (~1 KB) cost a few hundred microseconds round-trip.
``np.copyto`` into a pre-attached shared region is a memcpy: ~1 µs.

Weight updates
--------------
Trainer pushes fresh weights through ``weights_queue`` (mp.Queue, low
frequency so no contention). Server polls non-blockingly between batches and
reloads. Lockstep AlphaZero — see CLAUDE.md.
"""

from __future__ import annotations

import multiprocessing as mp
import multiprocessing.connection as mpc
import queue
import time
import uuid
from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from typing import Any

import numpy as np
import torch

from morris_rl.env.board import ACTION_SPACE_SIZE
from morris_rl.network.resnet import MorrisResNet
from morris_rl.utils.logging import logger

_NUM_PLANES = 8


@dataclass(frozen=True)
class ShmNames:
    """Names of the three shared memory regions backing the request/reply slots."""

    request_states: str
    reply_probs: str
    reply_values: str


@dataclass(frozen=True)
class InferenceRequest:
    """Tiny control message: state lives in shared memory at slot worker_id."""

    worker_id: int
    req_id: int
    legal_actions: tuple[int, ...]


@dataclass(frozen=True)
class InferenceReply:
    """Tiny control message: probs/value live in shared memory at slot worker_id."""

    req_id: int


@dataclass(frozen=True)
class ServerError:
    """Sentinel sent on a worker's reply pipe when the server crashed."""

    req_id: int
    message: str


# ---------------------------------------------------------------------------
# Shared memory helpers
# ---------------------------------------------------------------------------


def _alloc_shm(num_workers: int) -> tuple[ShmNames, list[SharedMemory]]:
    req_bytes = num_workers * _NUM_PLANES * 24 * 4
    probs_bytes = num_workers * ACTION_SPACE_SIZE * 4
    values_bytes = num_workers * 4

    suffix = uuid.uuid4().hex[:12]
    req_shm = SharedMemory(create=True, size=req_bytes, name=f"morris_req_{suffix}")
    probs_shm = SharedMemory(create=True, size=probs_bytes, name=f"morris_probs_{suffix}")
    values_shm = SharedMemory(create=True, size=values_bytes, name=f"morris_vals_{suffix}")

    np.ndarray(req_bytes, dtype=np.uint8, buffer=req_shm.buf).fill(0)
    np.ndarray(probs_bytes, dtype=np.uint8, buffer=probs_shm.buf).fill(0)
    np.ndarray(values_bytes, dtype=np.uint8, buffer=values_shm.buf).fill(0)

    names = ShmNames(
        request_states=req_shm.name,
        reply_probs=probs_shm.name,
        reply_values=values_shm.name,
    )
    return names, [req_shm, probs_shm, values_shm]


def _attach_views(
    names: ShmNames, num_workers: int
) -> tuple[list[SharedMemory], np.ndarray, np.ndarray, np.ndarray]:
    req_shm = SharedMemory(name=names.request_states)
    probs_shm = SharedMemory(name=names.reply_probs)
    values_shm = SharedMemory(name=names.reply_values)
    request_states = np.ndarray(
        (num_workers, _NUM_PLANES, 24), dtype=np.float32, buffer=req_shm.buf
    )
    reply_probs = np.ndarray(
        (num_workers, ACTION_SPACE_SIZE), dtype=np.float32, buffer=probs_shm.buf
    )
    reply_values = np.ndarray((num_workers,), dtype=np.float32, buffer=values_shm.buf)
    return [req_shm, probs_shm, values_shm], request_states, reply_probs, reply_values


def _build_server_network(network_cfg: dict[str, Any]) -> MorrisResNet:
    return MorrisResNet(
        num_blocks=network_cfg["num_blocks"],
        num_channels=network_cfg["num_channels"],
        num_planes=network_cfg.get("num_planes", _NUM_PLANES),
        policy_head_hidden=network_cfg["policy_head_hidden"],
        value_head_hidden=network_cfg["value_head_hidden"],
    )


def _drain_weight_update(weights_queue: mp.Queue, network: torch.nn.Module) -> bool:  # type: ignore[type-arg]
    try:
        update = weights_queue.get_nowait()
    except queue.Empty:
        return False
    if update is None:
        return False
    network.load_state_dict(update)
    network.eval()
    return True


# ---------------------------------------------------------------------------
# Server-side
# ---------------------------------------------------------------------------


def _gather_batch(
    req_recv_conns: list[mpc.Connection],
    shutdown_recv: mpc.Connection,
    max_batch: int,
    max_wait_ms: float,
) -> list[InferenceRequest] | None:
    """Block until at least one request arrives, then drain to ``max_batch``.

    Returns:
        The collected batch (non-empty), or ``None`` if shutdown was signalled.
    """
    wait_list: list[mpc.Connection] = [*req_recv_conns, shutdown_recv]
    ready = mpc.wait(wait_list)
    if shutdown_recv in ready:
        return None

    batch: list[InferenceRequest] = []
    for conn in ready:
        if conn is shutdown_recv:
            continue
        try:
            batch.append(conn.recv())
        except EOFError:
            pass

    deadline = time.perf_counter() + max_wait_ms / 1000.0
    while len(batch) < max_batch:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            break
        more = mpc.wait(req_recv_conns, timeout=remaining)
        if not more:
            break
        for conn in more:
            try:
                batch.append(conn.recv())
            except EOFError:
                pass
            if len(batch) >= max_batch:
                break
    return batch


def _server_loop(
    req_recv_conns: list[mpc.Connection],
    reply_send_conns: list[mpc.Connection],
    shutdown_recv: mpc.Connection,
    weights_queue: mp.Queue,  # type: ignore[type-arg]
    shm_names: ShmNames,
    num_workers: int,
    network_cfg: dict[str, Any],
    device_str: str,
    max_batch: int,
    max_wait_ms: float,
    log_interval_batches: int,
    log_file: str | None,
) -> None:
    """Inference server main loop. Runs until a value lands on shutdown_recv."""
    # File-only logging from this process so server stats don't fight the
    # trainer's tqdm bar on stderr.
    if log_file is not None:
        from loguru import logger as _log
        _log.remove()
        _log.add(
            log_file,
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            enqueue=True,  # async write so loguru never blocks the forward loop
            filter=lambda r: r["name"].startswith("morris_rl"),
        )

    device = torch.device(device_str)
    network = _build_server_network(network_cfg)
    network.to(device).eval()

    handles, request_states, reply_probs, reply_values = _attach_views(
        shm_names, num_workers
    )

    initial = weights_queue.get()
    if initial is None:
        for h in handles:
            h.close()
        return
    network.load_state_dict(initial)
    network.eval()

    # Dedicated CUDA stream so the inference forward overlaps with the
    # trainer's gradient steps on the default stream instead of serialising.
    use_cuda = device.type == "cuda"
    inference_stream = torch.cuda.Stream(device=device) if use_cuda else None

    # Pinned host buffer for staging inputs: pinned + non_blocking H2D yields
    # async transfer overlapped with kernel execution. Sized to max_batch so
    # we never reallocate inside the hot loop.
    if use_cuda:
        host_states = torch.empty(
            (max_batch, _NUM_PLANES, 24), dtype=torch.float32, pin_memory=True
        )
        host_states_view = host_states.numpy()
    else:
        host_states = None
        host_states_view = None

    batch_count = 0
    total_in_batches = 0
    total_forward_s = 0.0

    try:
        while True:
            batch = _gather_batch(
                req_recv_conns, shutdown_recv, max_batch, max_wait_ms
            )
            if batch is None:
                return

            _drain_weight_update(weights_queue, network)

            try:
                worker_ids = [req.worker_id for req in batch]
                n = len(batch)
                if use_cuda:
                    for i, wid in enumerate(worker_ids):
                        host_states_view[i] = request_states[wid]
                    states_t = host_states[:n].to(device, non_blocking=True)
                else:
                    states = np.stack(
                        [request_states[wid] for wid in worker_ids], axis=0
                    )
                    states_t = torch.from_numpy(states).to(device)

                mask = torch.zeros(
                    n, ACTION_SPACE_SIZE, dtype=torch.bool, device=device
                )
                for i, req in enumerate(batch):
                    if req.legal_actions:
                        mask[i, list(req.legal_actions)] = True

                t0 = time.perf_counter()
                if use_cuda:
                    with torch.cuda.stream(inference_stream):  # type: ignore[arg-type]
                        with (
                            torch.no_grad(),
                            torch.cuda.amp.autocast(dtype=torch.float16),
                        ):
                            log_policy, value = network(states_t, mask)
                        probs_t = log_policy.float().exp()
                        values_t = value.float()
                    inference_stream.synchronize()  # type: ignore[union-attr]
                    probs = probs_t.cpu().numpy()
                    values = values_t.cpu().numpy()
                else:
                    with torch.no_grad():
                        log_policy, value = network(states_t, mask)
                    probs = log_policy.exp().cpu().numpy()
                    values = value.cpu().numpy()
                total_forward_s += time.perf_counter() - t0
            except Exception as exc:
                for req in batch:
                    try:
                        reply_send_conns[req.worker_id].send(
                            ServerError(
                                req_id=req.req_id,
                                message=f"{type(exc).__name__}: {exc}",
                            )
                        )
                    except (BrokenPipeError, OSError):
                        pass
                logger.exception("Inference server forward pass failed")
                continue

            for i, req in enumerate(batch):
                # Write reply slots BEFORE notifying the worker — the pipe send
                # provides the release barrier the worker's recv pairs with.
                reply_probs[req.worker_id, :] = probs[i]
                reply_values[req.worker_id] = float(values[i])
                try:
                    reply_send_conns[req.worker_id].send(
                        InferenceReply(req_id=req.req_id)
                    )
                except (BrokenPipeError, OSError):
                    # Worker died; nothing useful we can do, drop the reply.
                    pass

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
    finally:
        for h in handles:
            h.close()


# ---------------------------------------------------------------------------
# Worker-side eval_fn
# ---------------------------------------------------------------------------


def make_remote_eval_fn(
    worker_id: int,
    req_send_conn: mpc.Connection,
    reply_recv_conn: mpc.Connection,
    shm_names: ShmNames,
    num_workers: int,
    encode_state: Any,
    get_legal_actions: Any,
    request_timeout_s: float = 60.0,
) -> Any:
    """Build an EvalFn that round-trips through the inference server.

    Attaches to the three shared memory regions on first call, then reuses the
    views for the lifetime of the worker. Bulky tensors travel via shared
    memory; only a small control message goes through the pipe.
    """
    handles, request_states, reply_probs, reply_values = _attach_views(
        shm_names, num_workers
    )
    counter = {"n": 0}

    def evaluate(state: Any) -> tuple[dict[int, float], float]:
        # Force closure capture so SharedMemory handles outlive this factory's
        # return — without this, GC unmaps the buffers and request_states
        # points to freed memory (silent SIGSEGV in daemon worker).
        _ = handles
        legal = get_legal_actions(state)
        encoded = encode_state(state).squeeze(0).numpy()
        np.copyto(request_states[worker_id], encoded.astype(np.float32, copy=False))
        counter["n"] += 1
        req_id = counter["n"]
        req_send_conn.send(
            InferenceRequest(
                worker_id=worker_id,
                req_id=req_id,
                legal_actions=tuple(legal),
            )
        )
        if not reply_recv_conn.poll(request_timeout_s):
            raise RuntimeError(
                f"inference reply timed out after {request_timeout_s}s "
                f"(worker {worker_id}, req {req_id})"
            )
        reply = reply_recv_conn.recv()
        if reply is None:
            raise SystemExit("inference server shutdown")
        if isinstance(reply, ServerError):
            raise RuntimeError(
                f"inference server error on req {reply.req_id}: {reply.message}"
            )
        if reply.req_id != req_id:
            raise RuntimeError(
                f"reply mismatch: expected {req_id}, got {reply.req_id} "
                "(worker reply pipe out of sync)"
            )
        probs_dict = {a: float(reply_probs[worker_id, a]) for a in legal}
        return probs_dict, float(reply_values[worker_id])

    return evaluate


# ---------------------------------------------------------------------------
# Public manager class
# ---------------------------------------------------------------------------


class InferenceServer:
    """Owns the inference subprocess, the per-worker pipes, and the shared memory.

    Args:
        network:        Reference network used to seed initial weights.
        network_cfg:    Architecture description for the server-side rebuild.
        num_workers:    Number of self-play workers that will connect.
        device:         "cuda" or "cpu" — where the server runs forwards.
        max_batch:      Upper bound on per-forward batch size.
        max_wait_ms:    Max time the batcher waits for more requests after the
                        first arrives.
        log_interval_batches: Log a stats line every N batches.
        log_file:       If set, server log goes only to this file (keeps stderr
                        clean for the trainer's tqdm bar).
    """

    def __init__(
        self,
        network: torch.nn.Module,
        network_cfg: dict[str, Any],
        num_workers: int,
        device: str = "cuda",
        max_batch: int = 32,
        max_wait_ms: float = 5.0,
        log_interval_batches: int = 2000,
        log_file: str | None = None,
    ) -> None:
        self._ctx = mp.get_context("spawn")
        self._num_workers = num_workers

        # Per-worker request and reply pipes (duplex=False for clarity).
        # Element i is (recv_end, send_end).
        self._req_pipes: list[tuple[mpc.Connection, mpc.Connection]] = [
            self._ctx.Pipe(duplex=False) for _ in range(num_workers)
        ]
        self._reply_pipes: list[tuple[mpc.Connection, mpc.Connection]] = [
            self._ctx.Pipe(duplex=False) for _ in range(num_workers)
        ]
        # Shutdown signal — pumping anything on shutdown_send wakes the server.
        self._shutdown_recv, self._shutdown_send = self._ctx.Pipe(duplex=False)

        self._weights_queue: mp.Queue = self._ctx.Queue(maxsize=1)  # type: ignore[type-arg]
        self._network = network
        self._network_cfg = network_cfg
        self._device_str = device
        # Each worker has at most one in-flight request (it blocks on the
        # reply), so the effective max batch is capped by num_workers. Without
        # this cap, the drain loop would happily wait the full max_wait_ms
        # hoping for a 9th request that can never arrive — burning ~5 ms per
        # cycle and crushing throughput in lockstep mode.
        self._max_batch = min(max_batch, num_workers)
        self._max_wait_ms = max_wait_ms
        self._log_interval_batches = log_interval_batches
        self._log_file = log_file
        self._process: Any = None
        self._running = False

        self._shm_names, self._shm_handles = _alloc_shm(num_workers)

    def start(self) -> None:
        if self._running:
            return
        initial_weights = {k: v.cpu() for k, v in self._network.state_dict().items()}
        self._weights_queue.put(initial_weights)
        req_recv_conns = [p[0] for p in self._req_pipes]
        reply_send_conns = [p[1] for p in self._reply_pipes]
        self._process = self._ctx.Process(
            target=_server_loop,
            args=(
                req_recv_conns,
                reply_send_conns,
                self._shutdown_recv,
                self._weights_queue,
                self._shm_names,
                self._num_workers,
                self._network_cfg,
                self._device_str,
                self._max_batch,
                self._max_wait_ms,
                self._log_interval_batches,
                self._log_file,
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
        # Wake the server loop.
        try:
            self._shutdown_send.send(None)
        except (BrokenPipeError, OSError):
            pass
        # Wake any worker still blocked on its reply pipe.
        for _, reply_send in self._reply_pipes:
            try:
                reply_send.send(None)
            except (BrokenPipeError, OSError):
                pass
        if self._process is not None:
            self._process.join(timeout=15)
            if self._process.is_alive():
                self._process.terminate()
        self._running = False
        self._cleanup_shm()

    def _cleanup_shm(self) -> None:
        for h in self._shm_handles:
            try:
                h.close()
            except Exception:
                pass
            try:
                h.unlink()
            except Exception:
                pass
        self._shm_handles = []

    def __del__(self) -> None:
        self._cleanup_shm()

    def worker_pipes(self, worker_id: int) -> tuple[mpc.Connection, mpc.Connection]:
        """Return (req_send, reply_recv) for *worker_id* — the worker's ends."""
        return self._req_pipes[worker_id][1], self._reply_pipes[worker_id][0]

    @property
    def shm_names(self) -> ShmNames:
        return self._shm_names

    @property
    def num_workers(self) -> int:
        return self._num_workers
