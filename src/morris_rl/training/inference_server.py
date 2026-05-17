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

_NUM_PLANES = 7


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
        value_head_type=network_cfg.get("value_head_type", "scalar"),
        aux_heads_enabled=bool(network_cfg.get("aux_heads_enabled", False)),
        aux_head_hidden=int(network_cfg.get("aux_head_hidden", 64)),
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
# CUDA-graphed forward (Tier 2.1)
# ---------------------------------------------------------------------------


def _bucket_sizes_for(max_batch: int) -> list[int]:
    """Powers of two up to max_batch (inclusive). Picked to balance memory
    vs padding waste — at most we round up by 2× to the next bucket."""
    sizes: list[int] = []
    b = 1
    while b < max_batch:
        sizes.append(b)
        b *= 2
    sizes.append(max_batch)
    return sizes


class GraphedForward:
    """Replays a CUDA-captured forward pass per bucketed batch size.

    Why CUDA Graphs here
    --------------------
    With a 16×192 ResNet at small batches (mean=6 in our trace), each forward
    issues ~80 CUDA kernels. The launch overhead floor (~5–10 µs/kernel) is
    a non-trivial fraction of the 1.5–2 ms forward, and serialises with
    Python-side bookkeeping. ``cudaGraph`` collapses all those launches into
    a single replay → fewer host↔device round trips, tighter inter-batch gap.

    Bucketing
    ---------
    CUDA Graphs require static shapes, but our batch size varies (1..max).
    We capture one graph per power-of-two bucket and pad smaller batches with
    zeros + an all-True mask in the unused rows. The padding is wasted compute
    but bounded to <2× compared to the snug batch.

    Weight updates
    --------------
    Captured graphs reference the *memory addresses* of the network's
    parameters. ``load_state_dict`` writes in-place by default, so the next
    replay automatically picks up fresh weights — no recapture needed.
    """

    def __init__(
        self,
        network: torch.nn.Module,
        max_batch: int,
        num_planes: int,
        action_space: int,
        device: torch.device,
    ) -> None:
        self._network = network
        self._device = device
        self._buckets = _bucket_sizes_for(max_batch)

        # Pinned host staging for inputs — built once, copied into the bucket's
        # static GPU tensor on each call.
        self._host_states = torch.empty(
            (max_batch, num_planes, 24), dtype=torch.float32, pin_memory=True
        )
        self._host_mask = torch.empty(
            (max_batch, action_space), dtype=torch.bool, pin_memory=True
        )

        # Per-bucket static GPU tensors and graph handles.
        self._static_states: dict[int, torch.Tensor] = {}
        self._static_mask: dict[int, torch.Tensor] = {}
        self._static_log_policy: dict[int, torch.Tensor] = {}
        self._static_value: dict[int, torch.Tensor] = {}
        self._graphs: dict[int, torch.cuda.CUDAGraph] = {}

        for b in self._buckets:
            self._static_states[b] = torch.zeros(
                (b, num_planes, 24), dtype=torch.float32, device=device
            )
            # Mask must be all-True in padding rows so the policy head's
            # masked log-softmax doesn't divide by zero.
            self._static_mask[b] = torch.ones(
                (b, action_space), dtype=torch.bool, device=device
            )

        self._capture()

    def _capture(self) -> None:
        # Warm cudnn / autotune on a side stream so the captured graph reuses
        # the selected algos. Required by torch.cuda.graph contract.
        s = torch.cuda.Stream(device=self._device)
        s.wait_stream(torch.cuda.current_stream(self._device))
        with torch.cuda.stream(s):
            for b in self._buckets:
                with (
                    torch.no_grad(),
                    torch.cuda.amp.autocast(dtype=torch.float16),
                ):
                    for _ in range(3):  # 3 warmup passes per bucket
                        _ = self._network(
                            self._static_states[b], self._static_mask[b]
                        )
        torch.cuda.current_stream(self._device).wait_stream(s)
        torch.cuda.synchronize(self._device)

        for b in self._buckets:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                with (
                    torch.no_grad(),
                    torch.cuda.amp.autocast(dtype=torch.float16),
                ):
                    log_policy, value = self._network(
                        self._static_states[b], self._static_mask[b]
                    )
                self._static_log_policy[b] = log_policy
                self._static_value[b] = value
            self._graphs[b] = graph

    def _bucket_for(self, n: int) -> int:
        for b in self._buckets:
            if b >= n:
                return b
        return self._buckets[-1]  # unreachable: n is bounded by max_batch

    def forward(
        self,
        request_states: np.ndarray,
        worker_ids: list[int],
        legal_actions_per_req: list[tuple[int, ...]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run forward on a packed batch and return (probs, values) as numpy."""
        n = len(worker_ids)
        bucket = self._bucket_for(n)

        # Stage inputs in pinned host buffers (CPU-side fill, async H2D copy).
        host_states_view = self._host_states.numpy()
        host_mask_view = self._host_mask.numpy()
        for i, wid in enumerate(worker_ids):
            host_states_view[i] = request_states[wid]
        host_mask_view[:n] = False
        for i, legal in enumerate(legal_actions_per_req):
            if legal:
                host_mask_view[i, list(legal)] = True
            else:
                # Terminal state slipped past MCTS: an all-False mask makes
                # masked log_softmax NaN. Release the constraint — the worker
                # iterates only over its own (empty) legal list, so the bogus
                # probs are never read; we just need the row to stay finite
                # to avoid poisoning the rest of the batch.
                host_mask_view[i, :] = True

        static_s = self._static_states[bucket]
        static_m = self._static_mask[bucket]
        static_s[:n].copy_(self._host_states[:n], non_blocking=True)
        if n < bucket:
            static_s[n:].zero_()
        static_m[:n].copy_(self._host_mask[:n], non_blocking=True)
        if n < bucket:
            static_m[n:].fill_(True)  # all-True so masked softmax is finite

        self._graphs[bucket].replay()

        # Outputs live in static_log_policy/static_value — clone+truncate then
        # bring back to CPU. .float() upcasts FP16 logits before exp() to keep
        # the existing numpy contract identical to the pre-graph code path.
        probs = self._static_log_policy[bucket][:n].float().exp().cpu().numpy()
        values = self._static_value[bucket][:n].float().cpu().numpy()
        return probs, values


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

    use_cuda = device.type == "cuda"

    # Build the CUDA-graphed forward once on GPU. CPU fallback uses a plain
    # eager forward (no graphs) — used for tests / smoke runs without a GPU.
    if use_cuda:
        graphed_forward = GraphedForward(
            network=network,
            max_batch=max_batch,
            num_planes=_NUM_PLANES,
            action_space=ACTION_SPACE_SIZE,
            device=device,
        )
    else:
        graphed_forward = None

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
                legal_per_req = [req.legal_actions for req in batch]
                n = len(batch)

                t0 = time.perf_counter()
                if use_cuda:
                    assert graphed_forward is not None
                    probs, values = graphed_forward.forward(
                        request_states, worker_ids, legal_per_req
                    )
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
                        else:
                            # Same defensive fallback as the CUDA-graph path:
                            # all-False rows make masked log_softmax NaN.
                            mask[i, :] = True
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
    get_legal_actions_no_rep: Any = None,
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
        if not legal and get_legal_actions_no_rep is not None:
            # Empty set means every movement candidate matched a recent position
            # (no-rep window). Fall back to rule-legal moves only — i.e. release
            # the repetition filter, not the rule-level legality. The state is
            # still terminal per is_terminal (piece-count tie-break) but MCTS
            # may visit it transiently and needs a non-empty prior support.
            legal = get_legal_actions_no_rep(state)
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
