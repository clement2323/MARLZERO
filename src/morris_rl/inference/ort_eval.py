"""ONNX Runtime eval_fn factory for CPU inference.

Drop-in replacement for ``_make_local_eval_fn`` in :mod:`morris_rl.mcts.search`.
Loads an ONNX model exported by ``scripts/export_reversi_onnx.py`` (raw logits,
no mask inside the graph), applies the legal-action mask in numpy, and returns
the same ``(action→prob dict, value)`` signature expected by MCTS.

Usage inside reversi_server.py::

    from morris_rl.inference.ort_eval import make_ort_eval_fn
    eval_fn = make_ort_eval_fn(
        onnx_path="model.onnx",
        encode_fn=encode_state,
        get_legal_fn=get_legal_actions,
        action_space_size=65,
        num_threads=2,
    )
    search = MorrisSearch(eval_fn=eval_fn, game_fns=reversi_fns, num_simulations=200)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

try:
    import onnxruntime as ort

    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False


def make_ort_eval_fn(
    onnx_path: str,
    encode_fn: Callable[..., Any],
    get_legal_fn: Callable[..., list[int]],
    action_space_size: int,
    num_threads: int = 2,
) -> Callable[..., tuple[dict[int, float], float]]:
    """Build an eval_fn backed by ONNX Runtime.

    The ONNX model must have been exported by ``export_reversi_onnx.py``:
    - Input:  ``state`` of shape (batch, num_planes, num_positions)
    - Outputs: ``logits`` (batch, action_space_size), ``value`` (batch,)

    The legal-action mask is applied here in numpy (not inside the graph),
    then log_softmax is computed and converted to probabilities.

    Args:
        onnx_path: Path to the ``.onnx`` model file.
        encode_fn: State → torch.Tensor encoder (returns shape (1, P, N)).
        get_legal_fn: State → list[int] legal-action function.
        action_space_size: Total number of actions (e.g. 65 for Reversi).
        num_threads: ORT intra-op parallelism threads.  Keep ≤ 2 on HF Spaces
                     (limited CPU) to avoid contention with the MCTS Python threads.

    Returns:
        An EvalFn compatible with :class:`~morris_rl.mcts.search.MorrisSearch`.

    Raises:
        ImportError: If ``onnxruntime`` is not installed.
    """
    if not _ORT_AVAILABLE:
        raise ImportError(
            "onnxruntime is not installed. "
            "Install it with: pip install onnxruntime"
        )

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = num_threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    sess = ort.InferenceSession(
        onnx_path,
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )

    def evaluate(state: Any) -> tuple[dict[int, float], float]:
        legal = get_legal_fn(state)
        x_np = encode_fn(state).numpy()  # (1, planes, positions)

        logits, value = sess.run(None, {"state": x_np})
        # logits: (1, action_space_size),  value: (1,) or (1,1)
        logits_1d: np.ndarray = logits[0]  # (action_space_size,)

        # Apply legal mask: set illegal actions to -inf
        mask = np.full(action_space_size, -np.inf, dtype=np.float32)
        for a in legal:
            mask[a] = logits_1d[a]

        # Numerically stable log_softmax
        valid = np.isfinite(mask)
        log_probs = np.full(action_space_size, -np.inf, dtype=np.float32)
        if valid.any():
            m = mask[valid].max()
            log_sum = np.log(np.exp(mask[valid] - m).sum()) + m
            log_probs[valid] = mask[valid] - log_sum

        probs = np.exp(log_probs)
        action_probs = {a: float(probs[a]) for a in legal}

        scalar = float(value.ravel()[0])
        return action_probs, scalar

    return evaluate
