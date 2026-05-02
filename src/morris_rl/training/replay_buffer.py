"""FIFO replay buffer with optional 8-fold dihedral symmetry augmentation.

Stores (encoded_state, policy_target, value_target) tuples produced by
self-play workers. Thread-safe for concurrent reads and writes.

With symmetry augmentation enabled (default), each game position is stored
as 8 symmetry-equivalent samples, effectively 8× the raw data volume.
The effective capacity (in raw positions) is therefore capacity // 8 when
augmentation is on.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch

from morris_rl.env.board import ACTION_SPACE_SIZE, NUM_POSITIONS
from morris_rl.env.symmetries import (
    SYMMETRY_PERMUTATIONS,
    transform_encoded_state,
    transform_policy,
)

_NUM_PLANES = 7


@dataclass
class SampleRecord:
    """One training sample from a self-play position."""

    encoded_state: npt.NDArray[np.float32]   # (7, 24)
    policy_target: npt.NDArray[np.float32]  # (ACTION_SPACE_SIZE,)
    value_target: float                     # in {-1.0, 0.0, 1.0}, current-player perspective
    legal_mask: npt.NDArray[np.bool_]       # (ACTION_SPACE_SIZE,) True on legal actions

    # Auxiliary supervision targets (KataGo-style aux heads). All scalar.
    # Defaults make the field backward-compatible when aux heads are disabled
    # — the trainer simply ignores them in compute_loss.
    aux_mill: float = 0.0           # number of mills the current player has on the board
    aux_pieces_diff: float = 0.0    # pieces_own − pieces_opp at game terminal (own POV)
    aux_capture: float = 0.0        # 1.0 if current player captures within next n_plies else 0.0


def _augment_sample(sample: SampleRecord) -> list[SampleRecord]:
    """Return the 7 non-identity symmetric variants of a sample.

    The legal_mask transforms exactly like the policy: a legal action under
    the original orientation maps to the same legal action under the rotated
    board (transform_policy is dtype-agnostic and works on bool arrays).

    Auxiliary scalars (mill / pieces_diff / capture) are *invariant* under
    spatial symmetry — they're global counts not tied to specific positions —
    so all 8 variants share the original sample's aux targets.
    """
    augmented = []
    for perm in SYMMETRY_PERMUTATIONS[1:]:
        augmented.append(
            SampleRecord(
                encoded_state=transform_encoded_state(sample.encoded_state, perm),
                policy_target=transform_policy(sample.policy_target, perm),
                value_target=sample.value_target,
                legal_mask=transform_policy(sample.legal_mask, perm),
                aux_mill=sample.aux_mill,
                aux_pieces_diff=sample.aux_pieces_diff,
                aux_capture=sample.aux_capture,
            )
        )
    return augmented


class ReplayBuffer:
    """Thread-safe FIFO replay buffer backed by a pre-allocated array.

    Old entries are evicted in FIFO order once capacity is reached.

    Args:
        capacity: Maximum number of samples to store. When augmentation is
            enabled, each raw position counts as 8 samples.
        use_symmetry_augmentation: If True, each added sample is stored
            alongside its 7 dihedral-symmetric variants.
    """

    def __init__(
        self,
        capacity: int,
        use_symmetry_augmentation: bool = True,
    ) -> None:
        self._capacity = capacity
        self._use_augmentation = use_symmetry_augmentation
        self._lock = threading.Lock()

        # Pre-allocate storage arrays for O(1) circular writes.
        self._states = np.zeros((capacity, _NUM_PLANES, NUM_POSITIONS), dtype=np.float32)
        self._policies = np.zeros((capacity, ACTION_SPACE_SIZE), dtype=np.float32)
        self._values = np.zeros(capacity, dtype=np.float32)
        # Legal-action mask, kept aligned with the trainer's masked log_softmax.
        # bool dtype = 1 byte/element; ~300 MB at capacity=500k. Could be packed
        # via np.packbits for ~8× compression if memory ever becomes tight.
        self._masks = np.zeros((capacity, ACTION_SPACE_SIZE), dtype=np.bool_)
        # Auxiliary scalar targets — three parallel float32 arrays. Each costs
        # ~2 MB at capacity=500k, total ~6 MB (negligible).
        self._aux_mill = np.zeros(capacity, dtype=np.float32)
        self._aux_pieces_diff = np.zeros(capacity, dtype=np.float32)
        self._aux_capture = np.zeros(capacity, dtype=np.float32)

        self._write_ptr = 0
        self._size = 0

    # ------------------------------------------------------------------
    # Write API
    # ------------------------------------------------------------------

    def add(self, sample: SampleRecord) -> None:
        """Add one sample (and its symmetric variants if augmentation is on)."""
        samples = [sample] + (_augment_sample(sample) if self._use_augmentation else [])
        with self._lock:
            for s in samples:
                self._write(s)

    def add_samples(self, samples: list[SampleRecord]) -> None:
        """Add a list of samples (e.g., all positions from one game)."""
        to_write: list[SampleRecord] = []
        for s in samples:
            to_write.append(s)
            if self._use_augmentation:
                to_write.extend(_augment_sample(s))
        with self._lock:
            for s in to_write:
                self._write(s)

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    def sample(
        self,
        batch_size: int,
        device: torch.device | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        """Return a random minibatch.

        Returns:
            (states, policies, values, masks, aux_targets) where aux_targets
            is a dict ``{"mill_count": ..., "pieces_diff_at_end": ...,
            "capture_in_n": ...}`` with each value of shape (batch_size,).

        Shapes:
            states:   (batch_size, 7, 24)
            policies: (batch_size, ACTION_SPACE_SIZE)
            values:   (batch_size,)
            masks:    (batch_size, ACTION_SPACE_SIZE)  bool

        Raises:
            ValueError: if the buffer contains fewer than batch_size samples.
        """
        with self._lock:
            if self._size < batch_size:
                raise ValueError(
                    f"Cannot sample {batch_size} items from buffer of size {self._size}."
                )
            indices = np.random.choice(self._size, size=batch_size, replace=False)
            states_np = self._states[indices].copy()
            policies_np = self._policies[indices].copy()
            values_np = self._values[indices].copy()
            masks_np = self._masks[indices].copy()
            aux_mill_np = self._aux_mill[indices].copy()
            aux_pieces_np = self._aux_pieces_diff[indices].copy()
            aux_capture_np = self._aux_capture[indices].copy()

        states = torch.from_numpy(states_np)
        policies = torch.from_numpy(policies_np)
        values = torch.from_numpy(values_np)
        masks = torch.from_numpy(masks_np)
        aux_targets = {
            "mill_count": torch.from_numpy(aux_mill_np),
            "pieces_diff_at_end": torch.from_numpy(aux_pieces_np),
            "capture_in_n": torch.from_numpy(aux_capture_np),
        }

        if device is not None:
            states = states.to(device)
            policies = policies.to(device)
            values = values.to(device)
            masks = masks.to(device)
            aux_targets = {k: t.to(device) for k, t in aux_targets.items()}

        return states, policies, values, masks, aux_targets

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        with self._lock:
            return self._size

    @property
    def capacity(self) -> int:
        return self._capacity

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | "Path") -> None:
        """Save buffer contents to *path* (npz). Only the live slice is stored."""
        from pathlib import Path
        with self._lock:
            np.savez_compressed(
                Path(path),
                states=self._states[: self._size],
                policies=self._policies[: self._size],
                values=self._values[: self._size],
                masks=self._masks[: self._size],
                aux_mill=self._aux_mill[: self._size],
                aux_pieces_diff=self._aux_pieces_diff[: self._size],
                aux_capture=self._aux_capture[: self._size],
                write_ptr=np.array([self._write_ptr], dtype=np.int64),
                size=np.array([self._size], dtype=np.int64),
                capacity=np.array([self._capacity], dtype=np.int64),
            )

    def load(self, path: str | "Path") -> None:
        """Restore buffer contents from a file written by :meth:`save`.

        Capacity must match. Stored entries are placed at the start of the
        circular buffer; the write pointer is restored so subsequent writes
        continue evicting in FIFO order.

        Backward compat: a buffer saved before legal_mask was introduced
        won't have a "masks" key — in that case we fall back to all-True
        masks (equivalent to the old full_mask behaviour for those samples).
        """
        from pathlib import Path
        data = np.load(Path(path))
        loaded_capacity = int(data["capacity"][0])
        if loaded_capacity != self._capacity:
            raise ValueError(
                f"Buffer capacity mismatch: file has {loaded_capacity}, "
                f"current buffer has {self._capacity}"
            )
        with self._lock:
            size = int(data["size"][0])
            self._states[:size] = data["states"]
            self._policies[:size] = data["policies"]
            self._values[:size] = data["values"]
            if "masks" in data.files:
                self._masks[:size] = data["masks"]
            else:
                self._masks[:size] = True
            # Aux arrays are also backward-compatible: a buffer saved before
            # aux heads existed has no aux_* keys, in which case we leave
            # the pre-allocated zeros — equivalent to "no aux supervision".
            for key, arr in (
                ("aux_mill", self._aux_mill),
                ("aux_pieces_diff", self._aux_pieces_diff),
                ("aux_capture", self._aux_capture),
            ):
                if key in data.files:
                    arr[:size] = data[key]
            self._size = size
            self._write_ptr = int(data["write_ptr"][0])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write(self, sample: SampleRecord) -> None:
        """Write one sample at the current write pointer (must hold lock)."""
        self._states[self._write_ptr] = sample.encoded_state
        self._policies[self._write_ptr] = sample.policy_target
        self._values[self._write_ptr] = sample.value_target
        self._masks[self._write_ptr] = sample.legal_mask
        self._aux_mill[self._write_ptr] = sample.aux_mill
        self._aux_pieces_diff[self._write_ptr] = sample.aux_pieces_diff
        self._aux_capture[self._write_ptr] = sample.aux_capture
        self._write_ptr = (self._write_ptr + 1) % self._capacity
        if self._size < self._capacity:
            self._size += 1
