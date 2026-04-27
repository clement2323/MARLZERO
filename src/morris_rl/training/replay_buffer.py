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

_NUM_PLANES = 8


@dataclass
class SampleRecord:
    """One training sample from a self-play position."""

    encoded_state: npt.NDArray[np.float32]   # (8, 24)
    policy_target: npt.NDArray[np.float32]  # (ACTION_SPACE_SIZE,)
    value_target: float                     # in {-1.0, 0.0, 1.0}, current-player perspective


def _augment_sample(sample: SampleRecord) -> list[SampleRecord]:
    """Return the 7 non-identity symmetric variants of a sample."""
    augmented = []
    for perm in SYMMETRY_PERMUTATIONS[1:]:
        augmented.append(
            SampleRecord(
                encoded_state=transform_encoded_state(sample.encoded_state, perm),
                policy_target=transform_policy(sample.policy_target, perm),
                value_target=sample.value_target,
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return a random minibatch as (states, policies, values) tensors.

        Shapes:
            states:   (batch_size, 8, 24)
            policies: (batch_size, ACTION_SPACE_SIZE)
            values:   (batch_size,)

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

        states = torch.from_numpy(states_np)
        policies = torch.from_numpy(policies_np)
        values = torch.from_numpy(values_np)

        if device is not None:
            states = states.to(device)
            policies = policies.to(device)
            values = values.to(device)

        return states, policies, values

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
                write_ptr=np.array([self._write_ptr], dtype=np.int64),
                size=np.array([self._size], dtype=np.int64),
                capacity=np.array([self._capacity], dtype=np.int64),
            )

    def load(self, path: str | "Path") -> None:
        """Restore buffer contents from a file written by :meth:`save`.

        Capacity must match. Stored entries are placed at the start of the
        circular buffer; the write pointer is restored so subsequent writes
        continue evicting in FIFO order.
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
        self._write_ptr = (self._write_ptr + 1) % self._capacity
        if self._size < self._capacity:
            self._size += 1
