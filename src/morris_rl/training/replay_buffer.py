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

from morris_rl.env.board import ACTION_SPACE_SIZE as _MORRIS_ACTION_SPACE_SIZE
from morris_rl.env.board import NUM_POSITIONS as _MORRIS_NUM_POSITIONS
from morris_rl.env.symmetries import (
    SYMMETRY_PERMUTATIONS as _MORRIS_SYMMETRY_PERMUTATIONS,
    transform_encoded_state as _morris_transform_encoded_state,
    transform_policy as _morris_transform_policy,
)

_DEFAULT_NUM_PLANES = 7

AugmentFn = "Callable[[SampleRecord], list[SampleRecord]]"


@dataclass
class SampleRecord:
    """One training sample from a self-play position.

    Auxiliary targets (mill_diff_target, pieces_diff_target) are signed and
    measured from the *current player's* perspective at sample creation time.
    Default NaN means "no aux supervision for this sample" — the trainer
    masks it out of the aux loss so the sample still contributes to policy
    and value losses.
    """

    encoded_state: npt.NDArray[np.float32]   # (num_planes, num_positions)
    policy_target: npt.NDArray[np.float32]   # (action_space_size,)
    value_target: float                      # in {-1.0, 0.0, 1.0}, current-player perspective
    legal_mask: npt.NDArray[np.bool_]        # (action_space_size,) True on legal actions
    mill_diff_target: float = float("nan")   # own_mills - opp_mills (current-player view)
    pieces_diff_target: float = float("nan") # own_pieces - opp_pieces (current-player view)


def _morris_augment_sample(sample: SampleRecord) -> list[SampleRecord]:
    """Return the 7 non-identity Morris D4 symmetric variants of a sample.

    mill_diff and pieces_diff are scalar invariants of the D4 group on the
    board layout, so we copy them unchanged across all symmetric variants.
    """
    return [
        SampleRecord(
            encoded_state=_morris_transform_encoded_state(sample.encoded_state, perm),
            policy_target=_morris_transform_policy(sample.policy_target, perm),
            value_target=sample.value_target,
            legal_mask=_morris_transform_policy(sample.legal_mask, perm),
            mill_diff_target=sample.mill_diff_target,
            pieces_diff_target=sample.pieces_diff_target,
        )
        for perm in _MORRIS_SYMMETRY_PERMUTATIONS[1:]
    ]


class ReplayBuffer:
    """Thread-safe FIFO replay buffer backed by a pre-allocated array.

    Old entries are evicted in FIFO order once capacity is reached.

    Args:
        capacity: Maximum number of samples to store. When augmentation is
            enabled, each raw position counts as 8 samples.
        use_symmetry_augmentation: If True, each added sample is stored
            alongside its 7 dihedral-symmetric variants.
        num_planes: Number of input planes in the encoded state tensor.
        num_positions: Number of board positions (game-specific).
        action_space_size: Total number of actions (game-specific).
        augment_fn: Custom augmentation function. Defaults to Morris D4
            symmetry when None and ``use_symmetry_augmentation`` is True.
    """

    def __init__(
        self,
        capacity: int,
        use_symmetry_augmentation: bool = True,
        num_planes: int = _DEFAULT_NUM_PLANES,
        num_positions: int = _MORRIS_NUM_POSITIONS,
        action_space_size: int = _MORRIS_ACTION_SPACE_SIZE,
        augment_fn: "Callable[[SampleRecord], list[SampleRecord]] | None" = None,
    ) -> None:
        self._capacity = capacity
        self._use_augmentation = use_symmetry_augmentation
        self._num_planes = num_planes
        self._augment_fn = augment_fn if augment_fn is not None else _morris_augment_sample
        self._lock = threading.Lock()

        self._states = np.zeros((capacity, self._num_planes, num_positions), dtype=np.float32)
        self._policies = np.zeros((capacity, action_space_size), dtype=np.float32)
        self._values = np.zeros(capacity, dtype=np.float32)
        self._masks = np.zeros((capacity, action_space_size), dtype=np.bool_)
        # NaN-initialised so any old buffer loaded without aux fields is
        # correctly treated as "no aux supervision" by the trainer.
        self._mill_diff = np.full(capacity, np.nan, dtype=np.float32)
        self._pieces_diff = np.full(capacity, np.nan, dtype=np.float32)

        self._write_ptr = 0
        self._size = 0

    # ------------------------------------------------------------------
    # Write API
    # ------------------------------------------------------------------

    def add(self, sample: SampleRecord) -> None:
        """Add one sample (and its symmetric variants if augmentation is on)."""
        samples = [sample] + (self._augment_fn(sample) if self._use_augmentation else [])
        with self._lock:
            for s in samples:
                self._write(s)

    def add_samples(self, samples: list[SampleRecord]) -> None:
        """Add a list of samples (e.g., all positions from one game)."""
        to_write: list[SampleRecord] = []
        for s in samples:
            to_write.append(s)
            if self._use_augmentation:
                to_write.extend(self._augment_fn(s))
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
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Return a random minibatch.

        Returns:
            ``(states, policies, values, masks, mill_diff, pieces_diff)``.
            The last two are NaN-padded when aux targets are absent; the
            trainer masks NaN entries out of the aux loss.

        Shapes:
            states:      (batch_size, num_planes, num_positions)
            policies:    (batch_size, ACTION_SPACE_SIZE)
            values:      (batch_size,)
            masks:       (batch_size, ACTION_SPACE_SIZE)  bool
            mill_diff:   (batch_size,)  float32, may contain NaN
            pieces_diff: (batch_size,)  float32, may contain NaN

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
            mill_np = self._mill_diff[indices].copy()
            pieces_np = self._pieces_diff[indices].copy()

        states = torch.from_numpy(states_np)
        policies = torch.from_numpy(policies_np)
        values = torch.from_numpy(values_np)
        masks = torch.from_numpy(masks_np)
        mill_diff = torch.from_numpy(mill_np)
        pieces_diff = torch.from_numpy(pieces_np)

        if device is not None:
            states = states.to(device)
            policies = policies.to(device)
            values = values.to(device)
            masks = masks.to(device)
            mill_diff = mill_diff.to(device)
            pieces_diff = pieces_diff.to(device)

        return states, policies, values, masks, mill_diff, pieces_diff

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
                mill_diff=self._mill_diff[: self._size],
                pieces_diff=self._pieces_diff[: self._size],
                write_ptr=np.array([self._write_ptr], dtype=np.int64),
                size=np.array([self._size], dtype=np.int64),
                capacity=np.array([self._capacity], dtype=np.int64),
            )

    def load(self, path: str | "Path") -> None:
        """Restore buffer contents from a file written by :meth:`save`.

        Capacity must match. Backward compat: files saved before the legal_mask
        field existed fall back to all-True masks; files saved before aux
        targets existed fall back to NaN (excluded from aux loss).
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
            if "mill_diff" in data.files:
                self._mill_diff[:size] = data["mill_diff"]
            else:
                self._mill_diff[:size] = np.nan
            if "pieces_diff" in data.files:
                self._pieces_diff[:size] = data["pieces_diff"]
            else:
                self._pieces_diff[:size] = np.nan
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
        self._mill_diff[self._write_ptr] = sample.mill_diff_target
        self._pieces_diff[self._write_ptr] = sample.pieces_diff_target
        self._write_ptr = (self._write_ptr + 1) % self._capacity
        if self._size < self._capacity:
            self._size += 1
