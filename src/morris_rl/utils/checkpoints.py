"""Versioned checkpoint I/O.

Checkpoints store a version tag, the global training step, the full config
dict, and the model state dict.  The version field lets us detect and reject
stale checkpoints if the format ever changes.
"""

from pathlib import Path
from typing import Any

import torch

_CHECKPOINT_VERSION = 1


def save_checkpoint(
    path: str | Path,
    state_dict: dict[str, Any],
    config: dict[str, Any],
    step: int,
) -> None:
    """Persist a versioned checkpoint to disk.

    Args:
        path: Destination file path (conventionally ends in .pt).
        state_dict: Model (or optimizer) state dict to serialise.
        config: Full training config at the time of saving.
        step: Global training step counter at the time of saving.
    """
    payload: dict[str, Any] = {
        "version": _CHECKPOINT_VERSION,
        "step": step,
        "config": config,
        "state_dict": state_dict,
    }
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, dest)


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a checkpoint written by :func:`save_checkpoint`.

    Args:
        path: Path to the .pt checkpoint file.

    Returns:
        Dictionary with keys: ``version``, ``step``, ``config``, ``state_dict``.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the checkpoint version does not match the current format.
    """
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"Checkpoint not found: {src}")
    # Force CPU loading so checkpoints saved on a GPU machine still load in
    # CPU-only environments (HF Spaces, CI). The trainer / agent moves the
    # network onto its target device after loading state_dict, so mapping to
    # CPU here is the safe default everywhere.
    payload: dict[str, Any] = torch.load(src, weights_only=False, map_location="cpu")
    if payload.get("version") != _CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint version {payload.get('version')!r}; "
            f"expected {_CHECKPOINT_VERSION}"
        )
    return payload


__all__ = ["save_checkpoint", "load_checkpoint"]
