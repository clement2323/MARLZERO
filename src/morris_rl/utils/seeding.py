"""Reproducible seeding utilities."""

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Set all random seeds for full reproducibility.

    Pins cuDNN to deterministic mode — this may reduce throughput slightly
    but guarantees bit-identical results across runs with the same seed.

    Args:
        seed: Integer seed value. Must fit in a 32-bit unsigned integer.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


__all__ = ["seed_everything"]
