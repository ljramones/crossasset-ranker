"""Reproducibility helpers."""

from __future__ import annotations

import random

import numpy as np


def seed_everything(seed: int) -> None:
    """Set random seeds across the libraries used in this project."""

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:  # pragma: no cover
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

