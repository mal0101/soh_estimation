"""Reproducibility utilities.

Seeds all random number generators used across the project: Python random,
NumPy, PyTorch (CPU and CUDA), and hash seeds. Call set_seed() at the
start of every training run for deterministic results.
"""

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 42, cuda_deterministic: bool = False) -> None:
    """Set random seeds for full reproducibility.

    Args:
        seed: The seed value to use across all generators.
        cuda_deterministic: If True, set torch.backends.cudnn.deterministic=True
            and torch.backends.cudnn.benchmark=False. This slows training but
            ensures CUDA reproducibility. Only enable for final runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if cuda_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def get_device() -> torch.device:
    """Detect the best available compute device.

    Priority: MPS (Apple Silicon) > CUDA > CPU.

    Returns:
        The optimal torch.device for training.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_worker(worker_id: int) -> None:
    """Seed a PyTorch DataLoader worker for reproducibility.

    Args:
        worker_id: The worker ID assigned by the DataLoader.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_generator(seed: int) -> torch.Generator:
    """Create a seeded torch.Generator for DataLoader workers.

    Args:
        seed: The seed value for the generator.

    Returns:
        A seeded torch.Generator instance.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g
