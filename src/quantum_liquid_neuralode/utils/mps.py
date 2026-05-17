from __future__ import annotations

import torch


def select_device(*, prefer_mps: bool = True) -> torch.device:
    """Select a torch device.

    Preference order:
    1) MPS (Apple Silicon) if prefer_mps=True and available
    2) CUDA if available
    3) CPU
    """
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
