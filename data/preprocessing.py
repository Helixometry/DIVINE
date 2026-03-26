"""
This file is intentionally lightweight.

Users are expected to preprocess their raw data and extract features themselves.
Add project-specific feature loading, normalization, padding, or masking here if needed.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def standardize_features(
    x: torch.Tensor,
    mean: Optional[torch.Tensor] = None,
    std: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if mean is None:
        mean = x.mean(dim=(0, 1), keepdim=True)
    if std is None:
        std = x.std(dim=(0, 1), keepdim=True)
    x_norm = (x - mean) / (std + eps)
    return x_norm, mean, std
