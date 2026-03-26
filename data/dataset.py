from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch.utils.data import Dataset


class FeatureDataset(Dataset):
    """
    Minimal dataset for already-extracted modality features.

    Users do their own preprocessing and feature extraction.
    This dataset only packages those features for training/evaluation.

    Expected:
      mod1_features: [N, T1, D1]
      mod2_features: [N, T2, D2]
      targets:       task-specific labels or values
    """
    def __init__(
        self,
        mod1_features: Optional[torch.Tensor] = None,
        mod2_features: Optional[torch.Tensor] = None,
        targets: Optional[Any] = None,
        mod1_present: Optional[torch.Tensor] = None,
        mod2_present: Optional[torch.Tensor] = None,
    ) -> None:
        self.mod1_features = mod1_features
        self.mod2_features = mod2_features
        self.targets = targets

        if mod1_features is None and mod2_features is None:
            raise ValueError("At least one modality must be provided.")

        n = mod1_features.shape[0] if mod1_features is not None else mod2_features.shape[0]

        self.mod1_present = torch.ones(n) if mod1_present is None else mod1_present.float()
        self.mod2_present = torch.ones(n) if mod2_present is None else mod2_present.float()

    def __len__(self) -> int:
        if self.mod1_features is not None:
            return self.mod1_features.shape[0]
        return self.mod2_features.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = {
            "mod1_x": None if self.mod1_features is None else self.mod1_features[idx],
            "mod2_x": None if self.mod2_features is None else self.mod2_features[idx],
            "mod1_present": self.mod1_present[idx],
            "mod2_present": self.mod2_present[idx],
        }

        if self.targets is not None:
            if isinstance(self.targets, dict):
                item["targets"] = {k: v[idx] for k, v in self.targets.items()}
            else:
                item["targets"] = self.targets[idx]

        return item
