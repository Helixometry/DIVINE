from __future__ import annotations
from typing import Dict
import torch
import torch.nn as nn

class IdentityHead(nn.Module):
    def forward(self, x: torch.Tensor):
        return x

class ClassificationHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, hidden_dim: int | None = None, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Linear(in_dim, num_classes) if hidden_dim is None else nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class RegressionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 1, hidden_dim: int | None = None, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Linear(in_dim, out_dim) if hidden_dim is None else nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, out_dim))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class MultiTaskHead(nn.Module):
    def __init__(self, heads: Dict[str, nn.Module]):
        super().__init__()
        self.heads = nn.ModuleDict(heads)
    def forward(self, x: torch.Tensor):
        return {name: head(x) for name, head in self.heads.items()}
