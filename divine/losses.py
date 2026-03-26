from __future__ import annotations
from typing import Any, Dict, Mapping
import torch
import torch.nn.functional as F

def compute_prediction_loss(predictions: Any, targets: Any, loss_type: str = "cross_entropy") -> torch.Tensor:
    if loss_type == "cross_entropy":
        return F.cross_entropy(predictions, targets)
    if loss_type == "binary_cross_entropy":
        return F.binary_cross_entropy_with_logits(predictions, targets.float())
    if loss_type == "mse":
        return F.mse_loss(predictions, targets.float())
    if loss_type == "l1":
        return F.l1_loss(predictions, targets.float())
    raise ValueError(f"Unsupported loss_type: {loss_type}")

def compute_auxiliary_loss(aux_losses: Mapping[str, torch.Tensor], weights: Mapping[str, float] | None = None) -> torch.Tensor:
    if weights is None:
        weights = {"local_utt": 1.0, "cycle": 0.1, "sparse": 0.1, "token": 0.04}
    total = 0.0
    for name, value in aux_losses.items():
        total = total + weights.get(name, 1.0) * value
    return total

def compute_total_loss(prediction_loss: torch.Tensor | Dict[str, torch.Tensor], aux_loss: torch.Tensor | None = None,
                       task_weights: Mapping[str, float] | None = None) -> torch.Tensor:
    if isinstance(prediction_loss, dict):
        task_weights = {k: 1.0 for k in prediction_loss} if task_weights is None else task_weights
        pred_total = 0.0
        for name, value in prediction_loss.items():
            pred_total = pred_total + task_weights.get(name, 1.0) * value
    else:
        pred_total = prediction_loss
    return pred_total if aux_loss is None else pred_total + aux_loss
