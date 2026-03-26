from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional
import torch
import torch.nn as nn
from .backbone import DIVINEBackbone

@dataclass
class DIVINEOutput:
    predictions: Any
    features: torch.Tensor
    backbone_outputs: Dict[str, Any]

class DIVINEModel(nn.Module):
    def __init__(self, backbone: DIVINEBackbone, head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head
    def forward(self, mod1_x: Optional[torch.Tensor] = None, mod2_x: Optional[torch.Tensor] = None,
                mod1_present: Optional[torch.Tensor] = None, mod2_present: Optional[torch.Tensor] = None) -> DIVINEOutput:
        backbone_outputs = self.backbone(mod1_x=mod1_x, mod2_x=mod2_x, mod1_present=mod1_present, mod2_present=mod2_present)
        features = backbone_outputs["backbone_features"]
        return DIVINEOutput(predictions=self.head(features), features=features, backbone_outputs=backbone_outputs)
