from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules import MLP, TemporalRefiner, TokenReasoner, UtteranceDisentangler, WindowVAE


@dataclass
class DIVINEConfig:
    mod1_in_dim: int
    mod2_in_dim: int
    refine_dim: int = 128
    window_latent_dim: int = 64
    shared_dim: int = 64
    private_dim: int = 64
    token_dim: int = 64
    num_tokens: int = 5
    dropout: float = 0.1

    use_local_vae: bool = True
    use_token_reasoner: bool = True
    use_cycle_loss: bool = True
    use_sparse_gating: bool = True

    beta_shared: float = 1.0
    beta_private: float = 1.0


class DIVINEBackbone(nn.Module):
    """
    Generic 2-modality DIVINE backbone.

    Users preprocess data themselves and provide extracted features:
      mod1_x: [B, T1, D1]
      mod2_x: [B, T2, D2]

    This class keeps the main DIVINE intuition fixed and returns:
    - fused backbone features
    - shared/private latents
    - auxiliary losses

    Users can attach any prediction head on top.
    """
    def __init__(self, cfg: DIVINEConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.mod1_refiner = TemporalRefiner(cfg.mod1_in_dim, cfg.refine_dim, dropout=cfg.dropout)
        self.mod2_refiner = TemporalRefiner(cfg.mod2_in_dim, cfg.refine_dim, dropout=cfg.dropout)

        self.mod1_window_vae = WindowVAE(cfg.refine_dim, cfg.window_latent_dim, dropout=cfg.dropout)
        self.mod2_window_vae = WindowVAE(cfg.refine_dim, cfg.window_latent_dim, dropout=cfg.dropout)

        self.utterance = UtteranceDisentangler(
            pooled_dim=cfg.window_latent_dim,
            shared_dim=cfg.shared_dim,
            private_dim=cfg.private_dim,
            beta_shared=cfg.beta_shared,
            beta_private=cfg.beta_private,
            dropout=cfg.dropout,
        )

        self.cross_decoder_1_to_2 = MLP(cfg.shared_dim, (cfg.shared_dim,), cfg.shared_dim, dropout=cfg.dropout)
        self.mod1_gate = nn.Linear(cfg.private_dim, cfg.shared_dim)
        self.mod2_gate = nn.Linear(cfg.private_dim, cfg.shared_dim)

        self.fused_to_token_dim = (
            nn.Identity()
            if cfg.shared_dim == cfg.token_dim
            else nn.Linear(cfg.shared_dim, cfg.token_dim)
        )
        self.token_reasoner = TokenReasoner(
            cfg.token_dim,
            cfg.num_tokens,
            hidden_dim=max(128, cfg.token_dim),
            dropout=cfg.dropout,
        )

    def _encode_modality(
        self,
        x: torch.Tensor,
        refiner: nn.Module,
        window_vae: nn.Module,
        modality_name: str,
    ) -> Dict[str, torch.Tensor]:
        refined = refiner(x)

        if self.cfg.use_local_vae:
            local = window_vae(refined)
            pooled = local["z_seq"].mean(dim=1)
        else:
            pooled = refined.mean(dim=1)
            zero = torch.tensor(0.0, device=x.device)
            local = {
                "z_seq": refined,
                "recon_seq": refined,
                "mu": refined,
                "logvar": refined,
                "loss_recon": zero,
                "loss_kl": zero,
                "loss_total": zero,
            }

        utter = self.utterance.forward_one(pooled, modality_name)

        return {
            "input": x,
            "refined": refined,
            "pooled": pooled,
            **{f"local_{k}": v for k, v in local.items()},
            **{f"utt_{k}": v for k, v in utter.items()},
        }

    def _local_and_utterance_loss(self, out: Dict[str, Any], device: torch.device) -> torch.Tensor:
        total = torch.tensor(0.0, device=device)
        if out.get("mod1") is not None:
            total = total + out["mod1"]["local_loss_total"] + out["mod1"]["utt_loss_total"]
        if out.get("mod2") is not None:
            total = total + out["mod2"]["local_loss_total"] + out["mod2"]["utt_loss_total"]
        return total

    def forward(
        self,
        mod1_x: Optional[torch.Tensor] = None,
        mod2_x: Optional[torch.Tensor] = None,
        mod1_present: Optional[torch.Tensor] = None,
        mod2_present: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        if mod1_x is None and mod2_x is None:
            raise ValueError("At least one modality must be provided.")

        device = mod1_x.device if mod1_x is not None else mod2_x.device
        batch_size = mod1_x.shape[0] if mod1_x is not None else mod2_x.shape[0]

        mod1_present = torch.ones(batch_size, 1, device=device) if mod1_present is None else mod1_present.float().view(batch_size, 1)
        mod2_present = torch.ones(batch_size, 1, device=device) if mod2_present is None else mod2_present.float().view(batch_size, 1)

        out: Dict[str, Any] = {}

        if mod1_x is not None:
            m1 = self._encode_modality(mod1_x, self.mod1_refiner, self.mod1_window_vae, "mod1")
            out["mod1"] = m1
            z1s, z1p = m1["utt_z_shared"], m1["utt_z_private"]
        else:
            out["mod1"] = None
            z1s = torch.zeros(batch_size, self.cfg.shared_dim, device=device)
            z1p = torch.zeros(batch_size, self.cfg.private_dim, device=device)

        if mod2_x is not None:
            m2 = self._encode_modality(mod2_x, self.mod2_refiner, self.mod2_window_vae, "mod2")
            out["mod2"] = m2
            z2s, z2p = m2["utt_z_shared"], m2["utt_z_private"]
        else:
            out["mod2"] = None
            z2s = torch.zeros(batch_size, self.cfg.shared_dim, device=device)
            z2p = torch.zeros(batch_size, self.cfg.private_dim, device=device)

        z1s, z1p = z1s * mod1_present, z1p * mod1_present
        z2s, z2p = z2s * mod2_present, z2p * mod2_present

        z2_hat_from_1 = self.cross_decoder_1_to_2(z1s)
        loss_cycle = F.mse_loss(z2_hat_from_1, z2s, reduction="mean") if self.cfg.use_cycle_loss else torch.tensor(0.0, device=device)

        if self.cfg.use_sparse_gating:
            g1 = torch.sigmoid(self.mod1_gate(z1p)) * mod1_present
            g2 = torch.sigmoid(self.mod2_gate(z2p)) * mod2_present
            h_fused = g1 * z1s + g2 * z2s
            loss_sparse = g1.abs().mean() + g2.abs().mean()
        else:
            g1 = mod1_present.expand(-1, self.cfg.shared_dim)
            g2 = mod2_present.expand(-1, self.cfg.shared_dim)
            h_fused = z1s + z2s
            loss_sparse = torch.tensor(0.0, device=device)

        if self.cfg.use_token_reasoner:
            token_out = self.token_reasoner(self.fused_to_token_dim(h_fused))
            backbone_features = token_out["h_fused_out"]
            loss_token = token_out["loss_token"]
        else:
            token_out = {}
            backbone_features = self.fused_to_token_dim(h_fused)
            loss_token = torch.tensor(0.0, device=device)

        out.update(
            {
                "z2_hat_from_1": z2_hat_from_1,
                "g_mod1": g1,
                "g_mod2": g2,
                "h_fused": h_fused,
                "backbone_features": backbone_features,
                "aux_losses": {
                    "cycle": loss_cycle,
                    "sparse": loss_sparse,
                    "token": loss_token,
                    "local_utt": self._local_and_utterance_loss(out, device),
                },
                "token_outputs": token_out,
            }
        )
        return out


class ClassificationHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, hidden_dim: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        if hidden_dim is None:
            self.net = nn.Linear(in_dim, num_classes)
        else:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DIVINEModel(nn.Module):
    """
    Backbone + prediction head wrapper.

    The backbone remains fixed.
    Users can replace the head with any custom task head.
    """
    def __init__(self, backbone: DIVINEBackbone, head: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(
        self,
        mod1_x: Optional[torch.Tensor] = None,
        mod2_x: Optional[torch.Tensor] = None,
        mod1_present: Optional[torch.Tensor] = None,
        mod2_present: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        backbone_outputs = self.backbone(
            mod1_x=mod1_x,
            mod2_x=mod2_x,
            mod1_present=mod1_present,
            mod2_present=mod2_present,
        )
        features = backbone_outputs["backbone_features"]
        predictions = self.head(features)
        return {
            "predictions": predictions,
            "features": features,
            "backbone_outputs": backbone_outputs,
        }
