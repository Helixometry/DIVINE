"""
DIVINE: reference PyTorch implementation from the paper
"DIVINE: Coordinating Multimodal Disentangled Representations for Oro-Facial Neurological Disorder Assessment"

Notes
-----
This script implements the architecture described in the uploaded paper:
- temporal refinement with 1D CNN blocks
- local/window VAE per modality
- global average pooling
- utterance-level shared/private disentanglement with tied shared encoder
- cross-modal alignment (video shared -> audio shared)
- sparse gated fusion using private latents
- learnable symptom tokens + dense reasoning
- multitask heads for diagnosis and severity

Because the paper does not fully specify a few low-level details
(e.g., exact utterance reconstruction decoder shape and the exact form of token loss),
this implementation uses reasonable, explicit choices:
- utterance reconstruction decodes concat(shared, private) back to the pooled utterance vector
- token regularization is implemented as token reconstruction MSE
- both diagnosis and severity are treated as categorical outputs

This file includes a random-input smoke test in `main()`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Utilities
# -----------------------------

def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


def kl_divergence_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    # mean over batch
    return 0.5 * torch.mean(torch.sum(torch.exp(logvar) + mu**2 - 1.0 - logvar, dim=-1))


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: Tuple[int, ...],
        out_dim: int,
        dropout: float = 0.1,
        activation: nn.Module = nn.ReLU,
        out_activation: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        dims = [in_dim, *hidden_dims, out_dim]
        layers = []
        for i in range(len(dims) - 2):
            layers += [
                nn.Linear(dims[i], dims[i + 1]),
                activation(),
                nn.Dropout(dropout),
            ]
        layers.append(nn.Linear(dims[-2], dims[-1]))
        if out_activation is not None:
            layers.append(out_activation())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -----------------------------
# Temporal refinement
# -----------------------------

class TemporalRefiner(nn.Module):
    """
    Input : [B, T, D_in]
    Output: [B, T_out, D_hidden]
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        kernel_size: int = 3,
        pool_kernel: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_dim, hidden_dim, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_kernel, stride=pool_kernel),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D] -> [B, D, T]
        x = x.transpose(1, 2)
        x = self.block(x)
        x = x.transpose(1, 2)  # [B, T_out, D_hidden]
        return x


# -----------------------------
# Local / window VAE
# -----------------------------

class WindowVAE(nn.Module):
    """
    Applies the same small VAE independently at each time step of a refined sequence.
    Input : [B, T, D_in]
    Output:
      z_seq       : [B, T, D_latent]
      recon_seq   : [B, T, D_in]
      mu/logvar   : [B, T, D_latent]
      loss dict
    """
    def __init__(
        self,
        in_dim: int,
        latent_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.enc_mu = MLP(in_dim, (hidden_dim,), latent_dim, dropout=dropout)
        self.enc_logvar = MLP(in_dim, (hidden_dim,), latent_dim, dropout=dropout)
        self.dec = MLP(latent_dim, (hidden_dim,), in_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        b, t, d = x.shape
        x_flat = x.reshape(b * t, d)

        mu = self.enc_mu(x_flat).reshape(b, t, -1)
        logvar = self.enc_logvar(x_flat).reshape(b, t, -1)
        z = reparameterize(mu, logvar)

        z_flat = z.reshape(b * t, -1)
        recon = self.dec(z_flat).reshape(b, t, d)

        recon_loss = F.mse_loss(recon, x, reduction="mean")
        kl_loss = kl_divergence_standard_normal(
            mu.reshape(b * t, -1), logvar.reshape(b * t, -1)
        )
        total = recon_loss + kl_loss

        return {
            "z_seq": z,
            "recon_seq": recon,
            "mu": mu,
            "logvar": logvar,
            "loss_recon": recon_loss,
            "loss_kl": kl_loss,
            "loss_total": total,
        }


# -----------------------------
# Utterance-level disentanglement
# -----------------------------

class GaussianEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        latent_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.mu = MLP(in_dim, (hidden_dim,), latent_dim, dropout=dropout)
        self.logvar = MLP(in_dim, (hidden_dim,), latent_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = reparameterize(mu, logvar)
        return z, mu, logvar


class UtteranceDisentangler(nn.Module):
    """
    Shared encoder is intended to be weight-tied across modalities.
    Reconstruction decodes concat(shared, private) back to pooled utterance embedding.
    """
    def __init__(
        self,
        pooled_dim: int,
        shared_dim: int,
        private_dim: int,
        hidden_dim: int = 128,
        beta_shared: float = 1.0,
        beta_private: float = 1.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.beta_shared = beta_shared
        self.beta_private = beta_private

        self.shared_encoder = GaussianEncoder(
            pooled_dim, shared_dim, hidden_dim=hidden_dim, dropout=dropout
        )
        self.private_encoder_video = GaussianEncoder(
            pooled_dim, private_dim, hidden_dim=hidden_dim, dropout=dropout
        )
        self.private_encoder_audio = GaussianEncoder(
            pooled_dim, private_dim, hidden_dim=hidden_dim, dropout=dropout
        )

        self.recon_video = MLP(shared_dim + private_dim, (hidden_dim,), pooled_dim, dropout=dropout)
        self.recon_audio = MLP(shared_dim + private_dim, (hidden_dim,), pooled_dim, dropout=dropout)

    def forward_one(
        self, pooled_x: torch.Tensor, modality: str
    ) -> Dict[str, torch.Tensor]:
        z_shared, mu_s, logvar_s = self.shared_encoder(pooled_x)

        if modality == "video":
            z_private, mu_p, logvar_p = self.private_encoder_video(pooled_x)
            recon = self.recon_video(torch.cat([z_shared, z_private], dim=-1))
        elif modality == "audio":
            z_private, mu_p, logvar_p = self.private_encoder_audio(pooled_x)
            recon = self.recon_audio(torch.cat([z_shared, z_private], dim=-1))
        else:
            raise ValueError(f"Unknown modality: {modality}")

        recon_loss = F.mse_loss(recon, pooled_x, reduction="mean")
        kl_s = kl_divergence_standard_normal(mu_s, logvar_s)
        kl_p = kl_divergence_standard_normal(mu_p, logvar_p)
        total = recon_loss + self.beta_shared * kl_s + self.beta_private * kl_p

        return {
            "z_shared": z_shared,
            "mu_shared": mu_s,
            "logvar_shared": logvar_s,
            "z_private": z_private,
            "mu_private": mu_p,
            "logvar_private": logvar_p,
            "recon": recon,
            "loss_recon": recon_loss,
            "loss_kl_shared": kl_s,
            "loss_kl_private": kl_p,
            "loss_total": total,
        }


# -----------------------------
# Token reasoning
# -----------------------------

class TokenDenseReasoner(nn.Module):
    """
    S = [T1, ..., TK, h_fused] in R[(K+1), ds]
    H_out = Dense(S)
    We implement Dense as row-wise MLP with residual norm.
    """
    def __init__(
        self,
        token_dim: int,
        num_tokens: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.tokens = nn.Parameter(torch.randn(num_tokens, token_dim) * 0.02)

        self.row_mlp = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, token_dim),
        )
        self.norm = nn.LayerNorm(token_dim)

    def forward(self, fused_h: torch.Tensor) -> Dict[str, torch.Tensor]:
        b, d = fused_h.shape
        token_bank = self.tokens.unsqueeze(0).expand(b, -1, -1)  # [B, K, D]
        fused_row = fused_h.unsqueeze(1)  # [B, 1, D]
        s = torch.cat([token_bank, fused_row], dim=1)  # [B, K+1, D]

        h_out = self.norm(s + self.row_mlp(s))
        h_fused_out = h_out[:, -1, :]  # final row per paper

        # Token reconstruction / specialization regularization:
        # keep transformed token rows close to the learned token templates.
        token_recon_loss = F.mse_loss(h_out[:, :-1, :], token_bank, reduction="mean")

        return {
            "S": s,
            "Hout": h_out,
            "h_fused_out": h_fused_out,
            "loss_token": token_recon_loss,
        }


# -----------------------------
# DIVINE
# -----------------------------

@dataclass
class DIVINEConfig:
    video_in_dim: int
    audio_in_dim: int
    refine_dim: int = 128
    window_latent_dim: int = 64
    shared_dim: int = 64
    private_dim: int = 64
    token_dim: int = 64
    num_symptom_tokens: int = 5
    num_classes: int = 3
    num_severity_levels: int = 3

    # loss hyperparameters from the paper
    alpha_severity: float = 2.0
    eps_regularization: float = 0.1
    lambda_token: float = 0.4

    beta_shared: float = 1.0
    beta_private: float = 1.0

    dropout: float = 0.1


class DIVINE(nn.Module):
    def __init__(self, cfg: DIVINEConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.video_refiner = TemporalRefiner(
            in_dim=cfg.video_in_dim,
            hidden_dim=cfg.refine_dim,
            dropout=cfg.dropout,
        )
        self.audio_refiner = TemporalRefiner(
            in_dim=cfg.audio_in_dim,
            hidden_dim=cfg.refine_dim,
            dropout=cfg.dropout,
        )

        self.video_window_vae = WindowVAE(
            in_dim=cfg.refine_dim,
            latent_dim=cfg.window_latent_dim,
            dropout=cfg.dropout,
        )
        self.audio_window_vae = WindowVAE(
            in_dim=cfg.refine_dim,
            latent_dim=cfg.window_latent_dim,
            dropout=cfg.dropout,
        )

        self.utterance = UtteranceDisentangler(
            pooled_dim=cfg.window_latent_dim,
            shared_dim=cfg.shared_dim,
            private_dim=cfg.private_dim,
            beta_shared=cfg.beta_shared,
            beta_private=cfg.beta_private,
            dropout=cfg.dropout,
        )

        # cross-modal decoder: video-shared -> audio-shared
        self.cross_decoder_v_to_a = MLP(
            cfg.shared_dim, (cfg.shared_dim,), cfg.shared_dim, dropout=cfg.dropout
        )

        # sparse gated fusion
        self.video_gate = nn.Linear(cfg.private_dim, cfg.shared_dim)
        self.audio_gate = nn.Linear(cfg.private_dim, cfg.shared_dim)

        self.token_reasoner = TokenDenseReasoner(
            token_dim=cfg.token_dim,
            num_tokens=cfg.num_symptom_tokens,
            hidden_dim=max(128, cfg.token_dim),
            dropout=cfg.dropout,
        )

        # if shared_dim != token_dim, project fused latent to token_dim
        self.fused_to_token_dim = (
            nn.Identity() if cfg.shared_dim == cfg.token_dim
            else nn.Linear(cfg.shared_dim, cfg.token_dim)
        )

        self.cls_head = nn.Linear(cfg.token_dim, cfg.num_classes)
        self.sev_head = nn.Linear(cfg.token_dim, cfg.num_severity_levels)

    def _encode_modality(
        self,
        x: torch.Tensor,
        refiner: nn.Module,
        window_vae: nn.Module,
        modality: str,
    ) -> Dict[str, torch.Tensor]:
        refined = refiner(x)                               # [B, T', D_ref]
        local = window_vae(refined)                       # z_seq [B, T'', D_w]
        pooled = local["z_seq"].mean(dim=1)               # global average pooling
        utter = self.utterance.forward_one(pooled, modality=modality)

        return {
            "input": x,
            "refined": refined,
            "pooled": pooled,
            **{f"local_{k}": v for k, v in local.items()},
            **{f"utt_{k}": v for k, v in utter.items()},
        }

    def forward(
        self,
        video_x: Optional[torch.Tensor] = None,
        audio_x: Optional[torch.Tensor] = None,
        video_present: Optional[torch.Tensor] = None,
        audio_present: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        if video_x is None and audio_x is None:
            raise ValueError("At least one modality must be provided.")

        device = video_x.device if video_x is not None else audio_x.device
        batch_size = video_x.shape[0] if video_x is not None else audio_x.shape[0]

        if video_present is None:
            video_present = torch.ones(batch_size, 1, device=device)
        else:
            video_present = video_present.float().view(batch_size, 1)

        if audio_present is None:
            audio_present = torch.ones(batch_size, 1, device=device)
        else:
            audio_present = audio_present.float().view(batch_size, 1)

        out: Dict[str, Any] = {}

        if video_x is not None:
            v = self._encode_modality(video_x, self.video_refiner, self.video_window_vae, "video")
            out["video"] = v
            z_v_shared = v["utt_z_shared"]
            z_v_private = v["utt_z_private"]
        else:
            z_v_shared = torch.zeros(batch_size, self.cfg.shared_dim, device=device)
            z_v_private = torch.zeros(batch_size, self.cfg.private_dim, device=device)
            out["video"] = None

        if audio_x is not None:
            a = self._encode_modality(audio_x, self.audio_refiner, self.audio_window_vae, "audio")
            out["audio"] = a
            z_a_shared = a["utt_z_shared"]
            z_a_private = a["utt_z_private"]
        else:
            z_a_shared = torch.zeros(batch_size, self.cfg.shared_dim, device=device)
            z_a_private = torch.zeros(batch_size, self.cfg.private_dim, device=device)
            out["audio"] = None

        # mask missing modalities at fusion time
        z_v_shared = z_v_shared * video_present
        z_v_private = z_v_private * video_present
        z_a_shared = z_a_shared * audio_present
        z_a_private = z_a_private * audio_present

        # cycle consistency: decode video-shared into audio-shared space
        z_a_hat_from_v = self.cross_decoder_v_to_a(z_v_shared)
        loss_cycle = F.mse_loss(z_a_hat_from_v, z_a_shared, reduction="mean")

        # sparse gated fusion
        g_v = torch.sigmoid(self.video_gate(z_v_private)) * video_present
        g_a = torch.sigmoid(self.audio_gate(z_a_private)) * audio_present

        h_fused = g_v * z_v_shared + g_a * z_a_shared
        loss_sparse = g_v.abs().mean() + g_a.abs().mean()

        # token reasoning
        h_fused_tok = self.fused_to_token_dim(h_fused)
        token_out = self.token_reasoner(h_fused_tok)
        h_final = token_out["h_fused_out"]

        logits_cls = self.cls_head(h_final)
        logits_sev = self.sev_head(h_final)

        out.update({
            "z_a_hat_from_v": z_a_hat_from_v,
            "g_video": g_v,
            "g_audio": g_a,
            "h_fused": h_fused,
            "h_final": h_final,
            "logits_cls": logits_cls,
            "logits_sev": logits_sev,
            "probs_cls": torch.softmax(logits_cls, dim=-1),
            "probs_sev": torch.softmax(logits_sev, dim=-1),
            "token_loss": token_out["loss_token"],
            "loss_cycle": loss_cycle,
            "loss_sparse": loss_sparse,
        })

        return out

    def compute_loss(
        self,
        outputs: Dict[str, Any],
        y_cls: torch.Tensor,
        y_sev: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Implements the paper's joint loss:

        L_total = L_cls + alpha * L_sev +
                  eps * (L_cycle + L_sparse + lambda * L_token) +
                  sum_m (L_w^m + L_u^m)

        where:
        - L_cls: classification CE
        - L_sev: severity CE
        - L_cycle: cross-modal alignment
        - L_sparse: gate sparsity
        - L_token: token reconstruction/specialization
        - L_w^m: local/window VAE loss
        - L_u^m: utterance-level VAE loss
        """
        loss_cls = F.cross_entropy(outputs["logits_cls"], y_cls)
        loss_sev = F.cross_entropy(outputs["logits_sev"], y_sev)

        local_and_utt = 0.0
        if outputs["video"] is not None:
            local_and_utt = local_and_utt + outputs["video"]["local_loss_total"] + outputs["video"]["utt_loss_total"]
        if outputs["audio"] is not None:
            local_and_utt = local_and_utt + outputs["audio"]["local_loss_total"] + outputs["audio"]["utt_loss_total"]

        reg = (
            outputs["loss_cycle"]
            + outputs["loss_sparse"]
            + self.cfg.lambda_token * outputs["token_loss"]
        )

        loss_total = (
            loss_cls
            + self.cfg.alpha_severity * loss_sev
            + self.cfg.eps_regularization * reg
            + local_and_utt
        )

        return {
            "loss_total": loss_total,
            "loss_cls": loss_cls,
            "loss_sev": loss_sev,
            "loss_cycle": outputs["loss_cycle"],
            "loss_sparse": outputs["loss_sparse"],
            "loss_token": outputs["token_loss"],
            "loss_local_utt": torch.as_tensor(local_and_utt, device=loss_total.device),
        }


# -----------------------------
# Random-input smoke test
# -----------------------------

def random_test() -> None:
    torch.manual_seed(7)

    cfg = DIVINEConfig(
        video_in_dim=512,
        audio_in_dim=768,
        refine_dim=128,
        window_latent_dim=64,
        shared_dim=64,
        private_dim=64,
        token_dim=64,
        num_symptom_tokens=5,
        num_classes=3,
        num_severity_levels=3,
        alpha_severity=2.0,
        eps_regularization=0.1,
        lambda_token=0.4,
        dropout=0.1,
    )

    model = DIVINE(cfg)

    # random frozen-feature sequences from upstream encoders
    batch_size = 4
    t_video = 40
    t_audio = 60

    video_feats = torch.randn(batch_size, t_video, cfg.video_in_dim)
    audio_feats = torch.randn(batch_size, t_audio, cfg.audio_in_dim)

    # optional modality masks to simulate missing modalities
    video_present = torch.tensor([1, 1, 0, 1], dtype=torch.float32)
    audio_present = torch.tensor([1, 0, 1, 1], dtype=torch.float32)

    y_cls = torch.tensor([0, 1, 2, 1], dtype=torch.long)
    y_sev = torch.tensor([0, 2, 1, 1], dtype=torch.long)

    outputs = model(
        video_x=video_feats,
        audio_x=audio_feats,
        video_present=video_present,
        audio_present=audio_present,
    )
    losses = model.compute_loss(outputs, y_cls=y_cls, y_sev=y_sev)

    print("=" * 80)
    print("DIVINE random-input smoke test")
    print("=" * 80)
    print(f"logits_cls shape : {tuple(outputs['logits_cls'].shape)}")
    print(f"logits_sev shape : {tuple(outputs['logits_sev'].shape)}")
    print(f"h_fused shape    : {tuple(outputs['h_fused'].shape)}")
    print(f"g_video shape    : {tuple(outputs['g_video'].shape)}")
    print(f"g_audio shape    : {tuple(outputs['g_audio'].shape)}")
    print("-" * 80)
    for k, v in losses.items():
        print(f"{k:16s}: {float(v):.6f}")
    print("-" * 80)
    print("Pred cls probs[0]:", outputs["probs_cls"][0].detach().cpu())
    print("Pred sev probs[0]:", outputs["probs_sev"][0].detach().cpu())

    # one backward pass
    losses["loss_total"].backward()
    print("Backward pass: OK")


def main() -> None:
    random_test()


if __name__ == "__main__":
    main()
