from __future__ import annotations
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    return mu + torch.randn_like(std) * std

def kl_divergence_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.mean(torch.sum(torch.exp(logvar) + mu**2 - 1.0 - logvar, dim=-1))

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: Tuple[int, ...], out_dim: int, dropout: float = 0.1,
                 activation: nn.Module = nn.ReLU, out_activation: Optional[nn.Module] = None) -> None:
        super().__init__()
        dims = [in_dim, *hidden_dims, out_dim]
        layers = []
        for i in range(len(dims) - 2):
            layers += [nn.Linear(dims[i], dims[i + 1]), activation(), nn.Dropout(dropout)]
        layers.append(nn.Linear(dims[-2], dims[-1]))
        if out_activation is not None:
            layers.append(out_activation())
        self.net = nn.Sequential(*layers)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class TemporalRefiner(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, kernel_size: int = 3, pool_kernel: int = 2, dropout: float = 0.1) -> None:
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
        return self.block(x.transpose(1, 2)).transpose(1, 2)

class WindowVAE(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int, hidden_dim: int = 128, dropout: float = 0.1) -> None:
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
        recon = self.dec(z.reshape(b * t, -1)).reshape(b, t, d)
        loss_recon = F.mse_loss(recon, x, reduction="mean")
        loss_kl = kl_divergence_standard_normal(mu.reshape(b * t, -1), logvar.reshape(b * t, -1))
        return {"z_seq": z, "recon_seq": recon, "mu": mu, "logvar": logvar, "loss_recon": loss_recon, "loss_kl": loss_kl, "loss_total": loss_recon + loss_kl}

class GaussianEncoder(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.mu = MLP(in_dim, (hidden_dim,), latent_dim, dropout=dropout)
        self.logvar = MLP(in_dim, (hidden_dim,), latent_dim, dropout=dropout)
    def forward(self, x: torch.Tensor):
        mu = self.mu(x)
        logvar = self.logvar(x)
        return reparameterize(mu, logvar), mu, logvar

class UtteranceDisentangler(nn.Module):
    def __init__(self, pooled_dim: int, shared_dim: int, private_dim: int, hidden_dim: int = 128,
                 beta_shared: float = 1.0, beta_private: float = 1.0, dropout: float = 0.1) -> None:
        super().__init__()
        self.beta_shared = beta_shared
        self.beta_private = beta_private
        self.shared_encoder = GaussianEncoder(pooled_dim, shared_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.private_encoder_mod1 = GaussianEncoder(pooled_dim, private_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.private_encoder_mod2 = GaussianEncoder(pooled_dim, private_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.recon_mod1 = MLP(shared_dim + private_dim, (hidden_dim,), pooled_dim, dropout=dropout)
        self.recon_mod2 = MLP(shared_dim + private_dim, (hidden_dim,), pooled_dim, dropout=dropout)
    def forward_one(self, pooled_x: torch.Tensor, modality: str) -> Dict[str, torch.Tensor]:
        z_shared, mu_s, logvar_s = self.shared_encoder(pooled_x)
        if modality == "mod1":
            z_private, mu_p, logvar_p = self.private_encoder_mod1(pooled_x)
            recon = self.recon_mod1(torch.cat([z_shared, z_private], dim=-1))
        else:
            z_private, mu_p, logvar_p = self.private_encoder_mod2(pooled_x)
            recon = self.recon_mod2(torch.cat([z_shared, z_private], dim=-1))
        loss_recon = F.mse_loss(recon, pooled_x, reduction="mean")
        loss_kl_shared = kl_divergence_standard_normal(mu_s, logvar_s)
        loss_kl_private = kl_divergence_standard_normal(mu_p, logvar_p)
        return {"z_shared": z_shared, "z_private": z_private, "mu_shared": mu_s, "logvar_shared": logvar_s,
                "mu_private": mu_p, "logvar_private": logvar_p, "recon": recon, "loss_recon": loss_recon,
                "loss_kl_shared": loss_kl_shared, "loss_kl_private": loss_kl_private,
                "loss_total": loss_recon + self.beta_shared * loss_kl_shared + self.beta_private * loss_kl_private}

class TokenReasoner(nn.Module):
    def __init__(self, token_dim: int, num_tokens: int, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.tokens = nn.Parameter(torch.randn(num_tokens, token_dim) * 0.02)
        self.row_mlp = nn.Sequential(nn.Linear(token_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, token_dim))
        self.norm = nn.LayerNorm(token_dim)
    def forward(self, fused_h: torch.Tensor) -> Dict[str, torch.Tensor]:
        b, _ = fused_h.shape
        token_bank = self.tokens.unsqueeze(0).expand(b, -1, -1)
        s = torch.cat([token_bank, fused_h.unsqueeze(1)], dim=1)
        h_out = self.norm(s + self.row_mlp(s))
        return {"S": s, "Hout": h_out, "h_fused_out": h_out[:, -1, :], "loss_token": F.mse_loss(h_out[:, :-1, :], token_bank, reduction="mean")}
