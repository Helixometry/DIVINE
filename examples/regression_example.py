import torch
from divine import DIVINEBackbone, DIVINEBackboneConfig, DIVINEModel, RegressionHead
from divine import compute_auxiliary_loss, compute_prediction_loss, compute_total_loss

torch.manual_seed(7)
cfg = DIVINEBackboneConfig(mod1_in_dim=256, mod2_in_dim=128)
model = DIVINEModel(DIVINEBackbone(cfg), RegressionHead(in_dim=cfg.token_dim, out_dim=1))
mod1_feats = torch.randn(6, 30, 256)
mod2_feats = torch.randn(6, 25, 128)
targets = torch.randn(6, 1)
out = model(mod1_x=mod1_feats, mod2_x=mod2_feats)
pred_loss = compute_prediction_loss(out.predictions, targets, "mse")
aux_loss = compute_auxiliary_loss(out.backbone_outputs["aux_losses"])
total_loss = compute_total_loss(pred_loss, aux_loss)
print("pred:", out.predictions.shape)
print("total_loss:", float(total_loss.detach()))
