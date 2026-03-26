import torch
from divine import DIVINEBackbone, DIVINEBackboneConfig, DIVINEModel, ClassificationHead
from divine import compute_auxiliary_loss, compute_prediction_loss, compute_total_loss

torch.manual_seed(7)
cfg = DIVINEBackboneConfig(mod1_in_dim=512, mod2_in_dim=768)
model = DIVINEModel(DIVINEBackbone(cfg), ClassificationHead(in_dim=cfg.token_dim, num_classes=4))
mod1_feats = torch.randn(8, 40, 512)
mod2_feats = torch.randn(8, 60, 768)
targets = torch.randint(0, 4, (8,))
out = model(mod1_x=mod1_feats, mod2_x=mod2_feats)
pred_loss = compute_prediction_loss(out.predictions, targets, "cross_entropy")
aux_loss = compute_auxiliary_loss(out.backbone_outputs["aux_losses"])
total_loss = compute_total_loss(pred_loss, aux_loss)
print("logits:", out.predictions.shape)
print("features:", out.features.shape)
print("total_loss:", float(total_loss.detach()))
