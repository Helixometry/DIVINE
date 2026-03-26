import torch
from divine import DIVINEBackbone, DIVINEBackboneConfig, DIVINEModel
from divine import ClassificationHead, RegressionHead, MultiTaskHead
from divine import compute_auxiliary_loss, compute_prediction_loss, compute_total_loss

torch.manual_seed(7)
cfg = DIVINEBackboneConfig(mod1_in_dim=300, mod2_in_dim=200)
head = MultiTaskHead({
    "class": ClassificationHead(in_dim=cfg.token_dim, num_classes=3),
    "score": RegressionHead(in_dim=cfg.token_dim, out_dim=1),
})
model = DIVINEModel(DIVINEBackbone(cfg), head)
mod1_feats = torch.randn(5, 32, 300)
mod2_feats = torch.randn(5, 18, 200)
targets = {"class": torch.randint(0, 3, (5,)), "score": torch.randn(5, 1)}
out = model(mod1_x=mod1_feats, mod2_x=mod2_feats)
pred_losses = {
    "class": compute_prediction_loss(out.predictions["class"], targets["class"], "cross_entropy"),
    "score": compute_prediction_loss(out.predictions["score"], targets["score"], "mse"),
}
aux_loss = compute_auxiliary_loss(out.backbone_outputs["aux_losses"])
total_loss = compute_total_loss(pred_losses, aux_loss, {"class": 1.0, "score": 0.5})
print("class logits:", out.predictions["class"].shape)
print("score pred:", out.predictions["score"].shape)
print("total_loss:", float(total_loss.detach()))
