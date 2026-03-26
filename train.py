from __future__ import annotations

import yaml
import torch
from torch.utils.data import DataLoader

from data.dataset import FeatureDataset
from models.divine import DIVINEBackbone, DIVINEConfig, DIVINEModel, ClassificationHead
from models.losses import compute_auxiliary_loss, compute_prediction_loss, compute_total_loss


def build_model_from_config(cfg: dict) -> DIVINEModel:
    model_cfg = DIVINEConfig(**cfg["model"])

    head_cfg = cfg["head"]
    if head_cfg["type"] != "classification":
        raise NotImplementedError(
            "This train.py demo uses ClassificationHead only. "
            "Users can replace it with their own regression or multitask head."
        )

    backbone = DIVINEBackbone(model_cfg)
    head = ClassificationHead(
        in_dim=model_cfg.token_dim,
        num_classes=head_cfg["num_classes"],
        hidden_dim=head_cfg.get("hidden_dim"),
    )
    return DIVINEModel(backbone, head)


def main() -> None:
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model = build_model_from_config(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])

    # Demo random extracted features. Replace with your own features.
    mod1_x = torch.randn(32, 40, cfg["model"]["mod1_in_dim"])
    mod2_x = torch.randn(32, 60, cfg["model"]["mod2_in_dim"])
    targets = torch.randint(0, cfg["head"]["num_classes"], (32,))

    dataset = FeatureDataset(mod1_features=mod1_x, mod2_features=mod2_x, targets=targets)
    loader = DataLoader(dataset, batch_size=cfg["training"]["batch_size"], shuffle=True)

    model.train()
    for epoch in range(cfg["training"]["epochs"]):
        epoch_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()

            outputs = model(
                mod1_x=batch["mod1_x"],
                mod2_x=batch["mod2_x"],
                mod1_present=batch["mod1_present"],
                mod2_present=batch["mod2_present"],
            )

            pred_loss = compute_prediction_loss(
                outputs["predictions"],
                batch["targets"],
                loss_type=cfg["loss"]["prediction"],
            )
            aux_loss = compute_auxiliary_loss(
                outputs["backbone_outputs"]["aux_losses"],
                weights=cfg["loss"]["aux_weights"],
            )
            total_loss = compute_total_loss(pred_loss, aux_loss)

            total_loss.backward()
            optimizer.step()
            epoch_loss += float(total_loss.detach())

        print(f"Epoch {epoch + 1}: loss={epoch_loss / len(loader):.4f}")

    torch.save(model.state_dict(), "divine_demo.pt")
    print("Saved checkpoint to divine_demo.pt")


if __name__ == "__main__":
    main()
