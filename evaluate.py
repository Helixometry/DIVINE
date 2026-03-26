from __future__ import annotations

import yaml
import torch
from torch.utils.data import DataLoader

from data.dataset import FeatureDataset
from models.divine import DIVINEBackbone, DIVINEConfig, DIVINEModel, ClassificationHead


def build_model_from_config(cfg: dict) -> DIVINEModel:
    model_cfg = DIVINEConfig(**cfg["model"])
    head_cfg = cfg["head"]

    backbone = DIVINEBackbone(model_cfg)
    head = ClassificationHead(
        in_dim=model_cfg.token_dim,
        num_classes=head_cfg["num_classes"],
        hidden_dim=head_cfg.get("hidden_dim"),
    )
    return DIVINEModel(backbone, head)


@torch.no_grad()
def evaluate_loader(model: DIVINEModel, loader: DataLoader, mode_name: str) -> None:
    model.eval()
    correct = 0
    total = 0

    for batch in loader:
        outputs = model(
            mod1_x=batch["mod1_x"],
            mod2_x=batch["mod2_x"],
            mod1_present=batch["mod1_present"],
            mod2_present=batch["mod2_present"],
        )
        preds = outputs["predictions"].argmax(dim=-1)
        correct += int((preds == batch["targets"]).sum())
        total += int(batch["targets"].shape[0])

    acc = correct / total if total > 0 else 0.0
    print(f"{mode_name} accuracy: {acc:.4f}")


def main() -> None:
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model = build_model_from_config(cfg)

    # If you already trained and saved a checkpoint:
    # model.load_state_dict(torch.load("divine_demo.pt", map_location="cpu"))

    n = 16
    mod1_x = torch.randn(n, 40, cfg["model"]["mod1_in_dim"])
    mod2_x = torch.randn(n, 60, cfg["model"]["mod2_in_dim"])
    targets = torch.randint(0, cfg["head"]["num_classes"], (n,))

    both_dataset = FeatureDataset(
        mod1_features=mod1_x,
        mod2_features=mod2_x,
        targets=targets,
        mod1_present=torch.ones(n),
        mod2_present=torch.ones(n),
    )

    mod1_only_dataset = FeatureDataset(
        mod1_features=mod1_x,
        mod2_features=mod2_x,
        targets=targets,
        mod1_present=torch.ones(n),
        mod2_present=torch.zeros(n),
    )

    mod2_only_dataset = FeatureDataset(
        mod1_features=mod1_x,
        mod2_features=mod2_x,
        targets=targets,
        mod1_present=torch.zeros(n),
        mod2_present=torch.ones(n),
    )

    evaluate_loader(model, DataLoader(both_dataset, batch_size=8), "Both modalities")
    evaluate_loader(model, DataLoader(mod1_only_dataset, batch_size=8), "Single modality: mod1 only")
    evaluate_loader(model, DataLoader(mod2_only_dataset, batch_size=8), "Single modality: mod2 only")


if __name__ == "__main__":
    main()
