# DIVINE Library

A reusable PyTorch package for **generic two-modality DIVINE modeling**.

You preprocess your data yourself and extract features for:
- modality 1
- modality 2

Then this library handles the main DIVINE backbone.

## Main idea

The backbone keeps the main intuition fixed:

- temporal refinement
- local latent modeling
- shared/private disentanglement
- cross-modal coordination
- sparse gated fusion
- token-based reasoning

Users can then attach:
- their own prediction head
- their own task loss
- classification, regression, or multitask outputs

## Public API

- `DIVINEBackbone`
- `DIVINEBackboneConfig`
- `DIVINEModel`
- `ClassificationHead`
- `RegressionHead`
- `MultiTaskHead`
- `compute_prediction_loss`
- `compute_auxiliary_loss`
- `compute_total_loss`

## Example

```python
from divine import DIVINEBackbone, DIVINEBackboneConfig, DIVINEModel, ClassificationHead

cfg = DIVINEBackboneConfig(mod1_in_dim=512, mod2_in_dim=768)
backbone = DIVINEBackbone(cfg)
head = ClassificationHead(in_dim=cfg.token_dim, num_classes=4)
model = DIVINEModel(backbone, head)
```
