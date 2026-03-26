<div align="center">

# DIVINE
## Coordinating Multimodal Disentangled Representations for Oro-Facial Neurological Disorder Assessment

[![Paper](https://img.shields.io/badge/Paper-PDF-red.svg)](#)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](#)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#)

</div>

---

## Overview

DIVINE is a multimodal framework for neurological disorder assessment from **audio** and **video** signals.  
It is designed to model complementary oro-facial information by learning:

- **shared representations** across modalities,
- **private modality-specific representations**,
- **cross-modal coordination** through alignment,
- **sparse gated fusion** for robust integration,
- and **symptom-aware reasoning** through learnable symptom tokens.

The model jointly predicts:

- **neurological disorder category**
- **severity level**

Unlike simple feature concatenation, DIVINE explicitly disentangles modality-invariant and modality-specific information, making fusion more structured and clinically meaningful. 

---

## Abstract

Neurological disorder assessment from speech and facial behavior requires integrating multiple modalities while preserving clinically relevant information unique to each. We propose **DIVINE**, a hierarchical multimodal disentanglement framework that coordinates audio and visual representations for oro-facial neurological disorder assessment. DIVINE first refines modality-specific temporal dynamics, then learns local latent representations, followed by utterance-level decomposition into **shared** and **private** latent factors. The shared factors capture common neuro-motor characteristics across audio and video, while the private factors preserve modality-specific information. A cross-modal alignment objective, sparse gated fusion, and learnable symptom tokens further improve robustness and interpretability. DIVINE jointly performs disorder classification and severity estimation, and outperforms standard multimodal fusion baselines. 

---

## Architecture

DIVINE consists of five key stages:

1. **Temporal Refinement**  
   Lightweight temporal convolution blocks refine frozen audio and visual feature sequences.

2. **Local Representation Learning**  
   A local VAE models short-term dynamics in each modality.

3. **Utterance-Level Disentanglement**  
   Each modality is decomposed into:
   - **shared latent** representation
   - **private latent** representation

4. **Cross-Modal Coordination and Fusion**  
   Shared latents are aligned across modalities, and private latents guide sparse gating for fusion.

5. **Symptom-Aware Prediction**  
   Learnable symptom tokens interact with the fused representation for:
   - disorder classification
   - severity prediction

This design lets DIVINE go beyond simple multimodal concatenation by explicitly separating common and modality-specific neurological cues. 

---
A reusable PyTorch implementation of the **DIVINE backbone** for generic two-modality feature fusion.

Users do their own:
- raw data preprocessing
- feature extraction
- padding / masking
- task-specific target creation

This repo keeps the **main DIVINE representation learning idea fixed** and lets users attach their own prediction layer for their own task.

---

## Repository Structure

```text
DIVINE/
├── models/
│   ├── divine.py
│   ├── modules.py
│   └── losses.py
├── data/
│   ├── dataset.py
│   └── preprocessing.py
├── configs/
│   └── default.yaml
├── train.py
├── evaluate.py
├── requirements.txt
└── README.md
```

---

## What stays fixed

The core DIVINE backbone remains the same:

- temporal refinement
- local latent modeling
- shared/private disentanglement
- cross-modal alignment
- sparse gated fusion
- token-based reasoning

This is implemented in `models/divine.py`.

---

## What users can customize

Users can change:

- modality feature dimensions
- hyperparameters
- whether they use both modalities or only one
- prediction head
- prediction loss
- task type

Examples of task types:
- classification
- regression
- multitask learning

---

## Install

```bash
pip install -r requirements.txt
```

---

## Main usage idea

You extract features yourself and then build the DIVINE backbone.

### Step 1: Create the DIVINE backbone

```python
from models.divine import DIVINEConfig, DIVINEBackbone

cfg = DIVINEConfig(
    mod1_in_dim=512,
    mod2_in_dim=768,
    refine_dim=128,
    window_latent_dim=64,
    shared_dim=64,
    private_dim=64,
    token_dim=64,
    num_tokens=5,
)

backbone = DIVINEBackbone(cfg)
```

### Step 2: Add your own prediction head

This repo includes a simple classification demo head, but users can replace it with any head they want.

```python
import torch.nn as nn
from models.divine import DIVINEModel

my_head = nn.Linear(cfg.token_dim, 4)
model = DIVINEModel(backbone, my_head)
```

Now the core DIVINE part stays the same, but the final layer is completely user-defined.

---

## Forward pass with both modalities

```python
import torch

mod1_feats = torch.randn(8, 40, 512)   # [batch, time, dim1]
mod2_feats = torch.randn(8, 60, 768)   # [batch, time, dim2]

outputs = model(
    mod1_x=mod1_feats,
    mod2_x=mod2_feats,
)

predictions = outputs["predictions"]
features = outputs["features"]
aux_losses = outputs["backbone_outputs"]["aux_losses"]
```

---

## Forward pass with a single modality

Even if your model was built for two modalities, you can evaluate using only one modality by setting the modality presence flags.

### mod1 only

```python
outputs = model(
    mod1_x=mod1_feats,
    mod2_x=mod2_feats,
    mod1_present=torch.ones(mod1_feats.size(0)),
    mod2_present=torch.zeros(mod2_feats.size(0)),
)
```

### mod2 only

```python
outputs = model(
    mod1_x=mod1_feats,
    mod2_x=mod2_feats,
    mod1_present=torch.zeros(mod1_feats.size(0)),
    mod2_present=torch.ones(mod2_feats.size(0)),
)
```

This keeps the same DIVINE backbone but disables one modality at fusion time.

---

## Training loss

The backbone returns auxiliary DIVINE losses:
- local + utterance losses
- cycle loss
- sparse gate loss
- token loss

Prediction loss is chosen by the user.

Example:

```python
from models.losses import (
    compute_prediction_loss,
    compute_auxiliary_loss,
    compute_total_loss,
)

pred_loss = compute_prediction_loss(predictions, targets, loss_type="cross_entropy")
aux_loss = compute_auxiliary_loss(aux_losses)
total_loss = compute_total_loss(pred_loss, aux_loss)
```

Users can replace:
- `cross_entropy`
- `mse`
- `l1`
- or their own custom loss

---

## Minimal training example

```python
import torch
import torch.nn as nn

from models.divine import DIVINEConfig, DIVINEBackbone, DIVINEModel
from models.losses import compute_prediction_loss, compute_auxiliary_loss, compute_total_loss

cfg = DIVINEConfig(mod1_in_dim=512, mod2_in_dim=768)
backbone = DIVINEBackbone(cfg)
head = nn.Linear(cfg.token_dim, 3)
model = DIVINEModel(backbone, head)

mod1_feats = torch.randn(8, 40, 512)
mod2_feats = torch.randn(8, 60, 768)
targets = torch.randint(0, 3, (8,))

outputs = model(mod1_x=mod1_feats, mod2_x=mod2_feats)

pred_loss = compute_prediction_loss(outputs["predictions"], targets, "cross_entropy")
aux_loss = compute_auxiliary_loss(outputs["backbone_outputs"]["aux_losses"])
loss = compute_total_loss(pred_loss, aux_loss)

loss.backward()
```

---

## How to use this as a library

The intended use is:

1. User preprocesses raw data on their own
2. User extracts `mod1` and `mod2` features on their own
3. User creates a DIVINE backbone
4. User adds any prediction head they want
5. User selects any task-specific loss they want

So the repo provides the **DIVINE representation engine**, not a fixed end-task model.

---

## Training demo

Run:

```bash
python train.py
```

This uses:
- random demo features
- classification demo head
- the configurable auxiliary DIVINE losses

Replace the demo tensors in `train.py` with your own extracted features.

---

## Evaluation demo

Run:

```bash
python evaluate.py
```

This demonstrates evaluation in three settings:
- both modalities
- mod1 only
- mod2 only

So users can compare multimodal and single-modal behavior using the same DIVINE backbone.

---

## Notes

- `models/divine.py` is the main file users import from
- `models/modules.py` contains reusable DIVINE blocks
- `models/losses.py` contains generic loss helpers
- `data/dataset.py` is only a lightweight wrapper around extracted features
- `data/preprocessing.py` is intentionally minimal because users usually have their own preprocessing pipeline

---
