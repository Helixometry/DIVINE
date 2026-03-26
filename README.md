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

## Model Summary

**Input**
- Audio feature sequence
- Video feature sequence

**Core components**
- Temporal 1D CNN refinement
- Local/window VAE
- Shared/private utterance-level VAE
- Cross-modal cycle/alignment loss
- Sparse gated fusion
- Symptom token reasoning
- Multi-task prediction heads

**Outputs**
- Disorder label
- Severity label

---

## Repository Structure

```bash
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