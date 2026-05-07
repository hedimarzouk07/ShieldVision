# Violence Against Women Detection in Videos — Deep Learning Benchmark

> Comparative study of deep learning architectures for **automatic detection of violence against women** in real-life and surveillance videos, using spatiotemporal modeling, transfer learning, and explainable AI.

This project was developed as part of the coursework for **Deep Learning / Computer Vision** at **Esprit School of Engineering**.

---

## Overview

Violence against women is a major global public health and human rights concern. Automated video-based detection systems can assist law enforcement, social services, and smart city platforms in identifying violent situations in real time — enabling faster intervention and better protection.

This project benchmarks **four deep learning architectures** on a unified multi-source dataset to identify the most effective approach for binary video classification: `Violence` vs `NonViolence`. The models learn to recognize patterns of physical aggression, assault, and threatening behavior from raw video footage.

Each model is evaluated end-to-end — from raw video ingestion to Grad-CAM explainability — ensuring a rigorous, reproducible, and ethically grounded comparison.

---

## Features

-  **Multi-dataset pipeline** — automatic unification, cleaning, and deduplication across 5 Kaggle datasets
-  **Balanced 50/50 splits** — stratified train / val / test (80 / 10 / 10)
-  **4 model architectures** compared side-by-side
-  **Full evaluation suite** — Accuracy, F1-score, AUC-ROC, Precision-Recall curves, Confusion Matrix
-  **Explainable AI** — Grad-CAM 3D visualizations on all models
-  **Training best practices** — LR warmup, early stopping, label smoothing, gradient clipping, mixed precision (AMP)
-  **Checkpoint system** — skip retraining if model already exists

---

##  Model Architectures

| Notebook | Architecture | Strategy | Input Size | Epochs |
|---|---|---|---|---|
| `resnet18_from_scratch.ipynb` | Custom CNN + BiGRU | Trained from scratch | 112×112 | 50 |
| `resnet18_pretrained.ipynb` | ResNet18 + BiGRU | Transfer learning | 224×224 | 24 |
| `3dcnn_gru.ipynb` | 3D-CNN + BiGRU | Spatiotemporal conv | 112×112 | 30 |
| `video-swin-transformer.ipynb` | Video Swin Transformer | Stage-wise fine-tuning | 224×224 | 5+8+… |

---

## Directory Structure

```
violence-against-women-detection/
│
├── 3dcnn_gru.ipynb                  # 3D-CNN + BiGRU model
├── resnet18_from_scratch.ipynb      # Custom CNN + BiGRU (no pretraining)
├── resnet18_pretrained.ipynb        # ResNet18 pretrained + BiGRU
├── video-swin-transformer.ipynb     # Video Swin Transformer fine-tuning
│
└── README.md
```

---

## Datasets

All models are trained on a unified dataset merged from 4 public Kaggle sources:

| Dataset | Source |
|---|---|
| Real Life Violence Situations | mohamedmustafa |
| Video Violence Detection | magicearth25 |
| Data by Hedi | marzoukhedi |
| SmartCity CCTV Violence Detection (SCVD) | toluwaniaremu |

Labels are automatically mapped using keyword matching:
- **Violence** → `violence`, `fight`, `weaponized`
- **NonViolence** → `nonviolence`, `nonfight`, `normal`, `nofight`

---

## Tech Stack

### Deep Learning
- **PyTorch** — model training, custom datasets, DataLoaders
- **TorchVision** — ResNet18 pretrained weights, transforms
- **pytorchvideo** — Video Swin Transformer backbone
- **timm** — auxiliary model components

### Video Processing
- **OpenCV (cv2)** — frame extraction, video reading
- **FFmpeg** — video format conversion

### Evaluation & XAI
- **scikit-learn** — classification report, ROC-AUC, confusion matrix
- **pytorch-grad-cam** — Grad-CAM 3D visualizations
- **Matplotlib / Seaborn** — training curves, metric plots

### Experiment Management
- **Kaggle Notebooks** — GPU P100 / T4 environment
- **tqdm** — progress bars
- **logging** — structured pipeline logs

---
##  Getting Started

### Prerequisites

```bash
pip install torch torchvision opencv-python scikit-learn matplotlib seaborn tqdm
pip install grad-cam pytorchvideo timm einops
```

### Running on Kaggle

1. Upload the notebooks to [Kaggle](https://www.kaggle.com)
2. Add the 4 datasets as Kaggle input sources
3. Enable GPU accelerator (P100 or T4)
4. Run cells top to bottom — the pipeline auto-skips steps already completed

### Force retraining

Inside each notebook, set the flags at the top:

```python
FORCE_RETRAIN   = True   # re-run training even if checkpoint exists
FORCE_REPROCESS = True   # re-unify and clean the dataset
```

---

## Evaluation Metrics

Each model is evaluated on the held-out test set with:

- **Accuracy** & **F1-Score** (macro)
- **AUC-ROC** curve
- **Precision-Recall** curve & Average Precision
- **Confusion Matrix**
- **Grad-CAM** heatmaps — visual explanation of model decisions

---

## Acknowledgments

This project was completed under the guidance of the faculty at **Esprit School of Engineering**.

Datasets provided by the Kaggle community: mohamedmustafa, magicearth25, marzoukhedi, and toluwaniaremu.

---

##  Topics

`python` · `deep-learning` · `computer-vision` · `violence-detection` · `violence-against-women` · `video-classification` · `pytorch` · `resnet18` · `swin-transformer` · `grad-cam` · `transfer-learning` · `spatiotemporal` · `artificial-intelligence` · `gender-based-violence` · `esprit-school-of-engineering`
