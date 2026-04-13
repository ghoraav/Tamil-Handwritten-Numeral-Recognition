# DSHTT — Dynamic Stroke-Aware Hybrid Token Transformer
### Tamil Handwritten Numeral Recognition (0–10)

A novel deep learning architecture for recognizing Tamil handwritten numerals, designed from scratch for low-resource, stroke-dominant scripts. Unlike generic CNN or ViT approaches, DSHTT explicitly models the **structural stroke properties** of Tamil numerals through a four-stage hybrid pipeline.

---

## Architecture Overview

```
Input Image (128×128)
       │
       ├──────────────────────────────────────┐
       │                                      │
  [Stage 1]                             [Stage 1]
  Stroke Descriptor Map (SDM)           Grayscale Image
  ┌──────────────────────────┐
  │  Ch0: Skeleton           │
  │  Ch1: Curvature Map      │
  │  Ch2: Direction Field    │
  └──────────────────────────┘
       │                         │                    │
  [Stage 2B]               [Stage 2A]           [SDM Encoder]
  Patch Token Encoder       Tiny CNN Branch      MLP on SDM
  (Linear Attention, ViT)   (DepthwiseSep Conv)
       │                         │                    │
       └──────────┬──────────────┘                    │
              [Stage 3]                               │
     Stroke-Guided Dynamic Gating Unit (SDGU) ◄───────┘
     Learns per-sample: "trust CNN or Transformer more?"
              │
          [Stage 4]
     Prototype-Based Classifier
     (Cosine Similarity Head)
              │
         Prediction (0–10)
```

| Stage | Component | Role |
|-------|-----------|------|
| 1 | Stroke Descriptor Map (SDM) | Skeletonization → structural priors |
| 2A | Tiny CNN Branch | Local texture & spatial features |
| 2B | Patch Token Encoder | Global attention over image patches |
| 3 | Stroke-Guided Dynamic Gating (SDGU) | Adaptive, learned branch fusion |
| 4 | Prototype-Based Classification | Cosine similarity head, low-parameter |

**Total Parameters: ~488K** — deliberately lightweight for a small, manually collected dataset.

---

## Dataset

- **Script:** Tamil numerals 0–10 (11 classes)
- **Collection:** Manually handwritten and scanned using Google's default scanner (lighten filter applied automatically during scanning)
- **Size:** 50 samples per class × 11 classes = **550 total images**
- **Format:** Resized to **264×264 px** (PNG)
- **Split:** 70% train / 15% validation / 15% test

```
Dataset/
  0/   1/   2/   3/   4/   5/   6/   7/   8/   9/   10/
  (50 images each)
```

---

## Results

### Baseline Comparisons (Validation Dataset Viability)

Before developing DSHTT, two baselines were trained to confirm the dataset quality:

| Model | Epochs | Final Train Acc | Final Val Acc |
|-------|--------|-----------------|---------------|
| Custom CNN (3 conv blocks) | 15 | ~80.2% | ~73.6% |
| EfficientNet-B0 (Transfer Learning, TF) | 30 | ~89.6% | ~89.0% |

These results confirmed the dataset is capable of producing meaningful signal.

### DSHTT (Custom Architecture)

Trained in two phases on a **GTX 1050 Ti** (CUDA 11.8):

| Phase | Epochs | Duration | Best Val Acc | Test Acc |
|-------|--------|----------|--------------|----------|
| Phase 1 | 300 | ~20 min | 27.78% | 22.89% |
| Phase 2 (resumed) | 300 | ~20 min | 86.92% | 83.4% |

**Final Test Accuracy: 74.70%** (83 test samples)

```
Per-Class Performance (Phase 2):

Class  0  →  Precision: 0.89  Recall: 0.80  F1: 0.84
Class  1  →  Precision: 0.83  Recall: 0.71  F1: 0.77
Class  2  →  Precision: 0.78  Recall: 0.78  F1: 0.78
Class  5  →  Precision: 0.75  Recall: 1.00  F1: 0.86
Class  7  →  Precision: 0.75  Recall: 1.00  F1: 0.86
Class  8  →  Precision: 1.00  Recall: 1.00  F1: 1.00
Class 10  →  Precision: 0.75  Recall: 1.00  F1: 0.86
```

---

## Key Novel Contributions

### 1. Stroke Descriptor Map (SDM)
Instead of passing raw pixels, the model computes a **3-channel structural descriptor** per image:
- **Ch 0 — Skeleton:** Binary stroke skeleton via morphological skeletonization
- **Ch 1 — Curvature:** Gradient magnitude (Sobel) as a curvature proxy
- **Ch 2 — Direction:** Normalized `arctan2` of gradient field

This gives the model structural intelligence about **where strokes are and how they curve** — critical for a script like Tamil where stroke topology distinguishes numerals.

### 2. Stroke-Guided Dynamic Gating Unit (SDGU)
Instead of concatenating CNN and Transformer features blindly, SDGU **dynamically learns per-sample gate weights** (`α` for CNN, `β` for Transformer) conditioned on the SDM. For highly cursive or ambiguous writing styles, the model learns to rely more on the Transformer's global context; for clearer strokes, the CNN path is weighted higher.

### 3. Prototype-Based Classifier
Rather than a standard fully-connected softmax head, the classifier uses **learnable class prototype embeddings** with cosine similarity scoring. This reduces parameters and improves generalization — important for a 550-sample dataset.

---

## Project Structure

```
DSHTT-Tamil-Numeral-Recognition/
│
├── Dataset/                         # 11 class folders (0–10), 50 images each
│
├── DSHTT_Tamil_Numeral_Recognition.ipynb   # Main DSHTT model: full pipeline
│
├── CNNtrain.py                      # Baseline 1: Custom 3-layer CNN (TensorFlow)
├── EfficientNET.py                  # Baseline 2: EfficientNet-B0 fine-tuning (TensorFlow)
│
├── dshtt_best.pt                    # Best saved model weights (PyTorch)
│
└── README.md
```

---

## Setup & Usage

### Requirements
```bash
pip install torch torchvision scikit-image scikit-learn einops matplotlib seaborn pillow
```

For baselines (TensorFlow):
```bash
pip install tensorflow
```

### Run DSHTT
Open `DSHTT_Tamil_Numeral_Recognition.ipynb` in **Google Colab** (recommended — GPU required) or locally with CUDA.

1. Mount Google Drive and set `DATA_ROOT` to your dataset path
2. Run all cells sequentially — the notebook trains for 300 epochs and saves `dshtt_best.pt`
3. To resume training (Phase 2), reload weights before the training cell and run again

### Run Baselines
```bash
# CNN Baseline
python CNNtrain.py

# EfficientNet Baseline
python EfficientNET.py
```
Both expect a `Dataset/` folder in the working directory with the class subfolders.

---

## Limitations & Future Work

- **Dataset size:** 550 images is very small. Collecting more samples (target: 500+ per class) would significantly improve DSHTT's potential.
- **SDGU gate imbalance:** During training, `α` (CNN) consistently dominated (~0.99) over `β` (Transformer). This suggests the tiny Transformer branch needs architectural tuning or dedicated pre-training.
- **Augmentation:** Currently limited to rotation and translation. More aggressive augmentation (elastic distortion, stroke-level noise) could help given the small dataset.
- **Gradio Demo:** A live inference UI using Gradio is planned.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Custom Model (DSHTT) | PyTorch 2.5.1, einops |
| CNN Baseline | TensorFlow / Keras |
| Transfer Learning Baseline | TensorFlow / Keras + EfficientNet-B0 |
| Stroke Processing | scikit-image (skeletonize, Sobel) |
| Training Environment | Google Colab, NVIDIA GTX 1050 Ti, CUDA 11.8 |
| Data Visualization | Matplotlib, Seaborn |
| Evaluation | scikit-learn (classification report, confusion matrix, t-SNE) |
