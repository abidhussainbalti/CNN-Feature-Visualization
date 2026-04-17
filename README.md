# CNN Activation Visualization — Cats vs Dogs

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/Keras-API-red?style=flat-square"/>
  <img src="https://img.shields.io/badge/Platform-Google%20Colab-yellow?style=flat-square&logo=googlecolab"/>
  <img src="https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square"/>
</p>

A complete implementation of a Convolutional Neural Network (CNN) for binary image classification, with full visualization of internal layer activations — showing exactly what the network learns at each depth.

---

## What this project does

This project trains a CNN from scratch on the Microsoft Cats vs Dogs dataset (23,262 images) and then opens the model's "black box" by extracting and visualizing the feature maps produced by every convolutional and pooling layer. The goal is to demonstrate that CNN representations are hierarchical: early layers detect simple edges and textures, while deeper layers encode abstract, class-discriminative patterns.

Three additional experiments are included as post-lab tasks:
- Comparing three different CNN architectures (shallow, deep, regularized)
- Analyzing activation responses across multiple input images
- Measuring the effect of data augmentation on generalization and activation quality

---

## Results

### Main model (5-block CNN with augmentation)

| Metric | Value |
|--------|-------|
| Training epochs | 50 |
| Best validation accuracy | 95.10% (epoch 46) |
| **Test accuracy** | **94.43%** |
| Test loss | 0.1560 |
| Training images | 16,000 |
| Validation images | 2,000 |
| Test images | 4,652 |

### Architecture comparison (30 epochs each)

| Architecture | Test Accuracy | Test Loss | Notes |
|---|---|---|---|
| A — Shallow (3 Conv blocks) | 87.75% | 0.2876 | Faster training, lower capacity |
| B — Deep (5 Conv blocks) | 93.49% | 0.1721 | Best accuracy, slightly more overfit |
| C — Lightweight + Dropout | 87.77% | 0.2806 | Smallest train/val gap |

### Augmentation experiment (30 epochs each)

| Setup | Test Accuracy | Test Loss |
|---|---|---|
| With augmentation | 93.92% | 0.1674 |
| Without augmentation | 89.55% | 0.2583 |

Without augmentation, training accuracy reached 99.24% at epoch 30 while validation accuracy plateaued at ~91% — a clear overfitting signature. With augmentation, both curves track closely.

### Activation map shapes (input image: 180×180×3)

| Layer | Output Shape | Filters |
|---|---|---|
| conv2d (Block 1) | (1, 178, 178, 32) | 32 |
| max_pooling2d | (1, 89, 89, 32) | 32 |
| conv2d_1 (Block 2) | (1, 87, 87, 64) | 64 |
| max_pooling2d_1 | (1, 43, 43, 64) | 64 |
| conv2d_2 (Block 3) | (1, 41, 41, 128) | 128 |
| max_pooling2d_2 | (1, 20, 20, 128) | 128 |
| conv2d_3 (Block 4) | (1, 18, 18, 256) | 256 |
| max_pooling2d_3 | (1, 9, 9, 256) | 256 |
| conv2d_4 (Block 5) | (1, 7, 7, 256) | 256 |

Spatial resolution drops from 178×178 → 7×7 through the network. Number of filters grows from 32 → 256.

---

## Repository structure

```
cnn-cats-dogs-activation-maps/
├── README.md                          ← This file
├── METHODOLOGY.md                     ← Full background and technical explanation
├── requirements.txt                   ← Python dependencies
├── .gitignore                         ← Excludes datasets, models, checkpoints
│
├── notebooks/
│   └── cnn_cats_dogs_activation.ipynb ← Main notebook — run this in Colab
│
├── results/
│   ├── training_metrics.md            ← Epoch-by-epoch training history
│   └── architecture_comparison.md    ← Task 1, 2, 3 detailed results
│
└── assets/
    └── model_architecture.md          ← Layer-by-layer architecture reference
```

---

## How to run

**Recommended: Google Colab**

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Upload `notebooks/cnn_cats_dogs_activation.ipynb`
3. Set runtime to GPU: Runtime → Change runtime type → T4 GPU
4. Run all cells (Runtime → Run all)

The notebook downloads the Cats vs Dogs dataset automatically via TensorFlow Datasets. Total runtime is approximately 45–60 minutes on a Colab T4 GPU.

**Local setup**

```bash
git clone https://github.com/YOUR_USERNAME/cnn-cats-dogs-activation-maps.git
cd cnn-cats-dogs-activation-maps
pip install -r requirements.txt
jupyter notebook notebooks/cnn_cats_dogs_activation.ipynb
```

> Note: Training for 50 epochs on CPU will take several hours. GPU is strongly recommended.

---

## Model architecture summary

```
Input (180 × 180 × 3)
  └─ Augmentation (RandomFlip + RandomRotation + RandomZoom)
  └─ Rescaling (÷ 255)
  └─ Conv2D(32, 3×3, ReLU) → MaxPooling2D(2×2)        Block 1
  └─ Conv2D(64, 3×3, ReLU) → MaxPooling2D(2×2)        Block 2
  └─ Conv2D(128, 3×3, ReLU) → MaxPooling2D(2×2)       Block 3
  └─ Conv2D(256, 3×3, ReLU) → MaxPooling2D(2×2)       Block 4
  └─ Conv2D(256, 3×3, ReLU)                            Block 5
  └─ Flatten
  └─ Dropout(0.5)
  └─ Dense(1, Sigmoid) + L2(0.001)

Loss:      Binary Cross-Entropy
Optimizer: RMSprop
Epochs:    50
Batch:     128
```

---

## Key concepts

- **CNN feature hierarchy** — early layers detect edges/textures; deep layers detect object parts
- **Activation visualization** — extracting intermediate layer outputs via a sub-model
- **Data augmentation** — RandomFlip, RandomRotation, RandomZoom applied in-model
- **Regularization** — Dropout(0.5) and L2 weight penalty to reduce overfitting
- **ModelCheckpoint** — saving the best model based on validation loss

For full explanations of every concept, see [METHODOLOGY.md](METHODOLOGY.md).

---

## Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| tensorflow | ≥ 2.12 | CNN building, training, Keras API |
| tensorflow-datasets | ≥ 4.9 | Cats vs Dogs dataset |
| numpy | ≥ 1.23 | Array operations |
| matplotlib | ≥ 3.7 | Visualization and plotting |
| Pillow | ≥ 9.0 | Image loading |

---

## References

- Chollet, F. (2021). *Deep Learning with Python, 2nd Edition.* Manning Publications.
- [TensorFlow Datasets — Cats vs Dogs](https://www.tensorflow.org/datasets/catalog/cats_vs_dogs)
- [Keras Functional API](https://keras.io/guides/functional_api/)
- Zeiler, M.D. & Fergus, R. (2014). *Visualizing and Understanding Convolutional Networks.* ECCV.

---

*NUST — SEECS | Deep Learning | Lab 04 | 2026*
