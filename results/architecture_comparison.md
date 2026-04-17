# Architecture Comparison & Experiment Results

Exact results from all three post-lab tasks.

---

## Task 1 — Architecture comparison (30 epochs each)

### Architecture A — Shallow (3 Conv blocks, Dropout 0.5)

Selected epoch highlights:

| Epoch | Train Acc | Train Loss | Val Acc | Val Loss |
|-------|-----------|------------|---------|----------|
| 1 | 58.13% | 0.6801 | 53.65% | 0.7111 |
| 5 | 73.15% | 0.5355 | 76.55% | 0.4866 |
| 10 | 78.64% | 0.4583 | 81.05% | 0.4171 |
| 16 | 81.54% | 0.4074 | 84.65% | 0.3487 |
| 20 | 83.14% | 0.3796 | 85.60% | 0.3231 |
| 25 | 84.21% | 0.3557 | 86.20% | 0.3276 |
| 27 | 84.48% | 0.3485 | 88.65% | 0.2807 |
| 30 | 85.57% | 0.3322 | 87.55% | 0.3056 |

**Test accuracy: 87.75% | Test loss: 0.2876**

### Architecture B — Deep (5 Conv blocks, L2 regularization)

Selected epoch highlights:

| Epoch | Train Acc | Train Loss | Val Acc | Val Loss |
|-------|-----------|------------|---------|----------|
| 1 | 53.05% | 0.6953 | 60.30% | 0.6752 |
| 5 | 72.94% | 0.5420 | 78.05% | 0.4695 |
| 10 | 80.46% | 0.4254 | 82.30% | 0.4103 |
| 18 | 87.31% | 0.2980 | 89.65% | 0.2624 |
| 19 | 87.87% | 0.2941 | 90.15% | 0.2416 |
| 21 | 88.94% | 0.2667 | 90.20% | 0.2388 |
| 24 | 90.48% | 0.2412 | 91.80% | 0.2081 |
| 27 | 91.44% | 0.2147 | 92.80% | 0.1905 |
| 30 | 92.18% | 0.2021 | 92.80% | 0.1855 |

**Test accuracy: 93.49% | Test loss: 0.1721**

### Architecture C — Lightweight + Dropout at each block

Selected epoch highlights:

| Epoch | Train Acc | Train Loss | Val Acc | Val Loss |
|-------|-----------|------------|---------|----------|
| 1 | 51.90% | 0.7034 | 61.00% | 0.6787 |
| 5 | 69.26% | 0.5804 | 69.80% | 0.5649 |
| 10 | 74.81% | 0.5144 | 77.85% | 0.4596 |
| 16 | 78.17% | 0.4544 | 82.65% | 0.3980 |
| 20 | 80.99% | 0.4184 | 85.35% | 0.3408 |
| 22 | 81.48% | 0.4030 | 86.40% | 0.3177 |
| 25 | 83.35% | 0.3760 | 86.85% | 0.3049 |
| 28 | 83.83% | 0.3654 | 86.30% | 0.3065 |
| 30 | 83.93% | 0.3601 | 88.00% | 0.2811 |

**Test accuracy: 87.77% | Test loss: 0.2806**

### Summary

| Architecture | Test Acc | Test Loss | Train/Val Gap (epoch 30) |
|---|---|---|---|
| A — Shallow | 87.75% | 0.2876 | 85.57% / 87.55% = ~2% |
| B — Deep | 93.49% | 0.1721 | 92.18% / 92.80% = ~0.6% |
| C — Dropout | 87.77% | 0.2806 | 83.93% / 88.00% = ~4% (val > train) |

Architecture B achieves highest accuracy. Architecture C shows healthiest regularization (validation accuracy exceeds training accuracy, indicating Dropout is working as intended). Architecture A is the fastest to train but limited in capacity.

---

## Task 2 — Multi-image activation analysis

Three images used: Cat 1 (from S3 URL), Dog 1 (from test dataset, label=1), Cat 2 (from test dataset, label=0).

All activations computed using the activation sub-model monitoring 9 layers.

### Layers compared

**Early layer: conv2d** — shape (1, 178, 178, 32)
- All three images produce feature maps with visible structural patterns (edges, gradients)
- Filters 0–7 show similar spatial patterns across all images
- Mean activation intensities are comparable across Cat 1, Dog 1, Cat 2

**Deep layer: conv2d_4** — shape (1, 7, 7, 256)
- Feature maps are very sparse (most values near zero)
- Filter activation patterns diverge significantly between cat and dog images
- The spatial patterns that survive at this depth are small, abstract, and class-specific

### Mean activation intensity by layer

Observation: Mean absolute activation values are similar across all three images in early layers and diverge in deep layers, confirming class-specific encoding at depth.

---

## Task 3 — Data augmentation experiment (30 epochs each)

### Model WITH augmentation

| Epoch | Train Acc | Train Loss | Val Acc | Val Loss |
|-------|-----------|------------|---------|----------|
| 1 | 53.74% | 0.6914 | 56.25% | 0.6633 |
| 5 | 73.56% | 0.5300 | 79.35% | 0.4710 |
| 10 | 82.16% | 0.4045 | 79.55% | 0.4872 |
| 17 | 87.38% | 0.3016 | 89.50% | 0.2679 |
| 19 | 88.79% | 0.2705 | 90.60% | 0.2305 |
| 20 | 89.76% | 0.2553 | 91.85% | 0.2203 |
| 23 | 90.53% | 0.2313 | 92.05% | 0.2007 |
| 25 | 91.18% | 0.2211 | 93.25% | 0.1839 |
| 28 | 92.26% | 0.1997 | 93.10% | 0.1777 |
| 30 | 92.34% | 0.1938 | 91.30% | 0.2262 |

**Test accuracy: 93.92% | Test loss: 0.1674**

### Model WITHOUT augmentation

| Epoch | Train Acc | Train Loss | Val Acc | Val Loss |
|-------|-----------|------------|---------|----------|
| 1 | 55.52% | 0.6829 | 61.20% | 0.6376 |
| 5 | 79.54% | 0.4464 | 78.85% | 0.4650 |
| 10 | 89.41% | 0.2612 | 87.55% | 0.3112 |
| 13 | 92.93% | 0.1825 | 89.65% | 0.2624 |
| 15 | 94.99% | 0.1324 | 89.50% | 0.3164 |
| 17 | 96.70% | 0.0961 | 88.45% | 0.3302 |
| 20 | 97.98% | 0.0683 | 87.50% | 0.5898 |
| 25 | 98.87% | 0.0421 | 90.80% | 0.4893 |
| 30 | 99.24% | 0.0328 | 90.80% | 0.4573 |

**Test accuracy: 89.55% | Test loss: 0.2583**

### Comparison

| Metric | With Augmentation | Without Augmentation |
|---|---|---|
| Final train acc (ep 30) | 92.34% | 99.24% |
| Final val acc (ep 30) | 91.30% | 90.80% |
| **Test accuracy** | **93.92%** | **89.55%** |
| **Test loss** | **0.1674** | **0.2583** |
| Overfitting indicator | Train ≈ Val | Train >> Val (gap 8.44%) |

Without augmentation: the model memorized training data (99.24% train vs 89.55% test — a 9.69% generalization gap). With augmentation: the model learned generalizable features (92.34% train vs 93.92% test — test even slightly exceeds training due to Dropout being disabled at inference).
