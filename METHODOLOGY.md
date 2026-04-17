# Methodology — CNN Activation Visualization

This document provides a complete technical and conceptual explanation of every component in this project — from the mathematics of convolution to the interpretation of feature maps. It is written to be self-contained: no prior deep learning knowledge is assumed.

---

## 1. Problem statement

The task is binary image classification: given a photo, predict whether it shows a cat (label 0) or a dog (label 1). Beyond achieving high accuracy, the project aims to *understand* what the trained network has learned by visualizing the internal representations it builds during inference.

---

## 2. Dataset

**Cats vs Dogs** is a standard computer vision benchmark originally released by Microsoft as part of the Kaggle Dogs vs Cats competition (2013). It contains 25,000 labelled images, of which 23,262 are usable (1,738 were found to be corrupted and automatically discarded by TensorFlow Datasets).

### Split used in this project

| Split | Size | Purpose |
|-------|------|---------|
| Training | 16,000 images | Weight updates during training |
| Validation | 2,000 images | Monitored during training; no weight updates |
| Test | 4,652 images | Evaluated once after training is complete |

The validation set is held out during training and used only to monitor overfitting. The test set is never seen by the model until the final evaluation.

### Image preprocessing

All images are resized to **180 × 180 pixels** before being fed to the model. This standardization is required because neural networks operate on fixed-size tensors. Each image is represented as a 3D array of shape (180, 180, 3), where the three channels correspond to red, green, and blue pixel intensities in the range [0, 255].

A **Rescaling layer** inside the model divides all pixel values by 255, mapping them to the range [0, 1]. This normalization accelerates gradient-based optimization and prevents numerical instability during training.

---

## 3. Data augmentation

Deep learning models generalize better when trained on diverse data. With only 16,000 training images, there is a risk that the model memorizes specific image configurations rather than learning true visual features. Data augmentation addresses this by randomly transforming each image every time it is loaded during training.

### Transformations applied

**RandomFlip("horizontal")** — mirrors the image left-to-right with 50% probability. Cats and dogs appear in natural photos facing either direction; this transformation teaches the model that horizontal orientation is not a discriminative feature.

**RandomRotation(0.1)** — rotates the image by a random angle uniformly sampled from [−36°, +36°] (10% of a full 360° rotation). This simulates the natural variation in camera tilt and subject posture.

**RandomZoom(0.2)** — randomly zooms in or out by up to 20%, simulating variation in the distance between the camera and the subject.

### Implementation detail

In Keras, augmentation layers are placed as the first layers inside the model graph (after the input). This means augmentation is applied automatically during `model.fit()` and is automatically disabled during `model.evaluate()` and `model.predict()`. No separate augmentation pipeline or manual switching is required.

---

## 4. CNN architecture

### Why CNNs for images?

A standard (fully-connected) neural network would treat each of the 180×180×3 = 97,200 pixel values as an independent input feature. This ignores the fundamental structure of images: nearby pixels are correlated (an edge is formed by adjacent pixels), and the same object can appear anywhere in the frame (translation invariance).

Convolutional Neural Networks address both issues through two key operations: **convolution** and **pooling**.

### Convolution operation

A convolutional layer applies a set of small **filters** (also called kernels) across the spatial dimensions of the input. Each filter is a 3D tensor of shape (kernel_height, kernel_width, input_channels) — in this project, all filters are 3×3 with the same channel depth as the input.

The filter slides across the input image in steps of 1 pixel (stride = 1). At each position, it computes the element-wise product between the filter weights and the overlapping input region, then sums all products into a single scalar. This scalar becomes one value in the output **feature map**.

The key insight is that the same filter weights are applied at every spatial position — this is called **weight sharing**. A filter that detects vertical edges will detect them whether they appear in the top-left or bottom-right of the image.

After convolution, a **ReLU activation function** is applied element-wise: `f(x) = max(0, x)`. This introduces non-linearity — without it, stacking multiple linear layers would still produce a linear function, severely limiting expressive power.

### Pooling operation

MaxPooling2D(pool_size=2) partitions each feature map into 2×2 non-overlapping windows and retains only the maximum value in each window. This reduces the spatial dimensions by a factor of 2 in each direction (e.g., 178×178 → 89×89), which:
- Reduces the number of parameters in subsequent layers
- Makes the representation slightly invariant to small translations (a feature present in slightly different positions in two images will still produce similar pooled outputs)
- Increases the receptive field of each neuron — after two pooling layers, each output value summarizes a 5×5 region of the original input

### Architecture blocks

The model follows a standard encoder structure with five convolutional blocks, each applying progressively more filters to capture increasingly abstract features:

| Block | Conv Filters | Output after Conv | Output after Pool |
|-------|-------------|------------------|------------------|
| 1 | 32 | (178, 178, 32) | (89, 89, 32) |
| 2 | 64 | (87, 87, 64) | (43, 43, 64) |
| 3 | 128 | (41, 41, 128) | (20, 20, 128) |
| 4 | 256 | (18, 18, 256) | (9, 9, 256) |
| 5 | 256 | (7, 7, 256) | — (no pool) |

After Block 5, the representation is flattened to a 1D vector of length 7×7×256 = 12,544 values. A Dropout layer and a Dense output layer complete the classification head.

### Regularization

**Dropout(0.5)** randomly sets 50% of the flattened vector's values to zero during each training step. This prevents individual neurons from becoming overly specialized and forces the network to learn redundant representations. During inference, Dropout is automatically disabled — all neurons are active, but their outputs are scaled by 0.5 to compensate for the doubling of active units.

**L2 regularization** adds a penalty term to the loss function proportional to the sum of squared weights in the final Dense layer. This discourages very large weight values, which tend to correspond to overfitting to specific training examples. The penalty coefficient is 0.001, meaning the loss is increased by `0.001 × sum(weights²)` at every training step.

---

## 5. Training procedure

### Loss function

Binary cross-entropy is the standard loss function for two-class classification with sigmoid output:

```
L = -[y × log(p) + (1-y) × log(1-p)]
```

where `y ∈ {0, 1}` is the true label and `p ∈ (0, 1)` is the model's predicted probability. When the prediction is confident and correct, L is near zero. When the prediction is confident and wrong, L is very large, forcing a strong gradient update.

### Optimizer

RMSprop (Root Mean Square Propagation) is an adaptive learning rate optimizer. It maintains a moving average of the squared gradients for each parameter and divides the gradient by the square root of this average before the update. This normalizes the gradient magnitude across parameters, which is particularly useful for networks with very different gradient scales at different layers (as is the case in deep CNNs).

### Training results (epoch summary)

| Epoch | Train Acc | Train Loss | Val Acc | Val Loss |
|-------|-----------|------------|---------|----------|
| 1 | 54.38% | 0.6923 | 63.65% | 0.6668 |
| 5 | 73.64% | 0.5326 | 76.35% | 0.5050 |
| 10 | 81.99% | 0.3993 | 81.30% | 0.4290 |
| 20 | 89.34% | 0.2626 | 91.40% | 0.2214 |
| 30 | 92.54% | 0.1892 | 92.95% | 0.2017 |
| 40 | 94.26% | 0.1532 | 94.70% | 0.1481 |
| 46 | 94.82% | 0.1408 | **95.10%** | **0.1530** |
| 50 | 95.38% | 0.1277 | 93.40% | 0.2527 |

The best validation loss of 0.1530 was recorded at epoch 46 (val accuracy 95.10%). The saved checkpoint from this epoch was loaded for final test evaluation.

### ModelCheckpoint callback

The `ModelCheckpoint` callback monitors `val_loss` after each epoch. When a new minimum is detected, the full model (weights + architecture + optimizer state) is saved to disk. This ensures that the final model used for evaluation corresponds to the generalization peak, not the last epoch — which may have slightly degraded due to late-stage overfitting.

---

## 6. Evaluation

The best checkpoint was loaded and evaluated on the 4,652 test images:

| Metric | Value |
|--------|-------|
| Test accuracy | **94.43%** |
| Test loss | **0.1560** |

This means 4,393 of 4,652 test images were correctly classified. The test accuracy being close to the best validation accuracy (95.10%) confirms that the model generalizes well and is not overfitting to the validation set.

---

## 7. Activation visualization

### What is an activation?

When an image is passed through a trained CNN, each layer produces an output — a multi-dimensional array of numbers. This output is called the **activation** of that layer for that input. It represents how much each learned detector (filter) responded to the input at each spatial position.

For a Conv2D layer with 32 filters operating on a 178×178 input, the activation is a 3D tensor of shape (178, 178, 32) — 32 separate 2D grids, one per filter, each showing where in the image that filter's pattern was detected.

### Building the activation model

Keras computes neural networks as **computational graphs**, where each layer's output is a tensor node. We can create a new model that shares the same input tensor as the trained model but outputs at any intermediate node, not just the final prediction layer.

```python
layer_outputs = [layer.output for layer in model.layers
                 if isinstance(layer, (Conv2D, MaxPooling2D))]
activation_model = keras.Model(inputs=model.input, outputs=layer_outputs)
```

When `activation_model.predict(img_tensor)` is called, it runs one forward pass and simultaneously returns the activations at all 9 monitored layers.

### Interpreting feature maps by depth

**Layer 1 — conv2d (178×178×32):** Feature maps are large and retain most of the spatial detail of the original image. Individual maps respond to low-level patterns: horizontal edges, vertical edges, colour contrasts, diagonal gradients. Many filters activate across large regions of the image.

**Layers 2–4 — after first two pooling layers (87×87×64 and 43×43×64):** Maps shrink and become more abstract. Filters now respond to combinations of the patterns detected in Layer 1 — corners, textures, repeated structures. The image content is harder to recognize in individual maps.

**Layers 5–7 — Blocks 3 and 4 (41×41×128 and 18×18×256):** Maps are small and highly abstract. Most filters produce sparse activations — the majority of values are near zero (black), with strong activations only in specific spatial regions. These filters respond to object-level structures: eye regions, fur texture clusters, ear shapes.

**Layer 9 — conv2d_4 (7×7×256):** The final convolutional output is extremely small and abstract. Only a handful of the 256 filters activate significantly for a given image. At this depth, the network has built a compact representation that captures the most class-discriminative information from the entire image.

### Visualization normalization

Raw activation values vary enormously in magnitude across filters. To make subtle patterns visible, each feature map is normalized independently before display:

```python
channel_image -= channel_image.mean()    # zero-center
channel_image /= channel_image.std()     # unit variance
channel_image *= 64                      # scale for display
channel_image += 128                     # shift to mid-gray
channel_image = np.clip(channel_image, 0, 255).astype("uint8")
```

This ensures that even low-magnitude activations with meaningful spatial structure become visible, while preventing high-magnitude activations from saturating the colour scale.

---

## 8. Architecture comparison experiments

Three CNN architectures were trained for 30 epochs each to study the effect of depth and regularization strategy.

### Architecture A — Shallow (3 Conv blocks)

Three convolutional blocks (32→64→128 filters), Dropout(0.5) at the classification head.

| Epoch | Train Acc | Val Acc |
|-------|-----------|---------|
| 10 | 78.64% | 81.05% |
| 20 | 83.14% | 85.60% |
| 27 | 84.48% | 88.65% |
| 30 | 85.57% | 87.55% |

**Test accuracy: 87.75% | Test loss: 0.2876**

The shallow model converges quickly but plateaus early. The limited representational capacity (fewer parameters, shallower hierarchy) means it cannot capture the full complexity of the cats vs dogs distinction.

### Architecture B — Deep (5 Conv blocks, L2 regularization)

Five convolutional blocks (32→64→128→256→256 filters), L2(0.001) on the Dense layer.

| Epoch | Train Acc | Val Acc |
|-------|-----------|---------|
| 10 | 80.46% | 82.30% |
| 20 | 88.24% | 89.85% |
| 27 | 91.44% | 92.80% |
| 30 | 92.18% | 92.80% |

**Test accuracy: 93.49% | Test loss: 0.1721**

The deep model achieves the highest accuracy. The additional depth allows it to learn more abstract, discriminative features. However, the training-validation accuracy gap (92.18% vs 92.80%) is larger than Architecture C, suggesting slightly more overfitting.

### Architecture C — Lightweight with Dropout at each block

Four convolutional blocks (32→64→128→128 filters), Dropout(0.3) after each pooling layer plus Dropout(0.5) at the Dense head.

| Epoch | Train Acc | Val Acc |
|-------|-----------|---------|
| 10 | 74.81% | 77.85% |
| 20 | 80.99% | 85.35% |
| 25 | 83.35% | 86.85% |
| 30 | 83.93% | 88.00% |

**Test accuracy: 87.77% | Test loss: 0.2806**

Aggressive Dropout significantly reduces the training-validation gap (training accuracy 83.93% vs validation 88.00% — validation actually exceeds training, a hallmark of well-regularized training). The cost is lower peak accuracy, since Dropout also limits the model's effective capacity during training.

---

## 9. Multi-image activation analysis

Activations were computed for three images: Cat 1, Cat 2, and Dog 1.

**Early layer (conv2d) observations:** All three images produce structurally similar feature maps in the first layer. The same filters activate for horizontal edges in all three images, for vertical edges in all three, and so on. This confirms that early-layer filters are general-purpose edge and texture detectors, not cat- or dog-specific.

**Deep layer (conv2d_4) observations:** Feature maps differ substantially between the cat images and the dog image. Filters that respond strongly to cat facial structure (compact, rounded) are mostly inactive for the dog image, and vice versa. The activations become class-discriminative at depth.

Mean activation intensity decreases and becomes sparser as depth increases, as filters become more selective — only a small fraction of the 256 filters in the final layer activate significantly for any given image.

---

## 10. Augmentation experiment

The same 5-block CNN was trained for 30 epochs with and without augmentation.

### Without augmentation

Training accuracy rose rapidly to 99.24% by epoch 30, while validation accuracy plateaued around 91%, producing a training/validation gap of ~8%. This gap widened throughout training, a clear overfitting signature. The model memorized training image details rather than learning transferable features.

| Metric | Value |
|--------|-------|
| Final train acc | 99.24% |
| Final val acc | 90.80% |
| **Test accuracy** | **89.55%** |
| Test loss | 0.2583 |

### With augmentation

Training accuracy grew more slowly, reaching 92.34% by epoch 30, while validation accuracy tracked closely at 91.30%. The train/validation gap remained small throughout, indicating the model was learning genuine features rather than memorizing specific images.

| Metric | Value |
|--------|-------|
| Final train acc | 92.34% |
| Final val acc | 91.30% |
| **Test accuracy** | **93.92%** |
| Test loss | 0.1674 |

The augmented model achieves 4.37% higher test accuracy despite lower training accuracy — a demonstration that augmentation forces the model to learn more general, robust representations.

---

## 11. Conclusions

- A 5-block CNN with data augmentation, Dropout, and L2 regularization achieves **94.43% test accuracy** on Cats vs Dogs.
- Activation visualization confirms the theoretical prediction of hierarchical feature learning: edges → textures → shapes → object parts as depth increases.
- Deeper architectures achieve higher accuracy but show more overfitting. Aggressive Dropout reduces overfitting at the cost of lower peak accuracy.
- Data augmentation is the single most impactful intervention for generalization: it reduces test error by 4.37% relative to an identical model trained without it.
- Feature maps in early layers are similar across different input images (generic detectors). Feature maps in deep layers diverge significantly between cats and dogs (class-specific detectors).
