# Model Architecture Reference

## Layer-by-layer shape flow

Input: (None, 180, 180, 3)

| # | Layer | Type | Output Shape | Parameters |
|---|-------|------|-------------|------------|
| 1 | input_1 | InputLayer | (None, 180, 180, 3) | 0 |
| 2 | sequential (augmentation) | Sequential | (None, 180, 180, 3) | 0 |
| 3 | rescaling | Rescaling | (None, 180, 180, 3) | 0 |
| 4 | conv2d | Conv2D(32, 3×3, ReLU) | (None, 178, 178, 32) | 896 |
| 5 | max_pooling2d | MaxPooling2D(2×2) | (None, 89, 89, 32) | 0 |
| 6 | conv2d_1 | Conv2D(64, 3×3, ReLU) | (None, 87, 87, 64) | 18,496 |
| 7 | max_pooling2d_1 | MaxPooling2D(2×2) | (None, 43, 43, 64) | 0 |
| 8 | conv2d_2 | Conv2D(128, 3×3, ReLU) | (None, 41, 41, 128) | 73,856 |
| 9 | max_pooling2d_2 | MaxPooling2D(2×2) | (None, 20, 20, 128) | 0 |
| 10 | conv2d_3 | Conv2D(256, 3×3, ReLU) | (None, 18, 18, 256) | 295,168 |
| 11 | max_pooling2d_3 | MaxPooling2D(2×2) | (None, 9, 9, 256) | 0 |
| 12 | conv2d_4 | Conv2D(256, 3×3, ReLU) | (None, 7, 7, 256) | 590,080 |
| 13 | flatten | Flatten | (None, 12,544) | 0 |
| 14 | dropout | Dropout(0.5) | (None, 12,544) | 0 |
| 15 | dense | Dense(1, Sigmoid) + L2 | (None, 1) | 12,545 |

Total parameters: 991,041

## Activation model monitored layers

| Index | Layer name | Output shape |
|-------|-----------|-------------|
| 0 | conv2d | (1, 178, 178, 32) |
| 1 | max_pooling2d | (1, 89, 89, 32) |
| 2 | conv2d_1 | (1, 87, 87, 64) |
| 3 | max_pooling2d_1 | (1, 43, 43, 64) |
| 4 | conv2d_2 | (1, 41, 41, 128) |
| 5 | max_pooling2d_2 | (1, 20, 20, 128) |
| 6 | conv2d_3 | (1, 18, 18, 256) |
| 7 | max_pooling2d_3 | (1, 9, 9, 256) |
| 8 | conv2d_4 | (1, 7, 7, 256) |

Spatial resolution: 178 → 89 → 87 → 43 → 41 → 20 → 18 → 9 → 7 pixels
Filter count:      32  → 32  → 64  → 64  → 128 → 128 → 256 → 256 → 256 channels
