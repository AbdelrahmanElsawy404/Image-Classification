# Cats vs. Dogs Image Classifier

A deep learning Convolutional Neural Network (CNN) trained with TensorFlow and Keras to classify images as cats or dogs. This model leverages real-time data augmentation and regularizations to achieve high validation accuracy on the classic Kaggle `cats_vs_dogs` dataset.

---

## Performance Summary
* **Dataset Size:** 23,262 images (split 80/20 into training and validation sets)
* **Training Accuracy:** **85.0%**
* **Validation Accuracy:** **82.5%**
* **Total Parameters:** ~19 Million (mainly in the dense classification head)
* **Frameworks:** TensorFlow, Keras, NumPy, Pandas, Matplotlib

---

## Model Architecture

The model is built with a custom sequential CNN architecture featuring convolutional, pooling, batch normalization, and dropout layers to stabilize training and prevent overfitting:

| Layer (Type) | Output Shape | Kernel Size | Activation | Details / Regularization |
| :--- | :--- | :--- | :--- | :--- |
| **Input** | `(150, 150, 3)` | - | - | RGB image inputs resized to 150x150 |
| **Conv2D_1** | `(148, 148, 32)` | `3x3` | ReLU | Feature extraction (32 filters) |
| **MaxPooling2D_1**| `(74, 74, 32)` | `2x2` | - | Spatial downsampling |
| **BatchNorm_1** | `(74, 74, 32)` | - | - | Standardizes outputs of activation maps |
| **Dropout_1** | `(74, 74, 32)` | - | - | Drops 20% of units to prevent overfitting |
| **Conv2D_2** | `(72, 72, 64)` | `3x3` | ReLU | Deep feature extraction (64 filters) |
| **MaxPooling2D_2**| `(36, 36, 64)` | `2x2` | - | Spatial downsampling |
| **BatchNorm_2** | `(36, 36, 64)` | - | - | Standardizes outputs |
| **Dropout_2** | `(36, 36, 64)` | - | - | Drops 20% of units |
| **Conv2D_3** | `(34, 34, 128)`| `3x3` | ReLU | Deep feature extraction (128 filters) |
| **MaxPooling2D_3**| `(17, 17, 128)`| `2x2` | - | Spatial downsampling |
| **BatchNorm_3** | `(17, 17, 128)`| - | - | Standardizes outputs |
| **Dropout_3** | `(17, 17, 128)`| - | - | Drops 20% of units |
| **Flatten** | `(36992)` | - | - | Flattens 3D feature maps to 1D |
| **Dropout_4** | `(36992)` | - | - | Drops 50% of units to prevent dense overfitting |
| **Dense_1** | `(512)` | - | ReLU | Fully connected hidden layer |
| **Dense_2** | `(1)` | - | Sigmoid | Classification output (0 = Cat, 1 = Dog) |

---

## Data Augmentation
To regularize the model and improve generalization, real-time data augmentation was applied via Keras `ImageDataGenerator` during training:
- **Rescaling:** Pixel values normalized to `[0, 1]` (multiplied by `1/255`).
- **Rotation Range:** Up to 10 degrees rotation.
- **Width & Height Shifts:** Up to 10% translation shifts.
- **Shear Range:** Up to 10% shear transformations.
- **Zoom Range:** Up to 10% zoom.
- **Horizontal Flip:** Enabled.

---

## How to Set Up & Run

### 1. Prerequisites
Ensure you have Python 3 installed. You can install all dependencies via pip:
```bash
pip install tensorflow tensorflow-datasets pandas numpy matplotlib pillow requests
