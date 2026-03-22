# 🍄 Mushroom Species Classifier

A deep learning image classification project that identifies mushroom species (genus-level) from photos using transfer learning with VGG16.

---

## Overview

This project uses the [Kaggle Mushrooms Classification dataset](https://www.kaggle.com/datasets/maysee/mushrooms-classification-common-genuss-images) containing images across 9 mushroom genera. A pre-trained VGG16 model is fine-tuned to classify mushroom images, with data augmentation applied to handle class imbalance.

---

## Features

- Image data loading and exploration with class distribution visualization
- Data augmentation (flip, rotation, zoom, color jitter) to balance underrepresented classes
- Transfer learning using VGG16 pre-trained on ImageNet
- Fine-tuning with a custom classification head (GlobalAveragePooling → Dropout → Softmax)
- Training with EarlyStopping and ModelCheckpoint callbacks
- Evaluation with accuracy/loss curves, classification report, and confusion matrix heatmap
- Single image prediction pipeline

---

## Dataset

- **Source:** [Kaggle — Mushrooms Classification (Common Genus Images)](https://www.kaggle.com/datasets/maysee/mushrooms-classification-common-genuss-images)
- **Classes (9 genera):** Agaricus, Amanita, Boletus, Cortinarius, Entoloma, Hygrocybe, Lactarius, Russula, Suillus

---

## Tech Stack

| Library | Purpose |
|---|---|
| TensorFlow / Keras | Model building and training |
| VGG16 (ImageNet) | Transfer learning base |
| scikit-learn | Label encoding, train/val/test split, metrics |
| pandas | Dataset management |
| matplotlib / seaborn | Visualizations |
| PIL / torchvision | Image loading and augmentation |

---

## Model Architecture

```
VGG16 (pre-trained, fully trainable)
    └── GlobalAveragePooling2D
    └── Dropout (0.5)
    └── Dense(9, activation='softmax')
```

- Input size: 256×256×3
- Optimizer: Adam (lr=1e-5)
- Loss: Categorical Crossentropy
- Epochs: up to 30 (with early stopping, patience=5)

---

## Results

- Training monitored via accuracy and loss curves
- Best model saved with `ModelCheckpoint` based on validation accuracy
- Evaluated on a held-out test set (20% of data)
- Metrics: per-class precision, recall, F1-score, confusion matrix

---

## How to Run

### 1. Get the dataset (Google Colab)
```python
from google.colab import files
files.upload()   # upload your kaggle.json API key
```
```bash
!kaggle datasets download -d maysee/mushrooms-classification-common-genuss-images
```

### 2. Run the notebook
Open `Mushroom_latest.ipynb` in Google Colab and run all cells in order.

### 3. Predict on a new image
```python
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

img = load_img("your_image.jpg", target_size=(256, 256))
img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

pred = model.predict(img_array)
print("Predicted:", label_encoder.inverse_transform([np.argmax(pred)])[0])
```

---

## Project Structure

```
mushroom-classifier/
├── Mushroom_latest.ipynb   # Full pipeline notebook
└── README.md
```

> **Note:** The trained model file (`mushroom_classifier.keras`) and dataset are not included due to size. Download the dataset from Kaggle using the instructions above.
