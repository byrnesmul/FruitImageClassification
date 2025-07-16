![](Pictures/UTA-DataScience-Logo.png)

# Fruit Image Classification with Transfer Learning

## One Sentence Summary
This project classifies images of fruits using transfer learning with three pretrained CNN models: ResNet50, EfficientNetB0, and MobileNetV2.

---

## Overview
The goal of this project is to build an image classification model that identifies fruit types from images using transfer learning. A custom fruit image dataset with five classes was used, with fewer than 100 images per class. The task was framed as a multi-class classification problem. We trained three different transfer learning models—ResNet50, EfficientNetB0, and MobileNetV2—and compared their performance using validation accuracy and ROC curves. Data augmentation was applied to evaluate its impact on model generalization.

---

## Summary of Work Done

### Data

- **Type**: Image dataset
- **Classes**: Apple_5, Banana_1, Cherry_1, Kiwi_1, Orange_1
- **Input size**: 100x100 or 224x224 RGB images
- **Subset**: ~100 images per class

---

### Preprocessing

- Limited dataset to 5 classes
- Resized images to 100x100 (MobileNetV2) and 224x224 (ResNet50, EfficientNet)
- Split dataset into training and validation using `image_dataset_from_directory`
- Used `tf.data` pipelines for performance optimization

---

### Data Visualization & Augmentation

- Augmentations applied: RandomFlip, RandomRotation, RandomZoom
- Augmented samples were visualized to verify transformations

---

### Problem Formulation

- **Input**: RGB image (100x100x3 or 224x224x3)
- **Output**: Class label (one of 5 fruit classes)
- **Task**: Multi-class image classification

---

## Models Trained

### Transfer Learning Models

- **ResNet50**
- **EfficientNetB0**
- **MobileNetV2**

All models used pre-trained ImageNet weights and a custom classification head.

---

## Training

- **Software**: Python 3, Google Colab
- **Libraries**: TensorFlow, Keras, matplotlib, scikit-learn
- **Epochs**: 10
- **Optimizer**: Adam
- **Loss**: SparseCategoricalCrossentropy
- **Environment**: Google Colab GPU

---

## Performance Comparison

All three models achieved **100% validation accuracy** after a few epochs. However, since the dataset was very small, the results may be **trivially high** and not generalize well to unseen data. Below is the ROC curve comparison:

![](Pictures/final_roccurves.png)

---

## Conclusions

All models achieved perfect accuracy (1.0) on the validation set, both with and without data augmentation. While this may seem ideal, it is likely a **trivial result due to the small dataset size and potential class imbalance**. For more meaningful insights, a larger and more balanced dataset is recommended. Data augmentation did not significantly impact performance because all models already achieved perfect accuracy without it.

---

## Future Work

- Use a larger fruit dataset (e.g., Fruits 360)
- Introduce regularization (dropout, L2) to reduce overfitting
- Evaluate on a held-out test set
- Apply model explainability tools like Grad-CAM

---

## How to Reproduce Results

1. Clone this repository  
2. Open notebooks in the following order:
    - `DataLoader.ipynb`
    - `TrainBaseModel.ipynb`
    - `TrainBaseModelAugmentation.ipynb`
    - `Train-ResNet50.ipynb`
    - `Train-EfficientNet.ipynb`
    - `Train-MobileNet.ipynb`
    - `CompareAugmentation.ipynb`
    - `CompareModels.ipynb`
3. Run all cells
4. *(Optional)* Generate ROC and accuracy comparison plots

---

## Files in Repository

| File                          | Description                                |
|-------------------------------|--------------------------------------------|
| `DataLoader.ipynb`            | Loads and prepares the dataset             |
| `TrainBaseModel.ipynb`        | Baseline training with MobileNetV2         |
| `TrainBaseModelAugmentation.ipynb` | MobileNetV2 with data augmentation   |
| `Train-ResNet50.ipynb`        | Transfer learning using ResNet50           |
| `Train-EfficientNet.ipynb`    | Transfer learning using EfficientNetB0     |
| `Train-MobileNet.ipynb`       | Transfer learning using MobileNetV2        |
| `CompareAugmentation.ipynb`   | ROC curve comparison (augmented vs. baseline) |
| `CompareModels.ipynb`         | ROC comparison of all three models         |
| `Pictures/`                   | Folder for plots and image outputs         |

---

## Software Setup

```bash
pip install tensorflow matplotlib scikit-learn
