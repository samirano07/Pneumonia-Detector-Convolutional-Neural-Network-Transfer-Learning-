# Pneumonia Detection from Chest X-Rays using Transfer Learning (EfficientNetB0)

A deep learning pipeline that classifies chest X-ray images as Normal or Pneumonia using transfer learning with EfficientNetB0 in TensorFlow/Keras.

## Description

This project implements an end-to-end medical image classification pipeline designed to detect pneumonia from chest X-ray scans. Using **EfficientNetB0** pre-trained on ImageNet, the model leverages transfer learning to achieve high diagnostic accuracy with reduced training overhead. The pipeline addresses a critical issue in the public Kaggle dataset by re-splitting the training data to fix the undersized original validation set (16 images). It includes data augmentation, training callbacks, extensive metric evaluations (ROC-AUC, confusion matrix, classification report), error analysis on misclassified samples, and **Grad-CAM (Gradient-weighted Class Activation Mapping)** heatmaps to explain model predictions by highlighting affected lung regions.

## Getting Started

### Dependencies

* Operating System: Windows 10/11, macOS, or Linux
* Python 3.8 or higher
* Required Python libraries:
  * `tensorflow` / `keras`
  * `kagglehub`
  * `scikit-learn`
  * `numpy`
  * `pandas`
  * `matplotlib`
  * `seaborn`
  * `opencv-python`

### Installing

* Clone this repository to your local machine:
  ```bash
  git clone https://github.com/your-username/pneumonia-detection-efficientnet.git
  cd pneumonia-detection-efficientnet
  ```
* Install the required dependencies:
  ```bash
  pip install tensorflow keras kagglehub scikit-learn numpy pandas matplotlib seaborn opencv-python
  ```

### Executing program

1. Download the dataset automatically via `kagglehub` within your script or run:
   ```python
   import kagglehub
   path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
   print("Dataset downloaded to:", path)
   ```
2. Run the main training and evaluation script:
   ```bash
   python main.py
   ```
3. To generate Grad-CAM visual explainability maps for test samples:
   ```bash
   python explainability.py
   ```


## Acknowledgments

Inspiration, datasets, and code structure reference:
* [Kaggle Chest X-Ray Images (Pneumonia) Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
* [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
* [Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization](https://arxiv.org/abs/1610.02391)
