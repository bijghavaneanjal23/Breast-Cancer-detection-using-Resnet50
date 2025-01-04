# Breast Cancer Detection using ResNet50 and LIME

This project implements a deep learning-based model for breast cancer detection using histopathology images. The model leverages ResNet50, a convolutional neural network (CNN), and integrates LIME (Local Interpretable Model-Agnostic Explanations) to provide high classification accuracy and model transparency.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [License](#license)

## Project Overview

This project aims to develop an interpretable breast cancer detection model using deep learning. The ResNet50 model, pre-trained on ImageNet, is fine-tuned on the BreakHis dataset, which consists of histopathology images labeled as benign or malignant. The use of LIME provides transparency, helping pathologists understand the reasoning behind the model’s predictions.

## Dataset

The model is trained on the **BreakHis dataset**, which contains 7,909 microscopic images of breast tissue samples. These images are categorized into two classes: benign and malignant. The dataset includes images captured at various magnifications: 40x, 100x, 200x, and 400x.

### Dataset Preparation

- **Resizing**: Images are resized to a consistent size for model processing.
- **Normalization**: Pixel values are normalized to a range of [0,1].
- **Augmentation**: Techniques like flipping, rotation, and zooming are applied to enhance the robustness of the model.

## Installation

### Prerequisites

Make sure to have Python 3.x installed along with the following libraries:

- TensorFlow
- Keras
- LIME
- NumPy
- Matplotlib
- Seaborn
- Pandas

You can install the dependencies via pip:

```bash
pip install tensorflow keras lime numpy matplotlib seaborn pandas
```

### Hardware Requirements

- **Google Colab** or any environment with GPU or TPU acceleration is recommended for faster training.
- **Local machine**: Can be used for testing and running smaller models, but training on large datasets may require high computational resources.

## Usage

To run the project, you need to:

1. **Load the dataset**: Ensure the images are structured in directories corresponding to the two classes (benign and malignant).
2. **Train the model**: Use the provided code to train the ResNet50 model with the specified configurations.
3. **Evaluate the model**: After training, evaluate the model’s performance on the test set using various metrics (accuracy, precision, recall, etc.).
4. **Visualize with LIME**: Use the LIME library to generate explanations for the predictions made by the model.

## Methodology

### 1. **Data Preprocessing**

The dataset is preprocessed by resizing the images, normalizing pixel values, and augmenting the dataset using basic transformations to increase model robustness.

### 2. **Model Architecture**

The model uses ResNet50, a deep CNN that includes:
- Convolutional layers to extract features.
- Residual blocks to mitigate vanishing gradient problems.
- Fully connected layers for classification.

### 3. **Explainability with LIME**

LIME is integrated to provide local, interpretable explanations for each prediction. It highlights the areas in the image that were most influential in the model's decision-making process.

### 4. **Training Configuration**

- **Optimizer**: Adam with learning rate of 1e-4.
- **Loss Function**: Sparse categorical cross-entropy.
- **Epochs**: 150 epochs.
- **Batch Size**: 32.

### 5. **Model Evaluation**

The model is evaluated using:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **Confusion Matrix**

## Evaluation Metrics

| Metric          | Benign    | Malignant | Accuracy |
|-----------------|----------|-----------|----------|
| Precision       | 0.98     | 0.96      | 0.97     |
| Recall          | 0.95     | 0.99      | 0.97     |
| F1-Score        | 0.97     | 0.98      | 0.97     |
| Overall Accuracy| 0.97     | 0.97      | 0.97     |

The confusion matrix and classification report show high classification performance, with minimal false positives and false negatives.

## Results

- **Training and Validation Accuracy**: The model shows high accuracy during training and validation.
- **Confusion Matrix**: The model achieves excellent classification performance.
- **LIME Visualizations**: LIME provides clear visual explanations for both benign and malignant images, helping interpret the model’s decision-making process.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

This README file includes all necessary sections for documentation, making it easier for other developers to understand, replicate, and extend your project.
