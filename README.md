# MNIST Classification with KNN and Logistic Regression

This repository contains a Jupyter notebook that performs classification on the MNIST dataset using K-Nearest Neighbors (KNN) and Logistic Regression. The notebook includes evaluation metrics and fine-tuning steps for both models.

## Introduction
The MNIST dataset is a widely-used dataset in the field of machine learning and computer vision. It consists of 70,000 images of handwritten digits (0-9), each of size 28x28 pixels. The goal of this project is to build and compare two different machine learning models, K-Nearest Neighbors (KNN) and Logistic Regression, for the task of classifying these handwritten digits.

K-Nearest Neighbors (KNN) is a simple yet powerful algorithm used for both classification and regression tasks. It works by finding the most similar instances (nearest neighbors) to a given data point based on a distance measure, such as Euclidean distance. In this project, we explore how KNN performs on the MNIST dataset and fine-tune its parameters to achieve optimal classification accuracy.

Logistic Regression, despite its name, is a linear model for binary classification tasks. It estimates probabilities using a logistic function and predicts the class with the highest probability. We apply Logistic Regression to the MNIST dataset and optimize its hyperparameters to enhance its performance.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Usage](#usage)
- [Model Training and Evaluation](#model-training-and-evaluation)
  - [K-Nearest Neighbors (KNN)](#k-nearest-neighbors-knn)
    - [Training](#training)
    - [Evaluation](#evaluation)
    - [Fine-Tuning](#fine-tuning)
  - [Logistic Regression](#logistic-regression)
    - [Training](#training-1)
    - [Evaluation](#evaluation-1)
    - [Fine-Tuning](#fine-tuning-1)
- [Results](#results)
- [Conclusion](#conclusion)
- [Acknowledgements](#acknowledgements)

## Dataset
The MNIST dataset consists of 70,000 images of handwritten digits (0-9), each of size 28x28 pixels. It is a benchmark dataset for evaluating machine learning algorithms and is widely used in the research community.

## Requirements
- Python 3.x
- Jupyter Notebook
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

You can install the required packages using:
```bash
pip install numpy pandas scikit-learn matplotlib
```
## Usage
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/mnist-classification.git
   cd mnist-classification


## Model Training and Evaluation

### K-Nearest Neighbors (KNN)

#### Training
The KNN model is trained on the MNIST dataset using a default or initial value of k.

#### Evaluation
The performance of the KNN model is evaluated using metrics such as accuracy, precision, recall, and F1-score.

#### Fine-Tuning
Fine-tuning the KNN model involves selecting the optimal value of k through grid search to improve performance.

### Logistic Regression

#### Training
The Logistic Regression model is trained on the MNIST dataset using default or initial hyperparameters.

#### Evaluation
The performance of the Logistic Regression model is evaluated using metrics such as accuracy, precision, recall, and F1-score.

#### Fine-Tuning
Fine-tuning the Logistic Regression model involves adjusting hyperparameters such as regularization strength (C) using randomized search due to CPU load constraints.

## Results
The results section includes a comparison of the performance metrics (accuracy, precision, recall, F1-score) for both KNN and Logistic Regression models before and after fine-tuning.

## Conclusion
This project demonstrates the effectiveness of KNN and Logistic Regression models for classifying handwritten digits from the MNIST dataset. Fine-tuning the models significantly improves their performance.

## Acknowledgements
- The MNIST dataset is publicly available and can be found [here](http://yann.lecun.com/exdb/mnist/).
- This project uses the Scikit-learn library for model training and evaluation.
