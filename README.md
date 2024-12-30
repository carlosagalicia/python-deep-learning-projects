# Deep Learning Projects Based on Deep Learning with Python by François Chollet

This repository contains four projects that serve as practical applications of deep learning concepts from the book Deep Learning with Python by François Chollet. Each project focuses on a different aspect of deep learning, including classification, regression, and working with Keras, a deep learning framework.

## Table of Contents
- Newswire Topic Classification
- Boston Housing Price Prediction
- MNIST Digit Classification
- IMDB Movie Review Sentiment Analysis

## Newswire Topic Classification
This project focuses on classifying Reuters newswires into 46 mutually exclusive topics. The model is built using Keras with a simple neural network consisting of fully connected layers. The data is preprocessed by converting words into vectors of 10,000 most frequent words, and the labels are one-hot encoded.

### Key Learnings
- Implemented multiclass classification using softmax activation.
- Applied word index decoding and vectorized the sequences for training.
- Built and trained a neural network model with Keras.

### Technologies
- **Libraries:** Keras, TensorFlow, Matplotlib, NumPy
- **Model Type:** Neural Network (Fully Connected Dense Layers)
- **Dataset:** Reuters newswires dataset

## Boston Housing Price Prediction
This project aims to predict the median value of houses in Boston using features like crime rate, property tax rate, etc., as inputs. A regression model is built using Keras, and the data is normalized before training. The model is validated using K-fold cross-validation.

### Key Learnings
- Applied data normalization to scale features.
- Built a regression model using fully connected layers.
- Implemented K-fold cross-validation for model validation.
- Analyzed and visualized model performance through Mean Absolute Error (MAE).

### Technologies
- **Libraries:** Keras, TensorFlow, Matplotlib, NumPy
- **Model Type:** Neural Network (Fully Connected Dense Layers)
- **Dataset:** Boston Housing dataset

## MNIST Digit Classification
This project demonstrates the classification of handwritten digits from the MNIST dataset. The neural network architecture is simple but effective, utilizing convolutional layers for better feature extraction, followed by fully connected layers for classification.

### Key Learnings
- Applied Convolutional Neural Networks (CNNs) for image classification.
- Utilized dropout regularization to prevent overfitting.
- Visualized training and validation accuracy and loss during training.

### Technologies
- **Libraries:** Keras, TensorFlow, Matplotlib, NumPy
- **Model Type:** Convolutional Neural Network (CNN)
- **Dataset:** MNIST dataset

## IMDB Movie Review Sentiment Analysis
This project is focused on binary sentiment classification of movie reviews from the IMDB dataset. The dataset consists of positive and negative reviews, and the model uses a neural network to classify the sentiment based on the words in the review text.

### Key Learnings
- Applied Natural Language Processing (NLP) techniques to process text data.
- Used embedding layers for efficient text representation.
- Built a simple binary classification model with Keras.

### Technologies
- **Libraries:** Keras, TensorFlow, Matplotlib, NumPy
- **Model Type:** Neural Network (Fully Connected Dense Layers)
- **Dataset:** IMDB movie reviews dataset


