# Email Spam Classification

Machine learning project for classifying emails as **spam** or **not spam** using **Word2Vec embeddings** and a **feedforward neural network** built with TensorFlow/Keras.

## Project Overview

This project builds an end-to-end NLP pipeline for email spam classification. The raw email text is converted into numerical vectors using **Word2Vec**, and each email is represented by the **average of its word embeddings**. These vectors are then used as input to a neural network for binary classification.

## Objective

The goal is to predict whether an email is:

- `1` → spam
- `0` → not spam

using only the text content of the email.

## Dataset

The dataset used in this project is:

- `spam_or_not_spam.csv`

It contains two columns:

- `email`: the raw email text
- `label`: the target class (`1` for spam, `0` otherwise)

## Methodology

### 1. Data Loading

The dataset is loaded with pandas, and the email text and labels are separated into input and output variables.

### 2. Text Tokenization

Each email is converted into a list of words using simple whitespace splitting.

### 3. Word Embeddings with Word2Vec

A **Word2Vec** model is trained on the tokenized emails in order to generate vector representations for individual words.

### 4. Email Vector Representation

Since each email contains multiple words, each email is converted into a single fixed-length vector by taking the **average of the word vectors** of all words in the email.

### 5. Data Preparation

The final email vectors are converted into NumPy arrays so they can be used as input for the neural network.

### 6. Train-Test Split

The dataset is split into:

- **75% training data**
- **25% test data**

### 7. Neural Network Model

The classification model is a feedforward neural network with the following architecture:

- Dense layer with **12 neurons** and **ReLU** activation
- Dense layer with **100 neurons** and **ReLU** activation
- Output layer with **1 neuron** and **sigmoid** activation

The model is compiled using:

- **Loss function:** `binary_crossentropy`
- **Optimizer:** `adam`
- **Metric:** `accuracy`

### 8. Training

The model is trained with:

- **Epochs:** 20
- **Batch size:** 100

### 9. Evaluation

The model is evaluated on the test set using:

- **Precision**
- **Recall**
- **F1-score**

## Repository Structure

```text
email-spam-classification/
├── data/
│   └── spam_or_not_spam.csv
├── src/
│   └── spam_word2vec_nn.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
