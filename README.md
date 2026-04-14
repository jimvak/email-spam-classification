# Email Spam Classification

Machine learning project for classifying emails as **spam** or **not spam** using **Word2Vec embeddings** and a **feedforward neural network** built with TensorFlow/Keras.

## Project Overview

This project builds an end-to-end NLP pipeline for email spam classification. Raw email text is transformed into numerical vectors using Word2Vec, and each email is represented by the average of its word embeddings. These vectors are then used as input to a neural network for binary classification.

## Objective

The goal is to predict whether an email is **spam** (`1`) or **not spam** (`0`) using only the email text.

## Dataset

The dataset used in this project is `spam_or_not_spam.csv`.

It contains two columns:

- `email`: raw email text
- `label`: target class (`1` for spam, `0` otherwise)

## Methodology

### 1. Data Loading
The dataset is loaded with pandas, and the email text and labels are separated into input and output variables.

### 2. Text Tokenization
Each email is split into tokens using simple whitespace splitting.

### 3. Word Embeddings
A Word2Vec model is trained on the tokenized emails to generate vector representations for words.

### 4. Email Representation
Each email is converted into a single fixed-length vector by taking the average of its word vectors.

### 5. Data Preparation
The final email vectors are converted into NumPy arrays so they can be used as input for the neural network.

### 6. Train-Test Split
The dataset is split into **75% training data** and **25% test data**.

### 7. Neural Network Model
The model architecture is:

- Dense layer with 12 neurons and ReLU activation
- Dense layer with 100 neurons and ReLU activation
- Output layer with 1 neuron and sigmoid activation

The model is compiled with:

- Loss: `binary_crossentropy`
- Optimizer: `adam`
- Metric: `accuracy`

### 8. Training
The model is trained for **20 epochs** with a **batch size of 100**.

### 9. Evaluation
The model is evaluated using:

- Precision
- Recall
- F1-score

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
## How to Run

```bash
git clone https://github.com/jimvak/email-spam-classification.git
cd email-spam-classification
pip install -r requirements.txt
python src/spam_word2vec_nn.py
```
