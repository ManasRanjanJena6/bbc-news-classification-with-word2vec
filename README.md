![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Project%20Status-Completed-brightgreen)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)

# 📰 BBC News Classification using Word2Vec

This project demonstrates a full pipeline for text classification on the **BBC News dataset** using Word2Vec embeddings and a Random Forest classifier.  
The goal is to classify news articles into categories like **business, entertainment, politics, sport,** and **tech**.

---

## 📁 Dataset

- **Source**: `bbc_text_cls.csv`  
- Each record contains:
  - `text`: The body of the news article.
  - `labels`: The corresponding news category.

---

## 🛠️ Workflow Overview

1. **Data Preprocessing**
   - Clean text: remove non-alphabetic characters, lowercase conversion.
   - Tokenization, stopword removal, and stemming (NLTK).

2. **Exploratory Data Analysis**
   - Word count distribution per category.
   - Sentence-level statistics.
   - Readability scores (Flesch, Dale-Chall).

3. **Word2Vec Embedding**
   - Trains a Word2Vec model from scratch on the preprocessed corpus.
   - Converts each article into a 100-dimensional vector by averaging word embeddings.

4. **Model Training**
   - `RandomForestClassifier` with 200 trees and max depth of 10.
   - Data split using `train_test_split`.

5. **Evaluation**
   - Train/Test accuracy.
   - Classification report: Precision, Recall, F1-score.
   - Confusion matrix.
   - Sample predictions (actual vs predicted labels).

---

## 📈 Final Results

- **Train Accuracy**: ~93%  
- **Test Accuracy**: ~94%

---

## 🔍 Sample Prediction Output

- **Overall Sample Accuracy**: ~76%

```text
Text snippet: market analysts are predicting that the economy...
👉 Actual: business, Predicted: business

Text snippet: the striker scored a hat-trick in yesterday's match...
👉 Actual: sport, Predicted: sport
