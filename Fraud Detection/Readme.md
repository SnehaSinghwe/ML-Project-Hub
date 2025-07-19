# Fraud Detection using Machine Learning
 "In a world of patterns, lies hide like shadows. With data as our lantern, we chase away the dark."

## Project Overview
This project focuses on detecting fraudulent financial transactions using machine learning techniques. In an age where digital payments are soaring, fraud detection is critical to prevent financial crimes and protect users.
The system is trained on historical transaction data to distinguish between legitimate and fraudulent behavior, with a strong emphasis on imbalanced data handling.

## Motivation
Fraudulent transactions are rare — but the cost of missing them is immense. Traditional rule-based systems struggle with new, evolving fraud techniques. Machine learning offers the ability to *adapt, detect, and learn* from patterns hidden deep within massive transaction logs.

## Problem Statement
*Binary classification* problem:
- *0*: Genuine transaction  
- *1*: Fraudulent transaction  
The dataset is highly imbalanced, with frauds comprising a tiny fraction of the total — making *precision, recall, and AUC* far more meaningful than mere accuracy.

## Dataset
- *Source*: A local copy of the dataset is included in this repository for convenience.
- *Features*:
  - 28 anonymized numerical features (PCA-transformed)
  - Time, Amount features
  - Class: Target (0 = normal, 1 = fraud)

| Feature Type       | Description                             |
|--------------------|-----------------------------------------|
| Anonymized V1–V28  | PCA-transformed original features       |
| Amount             | Transaction amount                      |
| Time               | Seconds since first transaction         |
| Class              | Target variable (0 = legit, 1 = fraud)  |

## ML Approach
### Data Preprocessing
- Standardization of Time and Amount
- Handling class imbalance using:
  - *SMOTE (Synthetic Minority Oversampling Technique)*
  - *Under-sampling* of majority class

### Models Used
- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- Isolation Forest (for anomaly detection)
- Autoencoders (for deep unsupervised fraud spotting)

### Evaluation Metrics
- Confusion Matrix
- Precision, Recall, F1-Score
- ROC-AUC Score
- PR Curve (Precision-Recall)

## How to Run
git clone https://github.com/SnehaSinghwe/ML-Project-Hub/tree/main/Fraud%20Detection
cd Fraud_detection.ipynb
