# Parkinson’s Disease Detection using Machine Learning
“In every quiver of the voice, in every subtle change, there lies a story. This model listens — not to words, but to the silence between them.”

## Project Overview
This project aims to detect *Parkinson’s disease*, a neurodegenerative disorder, using machine learning techniques applied to biomedical voice measurements. The goal is to build an intelligent system that can support early diagnosis and monitoring, aiding doctors and caregivers in improving patient outcomes.

## Why This Project Matters
Parkinson's disease affects *motor control, leading to tremors, stiffness, and voice modulation issues. Diagnosis is traditionally clinical, but machine learning enables **non-invasive, early, and **objective analysis* using subtle vocal indicators.
This model serves as a compassionate assistant — detecting the unseen, the unheard, and the yet-to-be-known.

## Dataset Overview
- *Source*: A local copy of the dataset is included in this repository for convenience.
- *Samples*: 195 voice recordings from 31 patients
- *Target Variable*: status (0 = healthy, 1 = Parkinson’s)

| Feature Name        | Description                               |
|---------------------|-------------------------------------------|
| MDVP:Fo(Hz)         | Average vocal fundamental frequency        |
| MDVP:Jitter(%)      | Variation in pitch (jitter)               |
| MDVP:Shimmer        | Variation in amplitude (shimmer)          |
| NHR / HNR           | Noise-to-Harmonics / Harmonics-to-Noise   |
| RPDE, DFA           | Nonlinear signal measures                 |
| PPE                 | Vocal fold signal deviation               |
| Status              | Diagnosis outcome (target)                |

## Machine Learning Pipeline

### Data Preprocessing
- Scaling (StandardScaler or MinMax)
- Handling missing values (if any)
- Stratified train-test split (to maintain class balance)

### Exploratory Data Analysis
- Correlation matrix
- Boxplots for key vocal features
- Class distribution

### Models Implemented
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- XGBoost
- K-Nearest Neighbors (KNN)
- Optional: Neural Network (MLP)

### Evaluation Metrics
- Accuracy
- Precision / Recall / F1-Score
- Confusion Matrix
- ROC-AUC Score

## Model Performance (Sample)
| Model              | Accuracy | ROC-AUC |
|--------------------|----------|---------|
| Logistic Regression| 87%      | 0.91    |
| SVM (RBF kernel)   | 89%      | 0.93    |
| XGBoost            | 91%      | 0.95    |
With tuning and cross-validation, performance may improve further.

## Visualizations Included
- Confusion matrix
- ROC curve
- Feature importance plots (Random Forest / XGBoost)
- Voice measure distributions per class

## How to Run
git clone https://github.com/SnehaSinghwe/ML-Project-Hub/tree/main/Parkinson%20diesease
cd Parkinsons-Disease-Prediction_using_Machine_Learning.ipynb
