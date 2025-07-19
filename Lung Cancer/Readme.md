# Lung Cancer Prediction using Machine Learning

## Project Overview
This project aims to *predict the likelihood of lung cancer* based on patient-reported symptoms, demographics, and lifestyle factors. Lung cancer is often diagnosed in later stages — this tool offers early warnings based on subtle health signals.
A machine learning model is trained on structured survey or clinical data to identify high-risk individuals, enabling proactive screening and early intervention.

## Motivation
Lung cancer is one of the leading causes of cancer-related deaths globally. However, early-stage detection is rare and often missed due to vague symptoms. This project envisions:
- Accessible pre-screening using simple health data
- Support for mass screening programs
- AI-powered healthcare in rural or underserved areas

## Problem Statement
A *binary classification* task:
- 0: Low Risk (Healthy / No signs of lung cancer)
- 1: High Risk (Symptoms or patterns pointing to lung cancer)

*Input:* Lifestyle choices, clinical indicators, and health symptoms  
*Output:* Risk prediction (Yes/No)

## Dataset
- *Source:* A local copy of the dataset is included in this repository for convenience.
- *Target Variable:* LUNG_CANCER (1 = Cancer risk, 0 = Healthy)

| Feature Name           | Description                         |
|------------------------|-------------------------------------|
| Age                    | Patient's age                       |
| Gender                 | Male/Female                         |
| Smoking                | Smoker or Non-smoker                |
| Yellow Fingers         | Nicotine staining                   |
| Anxiety                | Mental health factor                |
| Peer Pressure          | Environmental influence             |
| Chronic Disease        | Any underlying conditions           |
| Fatigue, Wheezing      | Physical symptoms                   |
| Shortness of Breath    | Respiratory indicator               |
| Chest Pain             | High-severity symptom               |
| Coughing of Blood      | Critical warning signal             |
| Alcohol Consumption    | Lifestyle indicator                 |

## ML Workflow
### Data Preprocessing
- Label Encoding of binary/categorical features
- Handling missing values
- Feature Scaling
- Train-Test Split (80-20 or Stratified)
### Models Used
- Logistic Regression
- Random Forest
- XGBoost Classifier
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
### Evaluation Metrics
- Confusion Matrix
- Accuracy
- Precision, Recall, F1-Score
- ROC-AUC Score
- PR Curve (optional for imbalanced data)

## Results

| Model           | Accuracy | F1-Score | AUC Score |
|----------------|----------|----------|-----------|
| Random Forest   | 93%      | 0.91     | 0.94      |
| XGBoost         | *95%*  | *0.93* | *0.96*  |
| Logistic Reg.   | 88%      | 0.86     | 0.89      |
| SVM             | 90%      | 0.87     | 0.90      |
XGBoost showed exceptional performance and stability, making it the preferred model.

## How to Run
git clone https://github.com/SnehaSinghwe/ML-Project-Hub/tree/main/Lung%20Cancer
cd Lung_cancer.ipynb
