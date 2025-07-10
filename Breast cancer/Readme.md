Breast cancer detection

# 🩺 Breast Cancer Detection Using Machine Learning

> "In data, we seek the silent warnings of disease. In code, we write the language of life."

## 📌 Project Overview

This project aims to build a machine learning model for early detection of breast cancer using structured clinical data. Early diagnosis can make a profound difference in treatment success, and this project is a small step towards harnessing AI for impactful healthcare.

## 💡 Motivation

Breast cancer is among the most common cancers affecting women worldwide. Timely detection through automated systems can assist medical professionals, reduce diagnostic errors, and bring healthcare closer to those in need. With this in mind, the model was trained to predict whether a tumor is benign or malignant using labeled medical data.

## 🧠 Techniques Used

- *Data Preprocessing*: Handling missing values, encoding categorical features, normalization
- *Exploratory Data Analysis (EDA)*: Statistical summaries, visualizations (correlation heatmaps, pairplots)
- *Model Building*:
  - Logistic Regression
  - Random Forest
  - Support Vector Machine
  - XGBoost
- *Model Evaluation*: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- *Hyperparameter Tuning*: GridSearchCV / RandomizedSearchCV for optimal performance

## 📂 Dataset

- *Source*: A local copy of the dataset is included in this repository for convenience.
- *Attributes*: 30 numeric features extracted from digitized images of a breast mass
- *Target Variable*: Diagnosis (M = Malignant, B = Benign)

## 🧪 How to Run

1. Clone the repository:
   
   git clone https://github.com/SnehaSinghwe/ML-Project-Hub/tree/main/Breast%20cancer
   cd Breast_Cancer_Prediction_test.ipynb
