# 🦋 Thyroid Cancer Detection using Machine Learning

> “In the hush of hormones and cells, whispers of imbalance arise. This model learns to listen — and to warn.”

## 📌 Project Overview

This project leverages machine learning to detect *thyroid cancer* using structured clinical data. By analyzing hormone levels and patient characteristics, it seeks to predict whether a thyroid condition may be benign or malignant, enabling *early intervention* and *targeted treatment*.

## 🌟 Why This Matters

Thyroid cancer is often treatable if caught early, but its symptoms are subtle and often misdiagnosed. With the rise of automated diagnostics, machine learning models can serve as an intelligent *second opinion* — flagging potential risks from patterns too complex for the human eye alone.

## 📊 Dataset Overview

- *Source*: A local copy of the dataset is included in this repository for convenience.
- *Samples*: ~1,000+ records
- *Target*: Binary (0 = Benign, 1 = Malignant)

| Feature             | Description                             |
|---------------------|-----------------------------------------|
| Age                | Patient age                             |
| Gender             | Male / Female                           |
| TSH                | Thyroid Stimulating Hormone              |
| T3 / T4            | Triiodothyronine / Thyroxine levels      |
| T4U                | Thyroxine Uptake                         |
| FTI                | Free Thyroxine Index                     |
| Tumor Size         | Nodule or lesion size (if available)     |
| Thyroid Cancer     | Target variable                          |

> Additional features may include biopsy results, imaging summaries, or hormone therapy history.

## 🧠 ML Workflow

### 🛠 1. Data Preprocessing
- Handling missing or noisy medical data
- Encoding categorical variables (Label or OneHot)
- Standardizing numerical features (MinMax / Z-score)

### 🔍 2. Exploratory Data Analysis (EDA)
- Class imbalance check
- Feature correlation heatmaps
- Distribution of hormone levels across classes

### 🤖 3. Model Building
- Logistic Regression
- Random Forest Classifier
- XGBoost / LightGBM
- Support Vector Machine (SVM)
- Optional: Neural Network (MLP)

### 📏 4. Evaluation Metrics
- Accuracy
- Precision, Recall, F1-Score
- ROC-AUC Curve
- Confusion Matrix

> For imbalanced data, *Recall* and *AUC* become critical — false negatives are costly in cancer detection.

## 📈 Results Snapshot

| Model              | Accuracy | ROC-AUC | F1-Score |
|-------------------|----------|---------|----------|
| Logistic Regression | 88%    | 0.91    | 0.87     |
| Random Forest       | 93%    | 0.95    | 0.91     |
| XGBoost             | 95%    | 0.97    | 0.93     |

## 📊 Visualizations

- ROC-AUC Curve
- Confusion Matrix
- Feature Importance (tree models)
- Hormone level distributions (boxplots)

## ⚙ How to Run
git clone https://github.com/SnehaSinghwe/ML-Project-Hub/tree/main/Thyriod%20cancer
cd thyroid-cancer-detection
