Forest cover prediction

# 🌲 Forest Cover Prediction using Machine Learning

> “Among the trees and terrain, data whispers the story of the earth — we listen through models.”
## 📌 Project Overview

This project predicts the *forest cover type* of a given geographic region based on topographic and environmental data. Using powerful machine learning algorithms, the model learns to distinguish forest types like spruce/fir, lodgepole pine, or cottonwood from terrain features.

## 🌱 Why It Matters

Forest cover type classification is crucial for:

- Environmental monitoring
- Wildlife conservation
- Wildfire risk management
- Sustainable land use planning

By using machine learning, we aim to automate and scale ecological insights from spatial datasets.

## 📊 Dataset

- *Source*: A local copy of the dataset is included in this repository for convenience.
- *Records*: 581,012 samples
- *Features*: 54 (10 continuous + 44 binary soil & wilderness types)
- *Target*: Cover_Type (1–7 classes)

| Cover Type Label | Forest Type           |
|------------------|------------------------|
| 1                | Spruce/Fir             |
| 2                | Lodgepole Pine         |
| 3                | Ponderosa Pine         |
| 4                | Cottonwood/Willow      |
| 5                | Aspen                  |
| 6                | Douglas-fir            |
| 7                | Krummholz              |

## 🧠 Approach

1. *EDA*: Histograms, pairplots, correlation analysis
2. *Preprocessing*:
   - StandardScaler for numerical features
   - OneHotEncoder / LabelEncoder for categorical (if needed)
   - Train-Test Split
3. *Models Implemented*:
   - Logistic Regression (Baseline)
   - Random Forest Classifier
   - XGBoost
   - K-Nearest Neighbors
   - SVM
4. *Metrics*:
   - Accuracy, Precision, Recall
   - Confusion Matrix
   - Classification Report

## ⚙ How to Run

git clone https://github.com/SnehaSinghwe/ML-Project-Hub/tree/main/Forest%20cover
cd Forest_cover_prediction.ipynb
