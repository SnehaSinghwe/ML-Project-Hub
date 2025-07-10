# ❤ Heart Disease Prediction using Machine Learning

> “Before a heart breaks, it sends a signal. In numbers and nuance, we listen, and we learn.”

## 📌 Project Overview

This project leverages machine learning to predict the likelihood of heart disease in a patient based on clinical data. By analyzing health indicators such as age, blood pressure, cholesterol, and more, the model aims to assist medical professionals in early diagnosis and treatment planning.

## 🌟 Why This Project?

Heart disease is a leading cause of death worldwide. Early detection can save lives — and machine learning offers an assistive tool for clinicians by identifying hidden patterns within patient data. This model aspires to become a proactive guardian of heart health.

## 🩺 Dataset Description

- *Source*: A local copy of the dataset is included in this repository for convenience.
- *Samples*: ~300 patients
- *Target*: 0 (No disease) / 1 (Disease)

| Feature | Description |
|--------|-------------|
| age | Age of the patient |
| sex | Gender |
| cp | Chest pain type |
| trestbps | Resting blood pressure |
| chol | Serum cholesterol |
| fbs | Fasting blood sugar |
| restecg | Resting ECG results |
| thalach | Max heart rate achieved |
| exang | Exercise-induced angina |
| oldpeak | ST depression |
| slope, ca, thal | Additional heart-related parameters |

## 🧠 ML Workflow

### 🔍 1. Data Preprocessing
- Handling missing values (if any)
- Encoding categorical features (LabelEncoding, OneHotEncoding)
- Feature scaling using StandardScaler

### 🧪 2. Exploratory Data Analysis (EDA)
- Correlation heatmaps
- Class distribution visualization
- Feature distributions and pairplots

### ⚙ 3. Model Training
- Logistic Regression
- K-Nearest Neighbors
- Support Vector Machine
- Random Forest
- XGBoost

### 🧾 4. Evaluation Metrics
- Accuracy
- Precision, Recall, F1-Score
- ROC-AUC Curve
- Confusion Matrix

## 📈 Results Snapshot

| Model              | Accuracy | ROC-AUC |
|-------------------|----------|---------|
| Logistic Regression | 86%      | 0.89    |
| Random Forest      | 89%      | 0.91    |
| XGBoost            | 92%      | 0.93    |

> Note: Model performance varies based on preprocessing and cross-validation strategy.

## 📊 Visualizations

- Confusion Matrix Heatmaps
- ROC-AUC Curves
- Feature Importance from Tree-based Models
- Heart disease risk comparison across demographics

## 💡 Future Scope

- Build an interactive *Streamlit app* for doctors and patients
- Add explainability using *SHAP / LIME*
- Extend model to support multi-class or stage-wise diagnosis
- Integrate with wearable data (e.g., Fitbit or Apple Health)

## ⚙ Technologies Used

- Python, Jupyter Notebook
- NumPy, Pandas, Scikit-learn
- XGBoost, Matplotlib, Seaborn

## 💻 How to Run
git clone https://github.com/SnehaSinghwe/ML-Project-Hub/tree/main/Heart%20disease
cd Heart_Disease_Prediction_using_ANN.ipynb
