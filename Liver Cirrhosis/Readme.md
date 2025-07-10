# 🩺 Liver Cirrhosis Stage Detection using Machine Learning

## 📌 Project Overview

This project aims to *predict the stage of liver cirrhosis* using structured patient data and machine learning models. Cirrhosis is a chronic and progressive condition—early detection of its stage can drastically influence treatment and patient outcomes.

The model learns from clinical features (e.g., lab results, symptoms) to classify cirrhosis as:
- *Early*
- *Moderate*
- *Severe*

---

## 🌱 Motivation

Liver cirrhosis is often diagnosed late when symptoms become severe. But clinical and biochemical signs manifest much earlier. This project is motivated by the desire to:
- Enable early detection using accessible data
- Support medical decision-making through AI
- Build cost-effective tools for regions with limited diagnostic access

---

## 🧠 Problem Statement

A *multi-class classification* task to determine cirrhosis stage:
- Class 0: Early Stage
- Class 1: Moderate Stage
- Class 2: Severe Stage

Input: Clinical features and health indicators  
Output: Stage classification (0, 1, or 2)

---

## 📊 Dataset

- *Source:* A local copy of the dataset is included in this repository for convenience.
- *Format:* CSV (structured data)
- *Target Variable:* Stage

### 🔑 Features Include:

| Feature Name             | Description                           |
|--------------------------|---------------------------------------|
| Age                     | Patient's age                         |
| Gender                  | Male/Female                           |
| Total Bilirubin         | Liver function marker                 |
| Direct Bilirubin        | Liver excretion efficiency            |
| Alkaline Phosphatase    | Bile duct obstruction marker          |
| SGPT, SGOT              | Enzymes indicating liver damage       |
| Total Proteins          | Protein synthesis capability          |
| Albumin                 | Blood protein levels                  |
| Albumin and Globulin Ratio | Indicates liver synthetic function |
| Class / Stage           | Target variable (0 = Early, 1 = Moderate, 2 = Severe) |

---

## 🔧 ML Workflow

### 1️⃣ Preprocessing
- Handle missing/null values
- Encode categorical variables
- Normalize numerical features
- Split dataset (Train/Test)

### 2️⃣ Model Building
- Logistic Regression
- Random Forest Classifier
- Support Vector Machine (SVM)
- XGBoost Classifier (Best performance)

### 3️⃣ Evaluation Metrics
- Accuracy
- Precision, Recall, F1-Score (Macro Avg)
- Confusion Matrix
- ROC-AUC (One-vs-Rest)

---

## 📈 Results

| Model              | Accuracy | F1-Score (Macro) |
|-------------------|----------|------------------|
| Random Forest      | 89%      | 0.88             |
| XGBoost            | *91%*  | *0.90*         |
| SVM                | 87%      | 0.86             |

---

## ⚙ How to Run
git clone https://github.com/your-username/liver-cirrhosis-detector.git
cd Liver_cirrhosis.ipynb
