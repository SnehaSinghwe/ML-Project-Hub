# 🚗 Vehicle Price Prediction using Machine Learning

## 📌 Project Overview

This project aims to *predict the price of a vehicle* based on various features such as brand, model, fuel type, engine capacity, and mileage. The objective is to build a robust regression model that estimates the fair market value of a used vehicle.

Such a system can assist:
- Car dealerships in pricing
- Buyers in checking fair value
- Sellers in setting expectations

---

## 💡 Motivation

With the rapid growth of the second-hand automobile market, pricing inconsistencies are common. Factors like brand premium, depreciation, mileage, and fuel efficiency all influence value — but not always transparently.

This project was born from the need to:
- Bring fairness and insight to automotive pricing
- Predict resale value using interpretable ML
- Support consumers and businesses alike

---

## 🧠 Problem Statement

A *regression* problem:
- Input: Vehicle features (make, model, year, fuel type, etc.)
- Output: Predicted price in INR/USD

The goal is to minimize prediction error and provide accurate price estimates based on real-world features.

---

## 📊 Dataset

- *Source:* A local copy of the dataset is included in this repository for convenience.
- *File Format:* CSV
- *Target Variable:* Price

### 🔑 Key Features:

| Feature Name       | Description                          |
|--------------------|--------------------------------------|
| Make / Brand       | Manufacturer (e.g., Maruti, Honda)   |
| Model              | Vehicle model name                   |
| Year               | Year of manufacture                  |
| Fuel Type          | Petrol, Diesel, CNG, Electric        |
| Transmission       | Manual / Automatic                   |
| Owner              | Number of previous owners            |
| Kilometers Driven  | Total distance driven (in km)        |
| Engine / Power     | Engine capacity & performance        |
| Mileage            | Fuel efficiency                      |
| Location           | Sale region                          |
| Price              | (Target) Selling price of vehicle    |

---

## 🔧 ML Workflow

### 🧹 Data Preprocessing
- Handle missing values & inconsistent formats
- One-Hot Encoding for categorical features
- Feature engineering (e.g., age of car = current year – year)
- Outlier treatment
- Log transformation of skewed data

### ⚙ Regression Models Tried
- Linear Regression
- Random Forest Regressor
- XGBoost Regressor
- Gradient Boosting Regressor
- Support Vector Regressor (SVR)

### 📏 Evaluation Metrics
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score (Coefficient of Determination)

---

## 📈 Results

| Model                 | RMSE     | R² Score |
|----------------------|----------|----------|
| Linear Regression     | 1.2 lakhs | 0.78     |
| Random Forest         | 0.85 lakhs | 0.91    |
| XGBoost               | *0.72 lakhs* | *0.94* |
| SVR                   | 1.1 lakhs | 0.80     |

> XGBoost gave the most reliable predictions with the lowest error and high interpretability.

---

## ⚙ How to Run
git clone https://github.com/yourusername/vehicle-price-prediction.git
cd Vehicle_price_prediction.ipynb
