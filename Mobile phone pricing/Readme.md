# Mobile Phone Price Prediction using Machine Learning
“Every feature tells a story — of power, pixels, and price. This model reads those stories to reveal the value behind the glass.”

## Project Overview
This project predicts the price *range of mobile phones* based on their technical specifications using classification and regression models. The goal is to help customers, retailers, or manufacturers understand how features translate into pricing tiers.
It can also serve as a recommendation system backend or market analysis tool for budget segmentation.

## Problem Statement
Given mobile phone specifications like RAM, processor speed, battery capacity, and more, predict the *price range* it falls into:
- 0: Low cost
- 1: Medium cost
- 2: High cost
- 3: Premium
Alternatively, this can also be modeled as a regression problem to predict the exact price.

## Dataset Overview
- *Source*: A local copy of the dataset is included in this repository for convenience.
- *Samples*: 2,000+ records
- *Target Variable*: price_range (0 to 3)

| Feature        | Description                        |
|----------------|------------------------------------|
| battery_power  | Total energy capacity              |
| dual_sim       | Whether dual SIM is supported      |
| four_g         | 4G support (1 = yes, 0 = no)        |
| internal_memory| Internal storage (in GB)           |
| mobile_wt      | Weight of mobile (grams)           |
| n_cores        | Number of processor cores          |
| px_height/width| Pixel resolution                   |
| ram            | RAM (in MB)                        |
| screen_width   | Screen size in pixels              |
| talk_time      | Talk time in hours                 |
| wifi, bluetooth, etc. | Boolean features             |

## ML Workflow
### 1. Data Preprocessing
- Feature normalization (StandardScaler/MinMax)
- Feature selection via correlation/importance
- Handling multicollinearity (if needed)

### 2. Exploratory Data Analysis (EDA)
- Price vs RAM/Storage correlation
- Heatmaps and KDE plots
- Feature importance ranking

### 3. Models Trained
- Logistic Regression
- Random Forest Classifier
- Decision Tree
- K-Nearest Neighbors
- XGBoost / LightGBM
- Optional: Linear Regression or SVR (for continuous price)

### 4. Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- ROC-AUC Curve (for binary splits)
- R² and RMSE (if using regression)

## Results Snapshot

| Model              | Accuracy |
|-------------------|----------|
| Logistic Regression | 86%     |
| Random Forest       | 91%     |
| XGBoost             | 93%     |
| KNN                 | 85%     |

## How to Run
git clone https://github.com/SnehaSinghwe/ML-Project-Hub/tree/main/Mobile%20phone%20pricing
cd Mobile_price_pricing.ipynb
