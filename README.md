# 🧠 Price Prediction by ML | YZTA Datathon 2025

This repository presents a machine learning pipeline developed for the **YZTA Datathon 2025**, where the goal was to predict the prices of retail products using historical and categorical features.

---

## 📌 Problem Statement

Given product-related features such as nutritional value, category, market, production origin, and date, the task is to **accurately estimate the product's sales price**.

---

## 📁 Dataset Description

The dataset contains three CSV files:

- `train.csv`: Historical data including product prices (target)
- `testFeatures.csv`: Unlabeled test set
- `sample_submission.csv`: Format required for prediction submissions

**Features included:**
- `tarih` – date of price record  
- `ürün` – product name  
- `ürün besin değeri` – nutritional value  
- `ürün kategorisi` – category  
- `ürün üretim yeri`, `market`, `şehir` – location-based attributes  
- `ürün fiyatı` – price (target)

---

## 🧠 Models Used

### 🔹 Linear Regression (Baseline)
- Implemented using `sklearn.linear_model.LinearRegression`
- Fast and interpretable
- Weak in capturing non-linear patterns

### 🔸 Random Forest Regressor
- Implemented via `RandomForestRegressor` from `sklearn.ensemble`
- Handles non-linearity and mixed feature types well
- Provided a significant boost over the baseline model

### 🚀 XGBoost with GridSearchCV (Final Model)
- Implemented using `XGBRegressor` from `xgboost`
- Hyperparameter tuning via `GridSearchCV` to improve RMSE
- Additional feature engineering was applied:
  - Extracted `year`, `month`, `day`, `weekday` from `tarih`
  - Created a combined `ürün_market` feature
- Achieved the best performance on leaderboard

---

## 🧪 Preprocessing

- Datetime conversion: extracted `year`, `month`, `day`, `weekday`
- Label encoding for all categorical variables
- Combined product and market as a new feature
- Final features used for model training:
