# 🧠 Price Prediction by ML (YZTA Datathon 2025)

This project was developed as part of the YZTA Datathon 2025 competition. The task was to predict product prices based on various features such as nutritional value, category, market, and more.

## 📁 Dataset Description

The dataset includes:
- `train.csv`: Labeled data with product prices
- `testFeatures.csv`: Unlabeled test data to predict
- `sample_submission.csv`: Format example for predictions

### Features:
- `tarih` (date)
- `ürün` (product name)
- `ürün besin değeri` (nutritional value)
- `ürün kategorisi` (category)
- `ürün fiyatı` (price – target)
- `ürün üretim yeri`, `market`, `şehir` (location attributes)

---

## 🛠 Models Used

### 1. 🔹 Linear Regression (Baseline)
- Trained a simple linear model
- Achieved basic accuracy with minimal feature engineering
- Good for interpretability, but limited on complex patterns

### 2. 🔸 Random Forest Regressor (Improved)
- Trained with `sklearn.ensemble.RandomForestRegressor`
- Handled non-linear relationships and categorical data better
- Significantly improved prediction performance (lower RMSE)

---

## 📊 Preprocessing Steps

1. Converted `tarih` to datetime, extracted `year`, `month`, `day`
2. Applied `LabelEncoder` to categorical features:
   - `ürün`, `ürün kategorisi`, `ürün üretim yeri`, `market`, `şehir`
3. Dropped unused columns (`tarih`, `id`)
4. Used the following features for training:
