# 🛒 Price Prediction by ML | YZTA Datathon 2025

This project was developed as part of the **YZTA (Yapay Zeka ve Teknoloji Akademisi) Datathon 2025**, where the goal was to predict the **sales price of retail products** based on various features such as category, market, date, and nutritional value using Machine Learning.

---

## 📌 Objective

To build a machine learning model that accurately predicts the price of a product using historical data and categorical features.

---

## 📁 Dataset Description

The dataset consists of three CSV files:

- `train.csv`: Contains historical product sales including actual prices.
- `testFeatures.csv`: Contains unseen product data (without prices) to predict.
- `sample_submission.csv`: Format for Kaggle submission.

### 🧾 Features:

- `tarih`: Date the product was recorded
- `ürün`: Product name
- `ürün besin değeri`: Nutritional value (numeric)
- `ürün kategorisi`: Product category (e.g. meat, vegetables)
- `ürün üretim yeri`: Production location
- `market`: Market name
- `şehir`: City of sale
- `ürün fiyatı`: (only in training set) — target variable to predict

---

## 🧠 Model Used

- **Linear Regression** (`sklearn.linear_model`)
  - Simple, interpretable baseline model
  - Categorical features were label encoded
  - Date features (year, month, day) were extracted from `tarih`

---

## 🧪 How to Run

1. Clone this repo or upload files to a Kaggle notebook or Colab
2. Ensure the data files are in the correct working directory
3. Run the notebook/script:
   ```bash
   python model.py  # or run all cells in notebook
