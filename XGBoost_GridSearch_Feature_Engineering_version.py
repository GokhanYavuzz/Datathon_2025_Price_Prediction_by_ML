# Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

# Load data
train = pd.read_csv("/kaggle/input/yzta-datathon-2025-dataset/train.csv")
test = pd.read_csv("/kaggle/input/yzta-datathon-2025-dataset/testFeatures.csv")

train_data = train.copy()
test_data = test.copy()

# Feature: datetime split
for df in [train_data, test_data]:
    df["tarih"] = pd.to_datetime(df["tarih"])
    df["year"] = df["tarih"].dt.year
    df["month"] = df["tarih"].dt.month
    df["day"] = df["tarih"].dt.day
    df["weekday"] = df["tarih"].dt.weekday  # Günün haftası

# Feature: combine product + market
train_data["ürün_market"] = train_data["ürün"] + "_" + train_data["market"]
test_data["ürün_market"] = test_data["ürün"] + "_" + test_data["market"]

# Label encode categorical columns
categorical_cols = ["ürün", "ürün kategorisi", "ürün üretim yeri", "market", "şehir", "ürün_market"]
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    train_data[col] = le.fit_transform(train_data[col])
    test_data[col] = le.transform(test_data[col])
    label_encoders[col] = le

# Prepare features and target
X = train_data.drop(columns=["ürün fiyatı", "tarih"])
y = train_data["ürün fiyatı"]
X_test = test_data.drop(columns=["id", "tarih"])

# XGBoost model
xgb = XGBRegressor(objective="reg:squarederror", random_state=42)

# GridSearch for hyperparameter tuning
param_grid = {
    "n_estimators": [100, 300],
    "max_depth": [5, 10],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 1.0]
}

grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring="neg_root_mean_squared_error",
    cv=3,
    n_jobs=-1,
    verbose=1
)

# Fit model and find best parameters
grid_search.fit(X, y)

# Print best score and parameters
print("Best RMSE:", -grid_search.best_score_)
print("Best Parameters:", grid_search.best_params_)

# Predict using the best model
best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test)

# Prepare submission
submission = pd.DataFrame({
    "id": test["id"],
    "ürün fiyatı": predictions
})
submission.to_csv("submission.csv", index=False)
