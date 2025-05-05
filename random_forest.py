# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# Load datasets
train = pd.read_csv("/kaggle/input/yzta-datathon-2025-dataset/train.csv")
test = pd.read_csv("/kaggle/input/yzta-datathon-2025-dataset/testFeatures.csv")
sample_submission = pd.read_csv("/kaggle/input/yzta-datathon-2025-dataset/sample_submission.csv")

# Make copies for processing
train_data = train.copy()
test_data = test.copy()

# Convert 'tarih' to datetime and extract year, month, day
for df in [train_data, test_data]:
    df["tarih"] = pd.to_datetime(df["tarih"])
    df["year"] = df["tarih"].dt.year
    df["month"] = df["tarih"].dt.month
    df["day"] = df["tarih"].dt.day

# Encode categorical columns
categorical_cols = ["ürün", "ürün kategorisi", "ürün üretim yeri", "market", "şehir"]
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    train_data[col] = le.fit_transform(train_data[col])
    test_data[col] = le.transform(test_data[col])
    label_encoders[col] = le

# Define features (X) and target (y)
X = train_data.drop(columns=["ürün fiyatı", "tarih"])
y = train_data["ürün fiyatı"]
X_test = test_data.drop(columns=["id", "tarih"])

# Train the Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42)
model.fit(X, y)

# Make predictions
predictions = model.predict(X_test)

# Create submission file
submission = pd.DataFrame({
    "id": test["id"],
    "ürün fiyatı": predictions
})

# Export to CSV
submission.to_csv("submission.csv", index=False)
