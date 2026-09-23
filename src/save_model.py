import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


# 1. Load feature-engineered data
df = pd.read_csv("data/creditcard_featured.csv")

X = df.drop("Class", axis=1)
y = df["Class"]


# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# 3. Calculate scale_pos_weight
negative = (y_train == 0).sum()
positive = (y_train == 1).sum()

scale_pos_weight = negative / positive


# 4. Create final tuned XGBoost model
model = XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.7,
    colsample_bytree=0.7,
    min_child_weight=5,
    gamma=0.1,
    scale_pos_weight=scale_pos_weight,
    objective="binary:logistic",
    eval_metric="aucpr",
    random_state=42,
    n_jobs=-1
)


# 5. Train final model
print("Training final XGBoost model...")

model.fit(X_train, y_train)

print("Training completed!")


# 6. Create models folder
import os

os.makedirs("models", exist_ok=True)


# 7. Save model
model_path = "models/fraud_detection_xgboost.joblib"

joblib.dump(model, model_path)


print(f"\nModel saved successfully!")
print(f"Model path: {model_path}")


# 8. Save feature names
feature_names = list(X.columns)

joblib.dump(
    feature_names,
    "models/feature_names.joblib"
)

print("Feature names saved successfully!")