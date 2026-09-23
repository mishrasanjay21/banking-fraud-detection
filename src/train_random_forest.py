import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score
)

# --------------------------------
# 1. Load Feature-Engineered Data
# --------------------------------

df = pd.read_csv("data/creditcard_featured.csv")

# --------------------------------
# 2. Features and Target
# --------------------------------

X = df.drop("Class", axis=1)
y = df["Class"]

# --------------------------------
# 3. Train/Test Split
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("X_train Shape:", X_train.shape)
print("X_test Shape:", X_test.shape)

# --------------------------------
# 4. Random Forest Model
# --------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# --------------------------------
# 5. Train Model
# --------------------------------

print("\nTraining Random Forest...")

model.fit(X_train, y_train)

print("Training Completed!")

# --------------------------------
# 6. Prediction
# --------------------------------

y_pred = model.predict(X_test)

# Fraud probability
y_proba = model.predict_proba(X_test)[:, 1]

# --------------------------------
# 7. Evaluation
# --------------------------------

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nROC-AUC Score:")
print(roc_auc_score(y_test, y_proba))