import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score
)

from xgboost import XGBClassifier


# --------------------------------
# 1. Load Data
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
# 4. Calculate Scale Position
# --------------------------------

negative = (y_train == 0).sum()
positive = (y_train == 1).sum()

scale_pos_weight = negative / positive

print("\nGenuine Transactions:", negative)
print("Fraud Transactions:", positive)
print("Scale Pos Weight:", scale_pos_weight)


# --------------------------------
# 5. XGBoost Model
# --------------------------------

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    objective="binary:logistic",
    eval_metric="aucpr",
    random_state=42,
    n_jobs=-1
)


# --------------------------------
# 6. Train
# --------------------------------

print("\nTraining XGBoost...")

model.fit(X_train, y_train)

print("Training Completed!")


# --------------------------------
# 7. Prediction
# --------------------------------

y_pred = model.predict(X_test)

y_proba = model.predict_proba(X_test)[:, 1]


# --------------------------------
# 8. Evaluation
# --------------------------------

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nROC-AUC Score:")
print(roc_auc_score(y_test, y_proba))