import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

from xgboost import XGBClassifier


# ==========================================
# 1. Load Dataset
# ==========================================

df = pd.read_csv("data/creditcard_featured.csv")

X = df.drop("Class", axis=1)
y = df["Class"]


# ==========================================
# 2. Train/Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)


# ==========================================
# 3. Calculate Class Weight
# ==========================================

negative = (y_train == 0).sum()
positive = (y_train == 1).sum()

scale_pos_weight = negative / positive

print("\nGenuine:", negative)
print("Fraud  :", positive)
print("Scale Pos Weight:", scale_pos_weight)


# ==========================================
# 4. Base XGBoost
# ==========================================

xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="aucpr",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1
)


# ==========================================
# 5. Hyperparameter Search Space
# ==========================================

param_grid = {
    "n_estimators": [200, 300, 400],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.03, 0.05, 0.1],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5],
    "gamma": [0, 0.1, 0.2]
}


# ==========================================
# 6. Randomized Search
# ==========================================

search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_grid,
    n_iter=10,
    scoring="average_precision",
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1
)


print("\nStarting Hyperparameter Tuning...")

search.fit(X_train, y_train)

print("\nHyperparameter Tuning Completed!")


# ==========================================
# 7. Best Parameters
# ==========================================

print("\nBest Parameters:")
print(search.best_params_)

print("\nBest Cross-Validation PR-AUC:")
print(search.best_score_)


# ==========================================
# 8. Best Model
# ==========================================

best_model = search.best_estimator_


# ==========================================
# 9. Test Prediction
# ==========================================

y_pred = best_model.predict(X_test)

y_proba = best_model.predict_proba(X_test)[:, 1]


# ==========================================
# 10. Evaluation
# ==========================================

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nROC-AUC:")
print(roc_auc_score(y_test, y_proba))

print("\nPR-AUC:")
print(average_precision_score(y_test, y_proba))