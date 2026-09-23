import pandas as pd
import joblib


# 1. Load trained model
model = joblib.load(
    "models/fraud_detection_xgboost.joblib"
)

# 2. Load feature names
feature_names = joblib.load(
    "models/feature_names.joblib"
)


# 3. Take one existing transaction for testing
df = pd.read_csv(
    "data/creditcard_featured.csv"
)

sample = df.drop("Class", axis=1).iloc[[0]]


# 4. Make sure feature order is correct
sample = sample[feature_names]


# 5. Predict fraud probability
fraud_probability = model.predict_proba(sample)[0][1]


# 6. Apply threshold
threshold = 0.5

if fraud_probability >= threshold:
    prediction = "FRAUD"
else:
    prediction = "GENUINE"


# 7. Print result
print("\nTransaction Prediction")
print("----------------------")

print(f"Fraud Probability: {fraud_probability:.4f}")
print(f"Threshold: {threshold}")
print(f"Prediction: {prediction}")