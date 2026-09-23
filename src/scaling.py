import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load feature-engineered dataset
df = pd.read_csv("data/creditcard_featured.csv")

# Features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Create scaler
scaler = StandardScaler()

# Fit only on training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform test data
X_test_scaled = scaler.transform(X_test)

# Print shapes
print("Original X_train Shape:", X_train.shape)
print("Scaled X_train Shape:", X_train_scaled.shape)

print("\nOriginal X_test Shape:", X_test.shape)
print("Scaled X_test Shape:", X_test_scaled.shape)

# Check first 5 rows
print("\nFirst 5 Scaled Training Rows:")
print(X_train_scaled[:5])