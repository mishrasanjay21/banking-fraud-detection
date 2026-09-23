import pandas as pd

from sklearn.model_selection import train_test_split

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

# Print shapes
print("X_train Shape:", X_train.shape)
print("X_test Shape:", X_test.shape)

print("y_train Shape:", y_train.shape)
print("y_test Shape:", y_test.shape)

# Check class distribution
print("\nTraining Class Distribution:")
print(y_train.value_counts())

print("\nTesting Class Distribution:")
print(y_test.value_counts())