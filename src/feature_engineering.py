# import pandas as pd

# # Load cleaned dataset
# df = pd.read_csv("data/creditcard_cleaned.csv")

# print("Original Shape:", df.shape)

# # Convert Time from seconds into hours
# df["Time_Hours"] = df["Time"] / 3600

# # Check new feature
# print("\nFirst 5 Time Values:")
# print(df[["Time", "Time_Hours"]].head())

# # Check columns

# print("\nColumns:")
# print(df.columns.tolist())


import pandas as pd

# Load cleaned dataset
df = pd.read_csv("data/creditcard_cleaned.csv")

print("Original Shape:", df.shape)

# Feature Engineering
df["Time_Hours"] = df["Time"] / 3600

# Separate features and target
X = df.drop("Class", axis=1)
y = df["Class"]

print("\nFeatures Shape (X):", X.shape)
print("Target Shape (y):", y.shape)

print("\nFeature Columns:")
print(X.columns.tolist())

print("\nTarget Distribution:")
print(y.value_counts())

# Save feature-engineered data
df.to_csv("data/creditcard_featured.csv", index=False)

print("\nFeature-engineered dataset saved successfully!")