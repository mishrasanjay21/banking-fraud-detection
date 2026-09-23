# import pandas as pd

# # Load dataset
# df = pd.read_csv("data/creditcard.csv")

# print("Original Shape:", df.shape)

# # Check duplicates
# print("Duplicate Rows:", df.duplicated().sum())

# # Remove duplicate rows
# df = df.drop_duplicates()

# print("Shape After Removing Duplicates:", df.shape)

# # Check missing values
# print("Missing Values:", df.isnull().sum().sum())

# # Check fraud distribution
# print("\nClass Distribution:")
# print(df["Class"].value_counts())


import pandas as pd

# Load dataset
df = pd.read_csv("data/creditcard.csv")

print("Original Shape:", df.shape)

# Check duplicates
print("Duplicate Rows:", df.duplicated().sum())

# Remove duplicates
df = df.drop_duplicates()

print("Shape After Removing Duplicates:", df.shape)

# Check missing values
print("Missing Values:", df.isnull().sum().sum())

# Check class distribution
print("\nClass Distribution:")
print(df["Class"].value_counts())

# Save cleaned dataset
df.to_csv("data/creditcard_cleaned.csv", index=False)

print("\nCleaned dataset saved successfully!")