import pandas as pd

# Load dataset
df = pd.read_csv("data/creditcard.csv")

print("========== DATASET INFORMATION ==========")

# Shape
print("\nDataset Shape:")
print(df.shape)

# Basic information
print("\nDataset Info:")
print(df.info())

# Statistical summary
print("\nStatistical Summary:")
print(df.describe())

# Duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Missing values
print("\nMissing Values:")
print(df.isnull().sum().sum())

# Target distribution
print("\nClass Distribution:")
print(df["Class"].value_counts())

# Target percentage
print("\nClass Percentage:")
print(df["Class"].value_counts(normalize=True) * 100)