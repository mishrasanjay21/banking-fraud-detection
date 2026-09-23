import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/creditcard.csv")

#-----------------------Fraud vs Genuine------------------------------
# plt.figure(figsize=(8, 5))

# sns.countplot(x="Class", data=df)

# plt.title("Genuine vs Fraud Transactions")
# plt.xlabel("Transaction Class")
# plt.ylabel("Number of Transactions")

# plt.xticks([0, 1], ["Genuine", "Fraud"])

# plt.show()

# --------------Transaction Amount Distribution---------------------

# plt.figure(figsize=(10, 5))

# sns.histplot(data=df, x="Amount", bins=50)

# plt.title("Transaction Amount Distribution")
# plt.xlabel("Transaction Amount")
# plt.ylabel("Number of Transactions")

# plt.show()


#--------------------Fraud vs Genuine Transaction Amount-----------------------

# plt.figure(figsize=(8, 5))

# sns.boxplot(data=df, x="Class", y="Amount")

# plt.title("Transaction Amount: Genuine vs Fraud")
# plt.xlabel("Transaction Class")
# plt.ylabel("Amount")

# plt.xticks([0, 1], ["Genuine", "Fraud"])

# plt.show()


#------------------Correlation Matrix--------------------------

# plt.figure(figsize=(14, 10))

# sns.heatmap(
#     df.corr(),
#     cmap="coolwarm",
#     linewidths=0.2
# )

# plt.title("Feature Correlation Matrix")

# plt.show()

