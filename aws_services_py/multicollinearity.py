import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset (example: using the Boston Housing dataset)
from sklearn.datasets import load_boston
boston = load_boston()

# Create a DataFrame from the dataset
df = pd.DataFrame(boston.data, columns=boston.feature_names)
df['PRICE'] = boston.target  # Add target variable

# Overview of the dataset
print(df.head())

# Compute the correlation matrix
corr_matrix = df.corr()

# Plotting the heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()

from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create a DataFrame of the independent variables (exclude the target 'PRICE')
X = df.drop('PRICE', axis=1)

# Calculate VIF for each predictor
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

# Display the VIF values
print(vif_data)

#Interpretation of VIF:
#VIF = 1: No correlation between this predictor and the others.
#1 < VIF < 5: Moderate correlation; may not be a serious issue.
#IF ≥ 5: Significant multicollinearity; consider removing or combining correlated features.

# Perform Singular Value Decomposition (SVD)
_, s, _ = np.linalg.svd(X)
condition_number = max(s) / min(s)
print(f'Condition Number: {condition_number}')