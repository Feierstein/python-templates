# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Setting up display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
sns.set(style="whitegrid")

# Load the dataset
# Athena Query? and cleanup
# result needs to be converted to a pandas dataframe
# df = pd.read_csv('your_dataset.csv')
# df = pd.read_excel('your_dataset.xlsx')

# 1. Overview of the dataset
print("Shape of the dataset: ", df.shape)  # Number of rows and columns
print("Column names: ", df.columns)  # Display all column names
print("Data types: \n", df.dtypes)  # Display data types of each column
print("First few rows: \n", df.head())  # Preview the first few rows of the data
print("Last few rows: \n", df.tail())  # Preview the last few rows of the data

# 2. Summary statistics
print("\nSummary statistics (numeric):")
print(df.describe())  # Summary stats for numeric columns

print("\nSummary statistics (categorical):")
print(df.describe(include=['object']))  # Summary stats for categorical columns

# 3. Checking for missing data
print("\nMissing values in each column:")
print(df.isnull().sum())

# Visualization of missing values
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Data Heatmap")
plt.show()

# 4. Univariate analysis
# Numerical features
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
for col in numeric_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()

# Categorical features
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(y=col, data=df, order=df[col].value_counts().index)
    plt.title(f'Count of categories in {col}')
    plt.show()

# 5. Bivariate analysis
# Correlation matrix for numeric features
plt.figure(figsize=(10, 6))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Categorical vs Numerical
for col in categorical_cols:
    for num_col in numeric_cols:
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=col, y=num_col, data=df)
        plt.title(f'{num_col} distribution across {col}')
        plt.show()

# 6. Multivariate analysis
# Pairplot for numeric data
sns.pairplot(df[numeric_cols])
plt.show()

# 7. Handling missing data
# Drop columns with too many missing values (threshold)
threshold = 0.6  # Adjust threshold as needed
df = df[df.columns[df.isnull().mean() < threshold]]

# Fill missing values with mean/median/mode
df.fillna(df.median(), inplace=True)  # Can also use df.fillna(df.mean()) or df.fillna(df.mode().iloc[0])

# 8. Outlier detection
for col in numeric_cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(df[col])
    plt.title(f'Outliers in {col}')
    plt.show()

# 9. Feature engineering (optional)
# - Feature scaling: StandardScaler, MinMaxScaler, etc.
# - Feature transformation (log, square root, etc.)
# - Creating new features from existing ones

# Example: Scaling features
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df[numeric_cols]), columns=numeric_cols)

# 10. Save the cleaned dataset (optional)
# df.to_csv('cleaned_dataset.csv', index=False)
