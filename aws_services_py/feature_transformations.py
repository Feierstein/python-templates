import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

# Function to add lag features
def add_lag_features(df, column, lag=1):
    df[f'{column}_lag_{lag}'] = df[column].shift(lag)
    return df

# Function to add rolling statistics (mean, std, min, max, etc.)
def add_rolling_features(df, column, window_size=3):
    df[f'{column}_rolling_mean_{window_size}'] = df[column].rolling(window=window_size).mean()
    df[f'{column}_rolling_std_{window_size}'] = df[column].rolling(window=window_size).std()
    return df

# Function to add time-based features (year, month, day, weekday)
def add_time_features(df, date_column):
    df['year'] = df[date_column].dt.year
    df['month'] = df[date_column].dt.month
    df['day'] = df[date_column].dt.day
    df['weekday'] = df[date_column].dt.weekday
    return df

# Function to apply one-hot encoding to categorical features
def one_hot_encode(df, categorical_columns):
    df = pd.get_dummies(df, columns=categorical_columns)
    return df

# Function to apply standard scaling to numeric columns
def standard_scale(df, numeric_columns):
    scaler = StandardScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    return df

# Function to apply min-max scaling to numeric columns
def min_max_scale(df, numeric_columns):
    scaler = MinMaxScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    return df

# Function to impute missing values
def impute_missing(df, columns, strategy='mean'):
    imputer = SimpleImputer(strategy=strategy)
    df[columns] = imputer.fit_transform(df[columns])
    return df


#non-linear relationships to test
# Function to add polynomial features (x^2, x^3, etc.)
def add_polynomial_features(df, column, degree=2):
    df[f'{column}_poly_{degree}'] = df[column] ** degree
    return df

# Function to add logarithmic transformation
def add_logarithmic_feature(df, column):
    df[f'{column}_log'] = np.log1p(df[column])  # log(1+x) handles zeros and negatives
    return df

# Function to add exponential transformation
def add_exponential_feature(df, column):
    df[f'{column}_exp'] = np.exp(df[column])
    return df

# Function to add square root transformation
def add_sqrt_feature(df, column):
    df[f'{column}_sqrt'] = np.sqrt(df[column].clip(lower=0))  # Clip to avoid sqrt of negative values
    return df

# Function to add reciprocal transformation
def add_reciprocal_feature(df, column):
    df[f'{column}_reciprocal'] = 1 / (df[column] + 1e-8)  # Add small constant to avoid division by zero
    return df



# Example feature engineering pipeline
def apply_all_transformations(df, date_column=None, categorical_columns=None, numeric_columns=None):
    # Add lag features for all numeric columns
    for col in numeric_columns:
        df = add_lag_features(df, col, lag=1)
    
    # Add rolling features for all numeric columns
    for col in numeric_columns:
        df = add_rolling_features(df, col, window_size=3)
    
    #these should be excluded ?
    #Add time-based features if a date column exists
    if date_column:
       df = add_time_features(df, date_column)
    
    #these should be excluded ?
    #Apply One-Hot Encoding for categorical columns
    # if categorical_columns:
    #   df = one_hot_encode(df, categorical_columns)
    
    # # Apply Standard Scaling for numeric columns
    # if numeric_columns:
    #     df = standard_scale(df, numeric_columns)
    
    # # Apply Min-Max Scaling for numeric columns
    # if numeric_columns:
    #     df = min_max_scale(df, numeric_columns)
    
    # # Impute missing values for all columns
    # df = impute_missing(df, df.columns)
    
    # Add Polynomial features for selected columns
    for col in numeric_columns:
        df = add_polynomial_features(df, col, degree=2)
    
    # Add Logarithmic transformation for numeric columns
    for col in numeric_columns:
        df = add_logarithmic_feature(df, col)
    
    # Add Exponential transformation for numeric columns
    for col in numeric_columns:
        df = add_exponential_feature(df, col)
    
    # Add Square Root transformation for numeric columns
    for col in numeric_columns:
        df = add_sqrt_feature(df, col)
    
    # Add Reciprocal transformation for numeric columns
    for col in numeric_columns:
        df = add_reciprocal_feature(df, col)
    
    return df

# Example usage with dummy data
if __name__ == "__main__":
    # Create a sample DataFrame
    df = pd.DataFrame({
        'date': pd.date_range(start='2020-01-01', periods=10),
        'sales': np.random.randint(100, 500, size=10),
        'store': ['A', 'B', 'A', 'B', 'A', 'C', 'C', 'A', 'B', 'C']
    })
    
    # AUTOMATED BY def classify_columns(df):
    # Define columns
    date_column = 'date'
    categorical_columns = ['store']
    numeric_columns = ['sales']
    
    # Apply all transformations
    df_transformed = apply_all_transformations(df, date_column, categorical_columns, numeric_columns)
    
    # Display transformed DataFrame
    print(df_transformed)


def classify_columns(df):
    # Define lists to hold the classified columns
    date_columns = []
    categorical_columns = []
    numeric_columns = []
    
    # Iterate over the DataFrame columns
    for column in df.columns:
        # Check if the column is of datetime type
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            date_columns.append(column)
        # Check if the column is categorical or object type
        elif pd.api.types.is_categorical_dtype(df[column]) or pd.api.types.is_object_dtype(df[column]):
            categorical_columns.append(column)
        # Check if the column is numeric
        elif pd.api.types.is_numeric_dtype(df[column]):
            numeric_columns.append(column)
    
    return date_columns, categorical_columns, numeric_columns