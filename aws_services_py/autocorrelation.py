# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pandas.plotting import lag_plot
from statsmodels.stats.stattools import durbin_watson

# Load your time series data
# Assuming df is a DataFrame with a 'value' column that contains the time series
# df = pd.read_csv('your_timeseries.csv')

# Example: Generate a random time series (for demonstration)
np.random.seed(0)
df = pd.DataFrame({'value': np.random.randn(100).cumsum()})

# 1. Plotting the autocorrelation function (ACF)
plt.figure(figsize=(10, 6))
plot_acf(df['value'], lags=40)
plt.title('Autocorrelation Function (ACF)')
plt.show()

# 2. Plotting the partial autocorrelation function (PACF)
plt.figure(figsize=(10, 6))
plot_pacf(df['value'], lags=40, method='ywm')
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()

# 3. Perform Durbin-Watson test for autocorrelation
dw_stat = durbin_watson(df['value'].diff().dropna())  # First-difference to check stationarity
print(f'Durbin-Watson statistic: {dw_stat}')

# Interpretation: Values around 2 suggest no autocorrelation. 
# Values < 2 indicate positive autocorrelation, and values > 2 indicate negative autocorrelation.

# 4. Visualize the lag plot to visually inspect autocorrelation
plt.figure(figsize=(6, 6))
lag_plot(df['value'])
plt.title('Lag Plot')
plt.show()

# 5. Check the autocorrelation at specific lags
# The Pandas autocorrelation function
for lag in range(1, 11):
    autocorr_value = df['value'].autocorr(lag=lag)
    print(f'Autocorrelation at lag {lag}: {autocorr_value}')
