import yfinance as  yf
import backtesting
import pandas as pd
import numpy as np
import sys

from backtesting import Backtest

from backtesting.lib import plot_heatmaps, crossover
from sklearn.impute import SimpleImputer
# setting path
sys.path.append('C:/Users/lqs/OneDrive - The Chinese University of Hong Kong/projects')
print(sys.path)
from strategies.BOLLStrategy import BOLLStrategy

# Download data from Yahoo Finance
base_df = yf.download('GC=F', start='2022-01-01', end='2022-12-31')
base_df['Date'] = base_df.index
# Read in CFTC data
df = pd.read_csv('../data/c_year_22.txt')
# Convert the column to datetime if not already in datetime format
df['Report_Date_as_YYYY-MM-DD'] = pd.to_datetime(df['Report_Date_as_YYYY-MM-DD'])  
# Set the date column as the index
start_date = pd.to_datetime('2022-01-01')
end_date = pd.to_datetime('2022-12-31')

# Boolean indexing to select rows within the date range
selected_rows = df[(df['Report_Date_as_YYYY-MM-DD'] >= start_date) & (df['Report_Date_as_YYYY-MM-DD'] <= end_date) & (df['Market_and_Exchange_Names'] == 'GOLD - COMMODITY EXCHANGE INC.')]
selected_rows.set_index('Report_Date_as_YYYY-MM-DD', inplace=True)

merged_df = base_df.merge(selected_rows, left_on=base_df.index, right_on='Report_Date_as_YYYY-MM-DD', how = 'outer')
merged_df.columns
merged_df.set_index('Date', inplace=True)


merged_df.interpolate(method='linear', inplace=True, limit_direction='forward')

merged_df.fillna(method='ffill', inplace=True)


imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

merged_df = pd.DataFrame(imp.fit_transform(merged_df), columns=merged_df.columns, index=merged_df.index)

# print(merged_df.head())

assert merged_df.isnull().sum().sum() == 0




test_strategy = BOLLStrategy

bt = Backtest(merged_df, test_strategy, cash=1e10, hedging=True,exclusive_orders=True, trade_on_close=True)

# results, heatmap = bt.optimize(threshold = list(np.arange(0,1,0.05)), exit_portion = list(np.arange(0,1,0.05)),maximize='Sharpe Ratio', return_heatmap=True)
# results, heatmap = bt.optimize(threshold = list(np.arange(0,1,0.05)), x = list(np.arange(0,1,0.05)),y = list(np.arange(0,1,0.05)), maximize='Sharpe Ratio', return_heatmap=True, constraint=lambda p: p.x + p.y <= 1 and p.x > p.y)
results = bt.run()

results, heatmap = bt.optimize(bollinger_std = list(np.arange(0,2.5,0.1)), bollinger_period = range(2,8), maximize='Sharpe Ratio', return_heatmap=True)

print(results)

bt.plot(results = results)
plot_heatmaps(heatmap)