# %%
import yfinance as yf
import backtesting
import pandas as pd
import numpy as np
import sys

# setting path
sys.path.append("C:/Users/lqs/OneDrive - The Chinese University of Hong Kong/projects")
print(sys.path)
base_df = yf.download("GC=F", start="2022-01-01", end="2022-12-31")
base_df["Date"] = base_df.index

df = pd.read_csv("../data/c_year_22.txt")
# Convert the column to datetime if not already in datetime format
df["Report_Date_as_YYYY-MM-DD"] = pd.to_datetime(df["Report_Date_as_YYYY-MM-DD"])

start_date = pd.to_datetime("2022-01-01")
end_date = pd.to_datetime("2022-12-31")

# Boolean indexing to select rows within the date range
selected_rows = df[
    (df["Report_Date_as_YYYY-MM-DD"] >= start_date)
    & (df["Report_Date_as_YYYY-MM-DD"] <= end_date)
    & (df["Market_and_Exchange_Names"] == "GOLD - COMMODITY EXCHANGE INC.")
]
selected_rows.set_index("Report_Date_as_YYYY-MM-DD", inplace=True)


merged_df = base_df.merge(
    selected_rows,
    left_on=base_df.index,
    right_on="Report_Date_as_YYYY-MM-DD",
    how="outer",
)
merged_df.columns
merged_df.set_index("Date", inplace=True)

merged_df.interpolate(method="linear", inplace=True, limit_direction="forward")
merged_df = merged_df.fillna(method="ffill")


from strategies.RSIStrategy import RSIStrategy
from backtesting import Backtest
from strategies.CompoundStrategy import CompoundStrategy
from backtesting.lib import plot_heatmaps, crossover

test_strategy = RSIStrategy

bt = Backtest(
    merged_df,
    test_strategy,
    cash=1e10,
    hedging=True,
    exclusive_orders=True,
    trade_on_close=True,
)
# results, heatmap = bt.optimize(threshold = list(np.arange(0,1,0.05)), exit_portion = list(np.arange(0,1,0.05)),maximize='Sharpe Ratio', return_heatmap=True)
# results, heatmap = bt.optimize(threshold = list(np.arange(0,1,0.05)), x = list(np.arange(0,1,0.05)),y = list(np.arange(0,1,0.05)), maximize='Sharpe Ratio', return_heatmap=True, constraint=lambda p: p.x + p.y <= 1 and p.x > p.y)
# results = bt.run()

# results, heatmap = bt.optimize(rsi_period = range(9,25), rsi_1 = range(17,39), rsi_2 = range(70,80), maximize='Sharpe Ratio', return_heatmap=True)
results, heatmap = bt.optimize(
    rsi_period=range(9, 10),
    rsi_1=range(17, 30),
    rsi_2=range(70, 80),
    maximize="Sharpe Ratio",
    return_heatmap=True,
)

print(results)

bt.plot(results=results)
plot_heatmaps(heatmap)
# print maxinum sharpe ratio and its corresponding parameters

# def rolling_test(bt:Backtest):
#     """This function will test the tratrategy on rolling windows

#     It will split the data into 3 parts, training, validation and testing.
#     """

# %%
print("Type of heatmap: ", type(heatmap))
# print the ind exes with maximum sharpe ratio
print("Max location: ", heatmap.idxmax())
# %%
print("Bt: ", bt._strategy.rsi_period)
bt._strategy.rsi_period = 9
print("Bt: ", bt._strategy.rsi_period)
members = [
    attr
    for attr in dir(bt._strategy)
    if not callable(getattr(bt._strategy, attr)) and not attr.startswith("_")
]

print(members)
# %%
