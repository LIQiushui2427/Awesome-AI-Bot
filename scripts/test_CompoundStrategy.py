import yfinance as yf
import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import plot_heatmaps

base_df = yf.download("ZW=F", start="2022-01-01", end="2022-12-31")
base_df["Date"] = base_df.index
# view_low_volume_df = base_df[base_df['Volume'] < 1000]
# view_low_volume_df
# base_df


df = pd.read_csv("./data/c_year_22.txt")
df["Report_Date_as_YYYY-MM-DD"] = pd.to_datetime(
    df["Report_Date_as_YYYY-MM-DD"]
)  # Convert the column to datetime if not already in datetime format

start_date = pd.to_datetime("2022-01-01")
end_date = pd.to_datetime("2022-12-31")

# Boolean indexing to select rows within the date range
selected_rows = df[
    (df["Report_Date_as_YYYY-MM-DD"] >= start_date)
    & (df["Report_Date_as_YYYY-MM-DD"] <= end_date)
    & (df["Market_and_Exchange_Names"] == "WHEAT-SRW - CHICAGO BOARD OF TRADE")
]
selected_rows.set_index("Report_Date_as_YYYY-MM-DD", inplace=True)
# selected_rows


merged_df = base_df.merge(
    selected_rows,
    left_on=base_df.index,
    right_on="Report_Date_as_YYYY-MM-DD",
    how="outer",
).fillna(method="bfill")
merged_df.columns
merged_df.set_index("Date", inplace=True)
# merged_df
# backtest dataframeindexdatetime

import sys

sys.path.append("C:/Users/lqs/OneDrive - The Chinese University of Hong Kong/projects")
from strategies.CompoundStrategy import CompoundStrategy


strategy = CompoundStrategy
bt = Backtest(
    merged_df,
    strategy,
    cash=1e8,
    hedging=True,
    exclusive_orders=True,
    trade_on_close=True,
)
results, heatmap = bt.optimize(
    n1=range(5, 20, 1),
    n2=range(5, 20, 1),
    exit_portion=list(np.arange(0, 1, 0.05)),
    maximize="Sharpe Ratio",
    constraint=lambda p: p.n1 < p.n2,
    return_heatmap=True,
)
bt.plot(results=results)
plot_heatmaps(heatmap)
print(results)
