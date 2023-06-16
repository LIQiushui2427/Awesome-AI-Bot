import pandas as pd
import sys
import argparse

from backtesting import Backtest

from backtesting.lib import plot_heatmaps, crossover


parser = argparse.ArgumentParser(description='Enter the data names and strategies to run a backtest.')

# Get data
df = pd.read_csv('./data/US#^GSPC.csv').set_index('Date')
df.index = pd.to_datetime(df.index)
df = df.query('Date > "2022-01-01"')

# Get Strategy
# setting path
sys.path.append('C:/Users/lqs/OneDrive - The Chinese University of Hong Kong/projects')
print(sys.path)
from strategies.DivergeWithMarketBreadth import *
test_strategy = DivergenceStrategy

# Run Backtest
bt = Backtest(df, test_strategy, cash=1e6, hedging=True, exclusive_orders=True, trade_on_close=True, commission=0.0002)


results, heatmap = bt.optimize(rsi_period = range(15,30),sma_period = range(8,25),maximize='Sharpe Ratio', return_heatmap=True)
print(bt._strategy)
print(results)

bt.plot(results = results)
plot_heatmaps(heatmap)