import yfinance as  yf
import pandas as pd
import numpy as np
import sys
 
# setting path
sys.path.append('C:/Users/lqs/OneDrive - The Chinese University of Hong Kong/projects')
print(sys.path)


from strategies.RSIStrategy import RSIStrategy
from backtesting import Backtest
from strategies.DivergeWithMarketBreadth import DivergeWithMarketBreadthStrategy
from backtesting.lib import plot_heatmaps, crossover

df = pd.read_csv('./data/US#^GSPC.csv')

df = df.query('Date > "2021-06-01"')

test_strategy = DivergeWithMarketBreadthStrategy

bt = Backtest(df, test_strategy, cash=1e10, hedging=True,exclusive_orders=True, trade_on_close=True, commission=0.0005)
# results, heatmap = bt.optimize(threshold = list(np.arange(0,1,0.05)), exit_portion = list(np.arange(0,1,0.05)),maximize='Sharpe Ratio', return_heatmap=True)
# results, heatmap = bt.optimize(threshold = list(np.arange(0,1,0.05)), x = list(np.arange(0,1,0.05)),y = list(np.arange(0,1,0.05)), maximize='Sharpe Ratio', return_heatmap=True, constraint=lambda p: p.x + p.y <= 1 and p.x > p.y)
# results = bt.run()

results, heatmap = bt.optimize(threshold = list(np.arange(0,0.2,0.01)), maximize='Sharpe Ratio', return_heatmap=True)

print(results)

bt.plot(results = results)
plot_heatmaps(heatmap)