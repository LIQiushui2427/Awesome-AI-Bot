from backtesting import Backtest, Strategy
from backtesting.lib import crossover, SignalStrategy, TrailingStrategy, plot_heatmaps, crossover
from backtesting.test import SMA

import pandas as pd

mydata = pd.read_excel('HSI.xlsx', index_col='Date', usecols=['Date', 'Open', 'High', 'Low', 'Close', '睇升', '睇跌'])

class SmaCross(SignalStrategy,
               TrailingStrategy):
    n1 = 10
    n2 = 25
    n_bullish = 8
    n_bearish = 14
    def init(self):
        # In init() and in next() it is important to call the
        # super method to properly initialize the parent classes
        super().init()
        
        print(self.data.df.睇升.head())
        # Precompute the two moving averages
        sma1 = self.I(SMA, self.data.Close, self.n1)
        sma2 = self.I(SMA, self.data.Close, self.n2)
        bullish = self.I(SMA, self.data.睇升, self.n_bullish)
        bearish = self.I(SMA, self.data.睇跌, self.n_bearish)

        # Where sma1 crosses sma2 upwards. Diff gives us [-1,0, *1*]
        signal = (pd.Series(sma1) > sma2).astype(int).diff().fillna(0)

        
        signal += pd.Series(bullish).diff().fillna(0)  # Add bullish signal
        signal += pd.Series(bearish).shift(2).diff().fillna(0)  # Add bearish signal

        signal = signal.replace(-1, 0)  # Upwards/long only
        
        
        # Use 95% of available liquidity (at the time) on each order.
        # (Leaving a value of 1. would instead buy a single share.)
        entry_size = signal * .95
                
        # Set order entry sizes using the method provided by 
        # `SignalStrategy`. See the docs.
        self.set_signal(entry_size=entry_size)
        
        # Set trailing stop-loss to 2x ATR using
        # the method provided by `TrailingStrategy`
        self.set_trailing_sl(8)
        self.set_atr_periods(50)
            
bt = Backtest(mydata, SmaCross, cash=1e12, hedging=True,exclusive_orders=True, trade_on_close=True) # 112%
stats = bt.run()
bt.plot()
print(stats)