from backtesting import Backtest, Strategy
from backtesting.lib import crossover, SignalStrategy, TrailingStrategy, plot_heatmaps, crossover
from backtesting.test import SMA
import pandas as pd
    
import pandas as pd
from backtesting.lib import SignalStrategy, TrailingStrategy


class SmaCross(SignalStrategy,
               TrailingStrategy):
    n1 = 10
    n2 = 25
    
    def init(self):
        # In init() and in next() it is important to call the
        # super method to properly initialize the parent classes
        super().init()
        
        # Precompute the two moving averages
        sma1 = self.I(SMA, self.data.Close, self.n1)
        sma2 = self.I(SMA, self.data.Close, self.n2)
        
        # Where sma1 crosses sma2 upwards. Diff gives us [-1,0, *1*]
        
        self.SMACrossSignal = (pd.Series(sma1) > sma2).astype(int).diff().fillna(0)
        
        # signal = signal.replace(-1, 0)  # Upwards/long only
        
        # Use 95% of available liquidity (at the time) on each order.
        # (Leaving a value of 1. would instead buy a single share.)
        self.entry_size = self.SMACrossSignal * 0.95
                
        # Set order entry sizes using the method provided by 
        # `SignalStrategy`. See the docs.
        # self.set_signal(entry_size=entry_size)
        
        # Set trailing stop-loss to 2x ATR using
        # the method provided by `TrailingStrategy`
        # self.set_trailing_sl(2)
    def apply(self):
        self.set_signal(entry_size=self.entry_size, exit_portion=[self.exit_portion for _ in range(len(entry_size))])
        self.set_trailing_sl(8)
        self.set_atr_periods(50)