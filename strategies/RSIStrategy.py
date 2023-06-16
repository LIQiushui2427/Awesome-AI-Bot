from backtesting import Backtest, Strategy
from backtesting.lib import crossover, SignalStrategy, TrailingStrategy
from backtesting.test import SMA
from strategies.utils import RSI
import pandas as pd
import pandas_ta as pta
import numpy as np

"""
RSI Strategy
This strategy will buy when the RSI is below 30 and sell when the RSI is above 70.

To initialize the strategy, you need to pass in the following parameters:
    rsi_period: the period of RSI
It require the inputed dataframe has the following columns:
    Close
"""

class RSIStrategy(SignalStrategy, TrailingStrategy):
    
    rsi_1 = 29
    rsi_2 = 74
    rsi_period = 16
    exit_portion = 0.5
    entry_size_ratio = 0.8
    
    def init(self):
        super().init()
                
        self.rsi = pta.rsi(self.data.df['Close'], length = self.rsi_period)
        
        # print(self.rsi.describe())
        
        # Buy when RSI is below 10
        self.buy_sig = self.rsi < self.rsi_1
        self.sell_sig = self.rsi > self.rsi_2
        
        self.RSIsignal = self.buy_sig * 1 - self.sell_sig * 0.8
        
        self.entry_size = self.RSIsignal * self.entry_size_ratio
        
        self.set_signal(entry_size = self.entry_size)
        
        self.set_trailing_sl(8)
        self.set_atr_periods(50)